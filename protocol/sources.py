"""Các nguồn điểm của giao thức StayReco. Mỗi nguồn: fit(...) -> self, score(users, items) -> np.ndarray."""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class BPR:
    """CF cho implicit feedback: xếp hotel đã đặt lên trên hotel chưa đặt (Rendle 2009)."""

    def __init__(self, seed=0, n_factors=32, n_epochs=30, lr=0.05, reg=1e-4, batch=4096):
        self.seed, self.f, self.epochs, self.lr, self.reg, self.batch = seed, n_factors, n_epochs, lr, reg, batch

    def fit(self, train):
        # một luồng: tổng nhiều luồng đổi thứ tự cộng số thực, làm cùng seed ra kết quả khác nhau
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return self._fit(train)
        finally:
            torch.set_num_threads(prev)

    def _fit(self, train):
        torch.manual_seed(self.seed)
        users, items = train["user_id"].astype(str), train["item_id"].astype(str)
        self.u_idx = {u: i for i, u in enumerate(users.unique())}
        self.i_idx = {i: j for j, i in enumerate(items.unique())}
        u = torch.tensor(users.map(self.u_idx).to_numpy())
        i = torch.tensor(items.map(self.i_idx).to_numpy())
        U = torch.nn.Parameter(0.1 * torch.randn(len(self.u_idx), self.f))
        I = torch.nn.Parameter(0.1 * torch.randn(len(self.i_idx), self.f))
        opt = torch.optim.Adam([U, I], lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(len(u), generator=g)
            for s in range(0, len(u), self.batch):
                b = order[s:s + self.batch]
                neg = torch.randint(len(self.i_idx), (len(b),), generator=g)
                eu = U[u[b]]
                diff = (eu * (I[i[b]] - I[neg])).sum(1)
                # shortcut: negatives drawn uniformly may include true positives; fine for a baseline
                loss = -torch.nn.functional.logsigmoid(diff).mean() + self.reg * eu.pow(2).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.U, self.I = U.detach().numpy(), I.detach().numpy()
        return self

    def score(self, users, items):
        ui = np.array([self.u_idx.get(str(x), -1) for x in users])
        ii = np.array([self.i_idx.get(str(x), -1) for x in items])
        out = np.full(len(ui), np.nan)
        ok = (ui >= 0) & (ii >= 0)
        out[ok] = (self.U[ui[ok]] * self.I[ii[ok]]).sum(1)
        return out


class ProfileContent:
    """Content: cosine giữa hồ sơ user (trung bình vector TF-IDF hotel đã đặt) và vector hotel."""

    def fit(self, hotels, train):
        self.idx = {str(i): j for j, i in enumerate(hotels["item_id"])}
        self.X = normalize(TfidfVectorizer().fit_transform(hotels["description"].fillna("")))
        hist = train.groupby(train["user_id"].astype(str))["item_id"].apply(list)
        self.profile = {}
        for u, its in hist.items():
            rows = [self.idx[str(i)] for i in its if str(i) in self.idx]
            if rows:
                p = np.asarray(self.X[rows].mean(axis=0)).ravel()
                n = np.linalg.norm(p)
                if n > 0:
                    self.profile[u] = p / n
        return self

    def score(self, users, items):
        out = np.zeros(len(users))
        users = np.asarray(users, dtype=str)
        for u in np.unique(users):  # một lần nhân ma trận thưa cho mỗi user
            p = self.profile.get(u)
            if p is None:
                continue
            pos = np.flatnonzero(users == u)
            rows = np.array([self.idx.get(str(items[k]), -1) for k in pos])
            ok = rows >= 0
            out[pos[ok]] = self.X[rows[ok]].dot(p)
        return out


class SegmentPopularity:
    """Demographic: số lần hotel được đặt trong cùng phân khúc của user (Pazzani 1999)."""

    def fit(self, train, segment_of):
        self.seg = {str(u): s for u, s in segment_of.items()}
        counts = {}
        for u, i in zip(train["user_id"].astype(str), train["item_id"].astype(str)):
            s = self.seg.get(u)
            if s is not None:
                counts[(s, i)] = counts.get((s, i), 0) + 1
        self.counts = counts
        return self

    def score(self, users, items):
        out = np.zeros(len(users))
        for k, (u, i) in enumerate(zip(users, items)):
            s = self.seg.get(str(u))
            if s is not None:
                out[k] = self.counts.get((s, str(i)), 0)
        return out


class _Pop:
    def __init__(self, counts):
        self.counts = counts

    def score(self, users, items):
        return np.array([float(self.counts.get(str(i), 0)) for i in items])


def hotel_popularity(train):
    """MostPop: số booking của hotel trong train (xếp hạng trong vùng ứng viên)."""
    return _Pop(train["item_id"].astype(str).value_counts().to_dict())
