"""Đánh giá giao thức StayReco: mỗi booking xếp hạng hotel cùng vùng, đo NDCG@10 và Recall@10."""

import numpy as np
import pandas as pd

from protocol.data import interaction_counts, user_group
from protocol.scoring import dynamic_weights, fixed_weights, fuse, zscore

K = 10


def candidates(by_region, region, seen):
    """Hotel cùng vùng mà user chưa có trong lịch sử."""
    return [h for h in by_region.get(region, []) if h not in seen]


def ndcg_recall_at_k(ranked, booked, k=K):
    """Một booking, một hotel liên quan: NDCG@k và Recall@k (0 nếu ngoài top-k)."""
    top = list(ranked[:k])
    if booked not in top:
        return 0.0, 0.0
    return 1.0 / np.log2(top.index(booked) + 2), 1.0


def _z(values):
    """z-score trên giá trị hữu hạn; NaN (không có biểu diễn) -> 0 sau chuẩn hóa."""
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = zscore(v[ok])
    return out


def dynamic_method(k, k_c):
    return lambda n, z: fuse(z["cf"], z["c"], z["d"], dynamic_weights(n, k, k_c))


def fixed_method(alpha, k_c):
    return lambda n, z: fuse(z["cf"], z["c"], z["d"], fixed_weights(alpha, n, k_c))


def single_method(name):
    """name ∈ {"cf", "c", "d"}: một nguồn duy nhất."""
    return lambda n, z: z[name]


def mostpop_method():
    return lambda n, z: z["pop"]


def evaluate_methods(methods, bookings, history, hotels, sources, seed=0, collect=None):
    """Chạy nhiều phương pháp trên cùng các booking, dùng chung điểm đã chuẩn hóa.

    methods: {tên: fn(n_u, z) -> điểm}, z = {"cf","c","d"} đã z-score và "pop" là số đếm thô.
    sources: {"cf","content","demo","pop"}, mỗi nguồn có score(users, items).
    collect: tùy chọn, gọi collect(n_u, z, y) cho mỗi booking (y[j]=1 nếu ứng viên j là hotel được đặt), dùng để huấn luyện FWLS.
    Trả {tên: DataFrame[user_id,item_id,n_u,group,is_new_hotel,pool,ndcg,recall]}; mỗi DataFrame có
    attrs["dropped"] = số booking bị bỏ; attrs["dropped_detail"] = DataFrame(reason, group) với reason ∈ {no_region, repeat_stay, not_in_pool}.
    """
    rng = np.random.RandomState(seed)
    region = hotels.set_index("item_id")["region"].to_dict()
    by_region = {}
    for h, r in region.items():
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            by_region.setdefault(r, []).append(str(h))
    for r in by_region:  # thứ tự cố định nhưng không theo id, để hòa điểm không thiên vị
        rng.shuffle(by_region[r])

    n_of = interaction_counts(history).to_dict()
    seen_of = history.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    known_items = set(history["item_id"].astype(str))
    first_ts = hotels.set_index("item_id")["first_ts"].to_dict() if "first_ts" in hotels else None
    times = bookings["timestamp"] if "timestamp" in bookings else [None] * len(bookings)

    rows = {name: [] for name in methods}
    dropped = []
    for u, booked, t in zip(bookings["user_id"].astype(str), bookings["item_id"].astype(str), times):
        r = region.get(booked)
        cand = candidates(by_region, r, seen_of.get(u, set())) if isinstance(r, str) else []
        if first_ts is not None and t is not None:  # chỉ tính hotel đã lên sàn (có review đầu tiên) tại thời điểm booking
            cand = [h for h in cand if first_ts.get(h, float("inf")) <= t]
        if booked not in cand:
            reason = ("no_region" if not isinstance(r, str)
                      else "repeat_stay" if booked in seen_of.get(u, set()) else "not_in_pool")
            dropped.append((reason, user_group(n_of.get(u, 0))))
            continue
        users = [u] * len(cand)
        z = {"cf": _z(sources["cf"].score(users, cand)),
             "c": _z(sources["content"].score(users, cand)),
             "d": _z(sources["demo"].score(users, cand)),
             "pop": np.asarray(sources["pop"].score(users, cand), dtype=float)}
        n = n_of.get(u, 0)
        if collect is not None:
            collect(n, z, np.array([c == booked for c in cand], dtype=int))
        meta = (u, booked, n, user_group(n), booked not in known_items, len(cand))
        for name, fn in methods.items():
            order = np.argsort(-np.asarray(fn(n, z), dtype=float), kind="stable")
            ndcg, rec = ndcg_recall_at_k([cand[j] for j in order[:K]], booked)
            rows[name].append(meta + (ndcg, rec))

    cols = ["user_id", "item_id", "n_u", "group", "is_new_hotel", "pool", "ndcg", "recall"]
    out = {}
    for name, rs in rows.items():
        df = pd.DataFrame(rs, columns=cols)
        df.attrs["dropped"] = len(dropped)
        df.attrs["dropped_detail"] = pd.DataFrame(dropped, columns=["reason", "group"])
        out[name] = df
    return out
