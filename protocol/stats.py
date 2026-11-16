"""Kiểm định, ước lượng k̂ và điểm giao n* của giao thức StayReco."""

import numpy as np
import pandas as pd
from scipy import stats


def paired_tests(a, b, n_boot=10_000, seed=0):
    """Paired t-test và paired bootstrap (kiểm định dịch chuyển về 0) trên a - b theo từng booking."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(d.mean())
    if np.allclose(d, d[0]):  # mọi chênh lệch như nhau (kể cả toàn 0): không có phương sai để kiểm định
        p = 1.0 if mean == 0 else 0.0
        return {"t_p": p, "boot_p": p, "mean_diff": mean}
    t_p = float(stats.ttest_rel(a, b).pvalue)
    rng = np.random.RandomState(seed)
    centered, hits, done = d - mean, 0, 0
    while done < n_boot:
        m = min(200, n_boot - done)  # chia khối để không dựng ma trận n_boot x n
        means = centered[rng.randint(0, len(d), size=(m, len(d)))].mean(axis=1)
        hits += int((np.abs(means) >= abs(mean)).sum())
        done += m
    return {"t_p": t_p, "boot_p": (hits + 1) / (n_boot + 1), "mean_diff": mean}


def estimate_k(x_by_user, prior_by_user, min_n=30):
    """Moment estimate k̂ = σ̂²/τ̂² (Eq. moments của bài). None nếu τ̂² <= 0.

    x_by_user: {user: mảng tín hiệu x của từng sự kiện}; prior_by_user: {user: ước lượng prior μ_u}.
    Chỉ dùng user có >= min_n sự kiện.
    """
    users = [u for u, x in x_by_user.items() if len(x) >= min_n and u in prior_by_user]
    if len(users) < 2:
        return None
    n = np.array([len(x_by_user[u]) for u in users], dtype=float)
    sigma2 = sum(((np.asarray(x_by_user[u]) - np.mean(x_by_user[u])) ** 2).sum() for u in users) / n.sum()
    gap = np.array([np.mean(x_by_user[u]) - prior_by_user[u] for u in users])
    tau2 = (gap ** 2).mean() - (sigma2 / n).mean()
    return None if tau2 <= 0 else float(sigma2 / tau2)


def crossover_n(ndcg_cf, ndcg_c, n_u, bins, alpha=0.05):
    """Nhóm n_u thấp nhất mà CF only cao hơn Content only có ý nghĩa; trả về cận dưới của nhóm.

    ndcg_cf, ndcg_c, n_u: mảng cùng độ dài theo booking. bins: danh sách (lo, hi) bao gồm hai đầu.
    """
    cf, c, n = (np.asarray(v) for v in (ndcg_cf, ndcg_c, n_u))
    for lo, hi in bins:
        m = (n >= lo) & (n <= hi)
        if m.sum() < 2:
            continue
        r = paired_tests(cf[m], c[m], n_boot=2000)
        if r["mean_diff"] > 0 and r["t_p"] < alpha:
            return lo
    return None


def holm(pvals):
    """Hiệu chỉnh Holm cho nhiều kiểm định; trả p đã hiệu chỉnh theo đúng thứ tự đầu vào."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def user_level(df, col):
    """Gộp các booking của cùng một user thành một giá trị (trung bình `col`), giữ n_u của user."""
    return df.groupby("user_id", as_index=False).agg(**{col: (col, "mean"), "n_u": ("n_u", "first")})


def khat_inputs(history):
    """Tín hiệu x (rating) theo user và prior μ_u = trung bình, trên các hotel của user, của rating
    trung bình LOẠI TRỪ chính rating của user (leave-one-out); hotel chỉ có 1 rating bị bỏ."""
    h = history[["user_id", "item_id", "rating"]].copy()
    g = h.groupby("item_id")["rating"].agg(["sum", "count"])
    h["loo"] = np.where(h["item_id"].map(g["count"]) > 1,
                        (h["item_id"].map(g["sum"]) - h["rating"]) / (h["item_id"].map(g["count"]) - 1), np.nan)
    x = {u: d["rating"].to_numpy() for u, d in h.groupby("user_id")}
    prior = h.groupby("user_id")["loo"].mean().dropna().to_dict()
    return x, prior


def crossover_table(ndcg_cf, ndcg_c, n_u, bins, alpha=0.05):
    """Bảng theo nhóm n_u (CF only trừ Content only) và n*: nhóm thấp nhất CF vượt Content có ý nghĩa
    mà không nhóm cao hơn nào Content vượt CF có ý nghĩa. Truyền mảng theo USER, không theo booking."""
    cf, c, n = (np.asarray(v, dtype=float) for v in (ndcg_cf, ndcg_c, n_u))
    rows = []
    for lo, hi in bins:
        m = (n >= lo) & (n <= hi)
        if m.sum() < 2:
            rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "mean_diff": np.nan, "t_p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "lo": lo})
            continue
        d = cf[m] - c[m]
        r = paired_tests(cf[m], c[m], n_boot=2000)
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "mean_diff": r["mean_diff"], "t_p": r["t_p"],
                     "ci_lo": r["mean_diff"] - 1.96 * se, "ci_hi": r["mean_diff"] + 1.96 * se, "lo": lo})
    t = pd.DataFrame(rows)
    win = (t["mean_diff"] > 0) & (t["t_p"] < alpha)
    lose = (t["mean_diff"] < 0) & (t["t_p"] < alpha)
    star = None
    for i in range(len(t)):
        if win[i] and not lose[i + 1:].any():
            star = int(t.loc[i, "lo"])
            break
    return t.drop(columns="lo"), star
