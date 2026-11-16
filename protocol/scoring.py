"""Chuẩn hóa điểm theo từng yêu cầu và trọng số α/β/γ của bài StayReco."""

import numpy as np


def zscore(scores):
    """z-score quần thể (ddof=0); σ=0 -> toàn 0, nguồn không có ý kiến."""
    s = np.asarray(scores, dtype=float)
    sd = s.std()
    return np.zeros_like(s) if sd == 0 else (s - s.mean()) / sd


def _split(alpha, n, k_c, declared):
    pi = 1.0 if declared else n / (n + k_c) if n + k_c > 0 else 0.0
    return alpha, (1 - alpha) * pi, (1 - alpha) * (1 - pi)


def dynamic_weights(n, k, k_c, declared=False):
    """(α, β, γ) với α = n/(n+k); phần còn lại chia theo π (Eq. alpha, pi, betagamma)."""
    return _split(n / (n + k), n, k_c, declared)


def fixed_weights(alpha, n, k_c, declared=False):
    """Giống dynamic nhưng α cố định; 1-α chia theo cùng π."""
    return _split(alpha, n, k_c, declared)


def fuse(z_cf, z_c, z_d, weights):
    a, b, g = weights
    return a * np.asarray(z_cf) + b * np.asarray(z_c) + g * np.asarray(z_d)


def pi_group(e_content, e_demo):
    """π_g theo sai số (Eq. bg): trọng số tỉ lệ nghịch với sai số 1 - NDCG@10 của mỗi nguồn."""
    return (1 / e_content) / (1 / e_content + 1 / e_demo)


def rrf(z_list, weights, kappa=60):
    """Reciprocal rank fusion có trọng số (Eq. rrf); nguồn không có thông tin (điểm hằng) bị bỏ."""
    out = np.zeros(len(z_list[0]))
    for z, w in zip(z_list, weights):
        z = np.asarray(z, dtype=float)
        if np.all(z == z[0]):
            continue
        rank = np.empty(len(z))
        rank[np.argsort(-z, kind="stable")] = np.arange(1, len(z) + 1)
        out += w / (kappa + rank)
    return out


def fwls_fit(n, z, y):
    """FWLS logistic (Eq. fwls): w_m(n) = a_m + b_m·log(1+n); trả hàm (n, z) -> điểm tuyến tính.

    z: (N, 3) điểm đã chuẩn hóa của CF, Content, Demographic; y: 1 nếu là hotel được đặt.
    shortcut: hệ số âm không bị chặn (bài chỉ nói báo cáo đường w_m(n) khi âm); nâng cấp nếu cần share α̂(n).
    """
    from sklearn.linear_model import LogisticRegression
    feats = lambda n_, z_: np.hstack([z_, z_ * np.log1p(np.asarray(n_, dtype=float))[:, None]])
    model = LogisticRegression(max_iter=1000).fit(feats(n, z), y)
    return lambda n_, z_: model.decision_function(feats(n_, z_))
