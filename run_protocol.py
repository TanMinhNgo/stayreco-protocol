"""Chạy giao thức đánh giá của bài StayReco trên một mẫu HotelRec.

    conda activate stayreco
    python run_protocol.py --sample datasets_raw/hotelrec_protocol_sample --tag taplus --seed 42

Output trong results/ (hậu tố _<tag>):
    protocol_ndcg / protocol_recall   — hàng = phương pháp, cột = nhóm n_u, hotel mới, toàn bộ
    protocol_params                   — k, k_c, α cố định đã chọn, k̂, n*
    protocol_stats                    — thống kê dữ liệu, mốc cắt, số booking bị bỏ
    protocol_tests                    — p-value từng cặp phương pháp
    run_meta_protocol                 — manifest (commit, thư viện)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from protocol.data import interaction_counts, load_sample, temporal_split
from protocol.evaluate import (dynamic_method, evaluate_methods, fixed_method,
                               mostpop_method, single_method)
from protocol.scoring import dynamic_weights, fuse, fwls_fit, pi_group, rrf
from protocol.data import segments_of, user_group
from protocol.sources import BPR, ProfileContent, SegmentPopularity, hotel_popularity
from protocol.stats import crossover_table, estimate_k, holm, khat_inputs, paired_tests, user_level
from utils import manifest

GRID_K = [5, 10, 25, 50, 100, 200]   # mở rộng lên 100, 200 ngay từ đầu (bài: mở rộng nếu tối ưu ở biên)
GRID_KC = [1, 2, 3, 5, 10]
FIXED_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
GROUPS = ["0", "1-4", "5-19", ">=20", "new", "all"]
N_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 10 ** 9)]


def build_sources(history, hotels, segment_of, seed, epochs):
    return {"cf": BPR(seed=seed, n_epochs=epochs).fit(history),
            "content": ProfileContent().fit(hotels, history),
            "demo": SegmentPopularity().fit(history, segment_of),
            "pop": hotel_popularity(history)}


def _mean(df):
    return float(df["ndcg"].mean()) if len(df) else float("nan")


def summarize(results, col):
    """{tên: df booking} -> bảng trung bình `col` theo nhóm n_u, hotel mới, toàn bộ."""
    rows = {}
    for name, df in results.items():
        r = {g: df.loc[df["group"] == g, col].mean() for g in GROUPS[:4]}
        r["new"] = df.loc[df["is_new_hotel"], col].mean()
        r["all"] = df[col].mean()
        r["n_bookings"] = len(df)
        rows[name] = r
    return pd.DataFrame(rows).T


def _subsample(df, n, seed):
    return df if n is None or len(df) <= n else df.sample(n, random_state=seed)


def run(sample, tag, seed=42, val_frac=0.1, test_frac=0.1, t_val=None, t_test=None,
        max_bookings=None, bpr_epochs=30, out_dir="results", min_n_khat=30, n_boot=10_000,
        region_min_count=30):
    os.makedirs(out_dir, exist_ok=True)
    reviews, hotels = load_sample(sample, region_min_count)
    if t_val is None or t_test is None:
        t_val, t_test = (int(reviews["timestamp"].quantile(q)) for q in (1 - val_frac - test_frac, 1 - test_frac))
    train, val, test = temporal_split(reviews, t_val, t_test)

    segment_of = segments_of(sample)  # user không có vùng -> Demographic không cho điểm (không gộp thành "unknown")

    manifest.write(os.path.join(out_dir, f"run_meta_protocol_{tag}.json"), config={
        "sample": sample, "t_val": t_val, "t_test": t_test, "grid_k": GRID_K, "grid_kc": GRID_KC,
        "fixed_alphas": FIXED_ALPHAS, "bpr_epochs": bpr_epochs, "n_boot": n_boot,
        "max_bookings": max_bookings, "min_n_khat": min_n_khat, "region_min_count": region_min_count}, seeds=[seed])

    # ── Validation: chọn k, k_c, α cố định ────────────────────────────
    val_b = _subsample(val, max_bookings, seed)
    grid = {f"dyn|{k}|{kc}": dynamic_method(k, kc) for k in GRID_K for kc in GRID_KC}
    grid.update({f"fix|{a}|{kc}": fixed_method(a, kc) for a in FIXED_ALPHAS for kc in GRID_KC})
    val_src = build_sources(train, hotels, segment_of, seed, bpr_epochs)
    val_res = evaluate_methods(grid, val_b, train, hotels, val_src, seed)
    val_score = {name: _mean(df) for name, df in val_res.items()}
    best_dyn = max((n for n in val_score if n.startswith("dyn")), key=val_score.get)
    best_fix = max((n for n in val_score if n.startswith("fix")), key=val_score.get)
    _, k, k_c = best_dyn.split("|")
    _, alpha, fix_kc = best_fix.split("|")
    k, k_c, alpha, fix_kc = float(k), float(k_c), float(alpha), float(fix_kc)

    # ── Test: lịch sử = train + validation, đánh giá một lần ──────────
    history = pd.concat([train, val])
    sources = build_sources(history, hotels, segment_of, seed, bpr_epochs)
    final = {"MostPop-City": mostpop_method(), "CF only (BPR)": single_method("cf"),
             "Content only": single_method("c"), "Demographic only": single_method("d"),
             f"Best fixed alpha (a={alpha}, k_c={fix_kc})": fixed_method(alpha, fix_kc),
             f"Dynamic alpha (k={k}, k_c={k_c})": dynamic_method(k, k_c)}
    # Biến thể của bài (mục 4.6 bước 3-4 và Eq. rrf): tham số học trên validation, đánh giá một lần trên test
    rows = []
    single = evaluate_methods({"c": single_method("c"), "d": single_method("d")}, val_b, train, hotels, val_src, seed,
                              collect=lambda n, z, y: rows.append((n, np.column_stack([z["cf"], z["c"], z["d"]]), y)))
    pi_by_group = {g: pi_group(max(1 - single["c"].loc[single["c"]["group"] == g, "ndcg"].mean(), 1e-6),
                               max(1 - single["d"].loc[single["d"]["group"] == g, "ndcg"].mean(), 1e-6))
                   for g in GROUPS[:4] if (single["c"]["group"] == g).any()}
    n_all = np.concatenate([np.full(len(y), n) for n, _, y in rows])
    z_all, y_all = np.vstack([z for _, z, _ in rows]), np.concatenate([y for _, _, y in rows])
    pick = np.random.RandomState(seed).permutation(len(y_all))[:300_000]
    fwls = fwls_fit(n_all[pick], z_all[pick], y_all[pick])

    def pi_g_method(n, z):
        a = n / (n + k)
        pi = pi_by_group.get(user_group(n), 0.5)
        return fuse(z["cf"], z["c"], z["d"], (a, (1 - a) * pi, (1 - a) * (1 - pi)))

    final["Dynamic + pi_g (error-based)"] = pi_g_method
    final["Dynamic + RRF"] = lambda n, z: rrf([z["cf"], z["c"], z["d"]], dynamic_weights(n, k, k_c))
    final["FWLS (logistic)"] = lambda n, z: fwls(np.full(len(z["cf"]), n), np.column_stack([z["cf"], z["c"], z["d"]]))
    test_b = _subsample(test, max_bookings, seed)
    res = evaluate_methods(final, test_b, history, hotels, sources, seed)
    if not len(res[next(iter(res))]):
        raise ValueError(f"no test bookings evaluated (dropped {res[next(iter(res))].attrs['dropped']}); lower --region-min-count or check the sample")
    names = list(res)
    fixed_name, dyn_name = names[4], names[5]

    summarize(res, "ndcg").to_csv(os.path.join(out_dir, f"protocol_ndcg_{tag}.csv"))
    summarize(res, "recall").to_csv(os.path.join(out_dir, f"protocol_recall_{tag}.csv"))

    # ── Kiểm định theo cặp, trên USER (gộp booking của cùng user), hiệu chỉnh Holm ──
    def mask_of(df, g):
        return (df["group"] == g) if g in GROUPS[:4] else (df["is_new_hotel"] if g == "new" else np.ones(len(df), bool))

    tests = []
    for other in [fixed_name, "MostPop-City", "CF only (BPR)", "Content only", "Demographic only"] + names[6:]:
        for g in GROUPS:
            m = mask_of(res[dyn_name], g)
            if not m.any():
                continue
            ua = user_level(res[dyn_name][m], "ndcg").set_index("user_id")["ndcg"]
            ub = user_level(res[other][m], "ndcg").set_index("user_id")["ndcg"].reindex(ua.index)
            if len(ua) < 2:
                continue
            r = paired_tests(ua.to_numpy(), ub.to_numpy(), n_boot=n_boot)
            tests.append({"a": dyn_name, "b": other, "group": g, "n_users": len(ua), "n_bookings": int(m.sum()), **r})
    tests = pd.DataFrame(tests)
    tests["t_p_holm"], tests["boot_p_holm"] = holm(tests["t_p"]), holm(tests["boot_p"])
    tests.to_csv(os.path.join(out_dir, f"protocol_tests_{tag}.csv"), index=False)

    # ── k̂ và n* (prior leave-one-out; n* trên user, kèm bảng từng nhóm) ──
    x_by_user, prior = khat_inputs(history)
    k_hat = estimate_k(x_by_user, prior, min_n=min_n_khat)
    cf_u = user_level(res["CF only (BPR)"], "ndcg").set_index("user_id")
    c_u = user_level(res["Content only"], "ndcg").set_index("user_id").reindex(cf_u.index)
    table, n_star = crossover_table(cf_u["ndcg"].to_numpy(), c_u["ndcg"].to_numpy(), cf_u["n_u"].to_numpy(), N_BINS)
    table.to_csv(os.path.join(out_dir, f"protocol_nstar_{tag}.csv"), index=False)
    pd.Series({**{f"pi_g_{g}": v for g, v in pi_by_group.items()}, "k": k, "k_c": k_c, "fixed_alpha": alpha,
               "fixed_k_c": fix_kc, "k_hat": k_hat, "n_star": n_star, "val_ndcg_dynamic": val_score[best_dyn],
               "val_ndcg_fixed": val_score[best_fix]}).to_csv(
        os.path.join(out_dir, f"protocol_params_{tag}.csv"), header=["value"])

    # ── Booking bị bỏ: lý do x nhóm n_u, so với tổng booking của nhóm ──
    n_hist = interaction_counts(history)
    total = test_b["user_id"].map(n_hist).fillna(0).astype(int).map(user_group).value_counts()
    drops = res[dyn_name].attrs["dropped_detail"].groupby(["group", "reason"]).size().unstack(fill_value=0)
    drops = drops.reindex(GROUPS[:4]).fillna(0).astype(int)
    drops["total_bookings"] = total.reindex(GROUPS[:4]).fillna(0).astype(int)
    drops["drop_rate"] = drops.drop(columns="total_bookings").sum(axis=1) / drops["total_bookings"].replace(0, np.nan)
    drops.to_csv(os.path.join(out_dir, f"protocol_drops_{tag}.csv"))

    # ── Thống kê dữ liệu ──────────────────────────────────────────────
    stamp = lambda t: str(pd.to_datetime(t, unit="s").date())
    stats = {"users": reviews["user_id"].nunique(), "hotels": len(hotels), "reviews": len(reviews),
             "train": len(train), "val": len(val), "test": len(test),
             "date_min": stamp(reviews["timestamp"].min()), "date_max": stamp(reviews["timestamp"].max()),
             "t_val": t_val, "t_val_date": stamp(t_val), "t_test": t_test, "t_test_date": stamp(t_test),
             "hotels_without_region": int(hotels["region"].isna().sum()), "regions": int(hotels["region"].nunique()),
             "val_dropped": val_res[best_dyn].attrs["dropped"], "test_dropped": res[dyn_name].attrs["dropped"],
             "test_evaluated": len(res[dyn_name]),
             "test_pool_median": float(res[dyn_name]["pool"].median()), "test_pool_min": int(res[dyn_name]["pool"].min()),
             "test_share_pool_le_10": float((res[dyn_name]["pool"] <= 10).mean()),
             "users_with_demographics": int((pd.Series(segment_of) != "unknown").sum())}
    stats.update({f"test_bookings_group_{g}": int((res[dyn_name]["group"] == g).sum()) for g in GROUPS[:4]})
    stats["test_bookings_new_hotel"] = int(res[dyn_name]["is_new_hotel"].sum())
    pd.Series(stats).to_csv(os.path.join(out_dir, f"protocol_stats_{tag}.csv"), header=["value"])
    return {"ndcg": summarize(res, "ndcg"), "params": {"k": k, "k_c": k_c, "alpha": alpha, "k_hat": k_hat, "n_star": n_star},
            "stats": stats}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--t-val", type=int)
    p.add_argument("--t-test", type=int)
    p.add_argument("--max-bookings", type=int)
    p.add_argument("--bpr-epochs", type=int, default=30)
    p.add_argument("--min-n-khat", type=int, default=30)
    p.add_argument("--region-min-count", type=int, default=30)
    a = p.parse_args()
    out = run(a.sample, a.tag, a.seed, a.val_frac, a.test_frac, a.t_val, a.t_test,
              a.max_bookings, a.bpr_epochs, min_n_khat=a.min_n_khat, region_min_count=a.region_min_count)
    print(out["ndcg"].round(4).to_string())
    print(json.dumps(out["params"], default=str))
