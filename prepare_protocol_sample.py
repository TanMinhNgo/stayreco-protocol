"""Xuất mẫu HotelRec cho giao thức StayReco, chọn mẫu CHỈ bằng thông tin có sẵn tại mốc cắt.

Khác `prepare_hotelrec_full.py` (chọn top hotel/user theo số review cả đời, dùng cả dữ liệu tương lai):
  - Hotel: top N theo số review TRƯỚC t_val, cộng một mẫu ngẫu nhiên hotel mới (review đầu tiên sau t_val).
  - User: những user có >=1 review từ t_val trở đi (đúng quần thể có booking để đánh giá), lấy mẫu phân tầng
    theo số review TRƯỚC t_val (0, 1-4, 5-19, >=20) để mỗi nhóm n_u đủ booking test. Phân tầng theo quá khứ,
    không theo kết quả.
  - hotels.csv.gz có cột first_ts (thời điểm review đầu tiên): hotel chỉ vào tập ứng viên từ lúc đó.
shortcut: first_ts là proxy cho "ngày hotel lên sàn" vì HotelRec không có ngày đăng ký hotel.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

STRATA = [("0", 0, 0), ("1-4", 1, 4), ("5-19", 5, 19), (">=20", 20, 10 ** 9)]


def export(database, out, t_val, n_pop, n_new, per_stratum, seed):
    rng = np.random.RandomState(seed)
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    hot = pd.read_sql(
        "SELECT item_id, MIN(timestamp) first_ts, COUNT(*) n_all, SUM(timestamp < ?) n_pre FROM reviews GROUP BY item_id",
        db, params=(t_val,))
    pop = hot.sort_values(["n_pre", "item_id"], ascending=[False, True]).head(n_pop)
    new = hot[(hot["first_ts"] >= t_val) & (hot["n_all"] >= 5)]
    new = new.iloc[rng.permutation(len(new))[:n_new]]
    sel = pd.concat([pop, new])
    db.execute("CREATE TEMP TABLE sel(item_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO sel VALUES (?)", [(int(i),) for i in sel["item_id"]])

    users = pd.read_sql(
        """SELECT r.user_id, SUM(r.timestamp < ?1) n_pre FROM reviews r JOIN sel s ON s.item_id = r.item_id
           GROUP BY r.user_id HAVING SUM(r.timestamp >= ?1) >= 1""", db, params=(t_val,))
    chosen = []
    for _, lo, hi in STRATA:
        pool = users[(users["n_pre"] >= lo) & (users["n_pre"] <= hi)]
        chosen.append(pool.iloc[rng.permutation(len(pool))[:per_stratum]])
    chosen = pd.concat(chosen)
    db.execute("CREATE TEMP TABLE usel(user_id TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO usel VALUES (?)", [(u,) for u in chosen["user_id"]])

    rev = pd.read_sql(
        """SELECT r.user_id, r.item_id, r.rating, r.timestamp FROM reviews r
           JOIN usel u ON u.user_id = r.user_id JOIN sel s ON s.item_id = r.item_id ORDER BY r.timestamp""", db)
    rev = rev.drop_duplicates(["user_id", "item_id"])  # giữ review sớm nhất của mỗi cặp
    desc = pd.read_sql("SELECT item_id, description FROM hotels", db)
    hotels = sel[["item_id", "first_ts"]].merge(desc, on="item_id")
    hotels["is_new_pool"] = hotels["item_id"].isin(new["item_id"])

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rev.to_csv(out / "reviews_slim.csv.gz", index=False)
    hotels[["item_id", "description", "first_ts", "is_new_pool"]].to_csv(out / "hotels.csv.gz", index=False)
    meta = {"database": str(database), "t_val": t_val, "n_pop_hotels": len(pop), "n_new_hotels": len(new),
            "per_stratum": per_stratum, "seed": seed, "users": int(rev["user_id"].nunique()), "reviews": len(rev),
            "stratum_counts": {name: int(((chosen["n_pre"] >= lo) & (chosen["n_pre"] <= hi)).sum()) for name, lo, hi in STRATA},
            "selection": "hotels: top by reviews before t_val + random new hotels; users: random per stratum of pre-t_val reviews among users with a review >= t_val"}
    # prepare_hotelrec_demographics.py kiểm tra checksum này trước khi gắn nhân khẩu học
    with open(out / "reviews_slim.csv.gz", "rb") as fh:
        meta["prepared_sha256"] = {"reviews_slim.csv.gz": hashlib.file_digest(fh, "sha256").hexdigest()}
    (out / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="datasets_raw/hotelrec/full.sqlite")
    p.add_argument("--output", default="datasets_raw/hotelrec_protocol_sample_v2")
    p.add_argument("--t-val", type=int, default=1491004800)  # 2017-04-01
    p.add_argument("--n-pop-hotels", type=int, default=4400)
    p.add_argument("--n-new-hotels", type=int, default=600)
    p.add_argument("--per-stratum", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(json.dumps(export(a.database, a.output, a.t_val, a.n_pop_hotels, a.n_new_hotels, a.per_stratum, a.seed), indent=2))
