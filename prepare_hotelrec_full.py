"""Index all HotelRec reviews on disk, then export a bounded activity-based sample.

The sample uses the whole archive, but favors popular hotels and active users.
It is suitable for simulated cold-start experiments, not population estimates.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
import zipfile

try:
    import orjson
except ImportError:
    orjson = None

from prepare_hotelrec import SOURCE_URL, INVALID_AUTHORS, parse_review

SOURCE_MD5 = "671e0290514f8411fe780fdea236783a"


def load_review(line):
    if orjson is not None:
        try:
            return orjson.loads(line)
        except orjson.JSONDecodeError:
            pass  # Standard decoder also accepts NaN; parse_review rejects invalid ratings.
    return json.loads(line)


def archive_hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def connect(database):
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def sanitize_index(database):
    """Remove known missing-ID placeholders from a completed derived cache atomically."""
    connection = connect(database)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        if not row:
            raise ValueError("HotelRec full index is incomplete")
        source = json.loads(row[0])
        policy = sorted(INVALID_AUTHORS)
        if source.get("invalid_author_policy") == policy:
            return source
        # Discover actual spellings with an index-only pass; also catches legacy case variants.
        aliases = [row[0] for row in connection.execute("SELECT user_id FROM user_counts")
                   if row[0].strip().lower() in INVALID_AUTHORS]
        excluded = {}
        with connection:
            for author in aliases:
                count, pairs = connection.execute("""SELECT COUNT(*),COUNT(DISTINCT item_id)
                    FROM reviews WHERE user_id=?""", (author,)).fetchone()
                excluded[author] = {"reviews": count, "unique_pairs": pairs}
                connection.execute("DELETE FROM reviews WHERE user_id=?", (author,))
                connection.execute("DELETE FROM user_counts WHERE user_id=?", (author,))
            removed = sum(row["reviews"] for row in excluded.values())
            if removed:
                connection.execute("DROP TABLE hotel_counts")
                connection.execute("""CREATE TABLE hotel_counts AS
                    SELECT item_id,COUNT(DISTINCT user_id) AS n FROM reviews GROUP BY item_id""")
                stats = connection.execute("""SELECT COUNT(*),SUM(n),MAX(n),
                    SUM(n=1),SUM(n>=5),SUM(n>=20),SUM(n>=30) FROM user_counts""").fetchone()
                source["global_counts"] = dict(zip(("users", "unique_pairs", "max_user_degree", "users_with_one",
                                                    "users_ge5", "users_ge20", "users_ge30"), stats))
                source["hotels"], source["max_hotel_degree"] = connection.execute(
                    "SELECT COUNT(*),MAX(n) FROM hotel_counts").fetchone()
                source["reviews_kept"] -= removed
                source["invalid_reviews_skipped"] += removed
            source["invalid_author_policy"] = policy
            source["excluded_author_placeholders"] = excluded
            connection.execute("UPDATE metadata SET value=? WHERE key='source'", (json.dumps(source),))
        print(f"Excluded missing-ID placeholders: {excluded}", flush=True)
        return source
    finally:
        connection.close()


def index_archive(archive, database, expected_md5=SOURCE_MD5):
    archive, database = Path(archive), Path(database)
    print("Verifying full archive checksum...", flush=True)
    checksums = archive_hashes(archive)
    if checksums["md5"] != expected_md5:
        raise ValueError("Full HotelRec archive MD5 does not match source checksum")
    if database.exists():
        connection = connect(database)
        try:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='metadata'").fetchone():
                raise ValueError("Incomplete HotelRec index")
            row = connection.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
            if row and json.loads(row[0])["archive_checksums"] == checksums:
                print("Reusing completed HotelRec disk index", flush=True)
                return sanitize_index(database)
        finally:
            connection.close()
        raise ValueError("Incomplete or incompatible HotelRec index; preserve it and use another database path")

    database.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    connection = connect(database)
    try:
        connection.executescript("""
            CREATE TABLE reviews(user_id TEXT NOT NULL, item_id INTEGER NOT NULL,
                                 rating REAL NOT NULL, timestamp INTEGER NOT NULL);
            CREATE TABLE hotels(item_id INTEGER PRIMARY KEY, description TEXT NOT NULL);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        scanned, kept, skipped = 0, 0, 0
        batch, hotels = [], {}

        def flush():
            with connection:
                connection.executemany("INSERT INTO reviews VALUES (?,?,?,?)", batch)
                connection.executemany("INSERT OR IGNORE INTO hotels VALUES (?,?)", hotels.items())
            batch.clear()
            hotels.clear()

        with zipfile.ZipFile(archive) as source:
            with io.BufferedReader(source.open("HotelRec.txt"), 1024 * 1024) as stream:
                for scanned, line in enumerate(stream, 1):
                    parsed = parse_review(load_review(line))
                    if parsed is None:
                        skipped += 1
                    else:
                        user_id, item_id, rating, timestamp, description = parsed
                        item_id = int(item_id[1:])
                        batch.append((user_id, item_id, rating, timestamp))
                        hotels[item_id] = description
                        kept += 1
                    if scanned % 10000 == 0:
                        flush()
                    if scanned % 1000000 == 0:
                        elapsed = time.monotonic() - started
                        print(f"Indexed {scanned:,} reviews; valid {kept:,}; {scanned / elapsed:,.0f} rows/s", flush=True)
        flush()
        if not kept:
            raise ValueError("HotelRec archive contains no valid reviews")
        # Build indexes after append-only ingestion to avoid 50M random index updates.
        print("Building item/user disk index...", flush=True)
        connection.execute("CREATE INDEX reviews_item_user ON reviews(item_id,user_id)")
        connection.commit()
        print("Building user/item disk index...", flush=True)
        connection.execute("CREATE INDEX reviews_user_item ON reviews(user_id,item_id)")
        connection.commit()
        print("Counting distinct interactions per user/hotel...", flush=True)
        connection.execute("""CREATE TABLE user_counts AS
            SELECT user_id, COUNT(DISTINCT item_id) AS n FROM reviews GROUP BY user_id""")
        connection.execute("""CREATE TABLE hotel_counts AS
            SELECT item_id, COUNT(DISTINCT user_id) AS n FROM reviews GROUP BY item_id""")
        user_stats = connection.execute("""SELECT COUNT(*), SUM(n), MAX(n),
            SUM(n=1), SUM(n>=5), SUM(n>=20), SUM(n>=30) FROM user_counts""").fetchone()
        hotel_stats = connection.execute("SELECT COUNT(*), MAX(n) FROM hotel_counts").fetchone()
        meta = {
            "source_url": SOURCE_URL,
            "source_repository": "https://github.com/diegoantognini/HotelRec",
            "archive_checksums": checksums, "archive_bytes": archive.stat().st_size,
            "json_decoder": f"orjson {orjson.__version__}" if orjson is not None else "stdlib json",
            "integrity": "Source MD5 and complete HotelRec.txt ZIP CRC verified",
            "reviews_scanned": scanned, "reviews_kept": kept, "invalid_reviews_skipped": skipped,
            "global_counts": dict(zip(("users", "unique_pairs", "max_user_degree", "users_with_one",
                                       "users_ge5", "users_ge20", "users_ge30"), user_stats)),
            "hotels": hotel_stats[0], "max_hotel_degree": hotel_stats[1],
            "index_seconds": round(time.monotonic() - started, 1),
            "content": "Hotel name/location from URL only; no review text or ratings",
            "demographics": "Unavailable; unknown fallback only", "usage": "Academic research only",
            "invalid_author_policy": sorted(INVALID_AUTHORS),
            "excluded_author_placeholders": {},
        }
        with connection:
            connection.execute("INSERT INTO metadata VALUES ('source',?)", (json.dumps(meta),))
        print(json.dumps(meta, indent=2), flush=True)
        return meta
    finally:
        connection.close()


def export_sample(database, folder, max_users=1000, max_items=2000):
    if min(max_users, max_items) < 1:
        raise ValueError("Sample limits must be positive")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    targets = [folder / name for name in ("reviews_slim.csv.gz", "hotels.csv.gz", "source.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("HotelRec sample already exists; use another output folder")
    sanitize_index(database)
    connection = connect(database)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        if row is None:
            raise ValueError("HotelRec full index is incomplete")
        source = json.loads(row[0])
        print("Selecting popular hotels across full archive...", flush=True)
        connection.execute("CREATE TEMP TABLE selected_hotels(item_id INTEGER PRIMARY KEY)")
        connection.execute("""INSERT INTO selected_hotels
            SELECT item_id FROM hotel_counts ORDER BY n DESC, 'd'||item_id LIMIT ?""", (max_items,))
        print("Selecting active users within the bounded catalog...", flush=True)
        connection.execute("CREATE TEMP TABLE selected_users(user_id TEXT PRIMARY KEY, n INTEGER)")
        connection.execute("""INSERT INTO selected_users
            SELECT r.user_id, COUNT(DISTINCT r.item_id) AS n
            FROM selected_hotels h JOIN reviews r ON r.item_id=h.item_id
            GROUP BY r.user_id ORDER BY n DESC,r.user_id LIMIT ?""", (max_users,))
        selected = connection.execute("SELECT COUNT(*),MIN(n),MAX(n),SUM(n) FROM selected_users").fetchone()
        print(f"Selected user count/min/max degree/pairs: {selected}", flush=True)
        temporary = targets[0].with_suffix(".tmp")
        try:
            with gzip.open(temporary, "wt", encoding="utf8", newline="", compresslevel=1) as output:
                writer = csv.writer(output)
                writer.writerow(["user_id", "item_id", "rating", "timestamp"])
                query = """SELECT r.user_id, 'd'||r.item_id, r.rating, r.timestamp
                    FROM selected_users u JOIN reviews r ON r.user_id=u.user_id
                    JOIN selected_hotels h ON h.item_id=r.item_id
                    ORDER BY r.timestamp,r.rowid"""
                writer.writerows(connection.execute(query))
            with gzip.open(targets[1], "wt", encoding="utf8", newline="", compresslevel=1) as output:
                writer = csv.writer(output)
                writer.writerow(["item_id", "description"])
                writer.writerows(connection.execute("""SELECT 'd'||h.item_id,h.description
                    FROM hotels h JOIN selected_hotels s ON s.item_id=h.item_id ORDER BY h.item_id"""))
            temporary.replace(targets[0])
        finally:
            temporary.unlink(missing_ok=True)
        checksums = {}
        for path in targets[:2]:
            with path.open("rb") as handle:
                checksums[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        source.update({
            "sampling": "Full archive: most reviewed hotels, then most active users within selected catalog; ties by string ID",
            "sampling_limits": {"max_users": max_users, "max_items": max_items},
            "sample_before_core": dict(zip(("users", "min_user_degree", "max_user_degree", "unique_pairs"), selected)),
            "prepared_sha256": checksums,
        })
        targets[2].write_text(json.dumps(source, indent=2), encoding="utf8")
        print(f"Exported sample to {folder}", flush=True)
        return source
    finally:
        connection.close()


def profile_sample(folder, output, seeds=(42, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    """Check split coverage without training models or constructing candidate matrices."""
    from collections import Counter
    import pandas as pd
    from config import (TEST_RATIO, COLD_USER_RATIO, MAX_COLD_N, COLD_ITEM_RATIO,
                        WARM_MIN_N, COLD_MAX_N, VAL_RATIO)
    from data.loader import load_hotelrec
    from data.preprocessor import simulate_cold_start, assign_groups, split_validation
    from evaluation.ranking import RELEVANCE_THRESHOLD

    ratings, hotels, users = load_hotelrec(folder)
    user_degree = ratings.groupby("user_id").size()
    item_degree = ratings.groupby("item_id").size()
    rows = []
    for seed in seeds:
        train, held = simulate_cold_start(
            ratings, test_ratio=TEST_RATIO, cold_user_ratio=COLD_USER_RATIO,
            max_cold_n=MAX_COLD_N, cold_item_ratio=COLD_ITEM_RATIO, seed=seed)
        held = split_validation(assign_groups(held, train, warm_min=WARM_MIN_N,
                                              cold_max=COLD_MAX_N), val_ratio=VAL_RATIO, seed=seed)
        test = held[held["split"] == "test"]
        val = held[held["split"] == "val"]
        eligible = test.loc[test["rating"] >= RELEVANCE_THRESHOLD, "user_id"].unique()
        n_train = train.groupby("user_id").size().reindex(eligible, fill_value=0)
        rank_groups = Counter("warm" if n >= WARM_MIN_N else "cool" if n >= COLD_MAX_N else "cold"
                              for n in n_train)
        seen = pd.concat([train[["user_id", "item_id"]], val[["user_id", "item_id"]]]).drop_duplicates()
        candidate_pairs = len(eligible) * len(hotels) - int(seen["user_id"].isin(eligible).sum())
        train_pairs = set(zip(train["user_id"], train["item_id"]))
        assert not train_pairs.intersection(zip(held["user_id"], held["item_id"]))
        rows.append({
            "seed": seed, "train": len(train), "validation": len(val), "test": len(test),
            "ranking_users": len(eligible), "candidate_pairs": candidate_pairs,
            **{f"rating_{group}": int(test["group"].eq(group).sum())
               for group in ("cold_item", "cold", "cool", "warm")},
            **{f"ranking_{group}": rank_groups[group] for group in ("cold", "cool", "warm")},
        })
    profile = {
        "dataset_counts": {"users": len(users), "items": len(hotels), "ratings": len(ratings)},
        "user_degree": user_degree.describe().to_dict(), "item_degree": item_degree.describe().to_dict(),
        "rating_distribution": ratings["rating"].value_counts().sort_index().to_dict(),
        "split_config": {"test_ratio": TEST_RATIO, "val_ratio": VAL_RATIO,
                         "cold_user_ratio": COLD_USER_RATIO, "max_cold_n": MAX_COLD_N,
                         "cold_item_ratio": COLD_ITEM_RATIO, "warm_min_n": WARM_MIN_N,
                         "cold_max_n": COLD_MAX_N},
        "split_profiles": rows,
        "all_seeds_have_each_ranking_group": all(row[f"ranking_{g}"] > 0 for row in rows
                                                 for g in ("cold", "cool", "warm")),
        "models_trained": False,
        "dataset_details": ratings.attrs["dataset_details"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf8")
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps({k: v for k, v in profile.items() if k != "dataset_details"}, indent=2), flush=True)
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("datasets_raw/hotelrec/Full_HotelRec.zip"))
    parser.add_argument("--database", type=Path, default=Path("datasets_raw/hotelrec/full.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("datasets_raw/hotelrec_full_sample"))
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--max-items", type=int, default=2000)
    args = parser.parse_args()
    index_archive(args.archive, args.database)
    export_sample(args.database, args.output, args.max_users, args.max_items)
