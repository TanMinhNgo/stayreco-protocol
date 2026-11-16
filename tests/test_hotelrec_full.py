"""Full-file selection must include late reviews, count unique pairs and preserve IDs."""

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_hotelrec
from prepare_hotelrec_full import index_archive, export_sample, profile_sample, load_review, sanitize_index
from prepare_hotelrec import parse_review
from test_hotelrec import archive, review


def test_fast_decoder_preserves_standard_json_behavior():
    assert load_review(b'{"author":"u1","rating":4}') == {"author": "u1", "rating": 4}
    assert math.isnan(load_review(b'{"rating":NaN}')["rating"])
    with pytest.raises(ValueError):
        load_review(b'{invalid}')
    assert parse_review(review("/undefined", 1)) is None
    assert parse_review(review("undefined", 1)) is None


def test_legacy_cache_removes_shared_placeholder_and_recounts(tmp_path):
    path, database = tmp_path / "full.zip", tmp_path / "index.sqlite"
    path.write_bytes(archive([review("real", 1)]).getvalue())
    source = index_archive(path, database, expected_md5=hashlib.md5(path.read_bytes()).hexdigest())
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO reviews VALUES ('/undefined',1,5,0)")
        connection.execute("INSERT INTO user_counts VALUES ('/undefined',1)")
        connection.execute("UPDATE hotel_counts SET n=n+1 WHERE item_id=1")
        source["reviews_scanned"] += 1
        source["reviews_kept"] += 1
        source.pop("invalid_author_policy")
        connection.execute("UPDATE metadata SET value=? WHERE key='source'", (json.dumps(source),))
    cleaned = sanitize_index(database)
    assert cleaned["reviews_scanned"] == 2
    assert cleaned["reviews_kept"] == 1 and cleaned["invalid_reviews_skipped"] == 1
    assert cleaned["global_counts"]["users"] == 1 and cleaned["global_counts"]["unique_pairs"] == 1
    assert cleaned["excluded_author_placeholders"] == {"/undefined": {"reviews": 1, "unique_pairs": 1}}
    assert sanitize_index(database) == cleaned
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT user_id FROM reviews").fetchall() == [("real",)]
        assert connection.execute("SELECT n FROM hotel_counts").fetchone()[0] == 1


def test_full_archive_sample_uses_late_history_and_latest_rating(tmp_path):
    rows = [review("u1", 1), review("u2", 1), review("u3", 2), review("u1", 2),
            review("u2", 2), review("u3", 1), review("u1", 3),
            review("u2", 1, rating=2, date="2020-01-01T00:00:00"), review("invalid", 3, rating=0)]
    path, database, sample = tmp_path / "full.zip", tmp_path / "index.sqlite", tmp_path / "sample"
    path.write_bytes(archive(rows).getvalue())
    expected = hashlib.md5(path.read_bytes()).hexdigest()
    source = index_archive(path, database, expected_md5=expected)
    assert source["reviews_scanned"] == 9 and source["reviews_kept"] == 8
    assert source["global_counts"]["unique_pairs"] == 7
    assert source["global_counts"]["users"] == 3
    assert index_archive(path, database, expected_md5=expected) == source
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 8
    meta = export_sample(database, sample, max_users=2, max_items=2)
    assert meta["sample_before_core"]["unique_pairs"] == 4
    ratings, hotels, users = load_hotelrec(sample, min_user=2, min_item=2)
    assert set(users["user_id"]) == {"u1", "u2"}
    assert set(hotels["item_id"]) == {"d1", "d2"}
    assert len(ratings) == 4
    assert ratings.loc[(ratings.user_id == "u2") & (ratings.item_id == "d1"), "rating"].item() == 2
    with pytest.raises(FileExistsError):
        export_sample(database, sample)


def test_full_archive_checksum_and_incomplete_index_are_rejected(tmp_path):
    path, database = tmp_path / "full.zip", tmp_path / "index.sqlite"
    path.write_bytes(archive([review("u1", 1)]).getvalue())
    with pytest.raises(ValueError, match="MD5"):
        index_archive(path, database, expected_md5="wrong")
    assert not database.exists()
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
    expected = hashlib.md5(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Incomplete"):
        index_archive(path, database, expected_md5=expected)
    with pytest.raises(ValueError, match="incomplete"):
        export_sample(database, tmp_path / "sample")


def test_profile_matches_existing_candidate_builder(tmp_path):
    from evaluation.ranking import build_candidate_frame
    from data.preprocessor import simulate_cold_start, assign_groups, split_validation
    from config import TEST_RATIO, COLD_USER_RATIO, MAX_COLD_N, COLD_ITEM_RATIO, VAL_RATIO

    rows = [review(f"u{u}", i, rating=4 + (u + i) % 2) | {"text": "ignored"}
            for u in range(40) for i in range(1, 41)]
    path, database, sample = tmp_path / "full.zip", tmp_path / "index.sqlite", tmp_path / "sample"
    path.write_bytes(archive(rows).getvalue())
    index_archive(path, database, expected_md5=hashlib.md5(path.read_bytes()).hexdigest())
    export_sample(database, sample)
    output = tmp_path / "profile.json"
    result = profile_sample(sample, output, seeds=[42])
    ratings, hotels, _ = load_hotelrec(sample)
    train, held = simulate_cold_start(ratings, test_ratio=TEST_RATIO, cold_user_ratio=COLD_USER_RATIO,
                                      max_cold_n=MAX_COLD_N, cold_item_ratio=COLD_ITEM_RATIO, seed=42)
    held = split_validation(assign_groups(held, train), val_ratio=VAL_RATIO, seed=42)
    import pandas as pd
    seen = pd.concat([train, held[held["split"] == "val"]])
    frame, relevant = build_candidate_frame(train, held[held["split"] == "test"], hotels["item_id"], df_exclude=seen)
    assert result["split_profiles"][0]["candidate_pairs"] == len(frame)
    assert result["split_profiles"][0]["ranking_users"] == len(relevant)
    assert result["all_seeds_have_each_ranking_group"]
    assert result["models_trained"] is False
    assert json.loads(output.read_text())["dataset_counts"]["ratings"] == 1600
