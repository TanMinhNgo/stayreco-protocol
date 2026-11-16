"""HotelRec keeps real IDs, valid ratings, bounded cores and held-out text out of features."""

import io
import json
import csv
import gzip
import hashlib
from pathlib import Path
import sys
import zipfile

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_hotelrec
from prepare_hotelrec import prepare, zip_lines


def archive(rows, compression=zipfile.ZIP_DEFLATED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as output:
        output.writestr("HotelRec.txt", b"\n".join(json.dumps(row).encode() for row in rows))
    stream.seek(0)
    return stream


def review(user, item, rating=4, date="2019-01-01T00:00:00"):
    return {"author": user, "hotel_url": f"Hotel_Review-g1-d{item}-Reviews-Hotel_{item}-Rome_Italy.html",
            "rating": rating, "date": date, "text": "HELD_OUT_REVIEW_SECRET" * 5000}


def test_prepare_validates_and_never_uses_review_text(tmp_path):
    rows = [review("real_account", 1), review("", 1), review("unknown", 1),
            review("bad_rating", 1, float("nan")), review("bad_date", 1, date="invalid")]
    meta = prepare(archive(rows), tmp_path, len(rows))
    ratings = pd.read_csv(tmp_path / "reviews_slim.csv.gz")
    hotels = pd.read_csv(tmp_path / "hotels.csv.gz")
    assert meta["reviews_scanned"] == 5
    assert meta["invalid_reviews_skipped"] == 4
    assert ratings["user_id"].tolist() == ["real_account"]
    assert hotels.to_dict("records") == [{"item_id": "d1", "description": "Hotel 1 Rome Italy"}]
    assert "HELD_OUT_REVIEW_SECRET" not in hotels.to_string()
    with pytest.raises(FileExistsError):
        prepare(archive(rows), tmp_path, len(rows))


def test_loader_caps_deduplicates_and_iterates_core(tmp_path):
    rows = [review(user, item) for user in ("u1", "u2", "u3") for item in (1, 2, 3)]
    rows += [review("u1", 1, rating=2, date="2020-01-01T00:00:00"),
             review("lonely", 3), review("lonely", 4)]
    prepare(archive(rows), tmp_path, len(rows))
    ratings, hotels, users = load_hotelrec(tmp_path, min_user=2, min_item=2, max_users=3, max_items=3)
    assert len(users) == 3 and len(hotels) == 3 and len(ratings) == 9
    assert ratings.loc[(ratings.user_id == "u1") & (ratings.item_id == "d1"), "rating"].item() == 2
    assert ratings.groupby("user_id").size().min() >= 2
    assert ratings.groupby("item_id").size().min() >= 2
    assert users["demographic"].isna().all()
    assert ratings.attrs["dataset_details"]["source"]["reviews_scanned"] == 12
    with pytest.raises(ValueError, match="empty"):
        load_hotelrec(tmp_path, min_user=4)
    with (tmp_path / "reviews_slim.csv.gz").open("ab") as output:
        output.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_hotelrec(tmp_path)


def test_prepare_rejects_truncated_or_wrong_archive(tmp_path):
    with pytest.raises(ValueError, match="ZIP header"):
        list(zip_lines(io.BytesIO(b"PK")))
    with pytest.raises(ValueError, match="Expected unencrypted"):
        list(zip_lines(archive([review("u", 1)], zipfile.ZIP_STORED)))
    with pytest.raises(ValueError, match="Expected 2 reviews"):
        prepare(archive([review("u", 1)]), tmp_path, 2)
    assert not (tmp_path / "source.json").exists()
    assert not (tmp_path / "reviews_slim.csv.tmp").exists()


def test_loader_filters_legacy_placeholder_before_applying_source_limits(tmp_path):
    rows = [review(user, item) for user in ("u1", "u2", "u3") for item in (1, 2, 3)]
    meta = prepare(archive(rows), tmp_path, len(rows))
    path = tmp_path / "reviews_slim.csv.gz"
    with gzip.open(path, "at", encoding="utf8", newline="") as output:
        csv.writer(output).writerow(["/undefined", "d1", 5, 0])
    meta["prepared_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["sampling_limits"] = {"max_users": 2, "max_items": 2}
    (tmp_path / "source.json").write_text(json.dumps(meta), encoding="utf8")
    ratings, hotels, users = load_hotelrec(tmp_path, min_user=2, min_item=2)
    assert set(users["user_id"]) == {"u1", "u2"}
    assert set(hotels["item_id"]) == {"d1", "d2"}
    assert len(ratings) == 4
    assert ratings.attrs["dataset_details"]["invalid_author_rows_removed"] == 1


def test_group_plot_handles_missing_cool_ratings(tmp_path, monkeypatch):
    import run_experiment as runner

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(runner, "SUFFIX", "_missing_group")
    groups = ["cold_item", "cold", "warm", "cold (rank)", "cool (rank)", "warm (rank)"]
    frame = pd.DataFrame([{"method": "Popularity", "group": group, "rmse": 0.9,
                           "ndcg@10": 0.05, "n_rows": 10, "n_users": 5} for group in groups])
    runner.plot_by_group(frame, ["Popularity"])
    assert (tmp_path / "results/comparison_missing_group.png").stat().st_size > 0
