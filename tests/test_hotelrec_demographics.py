"""Use declared geography from exact usernames; missing/conflicting profiles fall back."""

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_hotelrec
from models.demographic import DemographicFilter
from prepare_hotelrec import prepare
from prepare_hotelrec_demographics import attach_demographics
from test_hotelrec import archive, review


@pytest.fixture
def enriched(tmp_path):
    rows = [review(f"u{u}", item) | {"text": "ignored"} for u in range(1, 9) for item in (1, 2, 3)]
    prepare(archive(rows), tmp_path, len(rows))
    authors = tmp_path / "authors.json.gz"
    profiles = [
        ("u1", "p1", " Hanoi,   Vietnam  "), ("u2", "p2", "Hanoi, Vietnam"),
        ("u3", "p3", None), ("u4", "p4", "Sydney, Australia"),
        ("u4", "different_id", "Sydney, Australia"),
        ("u5", "p5", "Auckland, New Zealand"), ("u5", "p5", "Wellington, New Zealand"),
        ("u6", None, "Invalid profile"), ("u7", "p7", "New York City, New York"),
        ("u8", "p8", "York, United Kingdom"), ("u8", "p8", "York, United Kingdom"),
        ("unrelated", "p9", "Paris, France"),
    ]
    with gzip.open(authors, "wt", encoding="utf8") as output:
        for username, profile_id, hometown in profiles:
            output.write(json.dumps({"username": username, "id": profile_id, "hometown": hometown}) + "\n")
    metadata = attach_demographics(tmp_path, authors)
    return tmp_path, authors, metadata


def test_exact_profiles_and_conflicts_are_loaded_with_real_regions(enriched):
    folder, authors, metadata = enriched
    ratings, _, users = load_hotelrec(folder)
    regions = users.set_index("user_id")["demographic"]
    assert regions["u1"] == regions["u2"] == "vietnam"
    assert regions["u7"] == "new york"  # A state/region, not a claim about nationality.
    assert regions["u8"] == "united kingdom"
    assert regions.loc[["u3", "u4", "u5", "u6"]].isna().all()
    assert metadata["counts"]["users_with_region"] == 4
    assert metadata["counts"]["known_region_groups"] == 3
    assert metadata["counts"]["invalid_matching_profiles"] == 1
    assert metadata["counts"]["match_status"]["ambiguous_username"] == 1
    assert metadata["counts"]["match_status"]["conflicting_hometown"] == 1
    assert metadata["profile_observed_at"] is None
    assert ratings.attrs["dataset_details"]["demographics"]["missing_users"] == 4
    with pytest.raises(FileExistsError):
        attach_demographics(folder, authors)


@pytest.mark.parametrize("failure", ["profiles", "target", "incomplete"])
def test_demographic_provenance_is_verified(enriched, failure):
    folder, _, metadata = enriched
    if failure == "profiles":
        with (folder / "users_demographics.csv.gz").open("ab") as output:
            output.write(b"corrupt")
    elif failure == "target":
        metadata["target_reviews_sha256"] = "wrong_sample"
        (folder / "demographics_source.json").write_text(json.dumps(metadata), encoding="utf8")
    else:
        (folder / "demographics_source.json").unlink()
    with pytest.raises(ValueError, match="demographic"):
        load_hotelrec(folder)


def test_demographic_cold_user_and_missing_profile_fallback():
    train = pd.DataFrame([("u1", "hotel", 5.0), ("u2", "hotel", 5.0), ("u3", "hotel", 1.0)],
                         columns=["user_id", "item_id", "rating"])
    profiles = pd.DataFrame({"user_id": ["u1", "u2", "u3", "new_user", "missing"],
                             "demographic": ["region_a", "region_a", "region_b", "region_a", pd.NA]})
    model = DemographicFilter(group_cols=("demographic",))
    model.fit(train, profiles)
    assert model.predict("new_user", "hotel") > model.predict("u3", "hotel")
    assert "missing" not in model.user_demo
    assert model.predict("missing", "hotel") == model.item_means["hotel"]
    assert model.predict("new_user", "new_hotel") == model.global_avg
    users = np.array(["new_user", "missing", "unknown"])
    items = np.array(["hotel", "hotel", "new_hotel"])
    np.testing.assert_allclose(model.predict_many(users, items),
                               [model.predict(u, i) for u, i in zip(users, items)])


def test_demographic_all_missing_profiles_use_item_fallback():
    train = pd.DataFrame({"user_id": ["u"], "item_id": ["i"], "rating": [4.0]})
    model = DemographicFilter(group_cols=("demographic",))
    model.fit(train, pd.DataFrame({"user_id": ["u"], "demographic": [pd.NA]}))
    assert not model.user_demo and not model.group_ratings
    assert model.predict("u", "i") == model.item_means["i"]
