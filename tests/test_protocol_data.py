import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.data import assign_regions, interaction_counts, temporal_split, user_group


def test_assign_regions_uses_longest_shared_suffix():
    names = [f"Hotel {i} London England" for i in range(5)] + [f"Inn {i} Leeds England" for i in range(5)]
    hotels = pd.DataFrame({"item_id": [str(i) for i in range(10)], "description": names})
    out = assign_regions(hotels, min_count=5)
    assert out.loc[out.item_id == "0", "region"].iat[0] == "london england"
    assert out.loc[out.item_id == "9", "region"].iat[0] == "leeds england"


def test_assign_regions_unsupported_is_none():
    hotels = pd.DataFrame({"item_id": ["a", "b"], "description": ["Solo Hotel Atlantis", "Other Inn Narnia"]})
    assert assign_regions(hotels, min_count=5)["region"].isna().all()


def test_temporal_split_no_leak():
    r = pd.DataFrame({"user_id": list("abcdef"), "item_id": list("123456"),
                      "rating": 5.0, "timestamp": [1, 2, 10, 11, 20, 21]})
    tr, va, te = temporal_split(r, t_val=10, t_test=20)
    assert tr.timestamp.max() < 10 <= va.timestamp.min() and va.timestamp.max() < 20 <= te.timestamp.min()
    assert len(tr) + len(va) + len(te) == len(r)


def test_interaction_counts_one_per_review():
    h = pd.DataFrame({"user_id": ["a", "a", "b"], "item_id": ["1", "2", "1"]})
    assert interaction_counts(h).to_dict() == {"a": 2, "b": 1}


@pytest.mark.parametrize("n,g", [(0, "0"), (1, "1-4"), (4, "1-4"), (5, "5-19"), (19, "5-19"), (20, ">=20")])
def test_user_group_boundaries(n, g):
    assert user_group(n) == g


def test_assign_regions_drops_region_that_ends_up_small():
    # "england" có 6 hotel chia sẻ hậu tố, nhưng 5 hotel được gán "leeds england" nên chỉ còn 1 hotel -> bỏ
    hotels = pd.DataFrame({"item_id": [str(i) for i in range(6)],
                           "description": [f"Inn {i} Leeds England" for i in range(5)] + ["Lone Place England"]})
    out = assign_regions(hotels, min_count=5)
    assert out.loc[out.item_id == "5", "region"].isna().all()
    assert (out.loc[out.item_id != "5", "region"] == "leeds england").all()


def test_segments_exclude_users_without_region(tmp_path):
    from protocol.data import segments_of
    pd.DataFrame({"user_id": ["a", "b"], "home_region": ["uk", None]}).to_csv(
        tmp_path / "users_demographics.csv.gz", index=False)
    assert segments_of(str(tmp_path)) == {"a": "uk"}
    assert segments_of(str(tmp_path / "nope")) == {}
