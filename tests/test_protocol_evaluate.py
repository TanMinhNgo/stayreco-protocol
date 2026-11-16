import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.evaluate import (candidates, dynamic_method, evaluate_methods, mostpop_method,
                               ndcg_recall_at_k, single_method)


@pytest.mark.parametrize("rank,expected", [(1, 1.0), (3, 0.5), (10, 0.2891)])
def test_ndcg_known_ranks(rank, expected):
    ranked = [f"h{i}" for i in range(1, 12)]
    ndcg, rec = ndcg_recall_at_k(ranked, f"h{rank}")
    assert ndcg == pytest.approx(expected, abs=1e-3) and rec == 1.0


def test_outside_topk_is_zero():
    assert ndcg_recall_at_k([f"h{i}" for i in range(1, 12)], "h11") == (0.0, 0.0)


def test_candidates_exclude_seen_and_other_region():
    by_region = {"uk": ["a", "b", "c"], "us": ["d"]}
    assert candidates(by_region, "uk", {"b"}) == ["a", "c"]
    assert candidates(by_region, "mars", set()) == []


class _Const:
    def __init__(self, value=None, table=None):
        self.value, self.table = value, table

    def score(self, users, items):
        if self.table is not None:
            return np.array([self.table.get(i, 0.0) for i in items])
        return np.full(len(items), self.value)


def _fixture():
    hotels = pd.DataFrame({"item_id": ["a", "b", "c", "d", "x"],
                           "region": ["r", "r", "r", "r", None]})
    history = pd.DataFrame({"user_id": ["u"], "item_id": ["a"]})
    bookings = pd.DataFrame({"user_id": ["u", "u", "u"], "item_id": ["b", "x", "a"]})
    return hotels, history, bookings


def test_evaluate_drops_booking_without_region_or_already_seen():
    hotels, history, bookings = _fixture()
    src = {"cf": _Const(1.0), "content": _Const(1.0), "demo": _Const(1.0), "pop": _Const(1.0)}
    out = evaluate_methods({"m": single_method("cf")}, bookings, history, hotels, src)
    assert len(out["m"]) == 1 and out["m"].attrs["dropped"] == 2


def test_all_equal_scores_no_nan():
    hotels, history, bookings = _fixture()
    src = {"cf": _Const(2.0), "content": _Const(2.0), "demo": _Const(2.0), "pop": _Const(0.0)}
    out = evaluate_methods({"dyn": dynamic_method(10, 3), "pop": mostpop_method()},
                           bookings, history, hotels, src)
    for df in out.values():
        assert not df[["ndcg", "recall"]].isna().any().any()


def test_better_source_scores_higher_ndcg():
    hotels, history, bookings = _fixture()
    bookings = bookings.iloc[:1]  # đặt "b"
    good = _Const(table={"b": 5.0, "c": 1.0, "d": 0.0})
    bad = _Const(table={"b": 0.0, "c": 1.0, "d": 5.0})
    src = {"cf": good, "content": bad, "demo": bad, "pop": bad}
    out = evaluate_methods({"cf": single_method("cf"), "c": single_method("c")},
                           bookings, history, hotels, src)
    assert out["cf"]["ndcg"].iat[0] == 1.0 and out["c"]["ndcg"].iat[0] < 1.0
    assert out["cf"]["group"].iat[0] == "1-4" and bool(out["cf"]["is_new_hotel"].iat[0]) is True


def test_results_report_candidate_pool_size():
    hotels, history, bookings = _fixture()
    src = {"cf": _Const(1.0), "content": _Const(1.0), "demo": _Const(1.0), "pop": _Const(1.0)}
    df = evaluate_methods({"m": single_method("cf")}, bookings.iloc[:1], history, hotels, src)["m"]
    assert df["pool"].iat[0] == 3  # vùng r có a,b,c,d; user đã có a


def test_pool_excludes_hotels_not_yet_listed_at_booking_time():
    hotels = pd.DataFrame({"item_id": ["a", "b", "c", "d"], "region": ["r"] * 4,
                           "first_ts": [0, 0, 0, 500]})          # d chỉ có review đầu tiên sau mốc 100
    history = pd.DataFrame({"user_id": ["u"], "item_id": ["a"]})
    bookings = pd.DataFrame({"user_id": ["u"], "item_id": ["b"], "timestamp": [100]})
    src = {"cf": _Const(1.0), "content": _Const(1.0), "demo": _Const(1.0), "pop": _Const(1.0)}
    df = evaluate_methods({"m": single_method("cf")}, bookings, history, hotels, src)["m"]
    assert df["pool"].iat[0] == 2  # chỉ b, c (a đã có trong lịch sử, d chưa lên sàn)


def test_drop_reasons_reported_by_group():
    hotels, history, bookings = _fixture()    # booking: b (ok), x (không vùng), a (đã ở lịch sử)
    src = {"cf": _Const(1.0), "content": _Const(1.0), "demo": _Const(1.0), "pop": _Const(1.0)}
    df = evaluate_methods({"m": single_method("cf")}, bookings, history, hotels, src)["m"]
    d = df.attrs["dropped_detail"]
    assert sorted(d["reason"]) == ["no_region", "repeat_stay"] and set(d["group"]) == {"1-4"}
