import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.sources import BPR, ProfileContent, SegmentPopularity, hotel_popularity


def _grouped_train(seed=0):
    rng = np.random.RandomState(seed)
    rows, held = [], []
    for u in range(40):
        own = np.arange(10) if u < 20 else np.arange(10, 20)
        pick = rng.choice(own, 7, replace=False)
        rows += [(f"u{u}", str(i)) for i in pick[:6]]
        held.append((f"u{u}", str(pick[6]), str(rng.choice(np.setdiff1d(np.arange(20), own)))))
    return pd.DataFrame(rows, columns=["user_id", "item_id"]), held


def test_bpr_prefers_own_group_items():
    train, held = _grouped_train()
    m = BPR(seed=0, n_epochs=60).fit(train)
    pos = m.score([u for u, _, _ in held], [p for _, p, _ in held])
    neg = m.score([u for u, _, _ in held], [n for _, _, n in held])
    ok = ~np.isnan(pos) & ~np.isnan(neg)
    assert (pos[ok] > neg[ok]).mean() > 0.8


def test_bpr_unseen_user_or_item_is_nan():
    train, _ = _grouped_train()
    m = BPR(seed=0, n_epochs=2).fit(train)
    assert np.isnan(m.score(["nobody"], ["1"])).all()
    assert np.isnan(m.score(["u0"], ["unseen-item"])).all()


def test_bpr_same_seed_same_scores():
    train, _ = _grouped_train()
    a = BPR(seed=3, n_epochs=3).fit(train).score(["u0", "u1"], ["1", "12"])
    b = BPR(seed=3, n_epochs=3).fit(train).score(["u0", "u1"], ["1", "12"])
    assert np.allclose(a, b)


def test_content_profile_prefers_similar_description():
    hotels = pd.DataFrame({"item_id": ["a", "b", "c"],
                           "description": ["beach resort spa", "beach resort pool", "city business tower"]})
    train = pd.DataFrame({"user_id": ["u"], "item_id": ["a"]})
    m = ProfileContent().fit(hotels, train)
    s = m.score(["u", "u"], ["b", "c"])
    assert s[0] > s[1]
    assert (m.score(["new"], ["b"]) == 0).all()


def test_segment_popularity_counts_within_segment():
    train = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "item_id": ["x", "x", "y"]})
    seg = {"u1": "uk", "u2": "uk", "u3": "us", "q": "uk"}
    s = SegmentPopularity().fit(train, seg).score(["q", "q", "ghost"], ["x", "y", "x"])
    assert list(s) == [2.0, 0.0, 0.0]  # user lạ không có segment -> 0


def test_hotel_popularity_counts_train_bookings():
    train = pd.DataFrame({"user_id": ["a", "b", "c"], "item_id": ["x", "x", "y"]})
    assert list(hotel_popularity(train).score(["z", "z", "z"], ["x", "y", "w"])) == [2.0, 1.0, 0.0]


def test_bpr_bit_identical_on_larger_data():
    rng = np.random.RandomState(0)
    train = pd.DataFrame({"user_id": rng.randint(0, 3000, 120_000).astype(str),
                          "item_id": rng.randint(0, 800, 120_000).astype(str)})
    a = BPR(seed=7, n_epochs=2).fit(train)
    b = BPR(seed=7, n_epochs=2).fit(train)
    assert np.array_equal(a.U, b.U) and np.array_equal(a.I, b.I)
