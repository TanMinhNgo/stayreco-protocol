import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.scoring import dynamic_weights, fixed_weights, fuse, zscore


def test_zscore_constant_is_zero():
    assert np.array_equal(zscore(np.array([3.0, 3.0, 3.0])), np.zeros(3))


def test_zscore_matches_paper_example():
    z = zscore(np.array([4.5, 4.0, 3.5, 4.0]))
    assert z == pytest.approx([1.414, 0.0, -1.414, 0.0], abs=1e-3)


def test_dynamic_weights_match_paper_table():
    assert dynamic_weights(0, 10, 3) == (0.0, 0.0, 1.0)
    assert dynamic_weights(0, 10, 3, declared=True) == (0.0, 1.0, 0.0)
    a, b, g = dynamic_weights(10, 10, 3)
    assert (round(a, 2), round(b, 2), round(g, 2)) == (0.50, 0.38, 0.12)
    assert dynamic_weights(10, 10, 3, declared=True)[:2] == (0.5, 0.5)
    assert round(dynamic_weights(30, 10, 3)[0], 2) == 0.75


@pytest.mark.parametrize("n", [0, 1, 7, 30, 500])
def test_weights_sum_to_one(n):
    assert sum(dynamic_weights(n, 10, 3)) == pytest.approx(1.0)
    assert sum(fixed_weights(0.3, n, 3)) == pytest.approx(1.0)


def test_fixed_alpha_one_is_cf_only():
    assert fixed_weights(1.0, 5, 3) == (1.0, 0.0, 0.0)


def test_fuse_reproduces_paper_example_ranking():
    cf = np.array([4.5, 4.0, 3.5, 4.0])      # A B C D
    c = np.array([0.60, 0.95, 0.70, 0.75])
    d = np.zeros(4)
    s = fuse(zscore(cf), zscore(c), zscore(d), (0.5, 0.5, 0.0))
    assert list(np.argsort(-s)) == [1, 0, 3, 2]  # B > A > D > C


from protocol.scoring import pi_group, rrf, fwls_fit


def test_pi_group_matches_paper_example():
    assert pi_group(0.9, 0.6) == pytest.approx(0.4)


def test_rrf_weighted_ranks():
    # nguồn 1 xếp A > B > C, nguồn 2 xếp C > B > A; trọng số lệch hẳn về nguồn 1
    s = rrf([np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])], (0.9, 0.1), kappa=60)
    assert list(np.argsort(-s)) == [0, 1, 2]
    assert s[0] == pytest.approx(0.9 / 61 + 0.1 / 63)


def test_rrf_drops_source_without_information():
    s = rrf([np.array([3.0, 2.0, 1.0]), np.zeros(3)], (0.5, 0.5), kappa=60)
    assert list(np.argsort(-s)) == [0, 1, 2] and s[0] == pytest.approx(0.5 / 61)


def test_fwls_learns_to_trust_informative_source():
    rng = np.random.RandomState(0)
    n = rng.randint(0, 40, 4000)
    z = rng.normal(size=(4000, 3))
    y = (z[:, 0] + 0.1 * rng.normal(size=4000) > 0.8).astype(int)  # chỉ nguồn CF có tín hiệu
    w = fwls_fit(n, z, y)
    s = w(np.array([5]), np.array([[1.0, 0.0, 0.0]]))[0]
    t = w(np.array([5]), np.array([[0.0, 1.0, 0.0]]))[0]
    assert s > t
