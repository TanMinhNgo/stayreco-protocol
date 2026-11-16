import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.stats import crossover_n, estimate_k, paired_tests


def test_paired_tests_detect_shift():
    rng = np.random.RandomState(0)
    b = rng.rand(300)
    r = paired_tests(b + 0.1 + rng.normal(0, 0.05, 300), b, n_boot=500)
    assert r["t_p"] < 0.05 and r["boot_p"] < 0.05 and r["mean_diff"] > 0.05


def test_paired_tests_no_difference():
    rng = np.random.RandomState(1)
    a = rng.rand(300)
    r = paired_tests(a, a + rng.normal(0, 0.05, 300), n_boot=500)
    assert r["t_p"] > 0.05 and r["boot_p"] > 0.05


def test_paired_tests_identical_inputs_do_not_crash():
    a = np.array([0.0, 1.0, 0.5])
    assert paired_tests(a, a)["t_p"] == 1.0


def test_estimate_k_known_variances():
    rng = np.random.RandomState(0)
    x, prior = {}, {}
    for u in range(3000):
        mu = rng.normal(3.0, 1.0)
        theta = mu + rng.normal(0, 2.0)  # tau^2 = 4
        x[u] = theta + rng.normal(0, np.sqrt(40.0), 40)  # sigma^2 = 40
        prior[u] = mu
    assert abs(estimate_k(x, prior, min_n=30) - 10) < 3


def test_estimate_k_none_when_tau_nonpositive():
    rng = np.random.RandomState(0)
    x = {u: rng.normal(3, 3, 40) for u in range(50)}  # user không lệch khỏi prior ngoài nhiễu
    prior = {u: float(np.mean(x[u])) for u in x}      # gap = 0 -> tau^2 < 0
    assert estimate_k(x, prior, min_n=30) is None


def test_crossover_none_when_cf_never_wins():
    n = np.repeat([0, 3, 10, 30], 50)
    rng = np.random.RandomState(0)
    c = rng.rand(200)
    assert crossover_n(c - 0.2, c, n, [(0, 0), (1, 4), (5, 19), (20, 10**9)]) is None


def test_crossover_finds_first_winning_bin():
    n = np.repeat([0, 3, 10, 30], 80)
    rng = np.random.RandomState(0)
    c = rng.rand(320)
    cf = c + np.where(n >= 10, 0.2, -0.2) + rng.normal(0, 0.05, 320)
    assert crossover_n(cf, c, n, [(0, 0), (1, 4), (5, 19), (20, 10**9)]) == 5


from protocol.stats import crossover_table, holm, khat_inputs, user_level


def test_holm_adjusts_and_keeps_order():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_user_level_averages_per_user():
    import pandas as pd
    df = pd.DataFrame({"user_id": ["a", "a", "b"], "ndcg": [1.0, 0.0, 0.5], "n_u": [3, 3, 7]})
    out = user_level(df, "ndcg")
    assert out.set_index("user_id")["ndcg"].to_dict() == {"a": 0.5, "b": 0.5}
    assert out.set_index("user_id")["n_u"].to_dict() == {"a": 3, "b": 7}


def test_khat_inputs_prior_excludes_own_rating():
    import pandas as pd
    h = pd.DataFrame({"user_id": ["a", "b", "b", "c"], "item_id": ["x", "x", "y", "y"],
                      "rating": [5.0, 1.0, 3.0, 4.0]})
    x, prior = khat_inputs(h)
    assert prior["a"] == 1.0          # chỉ b chấm x: leave-one-out của a là 1.0
    assert prior["c"] == 3.0          # chỉ b chấm y
    assert prior["b"] == pytest.approx((5.0 + 4.0) / 2)


def test_crossover_requires_no_later_significant_loss():
    n = np.repeat([0, 3, 10, 30], 80)
    rng = np.random.RandomState(0)
    c = rng.rand(320)
    cf = c + np.select([n == 3, n == 30], [0.2, -0.2], 0.0) + rng.normal(0, 0.05, 320)
    table, n_star = crossover_table(cf, c, n, [(0, 0), (1, 4), (5, 19), (20, 10**9)])
    assert n_star is None and list(table.columns[:3]) == ["bin", "n", "mean_diff"]
