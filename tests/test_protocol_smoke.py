import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_protocol import GROUPS, run


def _sample(path, seed=0):
    rng = np.random.RandomState(seed)
    regions = ["Alpha City", "Beta Town", "Gamma Bay"]
    hotels = pd.DataFrame({"item_id": [str(i) for i in range(45)],
                           "description": [f"Grand Hotel {i} {regions[i % 3]}" for i in range(45)]})
    rows = []
    for u in range(120):
        r = u % 3
        items = rng.choice([i for i in range(45) if i % 3 == r], 12, replace=False)
        for t, i in enumerate(items):
            rows.append((f"u{u}", str(i), float(rng.randint(3, 6)), 1_000_000 + 1000 * t + u))
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"]).to_csv(
        os.path.join(path, "reviews_slim.csv.gz"), index=False)
    hotels.to_csv(os.path.join(path, "hotels.csv.gz"), index=False)


def test_smoke_outputs_and_determinism(tmp_path):
    _sample(str(tmp_path / "s"))
    outs = []
    for tag in ("a", "b"):
        outs.append(run(str(tmp_path / "s"), tag, seed=1, bpr_epochs=3, out_dir=str(tmp_path / "out"),
                        min_n_khat=5, n_boot=200, region_min_count=5))
    for name in ("ndcg", "recall", "params", "stats", "tests"):
        assert (tmp_path / "out" / f"protocol_{name}_a.csv").exists()
    t = pd.read_csv(tmp_path / "out" / "protocol_ndcg_a.csv", index_col=0)
    assert list(t.columns[:6]) == GROUPS
    assert t[GROUPS].stack().dropna().between(0, 1).all()
    pd.testing.assert_frame_equal(outs[0]["ndcg"], outs[1]["ndcg"])
