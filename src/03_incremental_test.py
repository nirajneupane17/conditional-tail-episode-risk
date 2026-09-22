"""
03 -- Incremental information test.

Reads the per-observation absolute-error differences (augmented minus baseline)
from results/incremental_deltas.npz, aggregates equal-weight across assets, and
reports, per level, the mean MAE difference, a moving-block bootstrap 95% CI, and
a Diebold-Mariano p-value with a Newey-West (HAC) variance. Negative favours CTER.

Output: results/incremental_test.csv.
Run:  python src/03_incremental_test.py   (after 02_cter_pipeline.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from lib import RESULTS, H, SEED


def dm_pvalue(d, lag=H - 1):
    d = d[np.isfinite(d)]; n = len(d); m = d.mean(); dc = d - m
    var = (dc @ dc) / n
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * (dc[l:] @ dc[:-l]) / n
    se = np.sqrt(var / n)
    return m, se, 2 * (1 - norm.cdf(abs(m / se)))


def block_ci(d, B=2000, bl=20, seed=SEED):
    d = d[np.isfinite(d)]; n = len(d); rng = np.random.default_rng(seed)
    nb = n // bl; means = []
    for _ in range(B):
        starts = rng.integers(0, n - bl, nb)
        idx = np.concatenate([np.arange(s, s + bl) for s in starts])
        means.append(d[idx].mean())
    return np.percentile(means, [2.5, 97.5])


def main():
    z = np.load(RESULTS / "incremental_deltas.npz")
    rows = []
    for lv in [95.0, 97.5, 99.0]:
        d = np.concatenate([z[k] for k in z.files if k.endswith(f"_{lv}")])
        m, se, p = dm_pvalue(d); lo, hi = block_ci(d)
        rows.append(dict(level=f"{lv:g}%", mean_delta_MAE=round(m, 6),
                         ci_low=round(lo, 6), ci_high=round(hi, 6),
                         DM_pvalue=round(p, 3)))
        print(f"  {lv:>5}%  mean D = {m:+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  p = {p:.3f}")
    pd.DataFrame(rows).to_csv(RESULTS / "incremental_test.csv", index=False)
    print("\nSaved -> results/incremental_test.csv")


if __name__ == "__main__":
    main()
