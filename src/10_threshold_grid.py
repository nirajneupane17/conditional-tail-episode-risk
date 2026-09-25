"""
10 -- Threshold robustness grid (populates Section 9).

Fixes H=10 and sweeps the tail threshold alpha over {0.90, 0.925, 0.95, 0.975,
0.99} (finer than the paper's three levels), using the SAME full CTER feature set
and cross-asset pooling as src/09. Purpose: confirm the weak-discrimination and
null-incremental findings survive reasonable threshold choices rather than a
single favorable cut. The alpha=0.95/0.975/0.99 rows coincide with the H=10 rows
of the horizon grid.

Output: results/threshold_grid.csv
Run:  python src/10_threshold_grid.py
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from lib import ASSETS, W, BURN, SEED, RESULTS, load, load_stress, ewma_sigma
# reuse identical machinery from the horizon grid
import importlib.util
spec = importlib.util.spec_from_file_location("hg", str(Path(__file__).resolve().parent / "09_horizon_grid.py"))
hg = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg)

H = 10
ALPHAS = [0.90, 0.925, 0.95, 0.975, 0.99]


def main():
    stress = load_stress()[["date", "vix", "nfci", "stlfsi"]]
    sp = load("sp500"); sp["sp5"] = pd.Series(-sp["r"]).rolling(5).sum().values
    sp5full = sp[["date", "sp5"]]
    prep = {}
    for a in ASSETS:
        df = load(a); r = df["r"].values; L = -r
        st = df.merge(stress, on="date", how="left")
        for c in ["vix", "nfci", "stlfsi"]:
            st[c] = st[c].ffill()
        st["vix"] = st["vix"].fillna(st["vix"].median()); st[["nfci", "stlfsi"]] = st[["nfci", "stlfsi"]].fillna(0)
        spl = df.merge(sp5full, on="date", how="left")["sp5"].ffill().fillna(0).values
        prep[a] = (r, L, st, spl, ewma_sigma(r))
    rows = []; t0 = time.time()
    for alpha in ALPHAS:
        p = 1 - alpha; aucs = []; pooled = []
        for a in ASSETS:
            r, L, st, spl, esig = prep[a]; n = len(r)
            q = np.array([np.quantile(L[t - W:t], alpha) if t >= W else np.nan for t in range(n)])
            S, E = hg.targets(L, q, H); X = hg.features(r, L, q, spl, st)
            pt = hg.walk(X, E, H, proba=True); sa = hg.walk(X, S, H, proba=False)
            Xb = np.column_stack([esig, esig * abs(norm.ppf(p)), esig * norm.pdf(norm.ppf(p)) / p])
            sb = hg.walk(Xb, S, H, proba=False)
            ok = np.isfinite(pt) & np.isfinite(E)
            if len(np.unique(E[ok])) > 1:
                aucs.append(roc_auc_score(E[ok], pt[ok]))
            oki = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(S)
            pooled.append(np.abs(S[oki] - sa[oki]) - np.abs(S[oki] - sb[oki]))
        d = np.concatenate(pooled); md, pv = hg.dm(d, lag=H - 1); lo, hi = hg.block_ci(d)
        rows.append(dict(alpha=alpha, level=round(alpha * 100, 1), mean_AUC=round(float(np.mean(aucs)), 3),
                         incr_delta=round(float(md), 6), ci_low=round(float(lo), 6),
                         ci_high=round(float(hi), 6), DM_p=round(float(pv), 3)))
        print(f"  alpha={alpha:5}  AUC={np.mean(aucs):.3f}  delta={md:+.6f} [{lo:+.6f},{hi:+.6f}] p={pv:.3f}  ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "threshold_grid.csv", index=False)
    print("\n", R[["level", "mean_AUC", "incr_delta", "ci_low", "ci_high", "DM_p"]].to_string(index=False))
    print("\nSaved -> results/threshold_grid.csv")


if __name__ == "__main__":
    main()
