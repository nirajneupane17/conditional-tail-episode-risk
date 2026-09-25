"""
09 -- Horizon robustness grid (populates Section 9).

Re-runs the two headline results across H in {5,10,20,60} using the SAME full
CTER conditioning set as src/02 (returns, realized vol, drawdown, VIX and its
change, NFCI, STLFSI, S&P systemic proxy, exceedance frequency, run length) and
the SAME cross-asset pooling as the incremental test (src/03), so the H=10 row
reproduces the paper's Table 6. Reports episode-occurrence AUC (mean across
assets) and the pooled incremental severity delta (augmented - EWMA baseline)
with a moving-block bootstrap CI and a Diebold-Mariano p-value.

Output: results/horizon_grid.csv
Run:  python src/09_horizon_grid.py
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import (HistGradientBoostingClassifier as HGBC,
                              HistGradientBoostingRegressor as HGBR)
from sklearn.metrics import roc_auc_score
from lib import ASSETS, LEVELS, W, BURN, SEED, RESULTS, load, load_stress, ewma_sigma

HGB = dict(max_iter=200, max_depth=3, learning_rate=0.06, l2_regularization=1.0,
           early_stopping=False, random_state=SEED)
HORIZONS = [5, 10, 20, 60]


def targets(L, q, H):
    n = len(L); S = np.full(n, np.nan); E = np.full(n, np.nan)
    for t in range(n - H):
        qt = q[t]
        if not np.isfinite(qt):
            continue
        fut = L[t + 1:t + 1 + H]; exc = np.clip(fut - qt, 0, None); over = fut > qt
        best = cur = 0.0
        for e, o in zip(exc, over):
            cur = cur + e if o else 0.0
            best = max(best, cur)
        S[t] = best; E[t] = 1.0 if best > 0 else 0.0
    return S, E


def features(r, L, q, sp5, st):
    n = len(r); s = pd.Series(r)
    rv20 = s.rolling(20).std().values; rv60 = s.rolling(60).std().values
    wealth = np.exp(np.cumsum(r)); rollmax = pd.Series(wealth).rolling(60, min_periods=1).max().values
    dd = (rollmax - wealth) / rollmax
    l5 = pd.Series(L).rolling(5).sum().values; l10 = pd.Series(L).rolling(10).sum().values
    over = (L > np.where(np.isfinite(q), q, np.inf)).astype(float)
    exfreq = pd.Series(over).rolling(60).mean().values
    runlen = np.zeros(n)
    for t in range(1, n):
        runlen[t] = runlen[t - 1] + 1 if over[t] == 1 else 0
    return np.column_stack([L, l5, l10, rv20, rv60, rv20 / rv60, dd, st["vix"],
                            st["vix"].diff(5).values, st["nfci"], st["stlfsi"], sp5, exfreq, runlen])


def walk(X, y, H, proba, refit=378):
    n = len(y); pred = np.full(n, np.nan); first = W + BURN
    for T in range(first, n, refit):
        tr = np.arange(0, T - H); m = np.isfinite(y[tr]) & np.all(np.isfinite(X[tr]), axis=1); tr = tr[m]
        if len(tr) < 300:
            continue
        te = np.arange(T, min(T + refit, n)); te = te[np.all(np.isfinite(X[te]), axis=1)]
        if len(te) == 0:
            continue
        if proba:
            if len(np.unique(y[tr])) < 2:
                continue
            pred[te] = HGBC(**HGB).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            pred[te] = HGBR(**HGB).fit(X[tr], y[tr]).predict(X[te])
    return pred


def dm(d, lag):
    d = d[np.isfinite(d)]; n = len(d); m = d.mean(); dc = d - m
    var = (dc @ dc) / n
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * (dc[l:] @ dc[:-l]) / n
    se = np.sqrt(var / n) if var > 0 else np.nan
    return m, (2 * (1 - norm.cdf(abs(m / se))) if se and np.isfinite(se) else np.nan)


def block_ci(d, B=2000, bl=20, seed=SEED):
    d = d[np.isfinite(d)]; n = len(d); rng = np.random.default_rng(seed); nb = n // bl; out = []
    for _ in range(B):
        st = rng.integers(0, n - bl, nb)
        out.append(d[np.concatenate([np.arange(s, s + bl) for s in st])].mean())
    return np.percentile(out, [2.5, 97.5])


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
    for H in HORIZONS:
        for p in LEVELS:
            aucs = []; pooled = []
            for a in ASSETS:
                r, L, st, spl, esig = prep[a]; n = len(r)
                q = np.array([np.quantile(L[t - W:t], 1 - p) if t >= W else np.nan for t in range(n)])
                S, E = targets(L, q, H); X = features(r, L, q, spl, st)
                pt = walk(X, E, H, proba=True); sa = walk(X, S, H, proba=False)
                Xb = np.column_stack([esig, esig * abs(norm.ppf(p)), esig * norm.pdf(norm.ppf(p)) / p])
                sb = walk(Xb, S, H, proba=False)
                ok = np.isfinite(pt) & np.isfinite(E)
                if len(np.unique(E[ok])) > 1:
                    aucs.append(roc_auc_score(E[ok], pt[ok]))
                oki = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(S)
                pooled.append(np.abs(S[oki] - sa[oki]) - np.abs(S[oki] - sb[oki]))
            d = np.concatenate(pooled); md, pv = dm(d, lag=H - 1); lo, hi = block_ci(d)
            rows.append(dict(H=H, level=round((1 - p) * 100, 1), mean_AUC=round(float(np.mean(aucs)), 3),
                             incr_delta=round(float(md), 6), ci_low=round(float(lo), 6),
                             ci_high=round(float(hi), 6), DM_p=round(float(pv), 3)))
            print(f"  H={H:2d} lvl={round((1-p)*100,1):5}  AUC={np.mean(aucs):.3f}  "
                  f"delta={md:+.6f} [{lo:+.6f},{hi:+.6f}] p={pv:.3f}  ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "horizon_grid.csv", index=False)
    print("\nMean AUC by horizon x level:\n", R.pivot(index="H", columns="level", values="mean_AUC").to_string())
    print("\nIncremental delta by horizon x level (>0 = conditioning does not help):\n",
          R.pivot(index="H", columns="level", values="incr_delta").round(6).to_string())
    print("\nSaved -> results/horizon_grid.csv")


if __name__ == "__main__":
    main()
