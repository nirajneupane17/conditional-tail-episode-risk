"""
11 -- Conditional regression-GPD severity estimator (estimator robustness, Section 9).

Fixes the zero-inflation failure of the naive conditional GPD by fitting the
conditioning scale on the EPISODE SUBSAMPLE only (so the scale is the conditional
mean severity GIVEN an episode, bounded away from zero, rather than the collapsing
all-day mean). Episode severities are standardized by that scale, a static GPD-POT
tail is fit to the standardized exceedances, and conditional (VaR_beta, ES_beta)
of S* are recovered in closed form and scored with the FZ0 loss (reflected to the
upper tail). Augmented (full features) vs EWMA-only baseline; DM test.

Output: results/conditional_gpd.csv
Run:  python src/11_conditional_gpd.py
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm, genpareto
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from lib import ASSETS, LEVELS, W, BURN, SEED, RESULTS, load, load_stress, ewma_sigma, fz0
import importlib.util
spec = importlib.util.spec_from_file_location("hg", str(Path(__file__).resolve().parent / "09_horizon_grid.py"))
hg = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg)

H = 10; BETA = 0.95
MP = dict(max_iter=200, max_depth=3, learning_rate=0.06, l2_regularization=1.0,
          early_stopping=False, random_state=SEED)


def walk_gpd(X, S, refit=378):
    """Walk-forward conditional GPD: scale fit on episode-positive points only."""
    n = len(S); vhat = np.full(n, np.nan); ehat = np.full(n, np.nan); first = W + BURN
    for T in range(first, n, refit):
        tr = np.arange(0, T - H); fin = np.isfinite(S[tr]) & np.all(np.isfinite(X[tr]), axis=1); tr = tr[fin]
        pos = tr[S[tr] > 0]
        if len(pos) < 120:
            continue
        te = np.arange(T, min(T + refit, n)); te = te[np.all(np.isfinite(X[te]), axis=1)]
        if len(te) == 0:
            continue
        # scale = conditional mean severity GIVEN an episode (episode-subsample fit -> no collapse)
        m = HGBR(**MP).fit(X[pos], S[pos])
        floor = 0.25 * np.median(S[pos])                      # defensive positive floor
        sc_tr = np.maximum(m.predict(X[pos]), floor)
        sc_te = np.maximum(m.predict(X[te]), floor)
        z = S[pos] / sc_tr                                    # standardized episode severities
        u = np.quantile(z, 0.90); exc = z[z > u] - u; Nu = len(exc); nz = len(z)
        if Nu < 30:
            continue
        xi, _, sig = genpareto.fit(exc, floc=0)
        if not (xi < 0.9 and sig > 0):
            continue
        p = 1 - BETA
        vz = u + (sig / xi) * (((nz / Nu) * p) ** (-xi) - 1) if abs(xi) > 1e-6 else u - sig * np.log((nz / Nu) * p)
        ez = vz / (1 - xi) + (sig - xi * u) / (1 - xi)
        ez = max(ez, vz * 1.001)
        vhat[te] = vz * sc_te; ehat[te] = ez * sc_te
    return vhat, ehat


def fz_upper(S, v, e, beta):
    return fz0(-S, -v, -e, 1 - beta)


def main():
    stress = load_stress()[["date", "vix", "nfci", "stlfsi"]]
    sp = load("sp500"); sp["sp5"] = pd.Series(-sp["r"]).rolling(5).sum().values
    sp5full = sp[["date", "sp5"]]
    rows = []; t0 = time.time()
    for a in ASSETS:
        df = load(a); r = df["r"].values; L = -r; n = len(r)
        st = df.merge(stress, on="date", how="left")
        for c in ["vix", "nfci", "stlfsi"]:
            st[c] = st[c].ffill()
        st["vix"] = st["vix"].fillna(st["vix"].median()); st[["nfci", "stlfsi"]] = st[["nfci", "stlfsi"]].fillna(0)
        spl = df.merge(sp5full, on="date", how="left")["sp5"].ffill().fillna(0).values
        esig = ewma_sigma(r)
        for p in LEVELS:
            q = np.array([np.quantile(L[t - W:t], 1 - p) if t >= W else np.nan for t in range(n)])
            S, _ = hg.targets(L, q, H); X = hg.features(r, L, q, spl, st)
            va, ea = walk_gpd(X, S)
            Xb = np.column_stack([esig, esig * abs(norm.ppf(p)), esig * norm.pdf(norm.ppf(p)) / p])
            vb, eb = walk_gpd(Xb, S)
            ok = (S > 0) & np.isfinite(va) & np.isfinite(ea) & np.isfinite(vb) & np.isfinite(eb)
            if ok.sum() < 30:
                continue
            fa = fz_upper(S[ok], va[ok], ea[ok], BETA); fb = fz_upper(S[ok], vb[ok], eb[ok], BETA)
            md, pv = hg.dm(fa - fb, lag=H - 1)
            rows.append(dict(asset=a, level=round((1 - p) * 100, 1), n_episodes=int(ok.sum()),
                             FZ_aug=round(float(fa.mean()), 4), FZ_base=round(float(fb.mean()), 4),
                             delta=round(float(fa.mean() - fb.mean()), 4),
                             DM_p=round(float(pv), 3) if np.isfinite(pv) else np.nan))
        print(f"  {a:10s} done ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "conditional_gpd.csv", index=False)
    print("\nConditional regression-GPD FZ severity by level (lower better), aug vs base:")
    print(R.groupby("level")[["n_episodes", "FZ_aug", "FZ_base", "delta"]].mean().round(4))
    print("\naug better (d<0):", int((R.delta < 0).sum()), " aug worse (d>0):", int((R.delta > 0).sum()),
          " sig DM:", int((R.DM_p < 0.05).sum()), "of", R.DM_p.notna().sum())
    print("max |FZ| (stability check):", float(np.abs(R[['FZ_aug','FZ_base']].values).max()))
    print("Saved -> results/conditional_gpd.csv")


if __name__ == "__main__":
    main()
