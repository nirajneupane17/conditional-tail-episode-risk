"""
02 -- CTER out-of-sample pipeline.

For each asset and threshold level:
  * builds the dynamic threshold q_{t,alpha}, realized worst-episode severity
    S*_{t,H}, and occurrence O_{t,H} over H = 10 observations;
  * estimates episode occurrence p_t (gradient-boosted classifier) and episode
    severity (gradient-boosted regressor) by expanding-window walk-forward with a
    purge gap of H observations (no look-ahead);
  * records occurrence discrimination (Brier, AUC) and severity accuracy (MAE,
    RMSE), and the per-observation absolute-error DELTA between the fully
    conditioned model and an EWMA-only baseline (for the incremental test).

Outputs: results/cter_results.csv and results/incremental_deltas.npz.
Estimators use random_state=SEED and early_stopping=False for determinism.

Run:  python src/02_cter_pipeline.py
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import (HistGradientBoostingClassifier as HGBC,
                              HistGradientBoostingRegressor as HGBR)
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error
from lib import ASSETS, LEVELS, W, H, BURN, SEED, RESULTS, load, load_stress, ewma_sigma

HGB = dict(max_iter=200, max_depth=3, learning_rate=0.06, l2_regularization=1.0,
           early_stopping=False, random_state=SEED)


def targets(L, q):
    """Realized worst-episode severity S* and occurrence O over the next H obs."""
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


def features(r, L, q, sp_loss5, st):
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
    X = np.column_stack([L, l5, l10, rv20, rv60, rv20 / rv60, dd, st["vix"],
                         st["vix"].diff(5).values, st["nfci"], st["stlfsi"],
                         sp_loss5, exfreq, runlen])
    return X


def walk(X, y, proba, refit=378, first=None):
    n = len(y); pred = np.full(n, np.nan); first = first or (W + BURN)
    for T in range(first, n, refit):
        tr = np.arange(0, T - H)
        m = np.isfinite(y[tr]) & np.all(np.isfinite(X[tr]), axis=1); tr = tr[m]
        if len(tr) < 300:
            continue
        te = np.arange(T, min(T + refit, n))
        te = te[np.all(np.isfinite(X[te]), axis=1)]
        if len(te) == 0:
            continue
        if proba:
            if len(np.unique(y[tr])) < 2:
                continue
            pred[te] = HGBC(**HGB).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            pred[te] = HGBR(**HGB).fit(X[tr], y[tr]).predict(X[te])
    return pred


def run_asset(a, stress, sp_loss5_full):
    df = load(a); r = df["r"].values; L = -r; n = len(r)
    st = df.merge(stress, on="date", how="left")
    for c in ["vix", "nfci", "stlfsi"]:
        st[c] = st[c].ffill()
    st["vix"] = st["vix"].fillna(st["vix"].median())
    st[["nfci", "stlfsi"]] = st[["nfci", "stlfsi"]].fillna(0)
    spl = df.merge(sp_loss5_full, on="date", how="left")["sp_loss5"].ffill().fillna(0).values
    esig = ewma_sigma(r); rows = []; deltas = {}
    for p in LEVELS:
        q = np.array([np.quantile(L[t - W:t], 1 - p) if t >= W else np.nan for t in range(n)])
        S, E = targets(L, q); X = features(r, L, q, spl, st)
        pt = walk(X, E, proba=True)                 # occurrence
        st_hat = walk(X, S, proba=False)            # augmented severity
        Xb = np.column_stack([esig, esig * abs(norm.ppf(p)),
                              esig * norm.pdf(norm.ppf(p)) / p])
        sb_hat = walk(Xb, S, proba=False)           # EWMA-only baseline severity
        ok = np.isfinite(pt) & np.isfinite(S) & np.isfinite(E)
        oks = np.isfinite(st_hat) & np.isfinite(S)
        oki = np.isfinite(st_hat) & np.isfinite(sb_hat) & np.isfinite(S)
        auc = roc_auc_score(E[ok], pt[ok]) if len(np.unique(E[ok])) > 1 else np.nan
        rows.append(dict(asset=a, level=round((1 - p) * 100, 1), p=p, n=int(ok.sum()),
                         base_rate=round(float(E[ok].mean()), 4),
                         Brier=round(float(brier_score_loss(E[ok], np.clip(pt[ok], 0, 1))), 4),
                         AUC=round(float(auc), 4) if np.isfinite(auc) else np.nan,
                         MAE=round(float(mean_absolute_error(S[oks], st_hat[oks])), 5),
                         RMSE=round(float(mean_squared_error(S[oks], st_hat[oks]) ** 0.5), 5)))
        deltas[f"{a}_{round((1-p)*100,1)}"] = (np.abs(S[oki] - st_hat[oki]) -
                                               np.abs(S[oki] - sb_hat[oki]))
    return rows, deltas


def main():
    stress = load_stress()[["date", "vix", "nfci", "stlfsi"]]
    sp = load("sp500"); sp["sp_loss5"] = pd.Series(-sp["r"]).rolling(5).sum().values
    sp_loss5_full = sp[["date", "sp_loss5"]]
    t0 = time.time(); rows = []; deltas = {}
    for a in ASSETS:
        rw, dl = run_asset(a, stress, sp_loss5_full)
        rows += rw; deltas.update(dl); print(f"  {a:10s} done ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "cter_results.csv", index=False)
    np.savez(RESULTS / "incremental_deltas.npz", **deltas)
    print("\nMean AUC / Brier by level:\n",
          R.groupby("level")[["AUC", "Brier", "base_rate"]].mean().round(3))
    print("\nSaved -> results/cter_results.csv, results/incremental_deltas.npz")


if __name__ == "__main__":
    main()
