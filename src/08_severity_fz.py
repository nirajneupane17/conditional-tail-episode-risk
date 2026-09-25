"""
08 -- Episode-severity scoring with a Fissler-Ziegel score (implements Prop E(ii)).

On the episode subsample {O_{t,H}=1}, we forecast the conditional upper-tail
(VaR_beta, ES_beta) of the worst-episode severity S* and score the pair with the
strictly consistent 0-homogeneous Fissler-Ziegel loss (reflected to the upper
tail: the upper-beta tail of S* is the lower-(1-beta) tail of -S*). This replaces
the point-forecast MAE/RMSE of the CTER pipeline with a strictly consistent tail
score, as Proposition E(ii) requires. Augmented (full X,Z) vs EWMA-only baseline;
Diebold-Mariano test on the FZ differences (negative favors the augmented set).

Estimator (transparent, semiparametric):
  VaR_beta : walk-forward quantile regression on episode-positive points.
  ES_beta  : mean of S* over training episode points exceeding the predicted VaR
             (conditional mean-excess); falls back to VaR * global ES/VaR ratio
             when the exceedance training set is too thin. ES is floored at VaR.

Output: results/severity_fz.csv
Run:  python src/08_severity_fz.py
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from lib import ASSETS, LEVELS, W, H, BURN, SEED, RESULTS, load, load_stress, ewma_sigma, fz0

BETA = 0.95
QP = dict(loss="quantile", quantile=BETA, max_iter=200, max_depth=3,
          learning_rate=0.06, l2_regularization=1.0, early_stopping=False, random_state=SEED)
MP = dict(max_iter=200, max_depth=3, learning_rate=0.06, l2_regularization=1.0,
          early_stopping=False, random_state=SEED)


def targets(L, q):
    n = len(L); S = np.full(n, np.nan)
    for t in range(n - H):
        qt = q[t]
        if not np.isfinite(qt):
            continue
        fut = L[t + 1:t + 1 + H]; exc = np.clip(fut - qt, 0, None); over = fut > qt
        best = cur = 0.0
        for e, o in zip(exc, over):
            cur = cur + e if o else 0.0
            best = max(best, cur)
        S[t] = best
    return S


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
    return np.column_stack([L, l5, l10, rv20, rv60, rv20 / rv60, dd, st["vix"],
                            st["vix"].diff(5).values, st["nfci"], st["stlfsi"],
                            sp_loss5, exfreq, runlen])


def walk_varescond(X, S, refit=378):
    """Walk-forward conditional (VaR_beta, ES_beta) of S* on episode-positive points."""
    n = len(S); vhat = np.full(n, np.nan); ehat = np.full(n, np.nan); first = W + BURN
    for T in range(first, n, refit):
        tr = np.arange(0, T - H)
        fin = np.isfinite(S[tr]) & np.all(np.isfinite(X[tr]), axis=1); tr = tr[fin]
        pos = tr[S[tr] > 0]
        if len(pos) < 100:
            continue
        te = np.arange(T, min(T + refit, n)); te = te[np.all(np.isfinite(X[te]), axis=1)]
        if len(te) == 0:
            continue
        vm = HGBR(**QP).fit(X[pos], S[pos])
        vtr = vm.predict(X[pos])
        exc = pos[S[pos] >= vtr]                      # training exceedances above predicted VaR
        gratio = S[pos][S[pos] >= np.quantile(S[pos], BETA)].mean() / max(np.quantile(S[pos], BETA), 1e-9)
        if len(exc) >= 40:
            em = HGBR(**MP).fit(X[exc], S[exc])
            v = vm.predict(X[te]); e = em.predict(X[te])
        else:                                         # thin tail: scale VaR by global ES/VaR ratio
            v = vm.predict(X[te]); e = v * gratio
        e = np.maximum(e, v * 1.001)                  # ES must exceed VaR
        vhat[te] = v; ehat[te] = e
    return vhat, ehat


def fz_upper(S, v, e, beta):
    return fz0(-S, -v, -e, 1 - beta)


def dm(d, lag=H - 1):
    d = d[np.isfinite(d)]; n = len(d); m = d.mean(); dc = d - m
    var = (dc @ dc) / n
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * (dc[l:] @ dc[:-l]) / n
    se = np.sqrt(var / n) if var > 0 else np.nan
    return m, (2 * (1 - norm.cdf(abs(m / se))) if se and np.isfinite(se) else np.nan)


def run_asset(a, stress, sp_loss5_full):
    df = load(a); r = df["r"].values; L = -r; n = len(r)
    st = df.merge(stress, on="date", how="left")
    for c in ["vix", "nfci", "stlfsi"]:
        st[c] = st[c].ffill()
    st["vix"] = st["vix"].fillna(st["vix"].median()); st[["nfci", "stlfsi"]] = st[["nfci", "stlfsi"]].fillna(0)
    spl = df.merge(sp_loss5_full, on="date", how="left")["sp_loss5"].ffill().fillna(0).values
    esig = ewma_sigma(r); rows = []
    for p in LEVELS:
        q = np.array([np.quantile(L[t - W:t], 1 - p) if t >= W else np.nan for t in range(n)])
        S = targets(L, q); X = features(r, L, q, spl, st)
        va, ea = walk_varescond(X, S)
        Xb = np.column_stack([esig, esig * abs(norm.ppf(p)), esig * norm.pdf(norm.ppf(p)) / p])
        vb, eb = walk_varescond(Xb, S)
        ok = (S > 0) & np.isfinite(va) & np.isfinite(ea) & np.isfinite(vb) & np.isfinite(eb)
        fa = fz_upper(S[ok], va[ok], ea[ok], BETA); fb = fz_upper(S[ok], vb[ok], eb[ok], BETA)
        md, pv = dm(fa - fb)
        rows.append(dict(asset=a, level=round((1 - p) * 100, 1), beta=BETA, n_episodes=int(ok.sum()),
                         FZ_aug=round(float(fa.mean()), 4), FZ_base=round(float(fb.mean()), 4),
                         delta=round(float(fa.mean() - fb.mean()), 4), DM_p=round(float(pv), 3) if np.isfinite(pv) else np.nan))
    return rows


def main():
    stress = load_stress()[["date", "vix", "nfci", "stlfsi"]]
    sp = load("sp500"); sp["sp_loss5"] = pd.Series(-sp["r"]).rolling(5).sum().values
    sp_loss5_full = sp[["date", "sp_loss5"]]
    t0 = time.time(); rows = []
    for a in ASSETS:
        rows += run_asset(a, stress, sp_loss5_full); print(f"  {a:10s} done ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "severity_fz.csv", index=False)
    print("\nMean FZ severity score by level (lower better), aug vs base:")
    print(R.groupby("level")[["n_episodes", "FZ_aug", "FZ_base", "delta"]].mean().round(4))
    print("\nSaved -> results/severity_fz.csv")


if __name__ == "__main__":
    main()
