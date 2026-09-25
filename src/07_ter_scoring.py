"""
07 -- TER forecast scoring (implements Proposition D).

Produces genuine TER forecasts, TER_beta = Q_beta(S*_{t,H} | F_t), by walk-forward
quantile regression, and evaluates them with the strictly consistent pinball
(generalized piecewise-linear) loss and a coverage backtest, for both the fully
conditioned (augmented) information set and an EWMA-only baseline. A
Diebold-Mariano test on the pinball-loss differences gives the incremental value
of conditioning for TER itself (negative favors the augmented CTER information).

beta is the upper-tail level applied to the worst-episode severity S*. We use
beta = 0.95; because P(S*=0) is large (roughly 1 - base rate), a high beta is
required for TER to be non-degenerate at every threshold alpha.

Output: results/ter_scoring.csv
Run:  python src/07_ter_scoring.py   (after the data are in place)
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from lib import ASSETS, LEVELS, W, H, BURN, SEED, RESULTS, load, load_stress, ewma_sigma

BETA = 0.95          # upper-tail level on S* (the beta in TER = Q_beta(S*))
HGBQ = dict(loss="quantile", quantile=BETA, max_iter=200, max_depth=3,
            learning_rate=0.06, l2_regularization=1.0, early_stopping=False, random_state=SEED)


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


def walk_q(X, y, refit=378):
    n = len(y); pred = np.full(n, np.nan); first = W + BURN
    for T in range(first, n, refit):
        tr = np.arange(0, T - H)
        m = np.isfinite(y[tr]) & np.all(np.isfinite(X[tr]), axis=1); tr = tr[m]
        if len(tr) < 300:
            continue
        te = np.arange(T, min(T + refit, n)); te = te[np.all(np.isfinite(X[te]), axis=1)]
        if len(te) == 0:
            continue
        pred[te] = HGBR(**HGBQ).fit(X[tr], y[tr]).predict(X[te])
    return pred


def pinball(x, y, beta):
    return ((y <= x).astype(float) - beta) * (x - y)


def dm(d, lag=H - 1):
    d = d[np.isfinite(d)]; n = len(d); m = d.mean(); dc = d - m
    var = (dc @ dc) / n
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * (dc[l:] @ dc[:-l]) / n
    se = np.sqrt(var / n)
    return m, 2 * (1 - norm.cdf(abs(m / se)))


def run_asset(a, stress, sp_loss5_full):
    df = load(a); r = df["r"].values; L = -r; n = len(r)
    st = df.merge(stress, on="date", how="left")
    for c in ["vix", "nfci", "stlfsi"]:
        st[c] = st[c].ffill()
    st["vix"] = st["vix"].fillna(st["vix"].median())
    st[["nfci", "stlfsi"]] = st[["nfci", "stlfsi"]].fillna(0)
    spl = df.merge(sp_loss5_full, on="date", how="left")["sp_loss5"].ffill().fillna(0).values
    esig = ewma_sigma(r); rows = []
    for p in LEVELS:
        q = np.array([np.quantile(L[t - W:t], 1 - p) if t >= W else np.nan for t in range(n)])
        S = targets(L, q); X = features(r, L, q, spl, st)
        ter_aug = walk_q(X, S)                                   # augmented TER forecast
        Xb = np.column_stack([esig, esig * abs(norm.ppf(p)),
                              esig * norm.pdf(norm.ppf(p)) / p])
        ter_base = walk_q(Xb, S)                                 # EWMA-only baseline TER
        ok = np.isfinite(ter_aug) & np.isfinite(ter_base) & np.isfinite(S)
        pa = pinball(ter_aug[ok], S[ok], BETA); pb = pinball(ter_base[ok], S[ok], BETA)
        md, pv = dm(pa - pb)
        rows.append(dict(asset=a, level=round((1 - p) * 100, 1), beta=BETA, n=int(ok.sum()),
                         cover_aug=round(float((S[ok] <= ter_aug[ok]).mean()), 3),
                         cover_base=round(float((S[ok] <= ter_base[ok]).mean()), 3),
                         pinball_aug=round(float(pa.mean()), 5),
                         pinball_base=round(float(pb.mean()), 5),
                         delta=round(float(pa.mean() - pb.mean()), 5),
                         DM_p=round(float(pv), 3)))
    return rows


def main():
    stress = load_stress()[["date", "vix", "nfci", "stlfsi"]]
    sp = load("sp500"); sp["sp_loss5"] = pd.Series(-sp["r"]).rolling(5).sum().values
    sp_loss5_full = sp[["date", "sp_loss5"]]
    t0 = time.time(); rows = []
    for a in ASSETS:
        rows += run_asset(a, stress, sp_loss5_full); print(f"  {a:10s} done ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "ter_scoring.csv", index=False)
    print("\nCoverage (target = beta = %.2f) and mean pinball by level:" % BETA)
    print(R.groupby("level")[["cover_aug", "cover_base", "pinball_aug", "pinball_base", "delta"]].mean().round(4))
    print("\nSaved -> results/ter_scoring.csv")


if __name__ == "__main__":
    main()
