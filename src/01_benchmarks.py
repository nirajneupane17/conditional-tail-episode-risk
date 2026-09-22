"""
01 -- Conventional VaR/ES benchmarks scored with the FZ0 loss.

Four one-step-ahead forecasters (Historical Simulation, EWMA-Normal, EVT-POT,
GARCH(1,1)-t) are evaluated on every asset at the 95/97.5/99% levels using the
0-homogeneous Fissler-Ziegel loss. Output: results/benchmarks.csv.

Run:  python src/01_benchmarks.py
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
from arch import arch_model
from lib import ASSETS, LEVELS, W, RESULTS, load, ewma_sigma, hs, normal_var_es, evt_pot, fz0


def garch_sigma(r, refit_every=126):
    """1-step GARCH(1,1)-t volatility via O(N) filtering with periodic refits."""
    n = len(r); sig = np.full(n, np.nan); nu = np.full(n, np.nan)
    scale = 100.0; params = None; s2 = np.var(r[:W]) * scale ** 2
    for t in range(1, n):
        if t >= W and ((t - W) % refit_every == 0 or params is None):
            try:
                res = arch_model(r[t - W:t] * scale, mean="Zero", vol="GARCH",
                                 p=1, q=1, dist="t").fit(disp="off")
                params = (res.params["omega"], res.params["alpha[1]"],
                          res.params["beta[1]"], res.params["nu"])
            except Exception:
                pass
        if params is not None:
            om, al, be, ndf = params
            s2 = om + al * (r[t - 1] * scale) ** 2 + be * s2
            if t >= W:
                sig[t] = np.sqrt(s2) / scale; nu[t] = ndf
        else:
            s2 = 0.94 * s2 + 0.06 * (r[t - 1] * scale) ** 2
    return sig, nu


def gpd_params(L, refit_every=21, uq=0.90):
    n = len(L); out = np.full((n, 3), np.nan); cur = None
    for t in range(W, n):
        if (t - W) % refit_every == 0 or cur is None:
            Lw = L[t - W:t]; u = np.quantile(Lw, uq); exc = Lw[Lw > u] - u
            if len(exc) >= 30:
                xi, _, beta = stats.genpareto.fit(exc, floc=0)
                cur = (u, xi, beta) if (xi < 1 and beta > 0) else cur
        if cur:
            out[t] = cur
    return out


def run_asset(a):
    df = load(a); r = df["r"].values; L = -r; n = len(r)
    esig = ewma_sigma(r); gsig, gnu = garch_sigma(r); gp = gpd_params(L)
    idx = np.arange(W, n); rt = r[idx]; rows = []
    for p in LEVELS:
        hv = np.array([np.quantile(r[t - W:t], p) for t in idx])
        he = np.array([r[t - W:t][r[t - W:t] <= q].mean() for t, q in zip(idx, hv)])
        z = stats.norm.ppf(p); nv = esig[idx] * z; ne = -esig[idx] * stats.norm.pdf(z) / p
        u, xi, beta = gp[idx, 0], gp[idx, 1], gp[idx, 2]
        VaRl = u + (beta / xi) * (((p * W) / (W * 0.10)) ** (-xi) - 1)
        ESl = VaRl / (1 - xi) + (beta - xi * u) / (1 - xi)
        s, ndf = gsig[idx], gnu[idx]
        tq = stats.t.ppf(p, ndf) * np.sqrt((ndf - 2) / ndf); gv = s * tq
        ge = -s * (stats.t.pdf(stats.t.ppf(p, ndf), ndf) / p) * \
            ((ndf + stats.t.ppf(p, ndf) ** 2) / (ndf - 1)) * np.sqrt((ndf - 2) / ndf)
        models = {"HistoricalSim": (hv, he), "EWMA-Normal": (nv, ne),
                  "EVT-POT": (-VaRl, -ESl), "GARCH-t": (gv, ge)}
        for m, (v, e) in models.items():
            ok = np.isfinite(v) & np.isfinite(e) & np.isfinite(rt)
            if ok.sum() < 200:
                continue
            rows.append(dict(asset=a, level=round((1 - p) * 100, 1), p=p, model=m,
                             n=int(ok.sum()), breach_rate=round(float((rt[ok] <= v[ok]).mean()), 4),
                             exp_rate=p, FZ0=round(float(fz0(rt[ok], v[ok], e[ok], p).mean()), 4)))
    return rows


def main():
    t0 = time.time(); rows = []
    for a in ASSETS:
        rows += run_asset(a); print(f"  {a:10s} done ({time.time()-t0:4.0f}s)")
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "benchmarks.csv", index=False)
    piv = R.pivot_table(index="level", columns="model", values="FZ0", aggfunc="mean").round(3)
    print("\nMean FZ0 across assets (lower = better):\n", piv)
    print("\nSaved -> results/benchmarks.csv")


if __name__ == "__main__":
    main()
