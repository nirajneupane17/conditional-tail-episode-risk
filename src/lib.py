"""
Conditional Tail Episode Risk (CTER) -- shared library.

Data loaders, the Tail Episode Risk (TER) functionals, the strictly consistent
Fissler-Ziegel (FZ0) scoring function, and the conventional VaR/ES benchmark
models used throughout the paper.

All paths are resolved relative to the repository root, so the scripts run from
anywhere as `python src/0X_*.py`.

Reference: Neupane, N. (2026). Conditional Tail Episode Risk: A Path-Dependent
Framework for Extreme-Loss Episodes Beyond Value-at-Risk and Expected Shortfall.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths and global configuration (single source of truth for all scripts)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ASSETS = {
    "sp500": "S&P 500",
    "nasdaq": "Nasdaq",
    "tlt_bonds": "TLT",
    "eurusd": "EUR/USD",
    "usdjpy": "USD/JPY",
    "btc": "Bitcoin",
}

LEVELS = [0.05, 0.025, 0.01]  # tail probabilities p ; coverage alpha = 1 - p
W = 1250                       # rolling estimation / threshold window (observations)
H = 10                         # forecast horizon (observations)
BURN = 252                     # extra training burn-in before first out-of-sample forecast
SEED = 0                       # global seed for all stochastic components


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load(asset):
    """Load one asset's daily log returns as a DataFrame with columns [date, r]."""
    df = pd.read_csv(DATA / f"returns_{asset}.csv", parse_dates=["date"]).dropna()
    return df.rename(columns={"logret": "r"}).reset_index(drop=True)


def load_stress():
    """Merge the three per-series stress files (VIX daily; NFCI/STLFSI weekly).

    Weekly indices are carried forward to daily frequency by
    last-observation-carried-forward (LOCF), so only information available up to
    date t enters the feature set. The per-series files are authoritative; the
    merged `data/archive/stress_all.csv` is a derived convenience file only.
    """
    vix = pd.read_csv(DATA / "stress_vix.csv", parse_dates=["date"])
    nfci = pd.read_csv(DATA / "stress_nfci.csv", parse_dates=["date"])
    stlfsi = pd.read_csv(DATA / "stress_stlfsi.csv", parse_dates=["date"])
    s = vix.merge(nfci, on="date", how="outer").merge(stlfsi, on="date", how="outer")
    s = s.sort_values("date").reset_index(drop=True)
    s[["nfci", "stlfsi"]] = s[["nfci", "stlfsi"]].ffill()  # LOCF
    return s


# ---------------------------------------------------------------------------
# Tail Episode Risk functionals
# ---------------------------------------------------------------------------
def episode_severities(loss, q):
    """Cumulative excess severities of every contiguous exceedance episode."""
    over = loss > q
    exc = np.clip(loss - q, 0.0, None)
    eps, cur, running = [], 0.0, False
    for e, o in zip(exc, over):
        if o:
            cur += e
            running = True
        elif running:
            eps.append(cur)
            cur, running = 0.0, False
    if running:
        eps.append(cur)
    return eps


def worst_episode(loss, q):
    """S* : severity of the worst contiguous threshold-exceedance episode."""
    eps = episode_severities(loss, q)
    return max(eps) if eps else 0.0


# ---------------------------------------------------------------------------
# FZ0 : 0-homogeneous Fissler-Ziegel joint (VaR, ES) loss  (lower = better)
# Fissler & Ziegel (2016); Patton, Ziegel & Chen (2019).
# r, v, e are returns / VaR / ES, all negative in the lower tail; p = tail prob.
# ---------------------------------------------------------------------------
def fz0(r, v, e, p):
    e = np.minimum(e, -1e-8)
    hit = (r <= v).astype(float)
    return -(1.0 / (p * e)) * hit * (v - r) + v / e + np.log(-e) - 1.0


# ---------------------------------------------------------------------------
# Conventional one-step VaR/ES forecasters (return space; negative in the tail)
# ---------------------------------------------------------------------------
def hs(rw, p):
    """Historical Simulation VaR/ES from a return window."""
    v = np.quantile(rw, p)
    e = rw[rw <= v].mean() if (rw <= v).any() else v
    return v, e


def ewma_sigma(r, lam=0.94):
    """RiskMetrics EWMA volatility path."""
    s2 = np.empty(len(r))
    s2[0] = np.var(r[:20]) if len(r) >= 20 else np.var(r)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(s2)


def normal_var_es(sig, p):
    """Gaussian VaR/ES from a volatility forecast."""
    z = stats.norm.ppf(p)
    return sig * z, -sig * stats.norm.pdf(z) / p


def evt_pot(Lw, p, uq=0.90):
    """EVT peaks-over-threshold (GPD) VaR/ES from a loss window."""
    u = np.quantile(Lw, uq)
    exc = Lw[Lw > u] - u
    Nu, n = len(exc), len(Lw)
    if Nu < 30:
        return None
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    if xi >= 1 or beta <= 0:
        return None
    VaR = u + (beta / xi) * (((p * n) / Nu) ** (-xi) - 1)
    ES = VaR / (1 - xi) + (beta - xi * u) / (1 - xi)
    return -VaR, -ES  # to return space
