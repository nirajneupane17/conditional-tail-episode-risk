"""
04 -- Monte Carlo dependence experiment.

Simulates stationary AR(1) processes with standardized innovations so the
marginal variance is invariant to rho, and shows that marginal VaR/ES stay flat
while TER (especially TER99) rises sharply as serial dependence increases.
Gaussian and standardized Student-t innovations.

Outputs: results/simulation_gaussian.csv, results/simulation_student_t.csv.
Run:  python src/04_simulation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from lib import RESULTS, SEED, worst_episode

RNG = np.random.default_rng(20260920 + SEED)


def experiment(dist, rhos=(0.0, 0.3, 0.7, 0.9), Hh=10, n_windows=20000, df=5):
    rows = []
    for rho in rhos:
        s = np.sqrt(1 - rho ** 2); N = n_windows + Hh
        if dist == "normal":
            eps = RNG.standard_normal(N)
        else:
            eps = RNG.standard_t(df, N) / np.sqrt(df / (df - 2))  # unit variance
        x = np.empty(N); x[0] = eps[0]
        for t in range(1, N):
            x[t] = rho * x[t - 1] + s * eps[t]
        v = np.quantile(x, 0.95); e = x[x >= v].mean()
        S = np.array([worst_episode(x[i:i + Hh], v) for i in range(n_windows)])
        rows.append(dict(rho=rho, VaR95=round(v, 3), ES95=round(e, 3),
                         TER95=round(float(np.quantile(S, 0.95)), 3),
                         TER99=round(float(np.quantile(S, 0.99)), 3)))
    return pd.DataFrame(rows)


def main():
    g = experiment("normal"); t = experiment("t")
    g.to_csv(RESULTS / "simulation_gaussian.csv", index=False)
    t.to_csv(RESULTS / "simulation_student_t.csv", index=False)
    print("Gaussian AR(1):\n", g.to_string(index=False))
    print("\nStudent-t AR(1) (standardized):\n", t.to_string(index=False))
    print("\nSaved -> results/simulation_gaussian.csv, results/simulation_student_t.csv")


if __name__ == "__main__":
    main()
