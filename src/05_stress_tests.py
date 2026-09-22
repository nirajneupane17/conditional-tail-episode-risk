"""
05 -- Adversarial stress tests.

Six 60-observation loss paths evaluated under a common, fixed threshold q = 2.5
(standardized loss units), so differences reflect temporal organization rather
than a path-specific threshold. Shows that VaR/ES and the worst-episode severity
S* rank scenarios differently (e.g. separated tail-thickening vs a sustained
liquidity collapse).

Output: results/stress_tests.csv.
Run:  python src/05_stress_tests.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from lib import RESULTS, SEED, worst_episode

RNG = np.random.default_rng(11 + SEED)
Q = 2.5


def base():
    return np.abs(RNG.standard_normal(60)) * 0.8


def metrics(loss):
    v = np.quantile(loss, 0.95); e = loss[loss >= v].mean()
    return round(float(v), 2), round(float(e), 2), round(float(worst_episode(loss, Q)), 2)


def main():
    scen = {}
    p = base(); p[30] = 9.0;                          scen["Isolated crash"] = p
    p = base(); p[[8, 20, 33, 47, 58]] = 6.0;          scen["Tail thickening (separated spikes)"] = p
    p = base(); p[28:40] = 3.5;                        scen["Liquidity collapse (sustained run)"] = p
    p = base(); p[33:39] = RNG.uniform(5, 8, 6);       scen["Volatility explosion (cluster)"] = p
    p = base() + 2.0;                                  scen["Broad contagion (all elevated)"] = p
    p = base(); p[25:31] = np.linspace(2.6, 5.0, 6);   scen["Escalating deterioration"] = p
    rows = []
    for k, v in scen.items():
        va, es, ss = metrics(v)
        rows.append(dict(scenario=k, threshold=Q, VaR95=va, ES95=es, S_star=ss))
    df = pd.DataFrame(rows); df.to_csv(RESULTS / "stress_tests.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved -> results/stress_tests.csv")


if __name__ == "__main__":
    main()
