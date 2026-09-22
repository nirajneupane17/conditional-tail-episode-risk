# Paper → code → results map

Every table and figure in the manuscript, with the script that produces it and
the committed output it draws from. Run order for a full rebuild: `01 → 02 → 03`
then `04, 05, 06` (or simply `make all`).

## Tables

| Paper table | Content | Script | Output file |
| --- | --- | --- | --- |
| Table 1 | Monte Carlo, Gaussian AR(1) | `src/04_simulation.py` | `results/simulation_gaussian.csv` |
| Table 2 | Monte Carlo, Student-t AR(1) | `src/04_simulation.py` | `results/simulation_student_t.csv` |
| Table 3 | Adversarial stress scenarios | `src/05_stress_tests.py` | `results/stress_tests.csv` |
| Table 4 | VaR/ES benchmark FZ0 + breach rates | `src/01_benchmarks.py` | `results/benchmarks.csv` |
| Table 5 | CTER out-of-sample (Brier, AUC, MAE, RMSE) | `src/02_cter_pipeline.py` | `results/cter_results.csv` |
| Table 6 | Incremental information test | `src/03_incremental_test.py` | `results/incremental_test.csv` |

Worst realized S&P 500 episodes (discussed in the text) are in
`results/sp500_episodes.csv`. A consolidated dump of all headline numbers is in
`results/results.json`.

## Figures

| Paper figure | Content | Script | Output file |
| --- | --- | --- | --- |
| Figure 1 | A/B temporal-clustering illustration | `src/06_figures.py` | `figures/ab.pdf` |
| Figure 2 | Gaussian AR(1): TER rises, VaR/ES flat | `src/06_figures.py` | `figures/mc.pdf` |
| Figure 3 | S&P 500 realized episode severity, 2000–2026 | `src/06_figures.py` | `figures/episodes.pdf` |
| Figure 4 | Occurrence AUC by asset and threshold | `src/06_figures.py` | `figures/auc.pdf` |
| Figure 5 | Incremental performance with bootstrap CIs | `src/06_figures.py` | `figures/incremental.pdf` |

Each figure is written as both `.pdf` (used by the LaTeX source) and
`_png.png` (used by this repository's README and the Word manuscript).

## Determinism

`SEED = 0` is set once in `src/lib.py` and imported everywhere; the
gradient-boosted occurrence/severity models use `early_stopping=False`. The
Monte Carlo, stress, benchmark, and CTER tables reproduce exactly; the
incremental means and Diebold-Mariano p-values reproduce exactly, and the
block-bootstrap confidence intervals reproduce to within resampling noise. The
committed files under `results/` are the exact values reported in the paper.
