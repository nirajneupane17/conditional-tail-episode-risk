# Conditional Tail Episode Risk (CTER)

## A Path-Dependent Framework for Extreme-Loss Episodes Beyond Value-at-Risk and Expected Shortfall

<p align="center">

**Niraj Neupane**  
*Quantitative Finance · Financial Econometrics · Financial Risk Management · AI/ML*

</p>

<p align="center">

[![Status](https://img.shields.io/badge/Status-Working%20Paper-orange)](#research-status)
[![Research](https://img.shields.io/badge/Research-Quantitative%20Finance-1f6feb)](#research-overview)
[![Risk](https://img.shields.io/badge/Domain-Tail%20Risk-6f42c1)](#risk-measures)
[![Methodology](https://img.shields.io/badge/Methodology-Extreme%20Value%20Theory-8b5cf6)](#methodology)
[![Code](https://img.shields.io/badge/Code-Python-3776ab)](#reproducibility)
[![License](https://img.shields.io/badge/Code-MIT-green)](#license)

</p>

---

## Research at a Glance

> **Extreme losses are not only about how large losses become. They are also about how extreme losses organize themselves through time.**

Value-at-Risk (VaR) and Expected Shortfall (ES) are fundamental tools for measuring financial tail risk. However, they primarily characterize the **marginal distribution of losses**.

This research develops **Tail Episode Risk (TER)** and **Conditional Tail Episode Risk (CTER)** as a complementary framework for studying the **temporal structure and cumulative severity of extreme-loss episodes** over a future forecast horizon.

Instead of asking only:

> *How large can an individual tail loss be?*

the framework asks:

> *What happens when extreme losses persist and form an episode?*

The research does **not** position CTER as a replacement for VaR or ES. It investigates whether episode-based information represents a distinct and economically meaningful dimension of tail risk.

---

## Table of Contents

- [Research Overview](#research-overview)
- [The Problem](#the-problem)
- [Core Idea](#core-idea)
- [Tail Episode Risk (TER)](#tail-episode-risk-ter)
- [Conditional Tail Episode Risk (CTER)](#conditional-tail-episode-risk-cter)
- [VaR vs ES vs TER vs CTER](#var-vs-es-vs-ter-vs-cter)
- [Simple Example](#simple-example)
- [Research Contributions](#research-contributions)
- [Theoretical Properties](#theoretical-properties)
- [Methodology](#methodology)
- [Monte Carlo Experiments](#monte-carlo-experiments)
- [Adversarial Stress Tests](#adversarial-stress-tests)
- [Empirical Study](#empirical-study)
- [Backtesting Framework](#backtesting-framework)
- [Empirical Findings](#empirical-findings)
- [Incremental Information](#incremental-information)
- [Estimator Sensitivity](#estimator-sensitivity)
- [EVT/GPD Estimation Experiment](#evtgpd-estimation-experiment)
- [Relationship to Existing Literature](#relationship-to-existing-literature)
- [Novelty Position](#novelty-position)
- [Model Risk and Governance](#model-risk-and-governance)
- [Limitations](#limitations)
- [Future Research](#future-research)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Data](#data)
- [Research Workflow](#research-workflow)
- [Research Integrity](#research-integrity)
- [Why This Repository Matters](#why-this-repository-matters)
- [Citation](#citation)
- [Research Status](#research-status)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Author](#author)
- [The Core Idea](#the-core-idea)

---

# Research Overview

### Research Question

> **Can path-dependent information about extreme-loss episodes provide a distinct and economically meaningful dimension of tail risk beyond marginal VaR and Expected Shortfall?**

The research investigates this question through:

- mathematical formulation;
- theoretical analysis;
- Monte Carlo simulation;
- dependence experiments;
- adversarial stress testing;
- rolling out-of-sample forecasting;
- conventional VaR/ES benchmarks;
- multi-asset empirical evaluation;
- statistical inference;
- EVT/GPD estimation;
- robustness and sensitivity analysis.

---

# The Problem

Financial tail risk is often summarized using marginal measures.

### Value-at-Risk

VaR answers:

> **What loss threshold is exceeded with a specified probability?**

### Expected Shortfall

ES answers:

> **How severe is the loss beyond that threshold on average?**

These measures are essential, but they do not explicitly describe how extreme observations are **organized across time**.

Consider:

```text
Path A

4   0   4   0   4   0
↑       ↑       ↑
│       │       │
isolated exceedances
```

versus:

```text
Path B

4   4   4   0   0   0
↑───────────↑
persistent episode
```

The two paths contain the same individual extreme observations, but their temporal structures are different.

Path B contains a concentrated episode of extreme losses, while Path A contains isolated exceedances.

This motivates a path-dependent risk functional.

---

# Core Idea

The CTER framework has four conceptual stages:

```text
                 FUTURE LOSS PATH
                        │
                        ▼
                ┌───────────────┐
                │ Tail Threshold│
                │    q(t, α)    │
                └───────┬───────┘
                        │
                        ▼
              Identify Exceedances
                        │
                        ▼
             Form Contiguous Episodes
                        │
                        ▼
              Measure Episode Severity
                        │
                        ▼
               Select Worst Episode
                        │
                        ▼
              ┌──────────────────┐
              │ TER / CTER       │
              │ Path-Dependent   │
              │ Tail Risk        │
              └──────────────────┘
```

The framework therefore moves from:

**individual observations**

to

**episodes**

to

**cumulative episode severity**

to

**tail risk of the worst episode**.

---

# Tail Episode Risk (TER)

## 1. Future Loss Path

For a forecast origin `t` and forecast horizon `H`, define the future loss path as:

$$
\mathbf{L}_{t,H}
=
(L_{t+1},L_{t+2},\ldots,L_{t+H})
$$

Let:

$$
q_{t,\alpha}
$$

denote the tail-loss threshold at confidence level `α`.

---

## 2. Threshold Exceedance

For each future observation, define the positive threshold exceedance:

$$
(L_{t+k}-q_{t,\alpha})_+
=
\max(L_{t+k}-q_{t,\alpha},0)
$$

Only losses above the threshold contribute to episode severity.

---

## 3. Extreme-Loss Episodes

Let:

$$
\mathcal{E}_{t,H}
$$

denote the set of contiguous threshold-exceedance episodes within the forecast horizon.

For episode `j`, define cumulative severity as:

$$
S_j
=
\sum_{k\in j}
(L_{t+k}-q_{t,\alpha})_+
$$

Thus, `S_j` measures the cumulative amount by which losses exceed the tail threshold during episode `j`.

---

## 4. Worst Episode

The worst episode within the forecast horizon is:

$$
S^*_{t,H}
=
\max_{j\in\mathcal{E}_{t,H}}S_j
$$

If no threshold exceedance occurs, the episode burden is zero.

---

## 5. Tail Episode Risk

TER is defined as the upper-tail quantile of the worst episode severity:

$$
TER_{\alpha,\beta,H}
=
Q_\beta(S^*_{t,H})
$$

where:

- `α` = tail threshold level;
- `β` = upper-tail probability applied to episode severity;
- `H` = forecast horizon;
- `S*` = worst cumulative threshold-exceedance episode.

### Interpretation

TER answers:

> **How severe can the worst cumulative extreme-loss episode become over the forecast horizon?**

---

# Conditional Tail Episode Risk (CTER)

TER describes the distribution of the worst episode.

CTER adds a conditional decomposition.

Define the episode-occurrence indicator:

$$
E_{t,H}
=
\mathbf{1}
\left[
\max_{1\leq k\leq H}L_{t+k}
>
q_{t,\alpha}
\right]
$$

The proposed conditional formulation is:

$$
CTER_{\alpha,\beta,H,t}
=
P(E_{t,H}=1\mid X_t)
\cdot
ES_\beta
\left[
S^*_{t,H}
\mid
E_{t,H}=1,X_t,Z_t
\right]
$$

where:

- `X(t)` = information available at the forecast origin;
- `Z(t)` = episode-related state information;
- `E(t,H)` = indicator that an extreme-loss episode occurs;
- `ES(β)` = conditional expected shortfall of episode severity.

Conceptually:

```text
                         CTER
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
     Episode Probability          Episode Severity
             │                           │
             ▼                           ▼
     P(E = 1 | X)                ESβ(S* | E=1,X,Z)
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                          CTER
```

This decomposition separates two distinct questions:

### Question 1

> **How likely is an extreme-loss episode?**

### Question 2

> **If the episode occurs, how severe could it become?**

---

# VaR vs ES vs TER vs CTER

| Measure | Primary object | Perspective | Path-dependent? |
|---|---|---|---|
| **VaR** | Tail-loss threshold | Marginal | No |
| **ES** | Average severity beyond threshold | Marginal | No |
| **TER** | Upper tail of worst cumulative exceedance episode | Path-dependent | Yes |
| **CTER** | Episode probability × conditional episode severity | Conditional | Yes |

The objective is **not** to replace VaR or ES.

Instead, the framework investigates whether CTER captures an additional dimension of risk associated with the **temporal organization of extreme losses**.

---

# Simple Example

Consider:

```text
Path A = (4, 0, 4, 0, 4, 0)

Path B = (4, 4, 4, 0, 0, 0)
```

with:

```text
q = 2
```

The individual exceedances are:

```text
2   0   2   0   2   0
```

for Path A, while Path B produces:

```text
2   2   2   0   0   0
```

For Path A, the exceedances are separated. Therefore:

$$
S_A^*=2
$$

For Path B, the three exceedances form one episode:

$$
S_B^*=6
$$

Therefore:

$$
S_A^*\neq S_B^*
$$

even though the two paths contain the same individual loss observations.

This gives the core distinction:

$$
VaR_A = VaR_B
$$

and

$$
ES_A = ES_B
$$

while:

$$
TER_A \neq TER_B
$$

The difference arises from **temporal organization**, not marginal observations.

---

# Research Contributions

The research investigates the following contributions.

## 1. Path-Dependent Tail Functional

A formal framework based on the **worst cumulative threshold-exceedance episode** over a future horizon.

## 2. Conditional Decomposition

CTER separates:

- probability of an extreme-loss episode;
- conditional upper-tail severity of that episode.

## 3. Temporal Clustering

The framework explicitly distinguishes:

- isolated extreme observations;
- persistent extreme-loss episodes.

## 4. Simulation Evidence

Monte Carlo experiments investigate how dependence affects episode risk.

## 5. Adversarial Stress Testing

The framework is tested under different stylized market-stress environments.

## 6. Out-of-Sample Evaluation

The empirical study uses rolling forecasts over an out-of-sample period.

## 7. Incremental Information Testing

The research tests whether CTER provides additional information beyond conventional risk forecasts.

## 8. Estimator Sensitivity

Different estimation approaches are evaluated, including EVT/GPD-based experiments.

---

# Theoretical Properties

The research studies several properties of the proposed framework.

## Monotonicity

Conditional on a common threshold construction, increasing the loss path cannot reduce the corresponding episode burden.

## Positive Homogeneity

Under consistent scaling of losses and thresholds:

$$
TER(cL)=c\,TER(L)
$$

for:

$$
c>0
$$

## Translation Equivariance

When losses and the threshold are translated consistently, the threshold-exceedance structure is preserved.

## Temporal Clustering Sensitivity

TER and CTER respond to the temporal organization of extreme observations.

This is the defining distinction from purely marginal tail measures.

## Distinction from VaR and ES

Constructed paths demonstrate that the same marginal observations can produce different episode-risk values.

## Important Qualification

The current research does **not** claim that TER or CTER satisfy every property associated with coherent risk measures.

In particular, subadditivity and full coherence require separate mathematical analysis because:

- the threshold is distribution-dependent;
- episode construction is nonlinear;
- the worst-episode operator is nonlinear.

---

# Methodology

The research follows the following methodological sequence:

```text
Historical Market Data
        │
        ▼
Data Cleaning & Transformation
        │
        ▼
Rolling Estimation Window
        │
        ├───────────────┐
        ▼               ▼
   VaR / ES          TER / CTER
   Benchmarks         Framework
        │               │
        └───────┬───────┘
                ▼
        Out-of-Sample Forecasts
                │
                ▼
       Backtesting & Evaluation
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      VaR      ES      CTER
    Tests    Scores   Metrics
        │       │        │
        └───────┼────────┘
                ▼
       Statistical Inference
                │
                ▼
        Research Conclusions
```

---

# Monte Carlo Experiments

The simulation study examines the relationship between:

- marginal tail behavior;
- temporal dependence;
- episode severity.

Dependence levels include:

$$
\rho\in\{0.0,0.3,0.7,0.9\}
$$

under Gaussian and Student-t settings.

## Gaussian Dependence

| Correlation | VaR 95% | ES 95% | TER 95% | TER 99% |
|---:|---:|---:|---:|---:|
| 0.0 | 1.648 | 2.067 | 0.969 | 1.477 |
| 0.3 | 1.650 | 2.063 | 0.997 | 1.681 |
| 0.7 | 1.641 | 2.054 | 1.101 | 2.752 |
| 0.9 | 1.655 | 2.079 | 1.167 | 4.617 |

## Student-t Dependence

| Correlation | VaR 95% | ES 95% | TER 95% | TER 99% |
|---:|---:|---:|---:|---:|
| 0.0 | 2.013 | 2.864 | 1.986 | 3.941 |
| 0.3 | 2.021 | 2.888 | 2.148 | 4.355 |
| 0.7 | 2.020 | 2.909 | 2.291 | 6.566 |
| 0.9 | 1.990 | 2.855 | 1.929 | 9.010 |

### Simulation Interpretation

The simulations illustrate that stronger temporal dependence can materially change episode risk while marginal VaR and ES remain comparatively stable.

This provides controlled evidence for the distinction between:

```text
Marginal Tail Risk
        vs.
Path-Dependent Episode Risk
```

---

# Adversarial Stress Tests

The framework is also evaluated under stylized stress scenarios.

| Stress Scenario | VaR 95% | ES 95% | TER |
|---|---:|---:|---:|
| Sudden crash | 3.385 | 5.500 | 3.855 |
| Slow deterioration | 2.755 | 2.800 | 4.485 |
| Volatility explosion | 3.665 | 3.800 | 8.285 |
| Tail thickening | 2.865 | 3.000 | 1.355 |
| Liquidity collapse | 3.365 | 3.500 | 9.040 |
| Cross-asset contagion | 2.755 | 2.800 | 2.765 |

These scenarios illustrate that different stress mechanisms can affect:

- marginal loss severity;
- persistence;
- clustering;
- cumulative episode burden.

### Important Scope

The current TER implementation is primarily **univariate**.

The cross-asset contagion experiment should therefore be interpreted as a stress-test illustration rather than a complete multivariate CTER model.

---

# Empirical Study

## Assets

The empirical study covers:

- S&P 500
- Nasdaq
- TLT
- EUR/USD
- USD/JPY
- Bitcoin

## Out-of-Sample Period

**2020–2025**

## Forecast Horizon

**10 trading days**

## Rolling Training Window

**1,250 observations**

## Tail Levels

The primary analysis considers:

- 95%
- 97.5%
- 99%

## Benchmark Models

The benchmark set includes:

- Historical Simulation
- EWMA
- GARCH-t
- EVT / Peaks-over-Threshold where applicable

---

# Backtesting Framework

The empirical framework evaluates several dimensions of performance.

## VaR Evaluation

- Kupiec unconditional coverage
- Christoffersen independence
- Conditional coverage
- Quantile loss

## VaR / ES Evaluation

VaR and ES are evaluated jointly using strictly consistent scoring methods based on the Fissler-Ziegel framework.

## CTER Evaluation

### Episode Probability

- Brier score
- ROC/AUC
- calibration

### Episode Severity

- MAE
- RMSE

## Incremental Forecast Evaluation

The research compares:

```text
BASELINE
VaR / ES
```

with:

```text
EXTENDED
VaR / ES + CTER
```

The purpose is to determine whether CTER contributes additional predictive information.

---

# Empirical Findings

The current results are intentionally reported without selectively retaining only favorable findings.

The evidence indicates that CTER captures a distinct path-dependent dimension of extreme-loss behavior.

However:

> **The current empirical results do not establish universal incremental forecasting superiority over conventional VaR/ES frameworks.**

This distinction is central to the research.

---

# Episode Forecasting Results

Selected multi-asset results are shown below.

| Asset | Tail | MAE | RMSE | Brier | AUC |
|---|---:|---:|---:|---:|---:|
| S&P 500 | 95% | 0.01356 | 0.01753 | 0.200 | 0.659 |
| S&P 500 | 97.5% | 0.00498 | 0.01086 | 0.136 | 0.508 |
| S&P 500 | 99% | 0.00279 | 0.00909 | 0.079 | 0.401 |
| Nasdaq | 95% | 0.01224 | 0.01586 | 0.210 | 0.663 |
| Nasdaq | 97.5% | 0.00595 | 0.01095 | 0.161 | 0.575 |
| Nasdaq | 99% | 0.00226 | 0.00866 | 0.079 | 0.385 |
| TLT | 95% | 0.00908 | 0.01314 | 0.241 | 0.627 |
| EUR/USD | 95% | 0.00351 | 0.00398 | 0.231 | 0.580 |
| USD/JPY | 95% | 0.00672 | 0.00777 | 0.241 | 0.553 |
| Bitcoin | 95% | 0.03671 | 0.04838 | 0.200 | 0.632 |
| Bitcoin | 99% | 0.00709 | 0.02453 | 0.049 | 0.578 |

### Interpretation

The current experiments suggest that **episode-severity forecasting is generally more promising than episode-occurrence classification**.

However, performance varies across:

- assets;
- tail levels;
- market conditions;
- forecast specifications.

The results should therefore not be interpreted as evidence of universal early-warning capability.

---

# VaR / ES Benchmark Results

Average FZ0 scores from the current benchmark experiment:

| Tail Level | Historical Simulation | EWMA | EVT |
|---:|---:|---:|---:|
| 95% | -8.52 | -10.05 | -8.52 |
| 97.5% | -9.55 | -13.24 | -9.50 |
| 99% | -11.83 | -21.49 | -11.60 |

Lower values are preferred under the scoring convention used.

In the current benchmark experiment, EWMA produced the strongest average score among the evaluated benchmark specifications.

### Important

This result is **not** used to rank CTER against VaR/ES because CTER forecasts a different statistical target.

---

# Incremental Information

One of the most important empirical questions is:

> **Does adding CTER to a conventional risk forecast improve predictive performance?**

The current aggregate statistical inference does not establish statistically significant universal improvement.

## 95% Tail

Mean MAE difference:

$$
+0.0003196
$$

Bootstrap confidence interval:

$$
[0.000056,\;0.000754]
$$

## 97.5% Tail

Mean MAE difference:

$$
+0.0000742
$$

Bootstrap confidence interval:

$$
[-0.000011,\;0.000159]
$$

## 99% Tail

Mean MAE difference:

$$
-0.0000346
$$

Bootstrap confidence interval:

$$
[-0.0000877,\;0.0000057]
$$

### Interpretation

The aggregate evidence does **not** establish statistically significant incremental forecasting improvement.

The appropriate conclusion is:

> **CTER is a distinct path-dependent risk functional, but its incremental predictive value is asset-, horizon-, tail-, and estimator-dependent.**

---

# Estimator Sensitivity

A sensitivity analysis examined fixed combinations of baseline forecasts and CTER.

| Specification | MAE | RMSE |
|---|---:|---:|
| Current CTER | 0.004139 | 0.007248 |
| 50% Baseline + 50% CTER | 0.004030 | 0.007057 |
| 25% Baseline + 75% CTER | 0.004071 | 0.007139 |

These are **estimator sensitivity proxies**.

They are not treated as evidence of an optimized production combination model.

---

# EVT/GPD Estimation Experiment

A rolling Peaks-over-Threshold / Generalized Pareto Distribution experiment was also conducted.

The implementation used:

- historical episode frequency;
- rolling positive episode severities;
- a threshold based on the 75th percentile of positive episode severities;
- GPD fitting to exceedances;
- `β = 95%`;
- empirical fallback under problematic shape estimates.

Selected results:

| Asset | Tail | GPD MAE | Baseline MAE | CTER MAE |
|---|---:|---:|---:|---:|
| Bitcoin | 95% | 0.07837 | 0.00710 | 0.00720 |
| EUR/USD | 95% | 0.00685 | 0.00064 | 0.00064 |
| USD/JPY | 95% | 0.03119 | 0.00365 | 0.00366 |

Overall:

```text
GPD MAE       ≈ 0.038806
Baseline MAE  ≈ 0.003797
CTER MAE      ≈ 0.003836
```

### Interpretation

This is treated as an **estimator-specific negative result**.

It should not be interpreted as evidence that EVT/GPD methods generally perform poorly.

A key limitation is that this implementation was not a complete covariate-conditioned GPD model because the historical feature matrix required for such a specification was not retained in the relevant forecast file.

---

# What the Empirical Results Mean

The current evidence supports three important conclusions.

### 1. Episode structure is measurable

Extreme losses can be organized differently through time even when marginal tail characteristics are similar.

### 2. CTER provides a distinct risk dimension

CTER explicitly represents:

- episode occurrence;
- cumulative episode severity;
- temporal clustering.

### 3. Forecasting superiority is not universal

The current empirical results do not demonstrate that adding CTER always improves VaR/ES forecasting.

This is an important research finding rather than a result to be hidden.

---

# Relationship to Existing Literature

The framework is related to several established research areas.

## Value-at-Risk

VaR is a marginal quantile of the loss distribution.

## Expected Shortfall

ES measures expected severity beyond a specified tail threshold.

## Extreme Value Theory

EVT provides statistical methods for modeling rare and extreme observations.

## Clustered Exceedances

Extreme-value research has established methods for studying dependent and clustered exceedances.

## Aggregate Excess Measures

Aggregate excess approaches study cumulative severity associated with extreme events.

## Drawdown Risk

Drawdown measures focus on losses relative to previous wealth peaks.

## CTER

The present framework uses:

> **cumulative threshold exceedance within the worst contiguous extreme-loss episode over a forecast horizon.**

The state variable and economic interpretation therefore differ from conventional drawdown measures.

---

# Novelty Position

The research does **not** claim that:

- cumulative exceedances are new;
- clustered extremes are new;
- path-dependent risk measures are new;
- EVT is new;
- conditional tail risk is new.

These are established areas of research.

The potential contribution instead lies in the **specific financial-risk formulation and integration** of these concepts.

The proposed framework combines:

1. threshold exceedance identification;
2. contiguous episode construction;
3. cumulative episode severity;
4. worst-episode selection;
5. upper-tail modeling of worst-episode severity;
6. conditional episode probability;
7. conditional upper-tail episode severity.

The current literature search did not identify an existing financial risk measure using the exact names:

**Tail Episode Risk (TER)**

and

**Conditional Tail Episode Risk (CTER)**

with the same proposed formulation.

Accordingly, the research positions CTER as:

> **A specific path-dependent financial-risk formulation built from established concepts in extreme-value theory, clustered exceedances, aggregate excess severity, drawdown risk, and conditional tail-risk modeling.**

---

# Model Risk and Governance

Any financial risk measure intended for practical use must address model risk.

Important considerations include:

## Threshold Selection

Results depend on the selected threshold:

$$
q_{t,\alpha}
$$

## Forecast Horizon

Episode risk depends on:

$$
H
$$

A 10-day episode-risk measure is not automatically equivalent to a 1-day or 30-day measure.

## Tail Probability

The selected:

$$
\beta
$$

affects the severity distribution.

## Dependence Modeling

Incorrect dependence assumptions can materially affect episode risk.

## Parameter Instability

Extreme-event estimates can be unstable in small samples.

## Regime Changes

Relationships learned from historical data may not remain stable across market regimes.

## Data Quality

Extreme observations are particularly sensitive to:

- stale prices;
- market closures;
- liquidity effects;
- bad ticks;
- data revisions;
- market microstructure effects.

## Validation

A production CTER system would require:

- independent validation;
- backtesting;
- stress testing;
- sensitivity analysis;
- model governance;
- monitoring;
- documentation.

---

# Limitations

The current research has several limitations.

## 1. Univariate Core

The current TER implementation is primarily univariate.

A full multivariate formulation is needed for:

- portfolio-level episode risk;
- cross-asset contagion;
- systemic risk;
- multi-factor risk.

## 2. Threshold Dependence

Results depend on threshold selection.

## 3. Horizon Dependence

Episode risk depends on the forecast horizon.

## 4. Rare-Event Sample Size

Deep-tail analysis can suffer from limited observations.

## 5. Conditional Model Dependence

CTER depends on the models used to estimate:

$$
P(E_{t,H}=1|X_t)
$$

and

$$
ES_\beta(S^*_{t,H}|E_{t,H}=1,X_t,Z_t)
$$

## 6. Estimator Dependence

Different statistical and machine-learning estimators may produce different results.

## 7. Regime Instability

Historical relationships may not persist under structural market changes.

## 8. No Universal Superiority Claim

The current research does not establish that CTER universally improves conventional VaR/ES forecasting.

---

# Future Research

The framework provides several directions for future research.

## Multivariate CTER

Develop a portfolio-level formulation capable of modeling cross-asset extreme-loss episodes.

## Dynamic Thresholds

Investigate state-dependent and volatility-adjusted thresholds.

## Conditional EVT

Develop a fully covariate-conditioned GPD/EVT estimator for episode severity.

## Machine Learning

Evaluate:

- XGBoost
- LightGBM
- Random Forest
- LSTM
- Temporal Convolutional Networks
- Transformers

for:

- episode probability;
- episode severity;
- joint CTER estimation.

## Regime-Switching CTER

Incorporate market states such as:

- low volatility;
- normal conditions;
- crisis;
- liquidity stress;
- monetary-policy transitions.

## Portfolio Applications

Extend the framework to:

- multi-asset portfolios;
- hedge funds;
- bank trading books;
- market-making portfolios;
- derivatives portfolios;
- systematic trading strategies.

## High-Frequency Applications

Investigate CTER using intraday data for:

- liquidity shocks;
- order-book deterioration;
- volatility clustering;
- market microstructure stress;
- extreme intraday episodes.

## Regulatory Risk Applications

Investigate whether episode-based information can complement:

- market-risk capital;
- stress testing;
- liquidity-risk frameworks;
- risk appetite frameworks;
- model-risk governance.

---

# Repository Structure

```text
cter-paper1/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── .gitignore
│
├── configs/
│   └── experiment_manifest.json
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   ├── reproducibility.md
│   ├── data_sources.md
│   ├── experiment_manifest.json
│   └── paper_results_map.md
│
├── experiments/
│   ├── empirical/
│   ├── simulation/
│   └── robustness/
│
├── notebooks/
│   ├── exploratory/
│   ├── simulation/
│   └── empirical/
│
├── paper/
│   ├── manuscript.md
│   ├── sections/
│   ├── tables/
│   └── figures/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── benchmarks/
│   ├── evaluation/
│   └── visualization/
│
└── tests/
```

---

# Repository Components

| Directory | Purpose |
|---|---|
| `configs/` | Experiment configurations and manifests |
| `data/` | Raw, intermediate, and processed datasets |
| `docs/` | Reproducibility and data documentation |
| `experiments/` | Simulation, empirical, and robustness experiments |
| `notebooks/` | Exploratory and research notebooks |
| `paper/` | Manuscript, sections, tables, and figures |
| `results/` | Generated research results |
| `src/` | Core research implementation |
| `tests/` | Validation and testing |

---

# Reproducibility

The project follows a reproducible research workflow.

## Principles

### 1. Preserve Original Experiments

Research outputs should not be silently overwritten after observing results.

### 2. Separate Exploration from Final Evaluation

Exploratory experiments are distinguished from the frozen empirical protocol.

### 3. Preserve Negative Findings

Experiments that do not improve forecasting performance are retained where relevant.

### 4. Avoid Post-OOS Tuning

The final out-of-sample period is not repeatedly optimized after observing results.

### 5. Document Assumptions

Thresholds, horizons, rolling windows, estimators, and metrics are explicitly documented.

### 6. Track Experiments

Experiment configurations and outputs are mapped to the research results.

---

# Data

The empirical study uses financial-market data covering:

- S&P 500;
- Nasdaq;
- TLT;
- EUR/USD;
- USD/JPY;
- Bitcoin.

Stress indicators include:

- VIX;
- NFCI;
- STLFSI.

Additional regime information is used in the empirical research where applicable.

## Data Policy

Raw third-party market data may not be redistributed through the public repository where licensing restrictions apply.

The repository therefore separates:

```text
Raw Data
   ↓
Data Documentation
   ↓
Processing
   ↓
Research Outputs
```

See:

```text
docs/data_sources.md
```

for data-source and reproducibility documentation.

---

# Reproducing the Research

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/cter-paper1.git
cd cter-paper1
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then follow:

```text
docs/reproducibility.md
```

for the complete experiment sequence.

---

# Recommended Reproduction Sequence

```text
1. Data preparation
        ↓
2. Baseline VaR / ES estimation
        ↓
3. TER construction
        ↓
4. CTER estimation
        ↓
5. Monte Carlo experiments
        ↓
6. Adversarial stress tests
        ↓
7. Rolling OOS forecasts
        ↓
8. Calibration tests
        ↓
9. Incremental forecast comparison
        ↓
10. Statistical inference
        ↓
11. Tables and figures
        ↓
12. Manuscript results
```

---

# Research Workflow

```text
                 ┌──────────────────┐
                 │   MARKET DATA    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ DATA PROCESSING  │
                 └────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      ┌───────────────┐       ┌───────────────┐
      │ VaR / ES      │       │ TER / CTER    │
      │ Benchmarks    │       │ Framework     │
      └───────┬───────┘       └───────┬───────┘
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ OUT-OF-SAMPLE    │
                 │ FORECASTING      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ BACKTESTING      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ STATISTICAL      │
                 │ INFERENCE        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ RESEARCH RESULTS │
                 └──────────────────┘
```

---

# Why This Repository Matters

The repository is intended to serve three purposes.

## Academic Research

Provide a transparent implementation of a path-dependent tail-risk framework for further theoretical and empirical investigation.

## Quantitative Finance

Provide a framework that researchers can test against:

- VaR;
- ES;
- EVT;
- volatility models;
- machine-learning models;
- alternative path-dependent measures.

## Financial Risk Management

Provide a foundation for exploring whether episode-based risk information can complement:

- market-risk models;
- stress-testing frameworks;
- liquidity-risk analysis;
- model-risk systems.

---

# Research Integrity

This repository intentionally preserves both positive and negative findings.

The project does **not**:

- selectively remove unfavorable experiments;
- claim universal superiority over VaR or ES;
- describe exploratory results as confirmatory evidence;
- tune models using future out-of-sample information;
- present CTER as a replacement for established regulatory risk measures.

The objective is to provide a research framework that can be:

- replicated;
- criticized;
- independently validated;
- extended;
- applied to new datasets.

---

# Research Status

| Component | Status |
|---|:---:|
| TER formulation | ✅ Complete |
| CTER formulation | ✅ Complete |
| Theoretical analysis | ✅ Complete |
| Monte Carlo experiments | ✅ Complete |
| Adversarial stress tests | ✅ Complete |
| Multi-asset empirical evaluation | ✅ Complete |
| VaR/ES benchmarks | ✅ Complete |
| Incremental testing | ✅ Complete |
| Statistical inference | ✅ Complete |
| Estimator sensitivity | ✅ Complete |
| EVT/GPD experiment | ✅ Complete |
| Reproducibility structure | ✅ Complete |
| Multivariate CTER | 🔬 Future Research |
| ML-based CTER | 🔬 Future Research |
| Intraday CTER | 🔬 Future Research |
| Production implementation | 🔬 Future Research |

### Current Stage

**Working Paper — September 2026**

The mathematical framework and primary experiments have been developed. Additional extensions—including multivariate, machine-learning, and high-frequency formulations—remain areas for future research.

---

# Citation

If you use the methodology, conceptual framework, code, or research results, please cite:

```bibtex
@article{neupane2026cter,
  author  = {Neupane, Niraj},
  title   = {Conditional Tail Episode Risk: A Path-Dependent Framework for Extreme-Loss Episodes Beyond Value-at-Risk and Expected Shortfall},
  year    = {2026},
  month   = {September},
  note    = {Working Paper}
}
```

---

# Disclaimer

This repository is provided for academic and research purposes.

Nothing in this repository constitutes:

- investment advice;
- financial advice;
- trading advice;
- risk-management advice;
- regulatory advice;
- or a recommendation to deploy CTER in a live financial institution.

Historical and simulated results do not guarantee future performance.

A production implementation would require appropriate:

- model validation;
- governance;
- stress testing;
- backtesting;
- monitoring;
- documentation;
- regulatory review.

---

# License

The software and code components of this repository are released under the **MIT License**, unless otherwise specified.

See [`LICENSE`](LICENSE).

Third-party market data, external research materials, and other copyrighted materials may be subject to separate licensing restrictions.

---

# Author

## Niraj Neupane

**Chartered Accountant (ICAI)**  
**Financial Economics · Quantitative Finance · Financial Risk · AI/ML**

### Research Interests

- Quantitative Finance
- Financial Econometrics
- Extreme Value Theory
- Market Risk
- Derivatives
- Quantitative Trading
- Machine Learning for Finance
- Financial Engineering
- Model Risk
- Market Microstructure
- Risk Management

---

# The Core Idea

Traditional tail-risk measures primarily ask:

> **How large can an individual extreme loss become?**

TER asks:

> **How severe can the worst cumulative extreme-loss episode become?**

CTER goes one step further:

> **How likely is the episode, and how severe could it become if it occurs?**

In simplified form:

```text
                         TAIL RISK
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
         VaR                ES               CTER
          │                 │                 │
          ▼                 ▼                 ▼
   Tail threshold     Tail severity     Episode risk
                                            │
                                    ┌───────┴───────┐
                                    │               │
                                    ▼               ▼
                              Probability       Severity
                                    │               │
                                    └───────┬───────┘
                                            ▼
                                           CTER
```

---

## CTER

### Conditional Tail Episode Risk

**A path-dependent framework for measuring and forecasting extreme-loss episodes beyond Value-at-Risk and Expected Shortfall.**

---

<p align="center">

**Research • Quantitative Finance • Tail Risk • Financial Econometrics • Risk Management**

</p>
