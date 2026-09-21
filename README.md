# Conditional Tail Episode Risk (CTER)

## A Path-Dependent Framework for Extreme-Loss Episodes Beyond Value-at-Risk and Expected Shortfall

**Author:** Niraj Neupane  
**Status:** Working Paper / Research Repository  
**Version:** v0.1.0  
**Date:** September 2026  
**Research Area:** Quantitative Finance · Financial Risk Management · Financial Econometrics · Extreme Value Theory · Machine Learning · Market Risk

---

## Abstract

Value-at-Risk (VaR) and Expected Shortfall (ES) are central tools for measuring financial tail risk, but both are fundamentally marginal tail measures. They characterize the severity of losses relative to a distributional threshold without explicitly representing how extreme losses evolve across a forecast horizon.

This repository develops **Tail Episode Risk (TER)** and **Conditional Tail Episode Risk (CTER)** as a path-dependent framework for analyzing extreme-loss episodes.

Rather than evaluating extreme losses one observation at a time, the framework considers a future loss path

\[
\mathbf{L}_{t,H}
=
(L_{t+1},L_{t+2},\ldots,L_{t+H})
\]

and identifies contiguous periods in which losses exceed a specified tail threshold. The severity of each episode is measured by the cumulative threshold-exceedance loss:

\[
S_j
=
\sum_{k\in j}
(L_{t+k}-q_{t,\alpha})_+.
\]

The worst episode over the forecast horizon is then

\[
S^*_{t,H}
=
\max_{j\in\mathcal{E}_{t,H}} S_j,
\]

where \(\mathcal{E}_{t,H}\) denotes the set of contiguous exceedance episodes.

The proposed **Tail Episode Risk** measure is the upper-tail quantile of this worst-episode severity:

\[
TER_{\alpha,\beta,H}
=
Q_\beta(S^*_{t,H}).
\]

The conditional formulation is

\[
CTER_{\alpha,\beta,H,t}
=
P(E_{t,H}=1\mid X_t)
\cdot
ES_\beta
\left[
S^*_{t,H}
\mid
E_{t,H}=1,X_t,Z_t
\right],
\]

where \(E_{t,H}\) indicates whether at least one threshold exceedance occurs during the forecast horizon, \(X_t\) represents information available at the forecast origin, and \(Z_t\) represents episode-related state information such as persistence, duration, escalation, and clustering.

The objective is **not** to replace VaR or ES. Instead, CTER is designed to measure a different risk dimension: the potential severity of a concentrated extreme-loss episode over a future horizon.

---

# 1. Research Motivation

Financial risk can be path-dependent.

Two portfolios can have similar marginal loss distributions and therefore similar VaR and ES while exhibiting very different temporal structures.

Consider two six-period loss paths:

\[
A=(4,0,4,0,4,0)
\]

and

\[
B=(4,4,4,0,0,0).
\]

Suppose the tail threshold is

\[
q=2.
\]

Both paths contain the same individual observations and therefore have the same marginal tail characteristics.

However:

- Path A contains three isolated exceedances.
- Path B contains one persistent exceedance episode.
- The cumulative severity of the largest episode is therefore different.

For Path A:

\[
S^*_A=2.
\]

For Path B:

\[
S^*_B=6.
\]

Therefore,

\[
VaR_A=VaR_B,
\]

\[
ES_A=ES_B,
\]

while

\[
TER_A\neq TER_B.
\]

This illustrates the central motivation of the framework:

> **Marginal tail measures can be similar even when the temporal organization of extreme losses is materially different.**

CTER attempts to capture this additional path-dependent dimension.

---

# 2. Research Question

The central research question is:

> **Can a path-dependent measure of cumulative extreme-loss episodes provide information about future tail-loss behavior that is not captured by marginal VaR and Expected Shortfall alone?**

The research investigates this question through:

1. Mathematical construction of TER and CTER.
2. Theoretical analysis of their properties.
3. Monte Carlo simulations under different dependence structures.
4. Adversarial stress tests.
5. Out-of-sample empirical evaluation.
6. Comparison with conventional VaR and ES benchmarks.
7. Incremental forecast evaluation.
8. Estimation-layer sensitivity analysis.
9. Robustness and model-risk analysis.

---

# 3. Main Contributions

The repository develops and evaluates two related constructs.

## 3.1 Tail Episode Risk (TER)

TER measures the upper-tail severity of the **worst cumulative threshold-exceedance episode** within a future horizon.

\[
TER_{\alpha,\beta,H}
=
Q_\beta(S^*_{t,H}).
\]

It is therefore a path-dependent functional rather than a purely marginal tail statistic.

---

## 3.2 Conditional Tail Episode Risk (CTER)

CTER decomposes episode risk into two components:

1. The probability that an extreme-loss episode occurs.
2. The conditional upper-tail severity of that episode.

\[
CTER_{\alpha,\beta,H,t}
=
P(E_{t,H}=1|X_t)
\times
ES_\beta
[
S^*_{t,H}
|
E_{t,H}=1,X_t,Z_t
].
\]

This decomposition separates:

### Episode occurrence

\[
P(E_{t,H}=1|X_t)
\]

from

### Episode severity

\[
ES_\beta
[
S^*_{t,H}
|
E_{t,H}=1,X_t,Z_t
].
\]

This allows the framework to distinguish between:

- the likelihood of entering an extreme-loss episode, and
- the potential severity of that episode once it occurs.

---

# 4. TER Framework

## 4.1 Forecast Loss Path

For forecast origin \(t\) and horizon \(H\), define the future loss path:

\[
\mathbf{L}_{t,H}
=
(L_{t+1},L_{t+2},...,L_{t+H}).
\]

Let

\[
q_{t,\alpha}
\]

represent the tail-loss threshold at confidence level \(\alpha\).

---

## 4.2 Threshold Exceedance

Define the positive threshold exceedance as:

\[
(L_{t+k}-q_{t,\alpha})_+
=
\max(L_{t+k}-q_{t,\alpha},0).
\]

An observation contributes to episode severity only when the loss exceeds the threshold.

---

## 4.3 Exceedance Episodes

Let

\[
\mathcal{E}_{t,H}
\]

represent the collection of contiguous exceedance episodes within the forecast path.

For episode \(j\):

\[
S_j
=
\sum_{k\in j}
(L_{t+k}-q_{t,\alpha})_+.
\]

---

## 4.4 Worst Episode

Define:

\[
S^*_{t,H}
=
\max_{j\in\mathcal{E}_{t,H}} S_j.
\]

If no exceedance occurs, the episode burden is zero.

---

## 4.5 Tail Episode Risk

The TER measure is:

\[
TER_{\alpha,\beta,H}
=
Q_\beta(S^*_{t,H}).
\]

Here:

- \(\alpha\) = threshold confidence level.
- \(\beta\) = tail probability used for the episode-severity distribution.
- \(H\) = forecast horizon.
- \(S^*_{t,H}\) = worst cumulative threshold-exceedance episode.

---

# 5. Conditional Tail Episode Risk

The conditional formulation introduces an explicit episode-occurrence probability.

Define:

\[
E_{t,H}
=
1
\left[
\max_{1\leq k\leq H}
L_{t+k}
>
q_{t,\alpha}
\right].
\]

Then:

\[
CTER_{\alpha,\beta,H,t}
=
p_t
\cdot
ES_\beta(B_t|B_t>0,X_t,Z_t),
\]

where

\[
p_t
=
P(E_{t,H}=1|X_t)
\]

and \(B_t\) represents the relevant episode-burden variable.

The implementation used in the final empirical specification is severity-centered.

Episode-related information such as:

- duration,
- persistence,
- escalation,
- clustering,
- market state,

is treated as predictive state information rather than automatically multiplying the severity measure through mechanical penalty terms.

---

# 6. Conceptual Relationship to VaR and ES

The framework is intended to complement—not replace—standard tail-risk measures.

| Measure | Primary object | Marginal / Path-dependent |
|---|---|---|
| VaR | Tail threshold | Marginal |
| ES | Average tail severity | Marginal |
| TER | Upper tail of worst cumulative threshold-exceedance episode | Path-dependent |
| CTER | Episode probability × conditional tail episode severity | Path-dependent / Conditional |

### VaR

VaR answers:

> How large is the loss threshold associated with a specified tail probability?

### ES

ES answers:

> Conditional on being beyond the VaR threshold, how severe is the loss on average?

### TER

TER asks:

> How severe can the worst cumulative threshold-exceedance episode become over a future horizon?

### CTER

CTER asks:

> Given current information, what is the probability of entering an extreme-loss episode and how severe could that episode be in its upper tail?

These are related but distinct questions.

---

# 7. Theoretical Properties

The framework studies several structural properties.

## 7.1 Monotonicity

Conditional on a common threshold construction, increasing the underlying loss path cannot reduce the corresponding episode burden.

---

## 7.2 Positive Homogeneity

Under consistent scaling of losses and thresholds:

\[
TER(cL)=c\,TER(L),
\]

for \(c>0\).

The corresponding scaling behavior applies to CTER under consistent transformation of the threshold and severity components.

---

## 7.3 Translation Equivariance

When the threshold is translated consistently with the loss process, the exceedance structure is preserved.

---

## 7.4 Temporal Clustering Sensitivity

Unlike purely marginal measures, TER and CTER respond to the temporal arrangement of losses.

This is a central feature of the framework.

---

## 7.5 Distinction from VaR and ES

The constructed-path example demonstrates that identical marginal observations can produce different TER values.

Therefore, TER contains information about temporal organization that marginal VaR and ES do not directly encode.

---

## 7.6 Distinction from Drawdown Measures

Drawdown measures are based on cumulative declines from prior wealth peaks.

TER instead operates directly on threshold exceedances of a loss process.

The state variables and economic interpretation are therefore different.

---

# 8. Important Scope of the Claims

The framework does **not** claim that:

- CTER universally outperforms VaR.
- CTER universally outperforms ES.
- CTER should replace regulatory VaR or ES.
- TER is a universally superior risk measure.
- The framework is the first path-dependent tail-risk measure.
- All forms of temporal dependence are captured by the current implementation.
- The empirical results establish universal forecasting superiority.

The research instead evaluates whether the proposed functional captures an economically meaningful dimension of tail risk that conventional marginal measures do not explicitly represent.

---

# 9. Novelty Position

The literature contains established work on:

- extreme-value theory,
- clustered exceedances,
- aggregate excess measures,
- drawdown risk,
- conditional tail risk,
- dynamic VaR/ES,
- path-dependent risk measures,
- machine-learning-based tail-risk forecasting.

In particular, aggregate excess severity has been studied in the extreme-event literature, and drawdown-based risk measures provide another important path-dependent framework.

The proposed contribution should therefore not be interpreted as claiming that cumulative exceedances themselves are entirely new.

The research contribution is instead centered on the specific financial-risk formulation:

1. identifying contiguous threshold-exceedance episodes,
2. aggregating threshold-exceedance severity within episodes,
3. taking the worst episode over a forecast horizon,
4. modeling the upper tail of that episode severity,
5. and decomposing conditional episode risk into occurrence probability and conditional upper-tail severity.

The current literature search did not identify an existing financial risk measure using the exact names **Tail Episode Risk (TER)** and **Conditional Tail Episode Risk (CTER)** with the proposed formulation.

Accordingly, the appropriate novelty statement is:

> **The proposed framework builds on established concepts from extreme-value theory, clustered exceedances, aggregate excess measures, drawdown risk, and conditional tail-risk modeling. The potential contribution lies in the specific integration and financial-risk formulation of these components, particularly the conditional probability × upper-tail severity construction applied to the maximum cumulative threshold-exceedance episode.**

---

# 10. Research Design

The empirical research follows a strict out-of-sample protocol.

## Assets

The empirical study covers:

- S&P 500
- Nasdaq
- U.S. Treasury ETF / TLT
- EUR/USD
- USD/JPY
- Bitcoin

---

## Out-of-Sample Period

Primary evaluation period:

**2020–2025**

---

## Forecast Horizon

The primary forecast horizon is:

\[
H=10
\]

trading days.

---

## Tail Levels

The empirical evaluation considers:

\[
\alpha\in
\{95\%,97.5\%,99\%\}.
\]

---

## Rolling Training Window

The primary rolling estimation window is:

\[
1250
\]

observations.

---

## Benchmark Models

The benchmark set includes:

- Historical Simulation (HS)
- Exponentially Weighted Moving Average (EWMA)
- GARCH-t
- EVT / Peaks-over-Threshold where applicable

Corresponding VaR and ES forecasts are evaluated.

---

# 11. Statistical Evaluation

The empirical framework evaluates multiple dimensions of performance.

## 11.1 VaR Calibration

VaR forecasts are evaluated using:

- Kupiec unconditional coverage test
- Christoffersen independence test
- Conditional coverage testing
- Quantile loss

---

## 11.2 VaR–ES Joint Evaluation

VaR and ES are evaluated jointly using strictly consistent scoring methods based on the Fissler-Ziegel framework.

---

## 11.3 Episode Probability

The CTER occurrence component is evaluated using:

- Brier score
- ROC/AUC
- episode-probability calibration

---

## 11.4 Episode Severity

Episode severity forecasts are evaluated using:

- MAE
- RMSE

---

## 11.5 Incremental Information

The research compares:

### Baseline

\[
VaR/ES
\]

against

### Extended specification

\[
VaR/ES + CTER.
\]

The objective is to determine whether adding the CTER information provides incremental forecasting value.

---

## 11.6 Statistical Inference

The analysis includes:

- paired out-of-sample loss comparisons,
- Diebold-Mariano-type comparisons where appropriate,
- block bootstrap confidence intervals,
- multiple-testing considerations.

No OOS tuning is performed after observing the final empirical results.

---

# 12. Monte Carlo Experiments

Monte Carlo experiments investigate whether temporal dependence affects episode risk differently from marginal tail measures.

The simulations include Gaussian and Student-t dependence structures with correlations:

\[
\rho\in
\{0.0,0.3,0.7,0.9\}.
\]

The simulation evaluates:

- VaR,
- ES,
- TER.

A central result is that increasing temporal dependence can materially change TER while marginal VaR and ES remain comparatively stable.

### Gaussian Simulation

| Correlation | VaR 95% | ES 95% | TER 95% | TER 99% |
|---:|---:|---:|---:|---:|
| 0.0 | 1.648 | 2.067 | 0.969 | 1.477 |
| 0.3 | 1.650 | 2.063 | 0.997 | 1.681 |
| 0.7 | 1.641 | 2.054 | 1.101 | 2.752 |
| 0.9 | 1.655 | 2.079 | 1.167 | 4.617 |

### Student-t Simulation

| Correlation | VaR 95% | ES 95% | TER 95% | TER 99% |
|---:|---:|---:|---:|---:|
| 0.0 | 2.013 | 2.864 | 1.986 | 3.941 |
| 0.3 | 2.021 | 2.888 | 2.148 | 4.355 |
| 0.7 | 2.020 | 2.909 | 2.291 | 6.566 |
| 0.9 | 1.990 | 2.855 | 1.929 | 9.010 |

These results provide a controlled demonstration of the distinction between marginal tail severity and temporal episode structure.

---

# 13. Adversarial Stress Tests

The framework is also evaluated under several stylized stress scenarios.

The experiments include:

1. Sudden crash
2. Slow deterioration
3. Volatility explosion
4. Tail thickening
5. Liquidity collapse
6. Cross-asset contagion

Illustrative results:

| Scenario | VaR 95% | ES 95% | TER |
|---|---:|---:|---:|
| Sudden crash | 3.385 | 5.500 | 3.855 |
| Slow deterioration | 2.755 | 2.800 | 4.485 |
| Volatility explosion | 3.665 | 3.800 | 8.285 |
| Tail thickening | 2.865 | 3.000 | 1.355 |
| Liquidity collapse | 3.365 | 3.500 | 9.040 |
| Cross-asset contagion | 2.755 | 2.800 | 2.765 |

These stress tests illustrate that different forms of stress can affect marginal tail severity and episode persistence differently.

The cross-asset contagion experiment should not be interpreted as a full multivariate CTER model. The current TER implementation is fundamentally univariate, with multivariate extensions identified as future research.

---

# 14. Empirical Results

The empirical results are intentionally reported without selecting only favorable outcomes.

The current evidence suggests that CTER can provide useful information about the structure and severity of extreme-loss episodes, but the incremental forecasting results are **mixed**.

---

## 14.1 Episode Forecasting

Across assets, episode severity forecasting is generally more promising than episode-occurrence classification.

For example, the final multi-asset evaluation produced the following results:

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

These results should not be interpreted as universal early-warning evidence.

---

# 15. VaR / ES Benchmark Results

The current benchmark evaluation produced the following average FZ0 scores:

| Tail Level | Historical Simulation | EWMA | EVT |
|---:|---:|---:|---:|
| 95% | -8.52 | -10.05 | -8.52 |
| 97.5% | -9.55 | -13.24 | -9.50 |
| 99% | -11.83 | -21.49 | -11.60 |

Lower values are preferred under the scoring convention used.

In the current benchmark experiment, EWMA produced the strongest average score among the evaluated benchmark specifications.

This repository does **not** interpret that result as evidence that CTER is inferior, because CTER forecasts a different target from the VaR/ES joint functional and therefore should not be directly ranked using the same scoring target.

---

# 16. Incremental Information Test

A key empirical question is whether adding CTER to an existing risk forecast improves the prediction of future losses.

The final statistical inference does **not** provide statistically significant aggregate evidence of universal incremental improvement.

Equal-weight aggregate comparisons produced:

### 95%

Mean MAE difference:

\[
+0.0003196
\]

with bootstrap confidence interval approximately:

\[
[0.000056,\;0.000754].
\]

### 97.5%

Mean MAE difference:

\[
+0.0000742
\]

with confidence interval approximately:

\[
[-0.000011,\;0.000159].
\]

### 99%

Mean MAE difference:

\[
-0.0000346
\]

with confidence interval approximately:

\[
[-0.0000877,\;0.0000057].
\]

The aggregate evidence therefore does not establish statistically significant incremental forecasting improvement.

This is an important result and is retained in the repository rather than excluded.

---

# 17. Asset-Level Results

The asset-level results are heterogeneous.

For example:

- Bitcoin at the 95% level showed a statistically detectable deterioration in the extended loss forecast in the current comparison.
- EUR/USD at the 95% level also showed a statistically detectable deterioration.
- Several other asset/tail combinations showed small and statistically insignificant differences.
- Some deeper-tail specifications showed small improvements, but these were generally not statistically significant.

The appropriate conclusion is therefore:

> **The current evidence supports CTER as a distinct path-dependent risk functional, but does not establish universal incremental forecasting superiority over existing risk forecasts.**

---

# 18. Estimator-Layer Sensitivity

A separate sensitivity analysis examined simple fixed combinations of baseline forecasts and CTER.

Current average results:

| Specification | MAE | RMSE |
|---|---:|---:|
| Current CTER | 0.004139 | 0.007248 |
| 50% Baseline + 50% CTER | 0.004030 | 0.007057 |
| 25% Baseline + 75% CTER | 0.004071 | 0.007139 |

These results are estimator sensitivity proxies rather than a fully optimized combination model.

The repository therefore does not treat these combinations as a final production estimator.

---

# 19. EVT / GPD Estimator Experiment

A genuine rolling Peaks-over-Threshold / Generalized Pareto Distribution experiment was also conducted.

The experiment used:

- historical episode frequency,
- rolling positive episode severities,
- a threshold based on the 75th percentile of positive episode severities,
- GPD fitting to exceedances,
- \(\beta=95\%\),
- empirical fallback under problematic shape estimates.

The current implementation produced higher MAE than the benchmark specifications in the tested configuration.

Illustrative results:

| Asset | Tail | GPD MAE | Baseline MAE | Current CTER MAE |
|---|---:|---:|---:|---:|
| Bitcoin | 95% | 0.07837 | 0.00710 | 0.00720 |
| EUR/USD | 95% | 0.00685 | 0.00064 | 0.00064 |
| USD/JPY | 95% | 0.03119 | 0.00365 | 0.00366 |

Overall:

\[
MAE_{GPD}\approx0.038806
\]

versus approximately:

\[
MAE_{baseline}\approx0.003797.
\]

This is **not** interpreted as evidence that EVT/GPD methods generally perform poorly.

The experiment has an important limitation:

> The current GPD implementation does not use the full covariate-conditioned feature matrix because the historical feature matrix required for a complete conditional GPD implementation was not retained in the relevant forecast file.

Accordingly, this result is treated as a negative estimator-specific finding rather than a rejection of EVT methods.

---

# 20. Why the Mixed Results Matter

The empirical results provide an important methodological distinction.

A new risk functional can be:

- mathematically distinct,
- economically interpretable,
- sensitive to temporal clustering,
- useful for describing extreme-loss episodes,

without necessarily producing lower predictive loss than an established forecasting model in every dataset.

The current results therefore support a more precise interpretation:

> **CTER provides a framework for measuring and modeling path-dependent extreme-loss episodes. Whether that information improves a specific production forecasting system depends on the asset, tail level, estimator, forecast horizon, and information set.**

This is a central principle of the research.

---

# 21. Relation to Existing Risk Measures

CTER should be understood relative to several established families of risk measures.

## Value-at-Risk

VaR is a marginal quantile of the loss distribution.

CTER additionally considers temporal organization across a forecast horizon.

---

## Expected Shortfall

ES summarizes average loss severity beyond a VaR threshold.

CTER instead focuses on the upper-tail severity of the worst cumulative threshold-exceedance episode.

---

## Drawdown Measures

Drawdown-based measures evaluate losses relative to previous wealth peaks.

CTER evaluates cumulative threshold exceedances in the loss process.

---

## Aggregate Excess Measures

Aggregate excess approaches sum extreme-event exceedances.

CTER incorporates cumulative exceedance into a specific financial path-dependent episode framework and applies an upper-tail functional to the worst episode.

---

## Extreme-Value Theory

EVT provides theoretical tools for modeling rare and extreme observations.

CTER can incorporate EVT-based estimation but is not itself limited to one particular EVT estimator.

---

# 22. Model-Risk and Governance Considerations

A risk measure intended for financial applications must be evaluated not only for mathematical properties but also for model risk.

Important considerations include:

### Threshold selection

Results depend on the threshold \(q_{t,\alpha}\).

### Forecast horizon

Episode behavior can change substantially with \(H\).

### Tail probability

The choice of \(\beta\) affects the severity functional.

### Dependence modeling

Incorrect dependence assumptions can materially affect episode risk.

### Parameter instability

Extreme-event estimates can be unstable in small samples.

### Regime changes

Relationships learned in one market regime may not persist in another.

### Data quality

Extreme observations can be particularly sensitive to stale prices, market closures, liquidity effects, and data errors.

### Backtesting

A production CTER system should be subject to continuous out-of-sample monitoring.

---

# 23. Limitations

The current research has several limitations.

## 23.1 Univariate Core

The current TER formulation is fundamentally univariate.

A full multivariate formulation is required to model:

- cross-asset contagion,
- portfolio-level episodes,
- systemic risk,
- dependence across risk factors.

---

## 23.2 Threshold Dependence

The measure depends on the selected tail threshold.

Different threshold estimation procedures may produce different episode structures.

---

## 23.3 Forecast Horizon

Episode risk is inherently horizon-dependent.

A measure calibrated for ten trading days cannot automatically be interpreted as equivalent to a one-day or thirty-day measure.

---

## 23.4 Rare-Event Sample Size

Deep-tail estimation at 99% and beyond can suffer from limited observations.

---

## 23.5 Conditional Model Specification

CTER performance depends on the quality of the models used to estimate:

\[
P(E_{t,H}=1|X_t)
\]

and

\[
ES_\beta(B_t|B_t>0,X_t,Z_t).
\]

---

## 23.6 Estimation Layer

The current research does not establish one universally optimal estimator for CTER.

Different model classes—including:

- GARCH,
- EVT,
- quantile regression,
- gradient boosting,
- neural networks,
- transformers,
- state-space models,

may produce different results.

---

## 23.7 No Universal Superiority Claim

The empirical evidence does not support a claim that CTER universally improves VaR/ES forecasting.

The framework should instead be evaluated as an additional risk dimension.

---

# 24. Future Research

Several extensions are planned.

## 24.1 Multivariate CTER

Develop a portfolio-level CTER formulation capable of modeling cross-asset extreme-loss episodes.

---

## 24.2 Dynamic Thresholds

Investigate state-dependent and volatility-adjusted thresholds.

---

## 24.3 Conditional EVT

Develop a fully covariate-conditioned GPD / EVT estimator for episode severity.

---

## 24.4 Machine Learning

Evaluate:

- XGBoost
- LightGBM
- Random Forest
- temporal neural networks
- LSTM
- Temporal Convolutional Networks
- Transformers

for:

1. episode probability,
2. episode severity,
3. joint CTER estimation.

---

## 24.5 Regime-Switching CTER

Incorporate market regimes such as:

- low volatility,
- normal volatility,
- crisis,
- liquidity stress,
- monetary-policy transitions.

---

## 24.6 Portfolio Applications

Extend CTER to:

- multi-asset portfolios,
- hedge funds,
- banks,
- market-making books,
- derivatives portfolios,
- systematic trading strategies.

---

## 24.7 Regulatory Risk Applications

Investigate whether episode-based risk information can complement existing:

- market-risk capital,
- stress testing,
- liquidity risk,
- counterparty risk,
- risk appetite frameworks.

---

## 24.8 Intraday and High-Frequency Applications

A natural extension is to investigate whether episode persistence at intraday frequencies provides information about:

- liquidity shocks,
- market microstructure stress,
- order-book deterioration,
- volatility clustering,
- flash-crash-type episodes.

---

# 25. Repository Structure

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
│   ├── README.md
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
