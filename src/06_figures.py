"""
06 -- Figures.

Regenerates every figure in the paper from data/ and results/:
  ab.pdf         A/B temporal-clustering illustration
  mc.pdf         Gaussian AR(1): TER rises with dependence, VaR/ES flat
  episodes.pdf   realized S&P 500 tail-episode severity, 2000-2026
  auc.pdf        episode-occurrence AUC by asset and threshold
  incremental.pdf incremental CTER forecast performance with bootstrap CIs

Both .pdf (for LaTeX) and _png.png (for the README / Word) are written.
Run:  python src/06_figures.py   (after 01-05)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from lib import ASSETS, RESULTS, FIGURES, W, H, BURN, load, worst_episode

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "font.family": "serif"})
C = ["#1b263b", "#415a77", "#778da9", "#b08968"]


def save(name):
    plt.savefig(FIGURES / f"{name}.pdf")
    plt.savefig(FIGURES / f"{name}_png.png", dpi=300)
    plt.close()


def fig_ab():
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, (lab, path, ss) in zip(axs, [("A = (4,0,4,0,4,0)", [4, 0, 4, 0, 4, 0], 2),
                                         ("B = (4,4,4,0,0,0)", [4, 4, 4, 0, 0, 0], 6)]):
        ax.bar(range(6), path, color=C[1], width=0.7); ax.axhline(2, ls="--", c=C[3], lw=1)
        ax.text(5.4, 2.15, "q=2", c=C[3], fontsize=8, ha="right")
        ax.set_title(f"{lab}\n$S^*$={ss}", fontsize=9); ax.set_xticks(range(6)); ax.set_xlabel("t")
    axs[0].set_ylabel("loss")
    fig.suptitle("Identical VaR & ES, different episode severity", fontsize=10, y=1.02)
    plt.tight_layout(); save("ab")


def fig_mc():
    mn = pd.read_csv(RESULTS / "simulation_gaussian.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(mn.rho, mn.VaR95, "o-", c=C[2], label="VaR 95%")
    ax.plot(mn.rho, mn.ES95, "s-", c=C[1], label="ES 95%")
    ax.plot(mn.rho, mn.TER95, "^--", c=C[3], label="TER 95%")
    ax.plot(mn.rho, mn.TER99, "D-", c=C[0], label="TER 99%")
    ax.set_xlabel(r"serial dependence $\rho$"); ax.set_ylabel("risk (std units)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Gaussian AR(1): TER rises with clustering, VaR/ES do not")
    plt.tight_layout(); save("mc")


def fig_episodes():
    sp = load("sp500"); L = (-sp["r"] * 100).values; dts = pd.to_datetime(sp["date"])
    q = np.quantile(L, 0.95)
    S = np.array([worst_episode(L[i:i + H], q) for i in range(len(L) - H)])
    ep = pd.read_csv(RESULTS / "sp500_episodes.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(dts[:len(S)], S, lw=0.6, c=C[1]); ax.fill_between(dts[:len(S)], S, color=C[2], alpha=0.4)
    for _, row in ep.head(5).iterrows():
        d = pd.Timestamp(row["date"])
        ax.annotate(d.strftime("%Y"), (d, row["S_star_pct"]), fontsize=7, ha="center", va="bottom", c=C[0])
    ax.set_ylabel(r"10-day episode severity $S^*$ (%)")
    ax.set_title("S&P 500 realized Tail Episode severity, 2000-2026")
    plt.tight_layout(); save("episodes")


def fig_auc():
    R = pd.read_csv(RESULTS / "cter_results.csv")
    short = {"sp500": "S&P", "nasdaq": "Nasdaq", "tlt_bonds": "TLT",
             "eurusd": "EURUSD", "usdjpy": "USDJPY", "btc": "BTC"}
    piv = R.pivot(index="asset", columns="level", values="AUC").reindex(list(ASSETS))
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(len(piv)); w = 0.26
    for i, lv in enumerate([95.0, 97.5, 99.0]):
        ax.bar(x + (i - 1) * w, piv[lv], w, label=f"{lv}%", color=C[i])
    ax.axhline(0.5, ls="--", c="grey", lw=1); ax.set_xticks(x)
    ax.set_xticklabels([short[a] for a in piv.index])
    ax.set_ylabel("AUC (episode occurrence)"); ax.legend(frameon=False, fontsize=8, title="threshold")
    ax.set_title("Occurrence discrimination is weak and decays into the tail"); ax.set_ylim(0.3, 0.75)
    plt.tight_layout(); save("auc")


def fig_incremental():
    inc = pd.read_csv(RESULTS / "incremental_test.csv"); ys = [2, 1, 0]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for (_, row), y in zip(inc.iterrows(), ys):
        ax.plot([row.ci_low, row.ci_high], [y, y], "-", c=C[1], lw=2.4)
        ax.plot(row.mean_delta_MAE, y, "o", c=C[0], ms=7, zorder=3)
        ax.annotate(f"$\\Delta$={row.mean_delta_MAE:+.4f}", (row.ci_high, y),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, c=C[0])
    ax.axvline(0, ls="--", c=C[3], lw=1.3)
    ax.set_yticks(ys); ax.set_yticklabels(inc.level); ax.set_ylabel("threshold")
    ax.set_xlabel(r"$\mathrm{MAE_{CTER\text{-}augmented}}-\mathrm{MAE_{baseline}}$   (negative favors CTER)")
    ax.set_title("Incremental CTER forecast performance", fontsize=11)
    ax.set_ylim(-0.55, 2.75); ax.set_xlim(-0.00055, 0.00265)
    ax.annotate("favors CTER", (0, 2.6), xytext=(-8, 0), textcoords="offset points",
                ha="right", fontsize=7.5, c="grey")
    ax.annotate("CTER worse", (0, 2.6), xytext=(8, 0), textcoords="offset points",
                ha="left", fontsize=7.5, c="grey")
    plt.tight_layout(); save("incremental")


def main():
    FIGURES.mkdir(exist_ok=True)
    fig_ab(); fig_mc(); fig_episodes(); fig_auc(); fig_incremental()
    print("Saved 5 figures (.pdf and _png.png) -> figures/")


if __name__ == "__main__":
    main()
