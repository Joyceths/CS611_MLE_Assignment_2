"""charts of model performance and stability over time, from the monitoring gold tables

usage: python utils/plot_monitoring.py --snapshotdate 2025-07-01   -> monitoring_report/*.png + monitoring_summary.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from utils import config as C
from utils import model_common as M

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#6b7280", "#e5e7eb"
GREEN_BG, AMBER_BG, RED_BG = "#e8f5ec", "#fdf3dc", "#fde7e4"
STATUS = {"GREEN": "#2f8f4e", "AMBER": "#d99a00", "RED": "#d6403a", "n/a": MUTED}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150})


def _dates(s):
    return pd.to_datetime(s)


def _month_axis(ax):
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))


def _switch_lines(ax, mon, y_text):
    """mark when a new model went live"""
    prev = None
    for _, r in mon.iterrows():
        if prev is not None and r["model_version"] != prev:
            ax.axvline(r["snapshot_date"], color=MUTED, ls=":", lw=1)
            ax.text(r["snapshot_date"], y_text, " " + r["model_version"], color=MUTED, fontsize=8, va="top")
        prev = r["model_version"]


def plot_performance(ax, mon):
    p = mon.dropna(subset=["auc"]).copy()
    p["cohort"] = _dates(p["perf_cohort"])
    lo, hi = 0.70, 0.92
    base = p["baseline_auc"]
    ax.fill_between(p["cohort"], base - C.AUC_DROP_AMBER, base + 0.1, color=GREEN_BG, lw=0)
    ax.fill_between(p["cohort"], base - C.AUC_DROP_RED, base - C.AUC_DROP_AMBER, color=AMBER_BG, lw=0)
    ax.fill_between(p["cohort"], lo, base - C.AUC_DROP_RED, color=RED_BG, lw=0)
    ax.plot(p["cohort"], base, color=MUTED, ls="--", lw=1.4, label="AUC at training (out-of-time)")
    ax.plot(p["cohort"], p["auc"], color=BLUE, lw=2.2, marker="o", ms=5, mfc="white", mew=1.8, label="Observed AUC")
    ax.set_ylim(lo, hi)
    ax.set_title("Performance: AUC by loan cohort")
    ax.set_ylabel("AUC")
    ax.set_xlabel("Loan application month (cohort)")
    _month_axis(ax)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="x", visible=False)
    last = p.iloc[-1]
    ax.annotate(f"{last['auc']:.2f}", (last["cohort"], last["auc"]), textcoords="offset points", xytext=(6, 4),
                color=INK, fontsize=9, fontweight="bold")


def plot_psi(ax, mon):
    s = mon.dropna(subset=["score_psi"])
    ax.axhspan(0, C.PSI_AMBER, color=GREEN_BG, lw=0)
    ax.axhspan(C.PSI_AMBER, C.PSI_RED, color=AMBER_BG, lw=0)
    ax.axhspan(C.PSI_RED, 5, color=RED_BG, lw=0)
    ax.plot(s["snapshot_date"], s["score_psi"], color=ORANGE, lw=2.2, marker="o", ms=5, mfc="white", mew=1.8,
            label="vs OOT reference (alerts)")
    if "score_psi_train_ref" in s:
        ax.plot(s["snapshot_date"], s["score_psi_train_ref"], color=MUTED, lw=1.4, ls="--", label="vs training (report only)")
        ax.legend(loc="upper right", frameon=False, fontsize=8, bbox_to_anchor=(1.0, 0.82))
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 0.1, 0.25, 1, 2])
    ax.set_yticklabels(["0", "0.10", "0.25", "1", "2"])
    ax.set_title("Stability: score PSI")
    ax.set_ylabel("PSI (log scale above 0.1)")
    ax.set_xlabel("Scoring month")
    ax.text(s["snapshot_date"].min(), 0.12, " amber ≥ 0.10", color=MUTED, fontsize=8, va="bottom")
    ax.text(s["snapshot_date"].min(), 0.29, " red ≥ 0.25", color=MUTED, fontsize=8, va="bottom")
    _month_axis(ax)
    ax.grid(axis="x", visible=False)
    _switch_lines(ax, s, 2.8)


def plot_cause(ax, mon):
    s = mon.dropna(subset=["has_clickstream_rate"])
    ax.fill_between(s["snapshot_date"], 0, s["has_clickstream_rate"] * 100, color=BLUE, alpha=0.18, lw=0)
    ax.plot(s["snapshot_date"], s["has_clickstream_rate"] * 100, color=BLUE, lw=2.2, marker="o", ms=5, mfc="white", mew=1.8)
    ax.set_ylim(0, 105)
    ax.set_title("Driver: applicants with clickstream data")
    ax.set_ylabel("% of applicants")
    ax.set_xlabel("Scoring month")
    _month_axis(ax)
    ax.grid(axis="x", visible=False)
    if (s["has_clickstream_rate"] == 0).any():
        first0 = s.loc[s["has_clickstream_rate"] == 0, "snapshot_date"].min()
        ax.annotate("clickstream feed stops", (first0, 2), xytext=(-110, 38), textcoords="offset points",
                    color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))


def plot_calibration(ax, mon, preds, labels):
    rows = []
    for c, g in preds.groupby("snapshot_date"):
        lab = labels[labels["loan_id"].isin(g["loan_id"])]
        if len(lab) == len(g):
            rows.append((pd.to_datetime(c), g["score"].mean(), lab["label"].mean()))
    if not rows:
        ax.set_axis_off()
        return
    r = pd.DataFrame(rows, columns=["cohort", "pred", "actual"]).sort_values("cohort")
    ax.plot(r["cohort"], r["actual"] * 100, color=INK, lw=2.2, marker="o", ms=5, mfc="white", mew=1.8, label="Actual default rate")
    ax.plot(r["cohort"], r["pred"] * 100, color=ORANGE, lw=2.2, ls="--", label="Average predicted risk")
    ax.set_title("Calibration: predicted vs. actual default rate")
    ax.set_ylabel("%")
    ax.set_xlabel("Loan application month (cohort)")
    ax.set_ylim(0, max(60, r[["pred", "actual"]].to_numpy().max() * 100 + 5))
    _month_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)


def plot_csi(ax, csi, n=12):
    latest = csi[csi["snapshot_date"] == csi["snapshot_date"].max()]
    order = latest.sort_values("importance", ascending=False)["feature"].head(n).tolist()
    piv = csi[csi["feature"].isin(order)].pivot_table(index="feature", columns="snapshot_date", values="csi").loc[order]
    cmap = LinearSegmentedColormap.from_list("csi", ["#e8f5ec", "#fdf3dc", "#f4a08a", "#b3261e"])
    vals = np.log10(piv.to_numpy().clip(0.01, 10))
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-2, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([pd.to_datetime(c).strftime("%b %y") for c in piv.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature drift: CSI of top features by month")
    return im


def main(snapshot_date):
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    mon = M.read_gold(C.GOLD_MONITORING, upto=snapshot_date)
    if mon.empty:
        print("no monitoring results yet")
        return
    mon["snapshot_date"] = _dates(mon["snapshot_date"])
    mon = mon.sort_values("snapshot_date")
    csi = M.read_gold(C.GOLD_CSI, upto=snapshot_date)
    preds = M.read_gold(C.GOLD_PREDICTIONS, upto=snapshot_date)
    labels = M.read_gold(C.GOLD_LABEL, upto=snapshot_date)

    cols = ["snapshot_date", "model_version", "model_name", "n_scored", "score_psi", "max_csi_top20", "n_csi_red_top20",
            "score_psi_train_ref", "has_clickstream_rate", "input_status", "perf_cohort", "perf_n", "auc", "gini", "ks", "brier",
            "calibration_gap", "baseline_auc", "auc_drop", "stability_status", "performance_status", "calibration_status",
            "overall_status"]
    mon[cols].round(4).to_csv(C.REPORT_DIR + "monitoring_summary.csv", index=False)

    figs = {"1_performance.png": (plot_performance, (mon,)), "2_stability_psi.png": (plot_psi, (mon,)),
            "3_drift_driver.png": (plot_cause, (mon,)), "4_calibration.png": (plot_calibration, (mon, preds, labels))}
    for name, (fn, args) in figs.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        fn(ax, *args)
        fig.tight_layout()
        fig.savefig(C.REPORT_DIR + name)
        plt.close(fig)
    if len(csi):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        im = plot_csi(ax, csi)
        cb = fig.colorbar(im, ax=ax, ticks=np.log10([0.01, 0.1, 0.25, 1, 10]), label="CSI")
        cb.ax.set_yticklabels(["0.01", "0.10", "0.25", "1", "10"])
        fig.tight_layout()
        fig.savefig(C.REPORT_DIR + "5_feature_csi_heatmap.png")
        plt.close(fig)

    # overview page
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.2))
    plot_performance(axs[0, 0], mon)
    plot_psi(axs[0, 1], mon)
    plot_cause(axs[1, 0], mon)
    plot_calibration(axs[1, 1], mon, preds, labels)
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR + "0_dashboard.png")
    plt.close(fig)
    print("saved monitoring charts to", C.REPORT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    os.chdir(C.BASE_DIR)
    main(a.snapshotdate)
