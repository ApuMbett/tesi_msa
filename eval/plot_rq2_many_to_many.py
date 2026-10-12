#!/usr/bin/env python3
"""
plot_rq2_many_to_many.py

Semantic MSA (RQ2) - Many-to-Many Retrieval Visualization Suite.
Generates publication-quality figures for large-scale cross-modal segment retrieval:
  1. Cumulative Match Characteristic (CMC / Recall@K) Curves:
     - Evaluates K from 1 to 25.
     - Curves: Audio -> Text (A2T), Text -> Audio (T2A), Bidirectional Mean.
     - Random Chance Baseline: K / M.
     - Highlighted annotations at K = 1, 5, 10.
  2. Cross-Track Semantic Confusion Matrices:
     - Analyzes domain consistency across 50 tracks:
       "When an audio/text segment misses its exact counterpart, what functional class does it retrieve?"
     - Row-normalized percentage heatmap for both A2T and T2A.
  3. Multi-System Comparative CMC Curve:
     - Side-by-side or overlaid comparison of FLASH vs. FUTGA retrieval power.

Outputs saved in:
  - rq2_many_to_many/output/<system>/plots/cmc_recall_curve.png (.pdf)
  - rq2_many_to_many/output/<system>/plots/cross_track_semantic_confusion.png (.pdf)
  - rq2_many_to_many/output/comparative_plots/cmc_comparison.png (.pdf)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STANDARD_ORDER = ["intro", "verse", "chorus", "bridge", "outro", "inst", "silence"]


def plot_cmc_curve(
    df_ranks: pd.DataFrame,
    system_key: str,
    out_dir: Path,
    k_max: int = 25,
    save_pdf: bool = True,
):
    """
    Renders Cumulative Match Characteristic (CMC) Recall@K curve.
    """
    M = len(df_ranks)
    ks = np.arange(1, k_max + 1)

    r_a2t = np.array([np.mean(df_ranks["rank_a2t"] <= k) * 100 for k in ks])
    r_t2a = np.array([np.mean(df_ranks["rank_t2a"] <= k) * 100 for k in ks])
    r_mean = 0.5 * (r_a2t + r_t2a)
    chance = (ks / float(M)) * 100

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=250)

    # Plot curves
    ax.plot(
        ks, r_a2t, "o-", color="#1f77b4", linewidth=2.2, markersize=5,
        label=f"Audio → Text (A2T)  [R@1={r_a2t[0]:.2f}%, R@5={r_a2t[4]:.2f}%, R@10={r_a2t[9]:.2f}%]"
    )
    ax.plot(
        ks, r_t2a, "s-", color="#ff7f0e", linewidth=2.2, markersize=5,
        label=f"Text → Audio (T2A)  [R@1={r_t2a[0]:.2f}%, R@5={r_t2a[4]:.2f}%, R@10={r_t2a[9]:.2f}%]"
    )
    ax.plot(
        ks, r_mean, "^-", color="#2ca02c", linewidth=2.2, markersize=5,
        label=f"Bidirectional Mean  [R@1={r_mean[0]:.2f}%, R@5={r_mean[4]:.2f}%, R@10={r_mean[9]:.2f}%]"
    )
    ax.plot(
        ks, chance, "--", color="#7f7f7f", linewidth=1.8,
        label=f"Theoretical Random Chance (K / {M})"
    )

    # Vertical reference lines for R@1, R@5, R@10
    for k_val in [1, 5, 10]:
        ax.axvline(x=k_val, color="#d62728", linestyle=":", alpha=0.45, linewidth=1.2)

    ax.set_xlabel("Rank Threshold (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Retrieval Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Many-to-Many Cross-Modal Retrieval CMC Curve: {system_key.upper()}\n"
        f"(Global Pool Size = {M} Segments across 50 Tracks)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xticks(range(1, k_max + 1, 2 if k_max <= 25 else 5))
    ax.set_xlim(0.5, k_max + 0.5)
    max_y = max(float(np.max(r_a2t)), float(np.max(r_t2a)))
    ax.set_ylim(0, max_y * 1.15)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cmc_recall_curve.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    if save_pdf:
        out_pdf = out_dir / "cmc_recall_curve.pdf"
        plt.savefig(out_pdf)
    plt.close()
    logging.info(f"Saved CMC curve: {out_png}")


def plot_semantic_confusion_matrix(
    df_ranks: pd.DataFrame,
    system_key: str,
    out_dir: Path,
    save_pdf: bool = True,
):
    """
    Renders 2-panel confusion matrix (A2T and T2A) of Query Class vs Top-1 Retrieved Class.
    """
    M = len(df_ranks)
    present_labels = [l for l in STANDARD_ORDER if l in df_ranks["standard_label"].unique()]
    if len(present_labels) < 2:
        return

    # Compute row-normalized percentages
    cm_a2t = pd.crosstab(df_ranks["standard_label"], df_ranks["top1_a2t_label"], normalize="index") * 100
    cm_a2t = cm_a2t.reindex(index=present_labels, columns=present_labels, fill_value=0.0)

    cm_t2a = pd.crosstab(df_ranks["standard_label"], df_ranks["top1_t2a_label"], normalize="index") * 100
    cm_t2a = cm_t2a.reindex(index=present_labels, columns=present_labels, fill_value=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=250)

    vmax_val = max(float(cm_a2t.values.max()), float(cm_t2a.values.max()), 50.0)

    def draw_cm(ax, data_df, title_str, xlabel_str, ylabel_str):
        im = ax.imshow(data_df.values, cmap="Blues", vmin=0, vmax=vmax_val)
        ax.set_xticks(range(len(present_labels)))
        ax.set_xticklabels([l.capitalize() for l in present_labels], rotation=45, ha="right", fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(present_labels)))
        ax.set_yticklabels([l.capitalize() for l in present_labels], fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel_str, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel(ylabel_str, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=10)

        for i in range(len(present_labels)):
            for j in range(len(present_labels)):
                val = data_df.values[i, j]
                text_col = "white" if val > (vmax_val * 0.5) else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_col, fontsize=8.5, fontweight="bold")
        return im

    im1 = draw_cm(
        ax1, cm_a2t,
        f"Audio → Text Top-1 Affinity ({system_key.upper()})",
        "Top-1 Retrieved Caption Label →",
        "← Ground Truth Audio Query Label"
    )
    im2 = draw_cm(
        ax2, cm_t2a,
        f"Text → Audio Top-1 Affinity ({system_key.upper()})",
        "Top-1 Retrieved Audio Label →",
        "← Ground Truth Caption Query Label"
    )

    # Shared colorbar
    fig.subplots_adjust(right=0.90, wspace=0.28, bottom=0.18, top=0.88)
    cbar_ax = fig.add_axes([0.92, 0.22, 0.02, 0.62])
    cb = fig.colorbar(im1, cax=cbar_ax)
    cb.set_label("Fraction of Query Class Top-1 Retrievals (%)", fontsize=10, fontweight="bold")
    cb.ax.tick_params(labelsize=8.5)

    fig.suptitle(
        f"Cross-Track Semantic Affinity in Global Pool ({system_key.upper()} - {M} Segments)",
        fontsize=13, fontweight="bold", y=0.97
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cross_track_semantic_confusion.png"
    plt.savefig(out_png, dpi=250)
    if save_pdf:
        out_pdf = out_dir / "cross_track_semantic_confusion.pdf"
        plt.savefig(out_pdf)
    plt.close()
    logging.info(f"Saved semantic confusion matrix: {out_png}")


def plot_comparative_cmc(
    dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
    k_max: int = 25,
    save_pdf: bool = True,
):
    """
    Overlays Bidirectional Mean CMC curves for multiple systems.
    """
    if len(dfs) < 2:
        return

    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=250)
    ks = np.arange(1, k_max + 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (sys_key, df) in enumerate(dfs.items()):
        M = len(df)
        r_a2t = np.array([np.mean(df["rank_a2t"] <= k) * 100 for k in ks])
        r_t2a = np.array([np.mean(df["rank_t2a"] <= k) * 100 for k in ks])
        r_mean = 0.5 * (r_a2t + r_t2a)
        chance = (ks / float(M)) * 100

        col = colors[idx % len(colors)]
        ax.plot(
            ks, r_mean, "o-", color=col, linewidth=2.2, markersize=5,
            label=f"{sys_key.upper()} (M={M}) [R@1={r_mean[0]:.2f}%, R@5={r_mean[4]:.2f}%, R@10={r_mean[9]:.2f}%]"
        )
        ax.plot(ks, chance, ":", color=col, alpha=0.6, linewidth=1.2, label=f"Caso {sys_key.upper()} (K/{M})")

    ax.set_xlabel("Soglia Recall (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recall@K Medio (%)", fontsize=11, fontweight="bold")
    # ax.set_title(
    #     "Comparative Cross-Modal Retrieval Performance (RQ2 Many-to-Many)\nBidirectional Mean CMC Curve Across Global Segment Pools",
    #     fontsize=12, fontweight="bold", pad=12
    # )
    ax.set_xticks(range(1, k_max + 1, 2 if k_max <= 25 else 5))
    ax.set_xlim(0.5, k_max + 0.5)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cmc_comparison.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    if save_pdf:
        out_pdf = out_dir / "cmc_comparison.pdf"
        plt.savefig(out_pdf)
    plt.close()
    logging.info(f"Saved comparative CMC curve: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality many-to-many retrieval plots.")
    parser.add_argument("--systems", nargs="+", default=["flash", "futga"], help="Systems to visualize.")
    parser.add_argument("--output-base", type=str, default=None, help="Base output dir (default: eval/rq2_many_to_many/output).")
    parser.add_argument("--k-max", type=int, default=25, help="Maximum rank K to plot for CMC curves.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip saving PDF vector files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    eval_dir = Path(__file__).resolve().parent
    base_dir = Path(args.output_base).resolve() if args.output_base else eval_dir / "rq2_many_to_many" / "output"
    save_pdf = not args.no_pdf

    dfs: Dict[str, pd.DataFrame] = {}

    for sys_key in args.systems:
        ranks_csv = base_dir / sys_key / f"{sys_key}_manytomany_ranks.csv"
        if not ranks_csv.is_file():
            logging.warning(f"Ranks CSV not found at: {ranks_csv}. Skipping system '{sys_key}'.")
            continue

        df = pd.read_csv(ranks_csv)
        dfs[sys_key] = df
        plot_dir = base_dir / sys_key / "plots"

        # 1. CMC Curve
        plot_cmc_curve(df, sys_key, plot_dir, k_max=args.k_max, save_pdf=save_pdf)

        # 2. Semantic Confusion Matrix
        plot_semantic_confusion_matrix(df, sys_key, plot_dir, save_pdf=save_pdf)

    # 3. Comparative CMC Curve if multiple systems available
    if len(dfs) >= 2:
        comp_dir = base_dir / "comparative_plots"
        plot_comparative_cmc(dfs, comp_dir, k_max=args.k_max, save_pdf=save_pdf)

    logging.info("Many-to-many plotting suite finished successfully!")


if __name__ == "__main__":
    main()
