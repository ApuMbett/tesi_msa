#!/usr/bin/env python3
"""
plot_rq2_intra_track.py

Semantic MSA (RQ2) - Intra-Track Heatmap Visualization Suite.
Generates publication-quality figures for intra-track CLAP semantic grounding:
  1. Enriched Single-Track Heatmaps:
     - Saved directly inside the matrices/ folder of each connected song:
       rq2_intra_track/output/<system>/matrices/<track_name>/<bhx_id>_heatmap.png (.pdf)
     - MIR structural annotation strips on axes X (captions) and Y (audio)
     - Timestamped tick labels with section names: [i] SECTION (mm:ss-mm:ss)
     - Target Grounding diagonal boxes (solid cyan)
     - Maximum Off-Diagonal competitor highlights (green if Same Semantic Class, yellow if Different)
     - In-cell numerical cosine similarity annotations
  2. Side-by-Side Comparative Heatmaps:
     - Direct comparison of Flash agent vs. FUTGA for the same track
     - Shared color scale and colorbar for honest visual assessment
     - Saved in rq2_intra_track/output/comparative_plots/<track_name>_flash_vs_futga.png (.pdf)
     - And copied into each song's matrix directory
  3. Corpus-Level Comparative Summary Charts:
     - Semantic Class Agreement vs. Chance Baseline across all 4 thresholds
     - Retrieval Metrics comparison (MRR, Top-1, Windowed Accuracy)
     - Saved in rq2_intra_track/output/comparative_plots/
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Standard MIR Functional Class Palette
LABEL_COLORS: Dict[str, str] = {
    "intro": "#2ca02c",      # Forest green
    "verse": "#1f77b4",      # Steel blue
    "chorus": "#ff7f0e",     # Vivid orange
    "bridge": "#9467bd",     # Purple
    "outro": "#d62728",      # Crimson red
    "inst": "#8c564b",       # Brown
    "solo": "#e377c2",       # Magenta / Pink
    "silence": "#7f7f7f",    # Neutral gray
}
DEFAULT_LABEL_COLOR = "#17becf"

DEFAULT_EXEMPLARY_TRACKS = [
    "Again",
    "Alejandro",
    "All Around Me",
    "Barbra Streisand",
    "Cooler Than Me",
    "Technologic",
    "Texas Flood",
    "Without You",
]


def format_sec(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m}:{s:02d}"


def load_track_matrix_and_meta(system_output_dir: Path, track_name: str) -> Optional[Tuple[np.ndarray, dict, str, Path]]:
    """
    Finds and loads M_clap matrix and metadata for a given track in system_output_dir.
    Returns: (M, meta, bhx_id, track_mat_dir)
    """
    track_mat_dir = system_output_dir / "matrices" / track_name
    if not track_mat_dir.is_dir():
        return None

    npy_files = list(track_mat_dir.glob("*.npy"))
    meta_files = list(track_mat_dir.glob("*_meta.json"))

    if not npy_files or not meta_files:
        return None

    mat_file = npy_files[0]
    meta_file = meta_files[0]
    bhx_id = mat_file.stem

    try:
        M = np.load(mat_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return M, meta, bhx_id, track_mat_dir
    except Exception as e:
        logging.warning(f"Error loading {mat_file}: {e}")
        return None


def plot_single_track_heatmap(
    M: np.ndarray,
    meta: dict,
    bhx_id: str,
    track_name: str,
    system_key: str,
    track_mat_dir: Path,
    plots_dir: Optional[Path] = None,
    vmax: Optional[float] = None,
    save_pdf: bool = True,
):
    """
    Renders an enriched intra-track heatmap with structural strips and bounding boxes.
    Saves directly in track_mat_dir as <bhx_id>_heatmap.png (.pdf) and <track_name>_heatmap.png.
    """
    segs = meta.get("segments", [])
    N = len(segs)
    if N == 0:
        return

    labels = [s.get("standard_label", "inst") for s in segs]
    starts = [float(s.get("start", 0.0)) for s in segs]
    ends = [float(s.get("end", 0.0)) for s in segs]
    tick_labels = [f"[{i}] {labels[i].upper()}\n({format_sec(starts[i])}-{format_sec(ends[i])})" for i in range(N)]

    if vmax is None:
        vmax = max(0.65, float(np.max(M)))

    fig_size = max(9.0, min(14.0, N * 0.95 + 2.5))
    fig = plt.figure(figsize=(fig_size, fig_size * 0.92), dpi=250)

    # GridSpec: top strip, left strip, main heatmap, colorbar
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[0.035, 1.0, 0.035],
        height_ratios=[0.035, 1.0],
        wspace=0.03, hspace=0.03,
        left=0.15, right=0.88, bottom=0.15, top=0.90
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Top strip (captions)
    top_colors = np.array([[plt.matplotlib.colors.to_rgba(LABEL_COLORS.get(l, DEFAULT_LABEL_COLOR)) for l in labels]])
    ax_top.imshow(top_colors, aspect="auto")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title("Text Captions (j) →", fontsize=10, fontweight="bold", pad=8)

    # Left strip (audio)
    left_colors = np.array([[plt.matplotlib.colors.to_rgba(LABEL_COLORS.get(l, DEFAULT_LABEL_COLOR))] for l in labels])
    ax_left.imshow(left_colors, aspect="auto")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_ylabel("← Audio Segments (i)", fontsize=10, fontweight="bold", labelpad=8)

    # Main heatmap
    im = ax_main.imshow(M, cmap="magma", vmin=0.0, vmax=vmax, aspect="equal")
    ax_main.set_xticks(range(N))
    ax_main.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_main.set_yticks(range(N))
    ax_main.set_yticklabels(tick_labels, fontsize=8)

    # Annotations & cell highlights
    for i in range(N):
        row = M[i, :].copy()
        row[i] = -np.inf
        j_star = int(np.argmax(row))
        is_concordant = (labels[j_star] == labels[i])

        # Diagonal box (Target Grounding)
        rect_diag = patches.Rectangle(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            fill=False, edgecolor="#00e5ff", linewidth=2.2
        )
        ax_main.add_patch(rect_diag)

        # Max off-diagonal box
        border_col = "#00ff66" if is_concordant else "#ffea00"
        rect_off = patches.Rectangle(
            (j_star - 0.46, i - 0.46), 0.92, 0.92,
            fill=False, edgecolor=border_col, linewidth=1.8, linestyle="--"
        )
        ax_main.add_patch(rect_off)

        for j in range(N):
            val = M[i, j]
            text_col = "white" if val < (vmax * 0.55) else "black"
            font_size = 7.5 if N > 7 else 9.0
            ax_main.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=font_size, fontweight="bold")

    # Colorbar
    cb = plt.colorbar(im, cax=ax_cbar)
    cb.set_label("CLAP Cosine Similarity $\\cos(a_i, t_j)$", fontsize=9.5, fontweight="bold")
    cb.ax.tick_params(labelsize=8)

    # Title & Subtitle
    fig.suptitle(
        f"CLAP Intra-Track Semantic Grounding: {track_name} ({bhx_id})\nSystem: {system_key.upper()} | Model: laion/clap-htsat-fused",
        fontsize=12, fontweight="bold", y=0.97
    )

    # Legend elements
    legend_elements = [
        patches.Patch(facecolor="#00e5ff", edgecolor="black", label="Target Grounding (i, i)"),
        patches.Patch(facecolor="none", edgecolor="#00ff66", linewidth=1.5, linestyle="--", label="Max Off-Diag: Same Class (Agreement)"),
        patches.Patch(facecolor="none", edgecolor="#ffea00", linewidth=1.5, linestyle="--", label="Max Off-Diag: Different Class"),
    ]
    unique_labels = sorted(list(set(labels)))
    for l in unique_labels:
        legend_elements.append(patches.Patch(facecolor=LABEL_COLORS.get(l, DEFAULT_LABEL_COLOR), edgecolor="black", label=l.capitalize()))

    ncol = min(len(legend_elements), 5)
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=ncol,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.51, 0.01)
    )

    # Save directly into song matrix folder
    out_png_matrix = track_mat_dir / f"{bhx_id}_heatmap.png"
    plt.savefig(out_png_matrix, dpi=250)
    if save_pdf:
        out_pdf_matrix = track_mat_dir / f"{bhx_id}_heatmap.pdf"
        plt.savefig(out_pdf_matrix)

    # Also save with track_name in matrix folder
    out_png_track = track_mat_dir / f"{track_name}_heatmap.png"
    if not out_png_track.exists():
        shutil.copyfile(out_png_matrix, out_png_track)

    # Optionally copy to plots/ folder
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_png_matrix, plots_dir / f"{track_name}_heatmap.png")
        if save_pdf and (track_mat_dir / f"{bhx_id}_heatmap.pdf").exists():
            shutil.copyfile(track_mat_dir / f"{bhx_id}_heatmap.pdf", plots_dir / f"{track_name}_heatmap.pdf")

    plt.close()
    logging.info(f"Saved heatmap to matrix dir: {out_png_matrix}")


def plot_side_by_side_comparison(
    track_name: str,
    sys1_key: str,
    sys1_data: Tuple[np.ndarray, dict, str, Path],
    sys2_key: str,
    sys2_data: Tuple[np.ndarray, dict, str, Path],
    out_dir: Path,
    save_pdf: bool = True,
):
    """
    Renders side-by-side comparison of two systems on the same track with shared colorbar.
    Saves in comparative_plots/ and in each system's track matrix directory.
    """
    M1, meta1, bhx1, mat_dir1 = sys1_data
    M2, meta2, bhx2, mat_dir2 = sys2_data

    vmax = max(float(np.max(M1)), float(np.max(M2)), 0.6)

    fig = plt.figure(figsize=(16, 7.5), dpi=250)
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.2, 0.7, 0.04],
        wspace=0.25,
        left=0.08, right=0.92, bottom=0.15, top=0.88
    )

    # Sub-gridspec for Sys 1
    gs1 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 0],
        width_ratios=[0.05, 1.0], height_ratios=[0.05, 1.0],
        wspace=0.03, hspace=0.03
    )
    ax1_top = fig.add_subplot(gs1[0, 1])
    ax1_left = fig.add_subplot(gs1[1, 0])
    ax1_main = fig.add_subplot(gs1[1, 1])

    # Sub-gridspec for Sys 2
    gs2 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 1],
        width_ratios=[0.08, 1.0], height_ratios=[0.08, 1.0],
        wspace=0.03, hspace=0.03
    )
    ax2_top = fig.add_subplot(gs2[0, 1])
    ax2_left = fig.add_subplot(gs2[1, 0])
    ax2_main = fig.add_subplot(gs2[1, 1])

    ax_cbar = fig.add_subplot(gs[0, 2])

    def render_sub(ax_m, ax_t, ax_l, M, meta, title_sub):
        segs = meta.get("segments", [])
        N = len(segs)
        lbls = [s.get("standard_label", "inst") for s in segs]
        sts = [float(s.get("start", 0.0)) for s in segs]
        ets = [float(s.get("end", 0.0)) for s in segs]
        ticks = [f"[{i}] {lbls[i].upper()}\n({format_sec(sts[i])}-{format_sec(ets[i])})" for i in range(N)]

        top_c = np.array([[plt.matplotlib.colors.to_rgba(LABEL_COLORS.get(l, DEFAULT_LABEL_COLOR)) for l in lbls]])
        ax_t.imshow(top_c, aspect="auto")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        ax_t.set_title(title_sub, fontsize=11, fontweight="bold", pad=8)

        left_c = np.array([[plt.matplotlib.colors.to_rgba(LABEL_COLORS.get(l, DEFAULT_LABEL_COLOR))] for l in lbls])
        ax_l.imshow(left_c, aspect="auto")
        ax_l.set_xticks([])
        ax_l.set_yticks([])

        im_sub = ax_m.imshow(M, cmap="magma", vmin=0.0, vmax=vmax, aspect="equal")
        ax_m.set_xticks(range(N))
        ax_m.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7.5)
        ax_m.set_yticks(range(N))
        ax_m.set_yticklabels(ticks, fontsize=7.5)

        for i in range(N):
            row = M[i, :].copy()
            row[i] = -np.inf
            j_star = int(np.argmax(row))
            is_concordant = (lbls[j_star] == lbls[i])

            rect_diag = patches.Rectangle((i - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#00e5ff", linewidth=2.0)
            ax_m.add_patch(rect_diag)
            border_col = "#00ff66" if is_concordant else "#ffea00"
            rect_off = patches.Rectangle((j_star - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=border_col, linewidth=1.8, linestyle="--")
            ax_m.add_patch(rect_off)

            for j in range(N):
                val = M[i, j]
                text_col = "white" if val < (vmax * 0.55) else "black"
                ax_m.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=7.5 if N > 5 else 9.5, fontweight="bold")
        return im_sub

    im1 = render_sub(ax1_main, ax1_top, ax1_left, M1, meta1, f"{sys1_key.upper()} ({len(meta1['segments'])} Segments)")
    im2 = render_sub(ax2_main, ax2_top, ax2_left, M2, meta2, f"{sys2_key.upper()} ({len(meta2['segments'])} Segments)")

    cb = plt.colorbar(im1, cax=ax_cbar)
    cb.set_label("CLAP Cosine Similarity $\\cos(a_i, t_j)$", fontsize=10, fontweight="bold")
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Comparative CLAP Intra-Track Grounding: {track_name} ({bhx1})\n{sys1_key.upper()} vs. {sys2_key.upper()}",
        fontsize=13, fontweight="bold", y=0.97
    )

    legend_elements = [
        patches.Patch(facecolor="#00e5ff", edgecolor="black", label="Target Grounding (i, i)"),
        patches.Patch(facecolor="none", edgecolor="#00ff66", linewidth=1.5, linestyle="--", label="Max Off-Diag: Same Class"),
        patches.Patch(facecolor="none", edgecolor="#ffea00", linewidth=1.5, linestyle="--", label="Max Off-Diag: Different Class"),
    ]
    all_labels = sorted(list(set([s["standard_label"] for s in meta1["segments"]] + [s["standard_label"] for s in meta2["segments"]])))
    for l in all_labels:
        legend_elements.append(patches.Patch(facecolor=LABEL_COLORS.get(l, DEFAULT_LABEL_COLOR), edgecolor="black", label=l.capitalize()))

    fig.legend(handles=legend_elements, loc="lower center", ncol=min(len(legend_elements), 6), fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.01))

    # Save in comparative_plots/
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{track_name}_{sys1_key}_vs_{sys2_key}.png"
    plt.savefig(out_png, dpi=250)
    if save_pdf:
        out_pdf = out_dir / f"{track_name}_{sys1_key}_vs_{sys2_key}.pdf"
        plt.savefig(out_pdf)

    # Also save a copy inside both systems' matrix folders for easy inspection
    out_copy1 = mat_dir1 / f"{bhx1}_{sys1_key}_vs_{sys2_key}.png"
    shutil.copyfile(out_png, out_copy1)
    out_copy2 = mat_dir2 / f"{bhx2}_{sys1_key}_vs_{sys2_key}.png"
    shutil.copyfile(out_png, out_copy2)

    plt.close()
    logging.info(f"Saved comparative heatmap: {out_png}")


def plot_corpus_comparative_summary(base_dir: Path, systems: List[str], save_pdf: bool = True):
    """
    Generates high-level comparative charts summarizing the corpus evaluation:
      1. Semantic Class Agreement vs. Chance Baseline across thresholds.
      2. Retrieval metrics comparison (MRR, Top-1, Windowed Acc).
    """
    comp_dir = base_dir / "comparative_plots"
    comp_dir.mkdir(parents=True, exist_ok=True)

    summary_dfs = {}
    for sys_key in systems:
        sum_csv = base_dir / sys_key / f"{sys_key}_eval_summary.csv"
        if sum_csv.is_file():
            df = pd.read_csv(sum_csv)
            summary_dfs[sys_key] = df.set_index("metric_key")

    if not summary_dfs:
        return

    # Chart 1: Semantic Class Agreement vs Baseline (Grouped Bar Chart)
    # Focus on Flash (or systems with active agreement)
    if "flash" in summary_dfs:
        df_flash = summary_dfs["flash"]
        thresh_keys = [
            ("fixed80", "Fixed 80% Diag"),
            ("calibrated", "Calibrated Mean"),
            ("p90", "Percentile 90"),
            ("p95", "Percentile 95"),
        ]

        obs_means = [df_flash.loc[f"semantic_agreement_{k}", "mean"] for k, _ in thresh_keys]
        obs_stds = [df_flash.loc[f"semantic_agreement_{k}", "std"] for k, _ in thresh_keys]
        base_means = [df_flash.loc[f"baseline_agreement_{k}", "mean"] for k, _ in thresh_keys]
        base_stds = [df_flash.loc[f"baseline_agreement_{k}", "std"] for k, _ in thresh_keys]

        labels = [name for _, name in thresh_keys]
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6), dpi=250)
        rects1 = ax.bar(x - width / 2, obs_means, width, yerr=obs_stds, capsize=4, label="Observed Agreement", color="#1f77b4", edgecolor="black")
        rects2 = ax.bar(x + width / 2, base_means, width, yerr=base_stds, capsize=4, label="Chance Baseline", color="#ff7f0e", edgecolor="black")

        ax.set_ylabel("Semantic Class Agreement (Fraction)", fontsize=11, fontweight="bold")
        ax.set_title("Semantic Class Agreement vs. Chance Baseline (FLASH Agent)\nComparison Across All 4 Calibration Thresholds (50 Tracks)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.legend(fontsize=10, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Values above bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_ylim(0, max(obs_means) * 1.35)
        plt.tight_layout()
        out_png = comp_dir / "semantic_agreement_vs_baseline_comparison.png"
        plt.savefig(out_png, dpi=250)
        if save_pdf:
            plt.savefig(comp_dir / "semantic_agreement_vs_baseline_comparison.pdf")
        plt.close()
        logging.info(f"Saved summary comparison chart: {out_png}")

    # Chart 2: Retrieval Metrics Comparison (MRR, Top-1, Windowed Acc)
    if len(summary_dfs) >= 2:
        metrics = [
            ("mrr", "MRR"),
            ("top1", "Top-1 Accuracy"),
            ("windowed_acc", "Windowed Acc (±1)"),
        ]
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=250)
        x = np.arange(len(metrics))
        n_sys = len(summary_dfs)
        width = 0.8 / n_sys
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, (s_name, s_df) in enumerate(summary_dfs.items()):
            means = [s_df.loc[m_k, "mean"] for m_k, _ in metrics]
            stds = [s_df.loc[m_k, "std"] for m_k, _ in metrics]
            offset = (idx - (n_sys - 1) / 2) * width
            rects = ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=s_name.upper(), color=colors[idx % len(colors)], edgecolor="black")
            for r in rects:
                h = r.get_height()
                ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylabel("Score (0 - 1)", fontsize=11, fontweight="bold")
        ax.set_title("Intra-Track Retrieval Metrics Comparison (Macro-averaged over 50 Tracks)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in metrics], fontsize=10, fontweight="bold")
        ax.legend(fontsize=10, frameon=True)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.tight_layout()
        out_png = comp_dir / "retrieval_metrics_comparison.png"
        plt.savefig(out_png, dpi=250)
        if save_pdf:
            plt.savefig(comp_dir / "retrieval_metrics_comparison.pdf")
        plt.close()
        logging.info(f"Saved retrieval metrics comparison chart: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality intra-track heatmaps.")
    parser.add_argument("--systems", nargs="+", default=["flash", "futga"], help="Systems to visualize.")
    parser.add_argument("--tracks", nargs="+", default=None, help="List of track names to plot (default: all 50 tracks).")
    parser.add_argument("--output-base", type=str, default=None, help="Base output dir (default: eval/rq2_intra_track/output).")
    parser.add_argument("--no-pdf", action="store_true", help="Skip saving PDF vector files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    eval_dir = Path(__file__).resolve().parent
    base_dir = Path(args.output_base).resolve() if args.output_base else eval_dir / "rq2_intra_track" / "output"

    save_pdf = not args.no_pdf

    # Discover all available track subdirectories under system matrices/
    sample_sys = args.systems[0]
    mat_dir = base_dir / sample_sys / "matrices"
    if not mat_dir.is_dir():
        logging.error(f"Matrices directory not found at: {mat_dir}")
        return

    available_tracks = sorted([d.name for d in mat_dir.iterdir() if d.is_dir()])
    if args.all_tracks:
        track_list = available_tracks
    elif args.tracks is not None:
        track_list = args.tracks
    else:
        track_list = [t for t in DEFAULT_EXEMPLARY_TRACKS if t in available_tracks]

    logging.info(f"Generating intra-track heatmaps for {len(track_list)} tracks across systems: {args.systems}")

    # 1. Single-system heatmaps saved directly in the song's matrix folder
    for sys_key in args.systems:
        sys_out_dir = base_dir / sys_key
        plots_dir = sys_out_dir / "plots"
        for tr in track_list:
            data = load_track_matrix_and_meta(sys_out_dir, tr)
            if data is not None:
                M, meta, bhx_id, track_mat_dir = data
                plot_single_track_heatmap(
                    M=M,
                    meta=meta,
                    bhx_id=bhx_id,
                    track_name=tr,
                    system_key=sys_key,
                    track_mat_dir=track_mat_dir,
                    plots_dir=plots_dir,
                    save_pdf=save_pdf,
                )

    # 2. Side-by-side comparison for all tracks if flash and futga are present
    if "flash" in args.systems and "futga" in args.systems:
        comp_dir = base_dir / "comparative_plots"
        flash_dir = base_dir / "flash"
        futga_dir = base_dir / "futga"

        for tr in track_list:
            f_data = load_track_matrix_and_meta(flash_dir, tr)
            u_data = load_track_matrix_and_meta(futga_dir, tr)
            if f_data is not None and u_data is not None:
                plot_side_by_side_comparison(
                    track_name=tr,
                    sys1_key="flash",
                    sys1_data=f_data,
                    sys2_key="futga",
                    sys2_data=u_data,
                    out_dir=comp_dir,
                    save_pdf=save_pdf,
                )

    # 3. High-level corpus comparative charts
    plot_corpus_comparative_summary(base_dir=base_dir, systems=args.systems, save_pdf=save_pdf)

    logging.info("Intra-track plotting suite finished successfully!")


if __name__ == "__main__":
    main()
