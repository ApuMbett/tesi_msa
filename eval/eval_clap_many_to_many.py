#!/usr/bin/env python3
"""
eval_clap_many_to_many.py

Semantic MSA (RQ2) - Many-to-Many CLAP Cross-Modal Retrieval Evaluation.
Evaluates whether segment captions and audio clips are distinctive enough to retrieve
their corresponding counterpart across the entire multi-track pool of all segments
(~554 segments for flash, ~150 for futga), rather than just within the same track.

Reuses precomputed CLAP embeddings cached by eval_clap_intra_track.py:
  rq2_intra_track/output/<system>/embeddings/<bhx_id>.npz
Zero new CLAP inferences required.

Evaluates in both directions:
  1. Audio -> Text (A2T): for each audio clip, retrieve correct caption from pool.
  2. Text -> Audio (T2A): for each caption, retrieve correct audio clip from pool.

Metrics computed for each direction:
  - Recall@1, Recall@5, Recall@10
  - MedR (Median Rank)
  - Mean Rank & MRR (Mean Reciprocal Rank)
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

# Reuse system registry, track collection, and formatters from intra-track evaluation
from eval_clap_intra_track import (
    SYSTEM_REGISTRY,
    PlainTextFormatter,
    TrackItem,
    collect_tracks,
)

# ==============================================================================
# Logging Configuration
# ==============================================================================

console = Console()


def setup_logger(output_dir: Optional[Path] = None, system_name: Optional[str] = None) -> logging.Logger:
    """
    Sets up a logger with both rich console output and clean plain-text file logging.
    """
    logger = logging.getLogger(f"eval_clap_many_to_many.{system_name}" if system_name else "eval_clap_many_to_many")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)

    # Clean plain-text file handler per system in a dedicated logs/ folder with run timestamp
    if output_dir is not None and system_name is not None:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"eval_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = PlainTextFormatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"File logging initialized at {log_file}")

    return logger


# ==============================================================================
# Metric Calculations
# ==============================================================================


def compute_retrieval_ranks(
    similarity_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes retrieval ranks and top-1 predictions for both directions:
      - Audio -> Text (A2T): row i is audio query, target is text i (column i)
      - Text -> Audio (T2A): column j is text query, target is audio j (row j)

    similarity_matrix S has shape (M, M), where S[i, j] = cos(audio_i, text_j).

    Returns:
      ranks_a2t, top1_idx_a2t, sim_top1_a2t,
      ranks_t2a, top1_idx_t2a, sim_top1_t2a
    """
    M = similarity_matrix.shape[0]
    ranks_a2t = np.zeros(M, dtype=np.float64)
    top1_idx_a2t = np.zeros(M, dtype=np.int64)
    sim_top1_a2t = np.zeros(M, dtype=np.float32)

    ranks_t2a = np.zeros(M, dtype=np.float64)
    top1_idx_t2a = np.zeros(M, dtype=np.int64)
    sim_top1_t2a = np.zeros(M, dtype=np.float32)

    # 1. Audio -> Text (A2T)
    for i in range(M):
        sims = similarity_matrix[i, :]
        target_sim = sims[i]
        top1_idx = int(np.argmax(sims))
        top1_idx_a2t[i] = top1_idx
        sim_top1_a2t[i] = float(sims[top1_idx])

        # Rank of target item with tie handling
        strictly_greater = np.sum(sims > target_sim)
        ties = np.sum((np.abs(sims - target_sim) <= 1e-6) & (np.arange(M) != i))
        ranks_a2t[i] = float(1 + strictly_greater + 0.5 * ties)

    # 2. Text -> Audio (T2A)
    for j in range(M):
        sims = similarity_matrix[:, j]
        target_sim = sims[j]
        top1_idx = int(np.argmax(sims))
        top1_idx_t2a[j] = top1_idx
        sim_top1_t2a[j] = float(sims[top1_idx])

        # Rank of target item with tie handling
        strictly_greater = np.sum(sims > target_sim)
        ties = np.sum((np.abs(sims - target_sim) <= 1e-6) & (np.arange(M) != j))
        ranks_t2a[j] = float(1 + strictly_greater + 0.5 * ties)

    return ranks_a2t, top1_idx_a2t, sim_top1_a2t, ranks_t2a, top1_idx_t2a, sim_top1_t2a


def compute_direction_summary(ranks: np.ndarray, pool_size: int, direction_name: str) -> Dict[str, Any]:
    """
    Aggregates Recall@K, MedR, Mean Rank, and MRR for a given rank array.
    """
    if len(ranks) == 0:
        return {
            "direction": direction_name,
            "pool_size": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "medr": 0.0,
            "mean_rank": 0.0,
            "mrr": 0.0,
        }

    r1 = float(np.mean(ranks <= 1.0))
    r5 = float(np.mean(ranks <= 5.0))
    r10 = float(np.mean(ranks <= 10.0))
    medr = float(np.median(ranks))
    mean_rank = float(np.mean(ranks))
    mrr = float(np.mean(1.0 / ranks))

    return {
        "direction": direction_name,
        "pool_size": pool_size,
        "recall_at_1": r1,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "medr": medr,
        "mean_rank": mean_rank,
        "mrr": mrr,
    }


# ==============================================================================
# Pipeline Execution per System
# ==============================================================================


def process_system_many_to_many(
    system_key: str,
    tracks: List[TrackItem],
    intra_output_dir: Path,
    system_output_dir: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes many-to-many cross-modal retrieval across all segments of the system.
    Reuses pre-computed embeddings from rq2_intra_track/output/<system>/embeddings/<bhx_id>.npz.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating Many-to-Many Retrieval for System: [bold magenta]{system_key.upper()}[/bold magenta]")
    logger.info(f"{'=' * 80}")

    system_emb_dir = intra_output_dir / system_key / "embeddings"
    if not system_emb_dir.is_dir():
        raise FileNotFoundError(
            f"Embeddings directory not found at: {system_emb_dir}. "
            f"Please run eval_clap_intra_track.py first for system '{system_key}'."
        )

    all_audio_embeds: List[np.ndarray] = []
    all_text_embeds: List[np.ndarray] = []
    segment_records: List[Dict[str, Any]] = []

    loaded_tracks = 0
    total_segments = 0

    for item in tracks:
        emb_file = system_emb_dir / f"{item.bhx_id}.npz"
        if not emb_file.is_file():
            logger.warning(f"Embedding file missing for track {item.track_name} ({item.bhx_id}) at {emb_file}. Skipping.")
            continue

        try:
            npz_data = np.load(emb_file, allow_pickle=True)
            a_emb = npz_data["audio_embeddings"]
            t_emb = npz_data["text_embeddings"]
            labels = npz_data["labels"]
            starts = npz_data["starts"]
            ends = npz_data["ends"]
            captions = npz_data["captions"]
        except Exception as e:
            logger.error(f"Error loading {emb_file}: {e}")
            continue

        n_seg = len(a_emb)
        for k in range(n_seg):
            seg_id = f"{item.bhx_id}_{k}"
            all_audio_embeds.append(a_emb[k])
            all_text_embeds.append(t_emb[k])
            segment_records.append(
                {
                    "segment_id": seg_id,
                    "track_name": item.track_name,
                    "bhx_id": item.bhx_id,
                    "segment_index": k,
                    "start": float(starts[k]),
                    "end": float(ends[k]),
                    "standard_label": str(labels[k]),
                    "caption": str(captions[k]),
                }
            )

        loaded_tracks += 1
        total_segments += n_seg

    M = len(segment_records)
    logger.info(f"Loaded [bold green]{loaded_tracks} tracks[/bold green] with [bold cyan]{M} total segments[/bold cyan] into retrieval pool.")
    if M == 0:
        raise ValueError(f"No segment embeddings found for system '{system_key}'. Cannot perform retrieval.")

    # Stack and normalize embeddings
    A = np.stack(all_audio_embeds, axis=0).astype(np.float32)  # (M, 512)
    T = np.stack(all_text_embeds, axis=0).astype(np.float32)   # (M, 512)

    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_A[norm_A == 0] = 1.0
    A = A / norm_A

    norm_T = np.linalg.norm(T, axis=1, keepdims=True)
    norm_T[norm_T == 0] = 1.0
    T = T / norm_T

    # Cross-modal similarity matrix: S[u, v] = cos(audio_u, text_v)
    logger.info(f"Computing ({M} x {M}) cross-modal cosine similarity matrix...")
    S = np.matmul(A, T.T)  # (M, M)

    # Compute retrieval ranks for both directions
    logger.info("Computing retrieval ranks for Audio->Text and Text->Audio...")
    (
        ranks_a2t,
        top1_idx_a2t,
        sim_top1_a2t,
        ranks_t2a,
        top1_idx_t2a,
        sim_top1_t2a,
    ) = compute_retrieval_ranks(S)

    # Build per-segment ranks DataFrame
    for idx in range(M):
        rec = segment_records[idx]
        rec["rank_a2t"] = int(ranks_a2t[idx]) if ranks_a2t[idx].is_integer() else float(ranks_a2t[idx])
        rec["sim_a2t_target"] = float(S[idx, idx])
        rec["top1_a2t_segment_id"] = segment_records[top1_idx_a2t[idx]]["segment_id"]
        rec["top1_a2t_label"] = segment_records[top1_idx_a2t[idx]]["standard_label"]
        rec["sim_a2t_top1"] = float(sim_top1_a2t[idx])

        rec["rank_t2a"] = int(ranks_t2a[idx]) if ranks_t2a[idx].is_integer() else float(ranks_t2a[idx])
        rec["sim_t2a_target"] = float(S[idx, idx])
        rec["top1_t2a_segment_id"] = segment_records[top1_idx_t2a[idx]]["segment_id"]
        rec["top1_t2a_label"] = segment_records[top1_idx_t2a[idx]]["standard_label"]
        rec["sim_t2a_top1"] = float(sim_top1_t2a[idx])

    df_ranks = pd.DataFrame(segment_records)

    # Save ranks CSV
    system_output_dir.mkdir(parents=True, exist_ok=True)
    ranks_csv_path = system_output_dir / f"{system_key}_manytomany_ranks.csv"
    df_ranks.to_csv(ranks_csv_path, index=False, float_format="%.4f")
    logger.info(f"Saved per-segment ranks to: [bold green]{ranks_csv_path}[/bold green]")

    # Aggregate retrieval metrics
    summary_a2t = compute_direction_summary(ranks_a2t, M, "audio_to_text")
    summary_t2a = compute_direction_summary(ranks_t2a, M, "text_to_audio")

    summary_mean = {
        "direction": "mean",
        "pool_size": M,
        "recall_at_1": float(0.5 * (summary_a2t["recall_at_1"] + summary_t2a["recall_at_1"])),
        "recall_at_5": float(0.5 * (summary_a2t["recall_at_5"] + summary_t2a["recall_at_5"])),
        "recall_at_10": float(0.5 * (summary_a2t["recall_at_10"] + summary_t2a["recall_at_10"])),
        "medr": float(0.5 * (summary_a2t["medr"] + summary_t2a["medr"])),
        "mean_rank": float(0.5 * (summary_a2t["mean_rank"] + summary_t2a["mean_rank"])),
        "mrr": float(0.5 * (summary_a2t["mrr"] + summary_t2a["mrr"])),
    }

    summary_rows = [summary_a2t, summary_t2a, summary_mean]
    df_summary = pd.DataFrame(summary_rows)

    # Save summary CSV
    summary_csv_path = system_output_dir / f"{system_key}_manytomany_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False, float_format="%.4f")
    logger.info(f"Saved aggregated retrieval summary to: [bold green]{summary_csv_path}[/bold green]")

    # Display Rich summary table
    table = Table(title=f"Many-to-Many Cross-Modal Retrieval: {system_key.upper()} (Pool Size = {M})")
    table.add_column("Direction", style="cyan", no_wrap=True)
    table.add_column("Pool Size", justify="right", style="dim")
    table.add_column("R@1 (%)", justify="right", style="bold green")
    table.add_column("R@5 (%)", justify="right", style="green")
    table.add_column("R@10 (%)", justify="right", style="yellow")
    table.add_column("MedR", justify="right", style="magenta")
    table.add_column("Mean Rank", justify="right", style="blue")
    table.add_column("MRR", justify="right", style="bold cyan")

    for r in summary_rows:
        table.add_row(
            r["direction"],
            str(r["pool_size"]),
            f"{r['recall_at_1'] * 100:.2f}%",
            f"{r['recall_at_5'] * 100:.2f}%",
            f"{r['recall_at_10'] * 100:.2f}%",
            f"{r['medr']:.1f}",
            f"{r['mean_rank']:.1f}",
            f"{r['mrr']:.4f}",
        )

    console.print(table)

    return df_ranks, df_summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ2: CLAP Many-to-Many Cross-Modal Segment Retrieval Evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--systems",
        nargs="+",
        default=["flash"],
        help=f"One or more system keys to evaluate. Available: {list(SYSTEM_REGISTRY.keys())} or 'all'.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Path to eval/ directory (default: auto-detected).",
    )
    parser.add_argument(
        "--intra-output-dir",
        type=str,
        default=None,
        help="Path to rq2_intra_track/output directory where .npz embeddings are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for many-to-many output files (default: eval/rq2_many_to_many/output).",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        help="Optional list of track names to evaluate (default: all 50 tracks).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine paths
    if args.eval_dir:
        eval_dir = Path(args.eval_dir).resolve()
    else:
        current_dir = Path.cwd().resolve()
        if (current_dir / "eval_processed").is_dir():
            eval_dir = current_dir
        elif (current_dir / "eval" / "eval_processed").is_dir():
            eval_dir = current_dir / "eval"
        else:
            eval_dir = Path(__file__).resolve().parent

    eval_processed_dir = eval_dir / "eval_processed"

    if args.intra_output_dir:
        intra_output_dir = Path(args.intra_output_dir).resolve()
    else:
        intra_output_dir = eval_dir / "rq2_intra_track" / "output"

    if args.output_dir:
        base_output_dir = Path(args.output_dir).resolve()
    else:
        base_output_dir = eval_dir / "rq2_many_to_many" / "output"

    base_output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = setup_logger()

    root_logger.info("Initializing RQ2 Many-to-Many Cross-Modal Retrieval Evaluation")
    root_logger.info(f"eval directory: [bold cyan]{eval_dir}[/bold cyan]")
    root_logger.info(f"intra-track embeddings source: [bold cyan]{intra_output_dir}[/bold cyan]")
    root_logger.info(f"output directory: [bold cyan]{base_output_dir}[/bold cyan]")

    # Resolve selected systems
    selected_systems = list(SYSTEM_REGISTRY.keys()) if "all" in args.systems else args.systems
    for sys_key in selected_systems:
        if sys_key not in SYSTEM_REGISTRY:
            root_logger.error(f"Unknown system key: '{sys_key}'. Available: {list(SYSTEM_REGISTRY.keys())}")
            sys.exit(1)

    root_logger.info(f"Systems to evaluate: [bold yellow]{selected_systems}[/bold yellow]")

    # Collect 50 tracks
    tracks = collect_tracks(eval_processed_dir, target_tracks=args.tracks)
    root_logger.info(f"Collected [bold cyan]{len(tracks)}[/bold cyan] tracks from {eval_processed_dir}")
    if len(tracks) == 0:
        root_logger.error("No tracks found to evaluate. Exiting.")
        sys.exit(1)

    # Process each selected system independently
    for sys_key in selected_systems:
        system_output_dir = base_output_dir / sys_key
        sys_logger = setup_logger(output_dir=system_output_dir, system_name=sys_key)

        process_system_many_to_many(
            system_key=sys_key,
            tracks=tracks,
            intra_output_dir=intra_output_dir,
            system_output_dir=system_output_dir,
            logger=sys_logger,
        )

    root_logger.info(f"\n[bold green]All many-to-many evaluations completed successfully![/bold green]")
    root_logger.info(f"Output files stored under: [bold cyan]{base_output_dir}[/bold cyan]")


if __name__ == "__main__":
    main()
