#!/usr/bin/env python3
"""
calculate_remaining_segments.py

Reads the full SongFormBench dataset (SongFormBench.jsonl), filters out the tracks
present in the sampled test set (sampled_eval_dataset.jsonl), and computes the average
and median segment counts across the remaining songs.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, mode, stdev


def find_default_path(relative_paths: list[str], base_dir: Path) -> Path | None:
    """Finds the first existing candidate path relative to base_dir or project root."""
    for rel in relative_paths:
        candidate = (base_dir / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def parse_args():
    script_dir = Path(__file__).resolve().parent

    default_full = find_default_path(
        [
            "../dataset_loaders/downloaded_data/data/SongFormBench.jsonl",
            "../downloaded_data/data/SongFormBench.jsonl",
            "../tmp/data/SongFormBench.jsonl",
        ],
        script_dir,
    )

    default_sampled = find_default_path(
        [
            "../dataset_loaders/sampled_eval_dataset.jsonl",
        ],
        script_dir,
    )

    parser = argparse.ArgumentParser(
        description="Calculate segment statistics on remaining SongFormBench songs after excluding the test set."
    )
    parser.add_argument(
        "--full-jsonl",
        type=Path,
        default=default_full,
        help="Path to full SongFormBench.jsonl",
    )
    parser.add_argument(
        "--sampled-jsonl",
        type=Path,
        default=default_sampled,
        help="Path to sampled test set jsonl (e.g. sampled_eval_dataset.jsonl)",
    )
    parser.add_argument(
        "--save-remaining-jsonl",
        type=Path,
        default=None,
        help="Optional path to save the remaining songs to a new JSONL file.",
    )

    return parser.parse_args()


def load_jsonl(file_path: Path) -> list[dict]:
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                print(f"Warning: Failed to parse line {line_num} in {file_path}: {err}", file=sys.stderr)
    return records


def compute_segment_metrics(labels: list[dict]) -> tuple[int, int, int]:
    """
    Computes segment counts for a single track:
      - structural: count of intervals (excluding terminal 'end' marker if present)
      - musical: count of non-'silence' intervals (excluding 'end')
      - raw: len(labels)
    """
    if not labels:
        return 0, 0, 0

    has_end = labels[-1].get("label") == "end"
    structural_count = len(labels) - 1 if has_end else len(labels)

    musical_count = len([
        l for l in labels
        if l.get("label") not in ("end", "silence")
    ])

    raw_count = len(labels)
    return structural_count, musical_count, raw_count


def print_stats_table(tracks: list[dict], group_name: str):
    print(f"\n{'=' * 72}")
    print(f"  {group_name} (N = {len(tracks)} songs)")
    print(f"{'=' * 72}")

    if not tracks:
        print("  No tracks in this subset.")
        return

    structural_counts = []
    musical_counts = []
    raw_counts = []

    for t in tracks:
        s, m, r = compute_segment_metrics(t.get("labels", []))
        structural_counts.append(s)
        musical_counts.append(m)
        raw_counts.append(r)

    def print_metric(label: str, data: list[int]):
        avg_val = mean(data)
        med_val = median(data)
        mod_val = mode(data)
        sd_val = stdev(data) if len(data) > 1 else 0.0
        min_val = min(data)
        max_val = max(data)

        print(f"\n* {label}:")
        print(f"    - Average (Mean):   {avg_val:.2f} ({avg_val:.4f})")
        print(f"    - Median:           {med_val:.1f}")
        print(f"    - Mode:             {mod_val}")
        print(f"    - Std Dev:          {sd_val:.2f}")
        print(f"    - Range (Min/Max):  {min_val} to {max_val}")

    print_metric(
        "1. Structural Segments (Intervals between boundaries, excluding 'end')",
        structural_counts,
    )
    print_metric(
        "2. Musical Segments (Excluding 'silence' intervals & 'end')",
        musical_counts,
    )
    print_metric(
        "3. Raw Labels Count (Including 'end' marker if present)",
        raw_counts,
    )

    print("\n  Structural Segment Count Distribution:")
    dist = sorted(Counter(structural_counts).items())
    dist_str = ", ".join(f"{segs} segs: {cnt} songs" for segs, cnt in dist)
    print(f"    {dist_str}")


def main():
    args = parse_args()

    if not args.full_jsonl or not args.full_jsonl.exists():
        print(f"Error: Full JSONL file not found at {args.full_jsonl}", file=sys.stderr)
        sys.exit(1)

    if not args.sampled_jsonl or not args.sampled_jsonl.exists():
        print(f"Error: Sampled JSONL file not found at {args.sampled_jsonl}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading full dataset from:    {args.full_jsonl}")
    full_tracks = load_jsonl(args.full_jsonl)
    print(f"  -> Loaded {len(full_tracks)} total tracks.")

    print(f"Loading test set from:        {args.sampled_jsonl}")
    sampled_tracks = load_jsonl(args.sampled_jsonl)
    print(f"  -> Loaded {len(sampled_tracks)} sampled test tracks.")

    # Collect IDs to exclude (handling prefixes like BHX_ if needed)
    sampled_ids = set()
    for t in sampled_tracks:
        tid = str(t.get("id", ""))
        sampled_ids.add(tid)
        if tid.startswith("BHX_"):
            sampled_ids.add(tid.replace("BHX_", ""))
        else:
            sampled_ids.add(f"BHX_{tid}")

    # Partition into HarmonixSet (SongFormBench-HS) and Other
    full_hs = [t for t in full_tracks if t.get("subset") == "HarmonixSet"]
    full_other = [t for t in full_tracks if t.get("subset") != "HarmonixSet"]

    # Filter out sampled test set
    remaining_hs = [t for t in full_hs if t.get("id") not in sampled_ids]
    remaining_all = [t for t in full_tracks if t.get("id") not in sampled_ids]

    print(f"\nFiltering Overview:")
    print(f"  - Full SongFormBench HarmonixSet (HS): {len(full_hs)} tracks")
    print(f"  - Excluded test tracks from HS:        {len(full_hs) - len(remaining_hs)} tracks")
    print(f"  - Remaining SongFormBench-HS tracks:   {len(remaining_hs)} tracks")
    print(f"  - Full SongFormBench (all subsets):    {len(full_tracks)} tracks")
    print(f"  - Remaining tracks across all subsets: {len(remaining_all)} tracks")

    # Print calculations
    print_stats_table(remaining_hs, "Remaining SongFormBench-HS (HarmonixSet - Test Set)")
    print_stats_table(remaining_all, "Remaining SongFormBench All Subsets (Full - Test Set)")
    print_stats_table(sampled_tracks, "Sampled Test Set Reference (For comparison)")

    # Save remaining tracks if requested
    if args.save_remaining_jsonl:
        args.save_remaining_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_remaining_jsonl, "w", encoding="utf-8") as f:
            for t in remaining_hs:
                f.write(json.dumps(t) + "\n")
        print(f"\nSaved {len(remaining_hs)} remaining HS tracks to: {args.save_remaining_jsonl}")


if __name__ == "__main__":
    main()
