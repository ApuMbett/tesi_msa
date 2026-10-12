#!/usr/bin/env python3
"""
to_msa_format.py

Converts final.json segment outputs from the Semantic MSA Multi-Agent Pipeline
into the standard SongFormer MSA TXT format:
    <start_time_seconds> <standard_label>
    ...
    <end_time_seconds> end

The exported TXT filename matches the .wav file found in each track's audio folder.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def parse_time_to_seconds(val: Any) -> float:
    """
    Parses a timestamp into float seconds.
    Supports float/int numbers and 'MM:SS' / 'MM:SS.SS' strings.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)

    val_str = str(val).strip()
    if ":" in val_str:
        parts = val_str.split(":")
        try:
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60.0 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600.0 + minutes * 60.0 + seconds
        except ValueError:
            pass

    try:
        return float(val_str)
    except ValueError:
        return 0.0


def export_to_songformer_txt(final_json_data: dict, output_txt_path: str) -> bool:
    """
    Converts semantic MSA segment output to the standard SongFormer MSA TXT format:
    <start_time_seconds> <standard_label>
    ...
    <end_time_seconds> end
    """
    segments = final_json_data.get("segments", [])
    if not segments:
        return False

    lines: List[str] = []
    for seg in segments:
        # Resolve start time in seconds
        raw_start = seg.get("start_time_seconds")
        if raw_start is None:
            raw_start = seg.get("start_time")
        if raw_start is None:
            raw_start = seg.get("start", 0.0)
        start_sec = parse_time_to_seconds(raw_start)

        label = seg.get("standard_label") or seg.get("label", "inst")
        label = str(label).lower().strip()
        lines.append(f"{start_sec:.2f} {label}")

    # Resolve terminal timestamp marker (end of song)
    track_identity = final_json_data.get("track_identity", {})
    raw_duration = track_identity.get("duration_seconds")
    if raw_duration is None:
        raw_duration = track_identity.get("duration")
    if raw_duration is None:
        raw_duration = final_json_data.get("post_processing", {}).get("track_duration")
    if raw_duration is None:
        raw_duration = final_json_data.get("metadata", {}).get("duration")
    if raw_duration is None and segments:
        last_seg = segments[-1]
        raw_duration = last_seg.get("end_time_seconds") or last_seg.get("end_time") or last_seg.get("end", 0.0)

    total_duration_sec = parse_time_to_seconds(raw_duration)
    lines.append(f"{total_duration_sec:.2f} end")

    out_path = Path(output_txt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return True


def find_audio_stem(track_dir: Path) -> str:
    """
    Locates the corresponding .wav file name in track_dir/audio/ or track_dir.
    Falls back to folder name if no .wav is found.
    """
    audio_dir = track_dir / "audio"
    if audio_dir.is_dir():
        wav_files = sorted(audio_dir.glob("*.wav"))
        if wav_files:
            return wav_files[0].stem

    # Check top-level of track_dir
    wav_files = sorted(track_dir.glob("*.wav"))
    if wav_files:
        return wav_files[0].stem

    return track_dir.name


def parse_args():
    script_dir = Path(__file__).resolve().parent

    default_processed = [
        script_dir / "eval_processed",
        script_dir.parent / "eval" / "eval_processed",
        script_dir.parent / "data" / "processed",
    ]

    default_output = script_dir / "inference_results" / "agents_txt"

    parser = argparse.ArgumentParser(
        description="Convert final.json segment annotations to SongFormer standard TXT format."
    )
    parser.add_argument(
        "--processed-folder",
        "--processed_folder",
        dest="processed_folder",
        type=str,
        default=None,
        help="Path to directory containing processed songs (default: eval/eval_processed).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=str(default_output),
        help=f"Destination directory for .txt files (default: {default_output}).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Also write the .txt file directly inside each song folder.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Convert only a specific track name.",
    )
    parser.add_argument(
        "--direct-audio",
        "--direct_audio",
        action="store_true",
        help="Convert direct audio post-processed JSONs (defaults output to eval/inference_results/direct_audio_txt).",
    )
    parser.add_argument(
        "--run",
        "--experiment",
        dest="run",
        type=str,
        default=None,
        help="Optional run or experiment name to convert (e.g. evaluation_max_v1).",
    )

    args = parser.parse_args()

    # Determine default output directory based on mode
    if args.direct_audio and (not args.output_dir or args.output_dir == str(default_output)):
        resolved_output = script_dir / "inference_results" / "direct_audio_txt"
    else:
        resolved_output = Path(args.output_dir).resolve()

    # Resolve processed folder
    processed_path = None
    if args.processed_folder:
        p = Path(args.processed_folder)
        if p.is_dir():
            processed_path = p.resolve()
        else:
            p_rel = (script_dir / args.processed_folder).resolve()
            if p_rel.is_dir():
                processed_path = p_rel
    else:
        for candidate in default_processed:
            if candidate.is_dir():
                processed_path = candidate.resolve()
                break

    if not processed_path or not processed_path.is_dir():
        print(f"Error: Processed folder not found: {args.processed_folder or default_processed}", file=sys.stderr)
        sys.exit(1)

    return args, processed_path, resolved_output


def main():
    args, processed_dir, output_dir = parse_args()

    print(f"\n========================================================")
    print(f" Converting final.json to SongFormer MSA TXT Format")
    print(f" Source:      {processed_dir}")
    print(f" Destination: {output_dir}")
    if args.in_place:
        print(f" In-Place:    Enabled (saving also in each song directory)")
    print(f"========================================================\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect track directories
    if (processed_dir / "final.json").is_file():
        track_dirs = [processed_dir]
    else:
        track_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])

    if args.track:
        track_dirs = [d for d in track_dirs if d.name == args.track]

    converted_count = 0
    skipped_count = 0

    for track_dir in track_dirs:
        final_json_path = None
        if args.direct_audio or args.run in ("direct_audio", "evaluation_direct_audio_flash_lite"):
            candidates = [
                track_dir / "direct_audio_postprocess.json",
                track_dir / "agents" / "evaluation_direct_audio_flash_lite" / "direct_audio_postprocess.json",
                track_dir / "final_evaluation_direct_audio_flash_lite.json",
                track_dir / "agents" / "evaluation_direct_audio_flash_lite" / "final.json",
            ]
            for cand in candidates:
                if cand.is_file():
                    final_json_path = cand
                    break
        elif args.run:
            candidates = [
                track_dir / f"final_{args.run}.json",
                track_dir / "agents" / args.run / "final.json",
            ]
            for cand in candidates:
                if cand.is_file():
                    final_json_path = cand
                    break
        else:
            candidates = [
                track_dir / "direct_audio_postprocess.json",
                track_dir / "final.json",
                track_dir / "agents" / "final.json",
            ]
            for cand in candidates:
                if cand.is_file():
                    final_json_path = cand
                    break

        if not final_json_path or not final_json_path.is_file():
            skipped_count += 1
            continue
        try:
            with open(final_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            audio_stem = find_audio_stem(track_dir)
            target_txt_name = f"{audio_stem}.txt"
            target_txt_path = output_dir / target_txt_name

            ok = export_to_songformer_txt(data, str(target_txt_path))
            if ok:
                converted_count += 1
                print(f"  ✓ {track_dir.name} -> {target_txt_name}")

                if args.in_place:
                    in_place_path = track_dir / target_txt_name
                    export_to_songformer_txt(data, str(in_place_path))
            else:
                print(f"  ⚠ {track_dir.name}: No segments found in final.json")
                skipped_count += 1

        except Exception as e:
            print(f"  ✖ {track_dir.name}: Failed to convert ({e})", file=sys.stderr)
            skipped_count += 1

    print(f"\nFinished!")
    print(f"  Total Converted: {converted_count}")
    print(f"  Skipped/Missing: {skipped_count}")
    print(f"  Output Folder:   {output_dir}\n")


if __name__ == "__main__":
    main()
