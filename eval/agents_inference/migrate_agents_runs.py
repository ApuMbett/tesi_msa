#!/usr/bin/env python3
"""
migrate_agents_runs.py

Non-destructive migration script to organize existing loose agent outputs
into an isolated run folder (e.g. agents/evaluation_max_v1/) and create
a root final_{run_name}.json copy for each track.
"""

import argparse
import shutil
import sys
from pathlib import Path

KNOWN_AGENT_FILES = [
    "profiler.json",
    "researcher.json",
    "captioner.json",
    "segmenter.json",
    "final.json",
]


def migrate_track(track_dir: Path, target_run: str, dry_run: bool = False) -> bool:
    agents_dir = track_dir / "agents"
    if not agents_dir.is_dir():
        return False

    # Check for loose agent files directly under agents/
    loose_files = [agents_dir / fname for fname in KNOWN_AGENT_FILES if (agents_dir / fname).is_file()]
    target_run_dir = agents_dir / target_run

    migrated_any = False

    if loose_files:
        print(f"\n[Track: {track_dir.name}]")
        print(f"  Target Run Folder: {target_run_dir}")
        if not dry_run:
            target_run_dir.mkdir(parents=True, exist_ok=True)

        for f in loose_files:
            dest = target_run_dir / f.name
            print(f"  -> Moving {f.name} to {target_run}/{f.name}")
            if not dry_run:
                shutil.move(str(f), str(dest))
            migrated_any = True

    # Also check if root final.json exists and create final_{target_run}.json
    root_final = track_dir / "final.json"
    run_final = track_dir / f"final_{target_run}.json"
    if root_final.is_file() and not run_final.is_file():
        print(f"  -> Copying final.json to {run_final.name}")
        if not dry_run:
            shutil.copy2(str(root_final), str(run_final))
        migrated_any = True

    return migrated_any


def main():
    parser = argparse.ArgumentParser(
        description="Migrate loose agents outputs into namespaced run directories."
    )
    parser.add_argument(
        "--processed-folder",
        type=str,
        default=None,
        help="Path to processed tracks directory (defaults to ../eval_processed).",
    )
    parser.add_argument(
        "--target-run",
        type=str,
        default="evaluation_max_v1",
        help="Run / experiment name to nest existing files into (default: evaluation_max_v1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate changes without moving or copying files.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.processed_folder:
        processed_dir = Path(args.processed_folder).resolve()
    else:
        candidates = [
            script_dir.parent / "eval_processed",
            script_dir.parent.parent / "data" / "processed",
        ]
        processed_dir = next((c.resolve() for c in candidates if c.is_dir()), None)

    if not processed_dir or not processed_dir.is_dir():
        print(f"Error: Processed folder not found: {args.processed_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Starting Migration ===")
    print(f"Target Directory: {processed_dir}")
    print(f"Target Run Name : {args.target_run}")
    print(f"Dry Run Mode    : {args.dry_run}")

    track_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    migrated_count = 0

    for track_dir in track_dirs:
        if migrate_track(track_dir, args.target_run, dry_run=args.dry_run):
            migrated_count += 1

    print("\n" + "=" * 40)
    print(f"Migration finished. {migrated_count} of {len(track_dirs)} tracks migrated.")
    if args.dry_run:
        print("Note: This was a DRY RUN. Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
