#!/usr/bin/env python3
"""
run_agents_pipeline.py

Batch runner for the Semantic MSA Multi-Agent Pipeline across all tracks in a processed folder.
Supports:
  - Custom configuration YAML files
  - Automatic track discovery (finding LLM.json in track subfolders)
  - Automatic checkpoint resumption (skipping completed tracks and intermediate steps)
  - Batch execution logging into eval/agents_inference/logs/log_<timestamp>.log
  - Per-track error isolation with batch execution summary
  - Single-track filtering and mock-mode for testing
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure semantic_msa package is importable regardless of current working directory
repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "semantic_msa" / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from semantic_msa.agents.pipeline import AgentsPipeline
from semantic_msa.utils.logger import PlainTextFormatter, logger


def resolve_path(input_path: Optional[str], default_candidates: List[Path]) -> Optional[Path]:
    """Resolves a user-provided path or tries candidate locations."""
    if input_path:
        p = Path(input_path)
        if p.is_file() or p.is_dir():
            return p.resolve()
        # Also check relative to script directory
        script_dir = Path(__file__).resolve().parent
        p_script = (script_dir / input_path).resolve()
        if p_script.is_file() or p_script.is_dir():
            return p_script
        return p.resolve()

    for candidate in default_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def parse_args():
    script_dir = Path(__file__).resolve().parent

    default_configs = [
        script_dir / "evaluation_direct_audio_flash_lite.yaml",
        script_dir / "evaluation_max_v1.yaml",
        script_dir.parent.parent / "semantic_msa" / "src" / "semantic_msa" / "config" / "pipeline.yaml",
    ]
    default_processed = [
        script_dir.parent / "eval_processed",
        script_dir.parent.parent / "data" / "processed",
    ]

    parser = argparse.ArgumentParser(
        description="Run Semantic MSA Multi-Agent Pipeline over a batch of processed tracks."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to pipeline config YAML (e.g. evaluation_direct_audio_flash_lite.yaml or evaluation_max_v1.yaml).",
    )
    parser.add_argument(
        "--processed-folder",
        "--processed_folder",
        dest="processed_folder",
        type=str,
        default=None,
        help="Path to directory containing processed track subfolders (e.g., ../eval_processed).",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Process only a specific track name (matching folder name).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional upper bound on number of tracks to process.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using offline deterministic StubProvider (no API keys required).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all tracks and steps, ignoring existing final.json or checkpoints.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable step-level checkpoint resumption.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode: runs only 1 song to safely test the pipeline before processing the whole batch.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run/experiment name for folder namespacing (defaults to config experiment_name).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Rate limit in requests per minute (e.g., 15 for Gemini Free Tier). Overrides config.",
    )

    args = parser.parse_args()

    if args.debug:
        args.limit = 1

    # Resolve config
    config_path = resolve_path(args.config, default_configs)
    if not config_path or not config_path.is_file():
        logger.error(f"Configuration file not found: {args.config or default_configs}")
        sys.exit(1)

    # Resolve processed folder
    processed_dir = resolve_path(args.processed_folder, default_processed)
    if not processed_dir or not processed_dir.is_dir():
        logger.error(f"Processed folder not found: {args.processed_folder or default_processed}")
        sys.exit(1)

    return args, config_path, processed_dir


def discover_tracks(
    processed_dir: Path,
    target_track: Optional[str] = None,
    require_audio: bool = False,
) -> List[Path]:
    """Finds all subfolders with an LLM.json file or audio files."""
    tracks: List[Path] = []

    # Check if the directory itself is a track folder
    if (processed_dir / "LLM.json").is_file():
        return [processed_dir / "LLM.json"]

    audio_extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")

    for item in sorted(processed_dir.iterdir()):
        if not item.is_dir():
            continue
        if target_track and item.name != target_track:
            continue

        llm_json = item / "LLM.json"
        if require_audio:
            # Check if track has an audio folder or audio file
            has_audio = False
            audio_dir = item / "audio"
            if audio_dir.is_dir():
                for ext in audio_extensions:
                    if list(audio_dir.glob(ext)):
                        has_audio = True
                        break
            if not has_audio:
                for ext in audio_extensions:
                    if list(item.glob(ext)):
                        has_audio = True
                        break

            if has_audio:
                if llm_json.is_file():
                    tracks.append(llm_json)
                else:
                    tracks.append(item)
        else:
            if llm_json.is_file():
                tracks.append(llm_json)

    return tracks


def main():
    # Initialize file logging immediately in eval/agents_inference/logs/log_<timestamp>.log
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"log_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        PlainTextFormatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.logger.addHandler(file_handler)

    try:
        args, config_path, processed_dir = parse_args()

        logger.info(f"\n[bold magenta]════════════ Starting Agents / Direct Audio Batch Pipeline ════════════[/bold magenta]")
        logger.info(f"Log file: [bold cyan]{log_file}[/bold cyan]")
        logger.info(f"Config: [bold cyan]{config_path}[/bold cyan]")
        logger.info(f"Processed Folder: [bold cyan]{processed_dir}[/bold cyan]")
        if args.mock:
            logger.warning("Running in MOCK mode (offline StubProvider)")
        if args.debug:
            logger.warning("[bold yellow]⚠ DEBUG MODE ACTIVE: Limiting run to 1 track only.[/bold yellow]")

        # 1. Instantiate pipeline early to check mode configuration
        try:
            pipeline = AgentsPipeline(
                config_path=config_path,
                mock_mode=args.mock,
                rate_limit_rpm=args.rpm,
            )
            if args.no_resume:
                pipeline.resume = False
        except Exception as e:
            logger.error(f"Failed to initialize AgentsPipeline: {e}", exc_info=True)
            sys.exit(1)

        is_direct_audio = pipeline.is_direct_audio
        if is_direct_audio:
            logger.info("[bold green]Execution Mode: DIRECT AUDIO BASELINE (sending audio directly to gemini-3.5-flash-lite)[/bold green]")
        else:
            logger.info("[bold cyan]Execution Mode: MULTI-AGENT PIPELINE (4 domain agents on DSP telemetry)[/bold cyan]")

        # 2. Discover tracks
        track_files = discover_tracks(
            processed_dir,
            target_track=args.track,
            require_audio=is_direct_audio,
        )
        if not track_files:
            logger.error(f"No valid tracks found in {processed_dir}")
            sys.exit(1)

        if args.limit:
            track_files = track_files[: args.limit]

        total_tracks = len(track_files)
        logger.info(f"Found [bold green]{total_tracks}[/bold green] track(s) to process.")

        active_run = args.run_name or pipeline.experiment_name or (
            "evaluation_direct_audio_flash_lite" if is_direct_audio else "default_pipeline"
        )
        logger.info(f"Active Run / Experiment: [bold yellow]{active_run}[/bold yellow]")
        if pipeline.rate_limit_rpm:
            logger.info(
                f"Rate Limiter: [bold green]{pipeline.rate_limit_rpm} RPM active[/bold green] "
                f"(min {60.0 / pipeline.rate_limit_rpm:.1f}s interval between requests)"
            )

        # 3. Process each track
        successful = []
        failed = []
        skipped = []

        batch_start = time.time()

        for idx, track_llm_path in enumerate(track_files, start=1):
            track_dir = track_llm_path if track_llm_path.is_dir() else track_llm_path.parent
            track_name = track_dir.name
            run_final_json_path = track_dir / f"final_{active_run}.json"
            agents_final_json_path = track_dir / "agents" / active_run / "final.json"

            # Check if already completed for this specific run
            is_already_done = (
                pipeline.resume
                and not args.force
                and (run_final_json_path.is_file() or agents_final_json_path.is_file())
            )

            logger.info(f"\n[bold blue]━━━ [{idx}/{total_tracks}] Track: {track_name} (Run: {active_run}) ━━━[/bold blue]")

            try:
                t0 = time.time()
                pipeline.run(
                    track_or_path=track_llm_path,
                    output_dir=None,  # Automatically defaults to track_dir
                    run_name=active_run,
                    force=args.force,
                    direct_audio=is_direct_audio,
                )
                elapsed = time.time() - t0

                if is_already_done:
                    skipped.append(track_name)
                    logger.info(f"Track '{track_name}' already complete for run '{active_run}' (checked in {elapsed:.2f}s).")
                else:
                    successful.append(track_name)
                    logger.success(f"Track '{track_name}' finished in {elapsed:.2f}s.")

            except Exception as e:
                failed.append((track_name, str(e)))
                logger.error(f"Track '{track_name}' failed: {e}", exc_info=True)

        # 4. Summary report
        total_elapsed = time.time() - batch_start
        logger.info("\n[bold magenta]════════════ Batch Execution Summary ════════════[/bold magenta]")
        logger.info(f"Total time elapsed: {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")
        logger.info(f"Total tracks: {total_tracks}")
        logger.info(f"Completed (Newly Processed): [bold green]{len(successful)}[/bold green]")
        logger.info(f"Skipped (Already Complete): [bold cyan]{len(skipped)}[/bold cyan]")
        logger.info(f"Failed: [bold red]{len(failed)}[/bold red]")

        if is_direct_audio:
            logger.info(f"Direct Audio Baseline Model: [bold cyan]{pipeline.direct_audio_agent.model}[/bold cyan]")

        if failed:
            logger.warning("Failed Tracks:")
            for name, err in failed:
                logger.warning(f"  - {name}: {err}")
            sys.exit(1)

    finally:
        file_handler.flush()
        file_handler.close()
        logger.logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()