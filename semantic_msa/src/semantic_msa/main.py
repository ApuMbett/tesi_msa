import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from semantic_msa.domain.workspace import TrackWorkspace
from semantic_msa.utils.logger import console, logger
from semantic_msa.utils.track_analyzer import TrackAnalyzer


def analyze_single_track(
    name: str,
    author: str,
    audio_path: Path,
    run_agents: bool = False,
    mock_agents: bool = False,
    config_path: Optional[str] = None,
):
    track_ws = TrackWorkspace(name, audio_path)

    # Set up file logger for this specific run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add_file_handler(track_ws.track_dir / "logs", run_id)

    logger.step(f"Starting analysis for {author} - {name}")

    extractor = TrackAnalyzer(track_ws, author)
    track = extractor.extract()

    full_json_path = track_ws.output_dir / "full.json"
    with open(full_json_path, "w", encoding="utf-8") as f:
        f.write(track.model_dump_json(indent=4, by_alias=True))
    logger.info(f"Full track JSON saved to {full_json_path}")

    LLM_json_path = track_ws.output_dir / "LLM.json"
    with open(LLM_json_path, "w", encoding="utf-8") as f:
        json.dump(track.LLM_representation(), f, indent=4)
    logger.info(f"LLM representation JSON saved to {LLM_json_path}")

    logger.success("DSP feature extraction complete.")

    if run_agents:
        from semantic_msa.agents import AgentsPipeline

        logger.step("Triggering Multi-Agent Pipeline...")
        pipeline = AgentsPipeline(config_path=config_path, mock_mode=mock_agents)
        pipeline.run(track, output_dir=track_ws.track_dir, run_id=run_id)


def main():
    parser = argparse.ArgumentParser(
        description="Process track audio analysis and multi-agent structural analysis."
    )
    parser.add_argument("--name", type=str, default=None, help="Track name")
    parser.add_argument("--author", type=str, default=None, help="Track author")
    parser.add_argument("--audio-path", type=str, default=None, help="Track audio path")
    parser.add_argument(
        "--run-agents",
        action="store_true",
        help="Run the multi-agent pipeline after audio extraction",
    )
    parser.add_argument(
        "--llm-json",
        type=str,
        default=None,
        help="Run multi-agent pipeline directly on an existing LLM.json file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use offline mock provider for agents (no API calls)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom pipeline.yaml config",
    )

    args = parser.parse_args()

    # Direct agent execution on precomputed LLM.json
    if args.llm_json:
        from semantic_msa.agents import AgentsPipeline

        try:
            pipeline = AgentsPipeline(config_path=args.config, mock_mode=args.mock)
            pipeline.run(args.llm_json)
        except Exception as e:
            logger.error(f"Agent pipeline crashed: {str(e)}", exc_info=True)
            sys.exit(1)
        return

    # Full audio analysis
    if not (args.name and args.author and args.audio_path):
        parser.error(
            "When --llm-json is not specified, --name, --author, and --audio-path are all required."
        )

    try:
        analyze_single_track(
            args.name,
            args.author,
            Path(args.audio_path),
            run_agents=args.run_agents,
            mock_agents=args.mock,
            config_path=args.config,
        )
    except Exception as e:
        logger.error(f"Pipeline crashed during execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
