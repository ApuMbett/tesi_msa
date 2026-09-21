import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from semantic_msa.agents.captioner import TokenCaptionerAgent
from semantic_msa.agents.core.factory import LLMFactory
from semantic_msa.agents.core.schemas import FinalTrackDTO
from semantic_msa.agents.core.synthesizer import TrackSynthesizer
from semantic_msa.agents.profiler import AcousticProfilerAgent
from semantic_msa.agents.researcher import CulturalResearcherAgent
from semantic_msa.agents.segmenter import SemanticSegmenterAgent
from semantic_msa.utils.logger import logger


class AgentsPipeline:
    """
    Main orchestrator for the Semantic MSA Multi-Agent Pipeline.
    Directs the 4 domain agents (Profiler, Researcher, Captioner, Segmenter),
    handles checkpoint resumption, retry resilience, logging, and synthesizes FinalTrackDTO.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        mock_mode: bool = False,
    ):
        cfg_file = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parent.parent / "config" / "pipeline.yaml"
        )
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        self.experiment_name = self.cfg.get("experiment_name", "default_pipeline")
        self.unit = self.cfg.get("unit", "minutes")
        self.mock_mode = mock_mode
        self.factory = LLMFactory(mock_mode=mock_mode)

        # Parse pipeline execution settings
        pipeline_settings = self.cfg.get("pipeline_settings", {})
        self.resume = bool(pipeline_settings.get("resume", True))
        self.save_intermediate_steps = bool(pipeline_settings.get("save_intermediate_steps", True))
        self.max_retries = int(pipeline_settings.get("max_retries", 2))

        # Propagate max_retries to agent configs if not specifically overridden
        agent_cfgs = self.cfg.get("agents", {})
        for agent_name in ["profiler", "researcher", "captioner", "segmenter"]:
            if agent_name in agent_cfgs and "max_retries" not in agent_cfgs[agent_name]:
                agent_cfgs[agent_name]["max_retries"] = self.max_retries

        self.profiler = AcousticProfilerAgent(
            agent_cfgs.get("profiler", {}), self.factory
        )
        self.researcher = CulturalResearcherAgent(
            agent_cfgs.get("researcher", {}), self.factory
        )
        self.captioner = TokenCaptionerAgent(
            agent_cfgs.get("captioner", {}), self.factory
        )
        self.segmenter = SemanticSegmenterAgent(
            agent_cfgs.get("segmenter", {}), self.factory
        )

    def run(
        self,
        track_or_path: Any,
        output_dir: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> FinalTrackDTO:
        """
        Executes the multi-agent pipeline on a Track domain object or an LLM.json file.
        Supports step-level checkpoint resumption, skipping completed steps if valid.
        """
        # 1. Load track data
        track_source_path: Optional[Path] = None
        if isinstance(track_or_path, (str, Path)):
            track_source_path = Path(track_or_path)
            if not track_source_path.is_file():
                raise FileNotFoundError(f"LLM json file not found: {track_source_path}")
            with open(track_source_path, "r", encoding="utf-8") as f:
                track_data = json.load(f)
        elif hasattr(track_or_path, "LLM_representation"):
            track_data = track_or_path.LLM_representation()
            if hasattr(track_or_path, "LLM_lyrics_profile"):
                track_data["lyrics_profile"] = track_or_path.LLM_lyrics_profile
        elif isinstance(track_or_path, dict):
            track_data = track_or_path
        else:
            raise ValueError(f"Unsupported track input: {type(track_or_path)}")

        track_name = track_data.get("name", "unnamed_track")

        # 2. Resolve directory layout
        if output_dir:
            out_path = Path(output_dir)
        elif track_source_path:
            out_path = track_source_path.parent
        else:
            repo_root = Path(__file__).resolve().parents[4]
            out_path = repo_root / "data" / "processed" / track_name

        logs_dir = out_path / "logs"
        agents_dir = out_path / "agents"
        logs_dir.mkdir(parents=True, exist_ok=True)
        agents_dir.mkdir(parents=True, exist_ok=True)

        track_final_json_path = out_path / "final.json"
        agents_final_json_path = agents_dir / "final.json"

        # Check if entire track output already exists and resumption is active
        if not force and self.resume and track_final_json_path.is_file():
            try:
                with open(track_final_json_path, "r", encoding="utf-8") as f:
                    final_data = json.load(f)
                final_track = FinalTrackDTO.model_validate(final_data)
                logger.success(
                    f"Track '{track_name}' output already exists at [bold cyan]{track_final_json_path}[/bold cyan]. "
                    f"Resuming/skipping pipeline execution."
                )
                return final_track
            except Exception as e:
                logger.warning(f"Existing final.json at {track_final_json_path} could not be validated ({e}). Re-running pipeline.")

        # 3. Initialize file logger for this run
        current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.add_file_handler(logs_dir, current_run_id)

        start_time = time.time()
        logger.step(f"Starting Multi-Agent Pipeline for track '{track_name}' (Run ID: {current_run_id})")

        # 4. Log configuration parameters
        logger.param(f"experiment: {self.experiment_name}")
        logger.param(f"pipeline_settings: resume={self.resume}, save_intermediate={self.save_intermediate_steps}, max_retries={self.max_retries}")
        logger.param(f"synthesis: unit={self.unit}")
        logger.param(f"profiler: model={self.profiler.model}, temp={self.profiler.temperature}, thinking_level={self.profiler.thinking_level}")
        logger.param(f"researcher: model={self.researcher.model}, temp={self.researcher.temperature}, thinking_level={self.researcher.thinking_level}, grounding={self.researcher.grounding}")
        logger.param(f"captioner: model={self.captioner.model}, temp={self.captioner.temperature}, thinking_level={self.captioner.thinking_level}, length={self.captioner.caption_length}")
        logger.param(f"segmenter: model={self.segmenter.model}, temp={self.segmenter.temperature}, thinking_level={self.segmenter.thinking_level}, granularity={self.segmenter.granularity}")

        # --- STEP 1: Acoustic Profiler ---
        profiler_file = agents_dir / "profiler.json"
        profiler_dict = None
        if not force and self.resume and profiler_file.is_file():
            try:
                with open(profiler_file, "r", encoding="utf-8") as f:
                    profiler_dict = json.load(f)
                logger.info("[1/4] Loaded existing profiler.json (skipping Profiler LLM call)")
            except Exception as e:
                logger.warning(f"Existing profiler.json could not be loaded ({e}). Re-running Profiler.")

        if profiler_dict is None:
            logger.step("[1/4] Running Acoustic Profiler...")
            p_t0 = time.time()
            p_input, profiler_dto, profiler_dict = self.profiler.run(track_data)
            logger.info(f"=== PROFILER INPUT PAYLOAD ===\n{p_input}")
            logger.info(f"=== PROFILER OUTPUT PAYLOAD ===\n{json.dumps(profiler_dict, indent=2)}")
            if self.save_intermediate_steps:
                with open(profiler_file, "w", encoding="utf-8") as f:
                    json.dump(profiler_dict, f, indent=2, ensure_ascii=False)
            logger.success(f"Acoustic Profiler complete in {time.time() - p_t0:.2f}s")

        # --- STEP 2: Cultural Researcher ---
        researcher_file = agents_dir / "researcher.json"
        researcher_dict = None
        if not force and self.resume and researcher_file.is_file():
            try:
                with open(researcher_file, "r", encoding="utf-8") as f:
                    researcher_dict = json.load(f)
                logger.info("[2/4] Loaded existing researcher.json (skipping Researcher LLM call)")
            except Exception as e:
                logger.warning(f"Existing researcher.json could not be loaded ({e}). Re-running Researcher.")

        if researcher_dict is None:
            logger.step("[2/4] Running Cultural Researcher...")
            r_t0 = time.time()
            r_input, researcher_dto, researcher_dict = self.researcher.run(track_data, profiler_dict)
            logger.info(f"=== RESEARCHER INPUT PAYLOAD ===\n{r_input}")
            logger.info(f"=== RESEARCHER OUTPUT PAYLOAD ===\n{json.dumps(researcher_dict, indent=2)}")
            if self.save_intermediate_steps:
                with open(researcher_file, "w", encoding="utf-8") as f:
                    json.dump(researcher_dict, f, indent=2, ensure_ascii=False)
            logger.success(f"Cultural Researcher complete in {time.time() - r_t0:.2f}s")

        # --- STEP 3: Token Captioner ---
        captioner_file = agents_dir / "captioner.json"
        captioner_dict = None
        if not force and self.resume and captioner_file.is_file():
            try:
                with open(captioner_file, "r", encoding="utf-8") as f:
                    captioner_dict = json.load(f)
                logger.info("[3/4] Loaded existing captioner.json (skipping Captioner LLM call)")
            except Exception as e:
                logger.warning(f"Existing captioner.json could not be loaded ({e}). Re-running Captioner.")

        if captioner_dict is None:
            logger.step("[3/4] Running Token Captioner...")
            c_t0 = time.time()
            c_input, captioner_dto, captioner_dict = self.captioner.run(
                track_data, profiler_dict, researcher_dict
            )
            logger.info(f"=== CAPTIONER INPUT PAYLOAD ===\n{c_input}")
            logger.info(f"=== CAPTIONER OUTPUT PAYLOAD ===\n{json.dumps(captioner_dict, indent=2)}")
            if self.save_intermediate_steps:
                with open(captioner_file, "w", encoding="utf-8") as f:
                    json.dump(captioner_dict, f, indent=2, ensure_ascii=False)
            logger.success(f"Token Captioner complete in {time.time() - c_t0:.2f}s")

        # --- STEP 4: Semantic Segmenter ---
        segmenter_file = agents_dir / "segmenter.json"
        segmenter_dict = None
        if not force and self.resume and segmenter_file.is_file():
            try:
                with open(segmenter_file, "r", encoding="utf-8") as f:
                    segmenter_dict = json.load(f)
                logger.info("[4/4] Loaded existing segmenter.json (skipping Segmenter LLM call)")
            except Exception as e:
                logger.warning(f"Existing segmenter.json could not be loaded ({e}). Re-running Segmenter.")

        if segmenter_dict is None:
            logger.step("[4/4] Running Semantic Segmenter...")
            s_t0 = time.time()
            s_input, segmenter_dto, segmenter_dict = self.segmenter.run(
                track_data,
                profiler_dict,
                researcher_dict,
                captioner_dict.get("token_captions", []),
            )
            logger.info(f"=== SEGMENTER INPUT PAYLOAD ===\n{s_input}")
            logger.info(f"=== SEGMENTER OUTPUT PAYLOAD ===\n{json.dumps(segmenter_dict, indent=2)}")
            if self.save_intermediate_steps:
                with open(segmenter_file, "w", encoding="utf-8") as f:
                    json.dump(segmenter_dict, f, indent=2, ensure_ascii=False)
            logger.success(f"Semantic Segmenter complete in {time.time() - s_t0:.2f}s")

        # --- STEP 5: Final Synthesis ---
        logger.step("Synthesizing final structured track...")
        final_track = TrackSynthesizer.synthesize(
            track_data=track_data,
            profiler_output=profiler_dict,
            researcher_output=researcher_dict,
            captioner_output=captioner_dict,
            segmenter_output=segmenter_dict,
            unit=self.unit,
            caption_length=self.captioner.caption_length,
            granularity=self.segmenter.granularity,
        )

        final_json_str = final_track.to_json()
        if self.save_intermediate_steps:
            with open(agents_final_json_path, "w", encoding="utf-8") as f:
                f.write(final_json_str)

        # Always write to track directory (primary benchmark/track final.json location)
        with open(track_final_json_path, "w", encoding="utf-8") as f:
            f.write(final_json_str)

        logger.success(f"Multi-Agent Pipeline finished in {time.time() - start_time:.2f}s!")
        logger.info(f"Final output written to [bold cyan]{track_final_json_path}[/bold cyan]")

        return final_track


def main_cli():
    """CLI entry point for running the multi-agent pipeline standalone."""
    parser = argparse.ArgumentParser(
        description="Run the Semantic MSA Multi-Agent Pipeline."
    )
    parser.add_argument(
        "--llm-json",
        type=str,
        default=None,
        help="Path to precomputed LLM.json file",
    )
    parser.add_argument(
        "--track-name",
        type=str,
        default=None,
        help="Track name to resolve data/processed/<track_name>/LLM.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline.yaml configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for logs and agent outputs",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using offline deterministic StubProvider (no API keys required)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all steps, ignoring existing checkpoint files",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from existing step outputs",
    )

    args = parser.parse_args()

    # Determine input track file
    target_path = None
    if args.llm_json:
        target_path = Path(args.llm_json)
    elif args.track_name:
        repo_root = Path(__file__).resolve().parents[4]
        target_path = repo_root / "data" / "processed" / args.track_name / "LLM.json"
    else:
        parser.error("Must provide either --llm-json or --track-name")

    try:
        pipeline = AgentsPipeline(config_path=args.config, mock_mode=args.mock)
        if args.no_resume:
            pipeline.resume = False
        pipeline.run(target_path, output_dir=args.output_dir, force=args.force)
    except Exception as e:
        logger.error(f"Pipeline crashed during execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
