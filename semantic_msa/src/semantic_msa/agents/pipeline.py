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
from semantic_msa.agents.core.schemas import FinalTrackDTO, FinalSegmentDTO, FinalTrackIdentityDTO
from semantic_msa.agents.core.synthesizer import TrackSynthesizer
from semantic_msa.agents.direct_audio import DirectAudioAgent
from semantic_msa.agents.profiler import AcousticProfilerAgent
from semantic_msa.agents.researcher import CulturalResearcherAgent
from semantic_msa.agents.segmenter import SemanticSegmenterAgent
from semantic_msa.utils.logger import logger


class AgentsPipeline:
    """
    Main orchestrator for the Semantic MSA Multi-Agent Pipeline.
    Directs the 4 domain agents (Profiler, Researcher, Captioner, Segmenter),
    handles checkpoint resumption, retry resilience, logging, and synthesizes FinalTrackDTO.
    Also supports end-to-end Direct Audio baseline execution with gemini-3.5-flash-lite.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        mock_mode: bool = False,
        rate_limit_rpm: Optional[int] = None,
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

        # Parse pipeline execution settings
        pipeline_settings = self.cfg.get("pipeline_settings", {})
        self.resume = bool(pipeline_settings.get("resume", True))
        self.save_intermediate_steps = bool(pipeline_settings.get("save_intermediate_steps", True))
        self.max_retries = int(pipeline_settings.get("max_retries", 2))

        # Check if direct audio baseline mode is active
        self.mode = str(self.cfg.get("mode", "multi_agent")).strip()
        mode_normalized = self.mode.lower().replace(" ", "_")
        self.is_direct_audio = (
            mode_normalized in ("direct_audio", "directaudio")
            or bool(self.cfg.get("direct_audio_enabled", False))
            or bool(pipeline_settings.get("direct_audio", False))
        )

        cfg_rpm = pipeline_settings.get("rate_limit_rpm")
        self.rate_limit_rpm = rate_limit_rpm if rate_limit_rpm is not None else (int(cfg_rpm) if cfg_rpm else None)

        self.factory = LLMFactory(mock_mode=mock_mode, rate_limit_rpm=self.rate_limit_rpm)

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

        # Initialize Direct Audio baseline agent
        direct_audio_cfg = dict(
            self.cfg.get("direct_audio")
            or agent_cfgs.get("direct_audio")
            or agent_cfgs.get("segmenter", {})
        )
        if "max_retries" not in direct_audio_cfg:
            direct_audio_cfg["max_retries"] = self.max_retries
        self.direct_audio_agent = DirectAudioAgent(
            direct_audio_cfg, self.factory
        )

    def run(
        self,
        track_or_path: Any,
        output_dir: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        force: bool = False,
        direct_audio: bool = False,
    ) -> FinalTrackDTO:
        """
        Executes the multi-agent pipeline or direct audio baseline on a Track domain object or an LLM.json file.
        Supports step-level checkpoint resumption, skipping completed steps if valid.
        Intermediate artifacts are saved in agents/<run_name>/, and the final output
        is saved as final_<run_name>.json and agents/<run_name>/final.json.
        """
        if direct_audio or self.is_direct_audio:
            return self.run_direct_audio(
                track_or_path=track_or_path,
                output_dir=output_dir,
                run_id=run_id,
                run_name=run_name,
                force=force,
            )

        active_run = run_name or self.experiment_name or "default_pipeline"

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
        agents_base_dir = out_path / "agents"
        run_agents_dir = agents_base_dir / active_run
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_agents_dir.mkdir(parents=True, exist_ok=True)

        # Output paths for this run
        run_final_json_path = out_path / f"final_{active_run}.json"
        agents_final_json_path = run_agents_dir / "final.json"
        track_final_json_path = out_path / "final.json"

        # Check if entire track output already exists for this run and resumption is active
        if not force and self.resume:
            candidate_final = None
            if run_final_json_path.is_file():
                candidate_final = run_final_json_path
            elif agents_final_json_path.is_file():
                candidate_final = agents_final_json_path

            if candidate_final:
                try:
                    with open(candidate_final, "r", encoding="utf-8") as f:
                        final_data = json.load(f)
                    final_track = FinalTrackDTO.model_validate(final_data)
                    logger.success(
                        f"Track '{track_name}' output for run '{active_run}' already exists at [bold cyan]{candidate_final}[/bold cyan]. "
                        f"Resuming/skipping pipeline execution."
                    )
                    return final_track
                except Exception as e:
                    logger.warning(f"Existing final JSON at {candidate_final} could not be validated ({e}). Re-running pipeline.")

        # 3. Initialize file logger for this run
        current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.add_file_handler(logs_dir, current_run_id)

        start_time = time.time()
        logger.step(f"Starting Multi-Agent Pipeline for track '{track_name}' (Run: {active_run}, Run ID: {current_run_id})")

        # 4. Log configuration parameters
        logger.param(f"experiment: {self.experiment_name} (active run: {active_run})")
        logger.param(f"pipeline_settings: resume={self.resume}, save_intermediate={self.save_intermediate_steps}, max_retries={self.max_retries}, rate_limit_rpm={self.rate_limit_rpm}")
        logger.param(f"synthesis: unit={self.unit}")
        logger.param(f"profiler: model={self.profiler.model}, temp={self.profiler.temperature}, thinking_level={self.profiler.thinking_level}")
        logger.param(f"researcher: model={self.researcher.model}, temp={self.researcher.temperature}, thinking_level={self.researcher.thinking_level}, grounding={self.researcher.grounding}")
        logger.param(f"captioner: model={self.captioner.model}, temp={self.captioner.temperature}, thinking_level={self.captioner.thinking_level}, length={self.captioner.caption_length}")
        logger.param(f"segmenter: model={self.segmenter.model}, temp={self.segmenter.temperature}, thinking_level={self.segmenter.thinking_level}, granularity={self.segmenter.granularity}")

        # --- STEP 1: Acoustic Profiler ---
        profiler_file = run_agents_dir / "profiler.json"
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
        researcher_file = run_agents_dir / "researcher.json"
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
        captioner_file = run_agents_dir / "captioner.json"
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
        segmenter_file = run_agents_dir / "segmenter.json"
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

        # Write run-specific final output: final_{active_run}.json
        with open(run_final_json_path, "w", encoding="utf-8") as f:
            f.write(final_json_str)

        # Always write to track directory (primary/latest final.json location)
        with open(track_final_json_path, "w", encoding="utf-8") as f:
            f.write(final_json_str)

        logger.success(f"Multi-Agent Pipeline finished in {time.time() - start_time:.2f}s!")
        logger.info(f"Final output written to [bold cyan]{run_final_json_path}[/bold cyan] and [bold cyan]{agents_final_json_path}[/bold cyan]")

        return final_track

    def _find_audio_file(self, track_dir: Path) -> Path:
        """Finds .wav, .mp3, .flac, or .ogg in track_dir/audio or track_dir."""
        audio_dir = track_dir / "audio"
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
            if audio_dir.is_dir():
                matches = sorted(audio_dir.glob(ext))
                if matches:
                    return matches[0]
            matches = sorted(track_dir.glob(ext))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No audio file found in {track_dir} or {audio_dir}")

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Gets audio duration in seconds from .wav or soundfile."""
        if audio_path.suffix.lower() == ".wav":
            try:
                import wave
                with wave.open(str(audio_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return round(frames / float(rate), 2)
            except Exception:
                pass
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return round(info.duration, 2)
        except Exception:
            pass
        return 0.0

    def run_direct_audio(
        self,
        track_or_path: Any,
        output_dir: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        force: bool = False,
    ) -> FinalTrackDTO:
        """
        Executes end-to-end direct audio baseline inference using gemini-3.5-flash-lite.
        Sends the raw audio directly with metadata (title, artist, duration) and saves
        the resulting segmentation as final_<run_name>.json and agents/<run_name>/final.json.
        """
        active_run = run_name or self.experiment_name or "evaluation_direct_audio_flash_lite"

        # Resolve track directory and audio path
        track_dir: Path
        llm_data: Dict[str, Any] = {}
        if isinstance(track_or_path, (str, Path)):
            p = Path(track_or_path).resolve()
            if p.is_file():
                if p.name == "LLM.json":
                    track_dir = p.parent
                    with open(p, "r", encoding="utf-8") as f:
                        llm_data = json.load(f)
                elif p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                    if p.parent.name == "audio":
                        track_dir = p.parent.parent
                    else:
                        track_dir = p.parent
                else:
                    track_dir = p.parent
            else:
                track_dir = p
                llm_file = track_dir / "LLM.json"
                if llm_file.is_file():
                    with open(llm_file, "r", encoding="utf-8") as f:
                        llm_data = json.load(f)
        elif isinstance(track_or_path, dict):
            llm_data = track_or_path
            track_name = llm_data.get("name", "unnamed_track")
            repo_root = Path(__file__).resolve().parents[4]
            track_dir = repo_root / "data" / "processed" / track_name
        else:
            raise ValueError(f"Unsupported track input: {type(track_or_path)}")

        track_name = llm_data.get("name") or track_dir.name
        out_path = Path(output_dir).resolve() if output_dir else track_dir

        logs_dir = out_path / "logs"
        agents_base_dir = out_path / "agents"
        run_agents_dir = agents_base_dir / active_run
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_agents_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        run_final_json_path = out_path / f"final_{active_run}.json"
        agents_final_json_path = run_agents_dir / "final.json"
        track_final_json_path = out_path / "final.json"
        direct_audio_json_path = run_agents_dir / "direct_audio.json"

        # Check resumption
        if not force and self.resume:
            candidate_final = None
            if run_final_json_path.is_file():
                candidate_final = run_final_json_path
            elif agents_final_json_path.is_file():
                candidate_final = agents_final_json_path

            if candidate_final:
                try:
                    with open(candidate_final, "r", encoding="utf-8") as f:
                        final_data = json.load(f)
                    final_track = FinalTrackDTO.model_validate(final_data)
                    logger.success(
                        f"Track '{track_name}' output for run '{active_run}' already exists at [bold cyan]{candidate_final}[/bold cyan]. "
                        f"Resuming/skipping direct audio execution."
                    )
                    return final_track
                except Exception as e:
                    logger.warning(f"Existing final JSON at {candidate_final} could not be validated ({e}). Re-running.")

        current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.add_file_handler(logs_dir, current_run_id)

        start_time = time.time()
        logger.step(f"Starting Direct Audio Baseline for track '{track_name}' (Run: {active_run}, Model: {self.direct_audio_agent.model})")

        audio_file = self._find_audio_file(track_dir)
        logger.info(f"Target Audio: [bold cyan]{audio_file}[/bold cyan]")

        title = llm_data.get("name") or track_dir.name
        artist = llm_data.get("author") or llm_data.get("artist") or "Unknown Artist"
        duration_ms = llm_data.get("duration_ms")
        if duration_ms:
            duration = round(duration_ms / 1000.0, 2)
        else:
            duration = self._get_audio_duration(audio_file)
            if duration <= 0.0 and llm_data.get("duration"):
                try:
                    duration = float(llm_data.get("duration"))
                except ValueError:
                    duration = 0.0

        logger.param(f"Track: '{title}' by {artist} (Duration: {duration:.2f}s)")
        logger.param(f"Model: {self.direct_audio_agent.model}, Temperature: {self.direct_audio_agent.temperature}")

        # Execute direct audio segmentation via direct audio agent
        user_prompt, parsed_dto, result_dict = self.direct_audio_agent.run(
            audio_path=audio_file,
            title=title,
            artist=artist,
            duration=duration,
        )

        logger.info(f"=== DIRECT AUDIO PROMPT ===\n{user_prompt}")
        logger.info(f"=== DIRECT AUDIO OUTPUT ===\n{json.dumps(result_dict['segments'], indent=2)}")

        # Save direct audio step artifact: direct_audio.json
        if self.save_intermediate_steps:
            with open(direct_audio_json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

        # Synthesize into FinalTrackDTO
        final_segments = []
        for idx, s in enumerate(result_dict["segments"]):
            final_segments.append(
                FinalSegmentDTO(
                    index=idx,
                    standard_label=s["label"],
                    name=s.get("label", "").capitalize(),
                    start_time=s["start"],
                    end_time=s["end"],
                    macro_caption=s.get("caption", ""),
                    rationale=f"Direct audio functional segmentation via {self.direct_audio_agent.model}.",
                    tokens=[],
                )
            )

        final_track = FinalTrackDTO(
            track_identity=FinalTrackIdentityDTO(
                name=title,
                author=artist,
                artist=artist,
                duration=duration,
                cultural_context={},
                acoustic_profile={},
            ),
            token_size_beats=16,
            lyrics_profile={},
            caption_length=0,
            segmenter_granularity="DIRECT_AUDIO",
            segments=final_segments,
        )

        final_json_data = final_track.model_dump()
        final_json_data["track_identity"]["duration_seconds"] = duration
        for seg_data in final_json_data.get("segments", []):
            seg_data["start_time_seconds"] = seg_data.get("start_time")
            seg_data["end_time_seconds"] = seg_data.get("end_time")

        final_json_str = json.dumps(final_json_data, indent=2, ensure_ascii=False)

        if self.save_intermediate_steps:
            with open(agents_final_json_path, "w", encoding="utf-8") as f:
                f.write(final_json_str)

        with open(run_final_json_path, "w", encoding="utf-8") as f:
            f.write(final_json_str)

        with open(track_final_json_path, "w", encoding="utf-8") as f:
            f.write(final_json_str)

        elapsed = time.time() - start_time
        logger.success(f"Direct Audio Baseline for '{track_name}' finished in {elapsed:.2f}s!")
        logger.info(f"Final output written to [bold cyan]{run_final_json_path}[/bold cyan] and [bold cyan]{agents_final_json_path}[/bold cyan]")

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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run/experiment name for folder namespacing (defaults to config experiment_name)",
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
        pipeline.run(
            target_path,
            output_dir=args.output_dir,
            run_name=args.run_name,
            force=args.force,
        )
    except Exception as e:
        logger.error(f"Pipeline crashed during execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
