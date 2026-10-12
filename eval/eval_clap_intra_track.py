#!/usr/bin/env python3
"""
eval_clap_intra_track.py

Semantic MSA (RQ2) - Intra-Track CLAP Caption-Audio Semantic Grounding Evaluation.
Evaluates whether segment captions are grounded in their corresponding audio segment
more than in any other segment of the same track.

Checkpoint: laion/clap-htsat-fused (feature-fusion CLAP, Larger CLAP), suitable for arbitrary audio durations.
Execution: CPU only.
Output: Independent per-system evaluation CSVs, summary CSVs, embeddings, and raw M_CLAP matrices.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from transformers import ClapModel, ClapProcessor

# ==============================================================================
# Logging Configuration
# ==============================================================================

console = Console()


class PlainTextFormatter(logging.Formatter):
    """
    Strips rich markup tags (e.g. [bold green]) and ANSI color escapes before
    writing to disk log files, keeping them clean plain-text like eval_processed.
    """

    def format(self, record: logging.LogRecord) -> str:
        record_copy = logging.makeLogRecord(record.__dict__)
        if isinstance(record_copy.msg, str):
            try:
                record_copy.msg = Text.from_markup(record_copy.msg).plain
            except Exception:
                record_copy.msg = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", record_copy.msg)
            # Remove any raw ANSI color escape sequences
            record_copy.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", record_copy.msg).strip()
        return super().format(record_copy)


def setup_logger(output_dir: Optional[Path] = None, system_name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("eval_clap_intra_track")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # Rich console handler (for terminal output with colors and progress)
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
# Data Structures
# ==============================================================================


@dataclass
class NormalizedSegment:
    start: float
    end: float
    standard_label: str
    caption: str
    index: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrackItem:
    track_name: str
    track_dir: Path
    audio_path: Path
    bhx_id: str


@dataclass
class SystemAdapter:
    name: str
    description: str
    loader_fn: Callable[[Path, str, Path], List[NormalizedSegment]]


# ==============================================================================
# System Adapter Registry
# ==============================================================================

SYSTEM_REGISTRY: Dict[str, SystemAdapter] = {}


def register_adapter(adapter: SystemAdapter):
    SYSTEM_REGISTRY[adapter.name] = adapter


# 1. flash (evaluation_flash_grounding_v1)
def load_flash_segments(track_dir: Path, bhx_id: str, eval_dir: Path) -> List[NormalizedSegment]:
    json_path = track_dir / "agents" / "evaluation_flash_grounding_v1" / "final.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Flash final.json not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[NormalizedSegment] = []
    for idx, seg in enumerate(data.get("segments", [])):
        st = float(seg.get("start_time", seg.get("start", 0.0)))
        et = float(seg.get("end_time", seg.get("end", 0.0)))
        lbl = str(seg.get("standard_label", seg.get("label", "inst"))).lower().strip()
        cap = str(seg.get("macro_caption", seg.get("caption", ""))).strip()
        if et <= st + 0.01:
            continue
        segments.append(NormalizedSegment(start=st, end=et, standard_label=lbl, caption=cap, index=idx))
    return segments


register_adapter(
    SystemAdapter(
        name="flash",
        description="Pipeline Flash Grounding v1 (gemini-2.5-flash + DSP telemetry)",
        loader_fn=load_flash_segments,
    )
)


# 2. lite (evaluation_flash_lite)
def load_lite_segments(track_dir: Path, bhx_id: str, eval_dir: Path) -> List[NormalizedSegment]:
    json_path = track_dir / "agents" / "evaluation_flash_lite" / "final.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Lite final.json not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[NormalizedSegment] = []
    for idx, seg in enumerate(data.get("segments", [])):
        st = float(seg.get("start_time", seg.get("start", 0.0)))
        et = float(seg.get("end_time", seg.get("end", 0.0)))
        lbl = str(seg.get("standard_label", seg.get("label", "inst"))).lower().strip()
        cap = str(seg.get("macro_caption", seg.get("caption", ""))).strip()
        if et <= st + 0.01:
            continue
        segments.append(NormalizedSegment(start=st, end=et, standard_label=lbl, caption=cap, index=idx))
    return segments


register_adapter(
    SystemAdapter(
        name="lite",
        description="Pipeline Flash-Lite (gemini-2.5-flash-lite + DSP telemetry)",
        loader_fn=load_lite_segments,
    )
)


# 3. max_v1 (evaluation_max_v1)
def load_max_v1_segments(track_dir: Path, bhx_id: str, eval_dir: Path) -> List[NormalizedSegment]:
    json_path = track_dir / "agents" / "evaluation_max_v1" / "final.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Max v1 final.json not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[NormalizedSegment] = []
    for idx, seg in enumerate(data.get("segments", [])):
        st = float(seg.get("start_time", seg.get("start", 0.0)))
        et = float(seg.get("end_time", seg.get("end", 0.0)))
        lbl = str(seg.get("standard_label", seg.get("label", "inst"))).lower().strip()
        cap = str(seg.get("macro_caption", seg.get("caption", ""))).strip()
        if et <= st + 0.01:
            continue
        segments.append(NormalizedSegment(start=st, end=et, standard_label=lbl, caption=cap, index=idx))
    return segments


register_adapter(
    SystemAdapter(
        name="max_v1",
        description="Pipeline Max v1 (gemini-2.5-pro/max agents + DSP telemetry)",
        loader_fn=load_max_v1_segments,
    )
)


# 4. direct_audio_lite (evaluation_direct_audio_flash_lite)
def load_direct_audio_lite_segments(track_dir: Path, bhx_id: str, eval_dir: Path) -> List[NormalizedSegment]:
    # Priority: postprocess.json resolves the minute notation drift documented in direct_audio_anomalies_report.md
    p_post = track_dir / "agents" / "evaluation_direct_audio_flash_lite" / "direct_audio_postprocess.json"
    p_final = track_dir / "agents" / "evaluation_direct_audio_flash_lite" / "final.json"

    target_path = p_post if p_post.is_file() else p_final
    if not target_path.is_file():
        raise FileNotFoundError(f"Direct Audio JSON not found in: {track_dir / 'agents' / 'evaluation_direct_audio_flash_lite'}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[NormalizedSegment] = []
    for idx, seg in enumerate(data.get("segments", [])):
        st = float(seg.get("start", seg.get("start_time", seg.get("start_time_seconds", 0.0))))
        et = float(seg.get("end", seg.get("end_time", seg.get("end_time_seconds", 0.0))))
        lbl = str(seg.get("standard_label", seg.get("label", "inst"))).lower().strip()
        cap = str(seg.get("caption", seg.get("macro_caption", ""))).strip()
        if et <= st + 0.01:
            continue
        segments.append(NormalizedSegment(start=st, end=et, standard_label=lbl, caption=cap, index=idx))
    return segments


register_adapter(
    SystemAdapter(
        name="direct_audio_lite",
        description="Baseline Gemini Flash-Lite with Raw Audio (zero DSP telemetry)",
        loader_fn=load_direct_audio_lite_segments,
    )
)


# 5. futga (eval/futga_inference/futga_inference_results_parsed/BHX_*.json)
def load_futga_segments(track_dir: Path, bhx_id: str, eval_dir: Path) -> List[NormalizedSegment]:
    json_path = eval_dir / "futga_inference" / "futga_inference_results_parsed" / f"{bhx_id}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"FUTGA parsed JSON not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[NormalizedSegment] = []
    for idx, seg in enumerate(data.get("segments", [])):
        st = float(seg.get("start_time_seconds", 0.0))
        et = float(seg.get("end_time_seconds", 0.0))
        # Filter degenerate/tail 0-duration sections (e.g. 100% tail outro with st == et)
        if et <= st + 0.01:
            continue
        lbl = str(seg.get("standard_label", seg.get("raw_label", "inst"))).lower().strip()
        # FUTGA segment caption is in 'description'; if empty, fallback to 'raw_label' prose header
        cap = str(seg.get("description", "")).strip()
        if not cap:
            cap = str(seg.get("raw_label", "")).strip()
        segments.append(NormalizedSegment(start=st, end=et, standard_label=lbl, caption=cap, index=idx))
    return segments


register_adapter(
    SystemAdapter(
        name="futga",
        description="FUTGA-7B Zero-Shot Music Captioning & Boundary Model",
        loader_fn=load_futga_segments,
    )
)


# ==============================================================================
# Model Wrapper
# ==============================================================================


class ClapEmbedder:
    """
    Wrapper for HuggingFace Transformers ClapModel and ClapProcessor.
    Enforces deterministic seed, CPU execution, and feature-fusion handling.
    """

    def __init__(self, model_checkpoint: str = "laion/clap-htsat-fused", seed: int = 42, logger: Optional[logging.Logger] = None):
        self.model_checkpoint = model_checkpoint
        self.seed = seed
        self.logger = logger or logging.getLogger("eval_clap_intra_track")
        self.device = torch.device("cpu")
        self._processor: Optional[ClapProcessor] = None
        self._model: Optional[ClapModel] = None

        self.set_seed(self.seed)

    def _ensure_loaded(self):
        if self._model is None:
            self.logger.info(f"Loading CLAP Processor from checkpoint: [bold green]{self.model_checkpoint}[/bold green]")
            self._processor = ClapProcessor.from_pretrained(self.model_checkpoint)

            self.logger.info(f"Loading CLAP Model from checkpoint: [bold green]{self.model_checkpoint}[/bold green] on [bold cyan]CPU[/bold cyan]")
            self._model = ClapModel.from_pretrained(self.model_checkpoint)
            self._model.to(self.device)
            self._model.eval()

    @property
    def processor(self) -> ClapProcessor:
        self._ensure_loaded()
        return self._processor  # type: ignore

    @property
    def model(self) -> ClapModel:
        self._ensure_loaded()
        return self._model  # type: ignore

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def embed_audio_clip(self, audio_clip: np.ndarray, sampling_rate: int = 48000) -> np.ndarray:
        """
        Embeds a single audio clip into a normalized 512-dim vector.
        """
        # Reset seed before each embedding call to maintain strict determinism across sampling crops
        self.set_seed(self.seed)

        inputs = self.processor(audio=audio_clip, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)
            tensor = out.pooler_output if hasattr(out, "pooler_output") else out
            tensor = tensor / tensor.norm(dim=-1, keepdim=True)

        return tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_captions(self, captions: List[str]) -> np.ndarray:
        """
        Embeds a list of N captions into normalized (N, 512) vectors.
        """
        self.set_seed(self.seed)

        inputs = self.processor(text=captions, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            tensor = out.pooler_output if hasattr(out, "pooler_output") else out
            tensor = tensor / tensor.norm(dim=-1, keepdim=True)

        return tensor.cpu().numpy().astype(np.float32)


# ==============================================================================
# Metric Calculations
# ==============================================================================


def compute_mrr(M_clap: np.ndarray) -> float:
    """
    MRR = (1/N) * sum_i (1 / rank_i)
    where rank_i is the 1-based rank of diagonal element M[i, i]
    among all N elements in row i, sorted in descending order.
    """
    N = M_clap.shape[0]
    if N == 0:
        return 0.0

    reciprocal_ranks: List[float] = []
    for i in range(N):
        row = M_clap[i, :]
        diag_val = row[i]
        # 1-based rank: number of elements strictly greater + 1
        rank = 1 + np.sum(row > diag_val)
        # In case of exact numerical ties with other items, add 0.5 tie penalty
        tie_count = np.sum((row == diag_val) & (np.arange(N) != i))
        if tie_count > 0:
            rank += 0.5 * tie_count
        reciprocal_ranks.append(1.0 / float(rank))

    return float(np.mean(reciprocal_ranks))


def compute_top1(M_clap: np.ndarray) -> float:
    """
    Top-1 accuracy: fraction of rows where diagonal element is the row argmax.
    """
    N = M_clap.shape[0]
    if N == 0:
        return 0.0
    correct = [1 if np.argmax(M_clap[i, :]) == i else 0 for i in range(N)]
    return float(np.mean(correct))


def compute_windowed_acc(M_clap: np.ndarray) -> float:
    """
    Windowed Accuracy: match counts as correct if peak similarity is at i-1, i, or i+1.
    Boundary conditions (i=0 and i=N-1) are handled cleanly.
    """
    N = M_clap.shape[0]
    if N == 0:
        return 0.0
    correct = [1 if abs(int(np.argmax(M_clap[i, :])) - i) <= 1 else 0 for i in range(N)]
    return float(np.mean(correct))


def compute_semantic_class_agreement(
    M_clap: np.ndarray,
    labels: List[str],
    threshold_mode: str,
    threshold_val: Optional[float] = None,
    calibrated_val: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Semantic Class Agreement & Chance Baseline:
    For each row i in track (N segments):
      p_i = (number of segments j != i in same track with labels[j] == labels[i]) / (N - 1)
      This represents the probability of matching the label by pure chance.

    If max_{j!=i} M[i, j] >= threshold:
      row i is significant:
        concordant += 1 if labels[j*] == labels[i]
        baseline_sum += p_i
        total_sig += 1

    Returns:
      (observed_agreement, baseline_agreement, total_sig)
      If total_sig == 0 or N <= 1, returns (None, None, 0).
    """
    N = M_clap.shape[0]
    if N <= 1:
        return None, None, 0

    # Precompute p_i for each row i
    p_chance: List[float] = []
    for i in range(N):
        match_count = sum(1 for j in range(N) if j != i and labels[j] == labels[i])
        p_chance.append(match_count / float(N - 1))

    concordant = 0
    baseline_sum = 0.0
    total_sig = 0
    effective_thresh_val = threshold_val if threshold_val is not None else calibrated_val

    for i in range(N):
        diag_val = M_clap[i, i]
        # Mask out diagonal element
        row = M_clap[i, :].copy()
        row[i] = -np.inf

        j_star = int(np.argmax(row))
        max_off_diag = row[j_star]

        # Determine significance threshold
        if threshold_mode == "fixed80":
            threshold = 0.8 * diag_val
        elif threshold_mode in ("calibrated", "threshold_val", "value"):
            if effective_thresh_val is None:
                raise ValueError("threshold_val must be provided for calibrated/value threshold mode.")
            threshold = effective_thresh_val
        else:
            raise ValueError(f"Unknown threshold mode: {threshold_mode}")

        if max_off_diag >= threshold:
            total_sig += 1
            if labels[j_star] == labels[i]:
                concordant += 1
            baseline_sum += p_chance[i]

    if total_sig == 0:
        return None, None, 0

    observed_agreement = float(concordant / total_sig)
    baseline_agreement = float(baseline_sum / total_sig)

    return observed_agreement, baseline_agreement, total_sig


# ==============================================================================
# Track Collection & Caching
# ==============================================================================


def collect_tracks(eval_processed_dir: Path, target_tracks: Optional[List[str]] = None) -> List[TrackItem]:
    """
    Finds all valid track directories in eval_processed/ and resolves the .wav file.
    """
    if not eval_processed_dir.is_dir():
        raise FileNotFoundError(f"eval_processed directory not found at: {eval_processed_dir}")

    track_dirs = sorted([d for d in eval_processed_dir.iterdir() if d.is_dir()])
    items: List[TrackItem] = []

    for td in track_dirs:
        if target_tracks and td.name not in target_tracks:
            continue

        audio_dir = td / "audio"
        wav_files = sorted(audio_dir.glob("*.wav")) if audio_dir.is_dir() else []
        if not wav_files:
            wav_files = sorted(td.glob("*.wav"))

        if not wav_files:
            continue

        audio_path = wav_files[0]
        bhx_id = audio_path.stem
        items.append(TrackItem(track_name=td.name, track_dir=td, audio_path=audio_path, bhx_id=bhx_id))

    return items


def compute_cache_hash(checkpoint: str, seed: int, segments: List[NormalizedSegment]) -> str:
    """
    Generates a deterministic SHA-256 hash representing model, seed, and segment definitions.
    """
    hasher = hashlib.sha256()
    hasher.update(checkpoint.encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    for s in segments:
        hasher.update(f"{s.start:.2f}_{s.end:.2f}_{s.standard_label}_{s.caption}".encode("utf-8"))
    return hasher.hexdigest()


# ==============================================================================
# Pipeline Execution for a Single System
# ==============================================================================


def process_system(
    system_key: str,
    adapter: SystemAdapter,
    tracks: List[TrackItem],
    embedder: ClapEmbedder,
    eval_dir: Path,
    system_output_dir: Path,
    force_recompute: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the CLAP intra-track evaluation for a single system over all tracks.
    Returns (per_track_df, summary_df).
    """
    logger.info(f"\n[bold magenta]==================================================================[/bold magenta]")
    logger.info(f" Starting Intra-Track CLAP Evaluation: [bold yellow]{system_key}[/bold yellow]")
    logger.info(f" Description: {adapter.description}")
    logger.info(f" Total Tracks: {len(tracks)}")
    logger.info(f" Output Directory: {system_output_dir}")
    logger.info(f"[bold magenta]==================================================================[/bold magenta]\n")

    embeddings_dir = system_output_dir / "embeddings"
    matrices_dir = system_output_dir / "matrices"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    track_results: List[Dict[str, Any]] = []
    all_off_diagonals: List[float] = []

    # Store loaded/computed data per track for second pass (calibrated threshold)
    track_cache: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------------------------
    # Pass 1: Compute / Load Embeddings & Matrices, Compute Primary Metrics
    # --------------------------------------------------------------------------
    cache_hits = 0
    cache_misses = 0

    for idx, item in enumerate(tracks, 1):
        try:
            segments = adapter.loader_fn(item.track_dir, item.bhx_id, eval_dir)
        except Exception as e:
            logger.error(f"[{idx}/{len(tracks)}] Error loading segments for track [bold red]{item.track_name}[/bold red]: {e}")
            continue

        if len(segments) == 0:
            logger.warning(f"[{idx}/{len(tracks)}] Track {item.track_name} has 0 valid segments. Skipping.")
            continue

        cache_hash = compute_cache_hash(embedder.model_checkpoint, embedder.seed, segments)
        emb_file = embeddings_dir / f"{item.bhx_id}.npz"

        # Per-song directory for matrices
        track_matrix_dir = matrices_dir / item.track_name
        track_matrix_dir.mkdir(parents=True, exist_ok=True)

        mat_file = track_matrix_dir / f"{item.bhx_id}.npy"
        mat_csv_file = track_matrix_dir / f"{item.bhx_id}.csv"
        meta_file = track_matrix_dir / f"{item.bhx_id}_meta.json"

        # Seamlessly migrate legacy flat matrix files if present
        legacy_mat_file = matrices_dir / f"{item.bhx_id}.npy"
        legacy_meta_file = matrices_dir / f"{item.bhx_id}_meta.json"
        legacy_csv_file = matrices_dir / f"{item.bhx_id}.csv"
        if not mat_file.is_file() and legacy_mat_file.is_file():
            try:
                shutil.move(str(legacy_mat_file), str(mat_file))
                if legacy_meta_file.is_file():
                    shutil.move(str(legacy_meta_file), str(meta_file))
                if legacy_csv_file.is_file():
                    shutil.move(str(legacy_csv_file), str(mat_csv_file))
            except Exception:
                pass

        is_cached = False
        if not force_recompute and emb_file.is_file() and mat_file.is_file() and meta_file.is_file():
            try:
                npz_data = np.load(emb_file, allow_pickle=True)
                saved_hash = str(npz_data["cache_hash"])
                if saved_hash == cache_hash:
                    audio_embeddings = npz_data["audio_embeddings"]
                    text_embeddings = npz_data["text_embeddings"]
                    M_clap = np.load(mat_file)
                    is_cached = True
                    cache_hits += 1
                    if not mat_csv_file.is_file():
                        seg_headers = [f"seg_{s.index}_{s.standard_label}" for s in segments]
                        df_mat = pd.DataFrame(M_clap, index=seg_headers, columns=seg_headers)
                        df_mat.to_csv(mat_csv_file, float_format="%.4f")
                    logger.info(f"[{idx}/{len(tracks)}] [bold green][CACHE HIT][/bold green] {item.track_name} ({item.bhx_id}) - {len(segments)} segments")
            except Exception as e:
                logger.warning(f"Failed to read cache for {item.track_name}: {e}. Recomputing.")

        if not is_cached:
            cache_misses += 1
            logger.info(f"[{idx}/{len(tracks)}] [bold yellow][CACHE MISS][/bold yellow] Computing CLAP embeddings for: {item.track_name} ({item.bhx_id}) - {len(segments)} segments")

            # Load full track audio once at 48kHz mono
            audio_full, sr = librosa.load(str(item.audio_path), sr=48000, mono=True)
            total_samples = len(audio_full)

            # Compute audio embeddings
            audio_embeds_list: List[np.ndarray] = []
            captions_list: List[str] = []
            labels_list: List[str] = []

            for seg in segments:
                start_samp = max(0, int(round(seg.start * sr)))
                end_samp = min(total_samples, int(round(seg.end * sr)))
                if end_samp <= start_samp:
                    # Fallback for boundary anomaly: ensure at least 0.5s clip
                    end_samp = min(total_samples, start_samp + int(sr * 0.5))

                clip = audio_full[start_samp:end_samp]
                audio_emb = embedder.embed_audio_clip(clip, sampling_rate=sr)
                audio_embeds_list.append(audio_emb)
                captions_list.append(seg.caption)
                labels_list.append(seg.standard_label)

            audio_embeddings = np.stack(audio_embeds_list, axis=0)  # (N, 512)
            text_embeddings = embedder.embed_captions(captions_list)  # (N, 512)

            # M_clap[i, j] = cos(audio_i, text_j)
            M_clap = np.matmul(audio_embeddings, text_embeddings.T)  # (N, N)

            # Save to disk cache
            np.savez_compressed(
                emb_file,
                audio_embeddings=audio_embeddings,
                text_embeddings=text_embeddings,
                labels=np.array(labels_list),
                starts=np.array([s.start for s in segments]),
                ends=np.array([s.end for s in segments]),
                captions=np.array(captions_list),
                cache_hash=cache_hash,
            )
            np.save(mat_file, M_clap)

            # Also export labeled CSV for easy heatmap plotting
            seg_headers = [f"seg_{s.index}_{s.standard_label}" for s in segments]
            df_mat = pd.DataFrame(M_clap, index=seg_headers, columns=seg_headers)
            df_mat.to_csv(mat_csv_file, float_format="%.4f")

            meta_content = {
                "track_name": item.track_name,
                "bhx_id": item.bhx_id,
                "n_segments": len(segments),
                "model_checkpoint": embedder.model_checkpoint,
                "seed": embedder.seed,
                "cache_hash": cache_hash,
                "segments": [
                    {
                        "index": s.index,
                        "start": s.start,
                        "end": s.end,
                        "standard_label": s.standard_label,
                        "caption": s.caption,
                    }
                    for s in segments
                ],
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_content, f, indent=2)

        # Collect off-diagonal elements for calibrated threshold calculation
        N = M_clap.shape[0]
        labels = [s.standard_label for s in segments]
        for i in range(N):
            for j in range(N):
                if i != j:
                    all_off_diagonals.append(float(M_clap[i, j]))

        # Calculate track-level metrics (fixed80 threshold)
        mrr = compute_mrr(M_clap)
        top1 = compute_top1(M_clap)
        win_acc = compute_windowed_acc(M_clap)
        sem_fixed80, base_fixed80, sig_fixed80 = compute_semantic_class_agreement(M_clap, labels, threshold_mode="fixed80")

        track_cache[item.bhx_id] = {
            "track_name": item.track_name,
            "bhx_id": item.bhx_id,
            "n_segments": N,
            "M_clap": M_clap,
            "labels": labels,
            "mrr": mrr,
            "top1": top1,
            "windowed_acc": win_acc,
            "semantic_agreement_fixed80": sem_fixed80,
            "baseline_agreement_fixed80": base_fixed80,
            "sig_activations_fixed80": sig_fixed80,
        }

    # --------------------------------------------------------------------------
    # Pass 2: Calibrated Thresholds on Off-Diagonal Elements (Mean, P90, P95)
    # --------------------------------------------------------------------------
    if all_off_diagonals:
        arr_off = np.array(all_off_diagonals, dtype=np.float32)
        calibrated_threshold = float(np.mean(arr_off))
        p90_threshold = float(np.percentile(arr_off, 90))
        p95_threshold = float(np.percentile(arr_off, 95))
    else:
        calibrated_threshold = 0.0
        p90_threshold = 0.0
        p95_threshold = 0.0

    total_corpus_rows = sum(c["n_segments"] for c in track_cache.values())
    total_sig_fixed80 = sum(c["sig_activations_fixed80"] for c in track_cache.values())
    total_sig_calib = 0
    total_sig_p90 = 0
    total_sig_p95 = 0

    # Complete per-track rows with calibrated agreement across all threshold variants
    for item in tracks:
        if item.bhx_id not in track_cache:
            continue
        c = track_cache[item.bhx_id]

        sem_calib, base_calib, sig_calib = compute_semantic_class_agreement(
            c["M_clap"],
            c["labels"],
            threshold_mode="calibrated",
            threshold_val=calibrated_threshold,
        )
        sem_p90, base_p90, sig_p90 = compute_semantic_class_agreement(
            c["M_clap"],
            c["labels"],
            threshold_mode="calibrated",
            threshold_val=p90_threshold,
        )
        sem_p95, base_p95, sig_p95 = compute_semantic_class_agreement(
            c["M_clap"],
            c["labels"],
            threshold_mode="calibrated",
            threshold_val=p95_threshold,
        )

        total_sig_calib += sig_calib
        total_sig_p90 += sig_p90
        total_sig_p95 += sig_p95

        track_results.append(
            {
                "track_name": c["track_name"],
                "bhx_id": c["bhx_id"],
                "n_segments": c["n_segments"],
                "mrr": c["mrr"],
                "top1": c["top1"],
                "windowed_acc": c["windowed_acc"],
                "semantic_agreement_fixed80": c["semantic_agreement_fixed80"],
                "baseline_agreement_fixed80": c["baseline_agreement_fixed80"],
                "sig_activations_fixed80": c["sig_activations_fixed80"],
                "semantic_agreement_calibrated": sem_calib,
                "baseline_agreement_calibrated": base_calib,
                "sig_activations_calibrated": sig_calib,
                "semantic_agreement_p90": sem_p90,
                "baseline_agreement_p90": base_p90,
                "sig_activations_p90": sig_p90,
                "semantic_agreement_p95": sem_p95,
                "baseline_agreement_p95": base_p95,
                "sig_activations_p95": sig_p95,
            }
        )

    pct_fixed80 = (100.0 * total_sig_fixed80 / total_corpus_rows) if total_corpus_rows > 0 else 0.0
    pct_calib = (100.0 * total_sig_calib / total_corpus_rows) if total_corpus_rows > 0 else 0.0
    pct_p90 = (100.0 * total_sig_p90 / total_corpus_rows) if total_corpus_rows > 0 else 0.0
    pct_p95 = (100.0 * total_sig_p95 / total_corpus_rows) if total_corpus_rows > 0 else 0.0

    logger.info(f"\n[bold cyan]Threshold Calibration Summary for '{system_key}'[/bold cyan] ({len(all_off_diagonals)} off-diagonal elements across {total_corpus_rows} total rows):")
    logger.info(f"  • Fixed 80% Diagonal Threshold:             {total_sig_fixed80}/{total_corpus_rows} activations ({pct_fixed80:.1f}%)")
    logger.info(f"  • Off-Diagonal Mean Threshold ({calibrated_threshold:.4f}):          {total_sig_calib}/{total_corpus_rows} activations ({pct_calib:.1f}%)")
    logger.info(f"  • Off-Diagonal 90th Percentile ({p90_threshold:.4f}):   {total_sig_p90}/{total_corpus_rows} activations ({pct_p90:.1f}%)")
    logger.info(f"  • Off-Diagonal 95th Percentile ({p95_threshold:.4f}):   {total_sig_p95}/{total_corpus_rows} activations ({pct_p95:.1f}%)")
    logger.info(f"Cache Performance: [bold green]{cache_hits} hits[/bold green], [bold yellow]{cache_misses} misses[/bold yellow].")

    # --------------------------------------------------------------------------
    # Build Per-Track & Summary DataFrames
    # --------------------------------------------------------------------------
    df_eval = pd.DataFrame(track_results)

    # Export per-track CSV
    eval_csv_path = system_output_dir / f"{system_key}_eval.csv"
    df_eval.to_csv(eval_csv_path, index=False, float_format="%.4f")
    logger.info(f"Saved per-track evaluation results to: [bold green]{eval_csv_path}[/bold green]")

    # Macro-averaging summary across tracks
    summary_rows = []
    metrics_to_summarize = [
        ("mrr", "MRR (Mean Reciprocal Rank)"),
        ("top1", "Top-1 Retrieval Accuracy"),
        ("windowed_acc", "Windowed Accuracy (±1 offset)"),
        ("semantic_agreement_fixed80", "Semantic Class Agreement (Fixed 80% Threshold)"),
        ("baseline_agreement_fixed80", "Baseline Class Agreement (Fixed 80% Threshold)"),
        ("semantic_agreement_calibrated", f"Semantic Class Agreement (Calibrated Mean: {calibrated_threshold:.4f})"),
        ("baseline_agreement_calibrated", f"Baseline Class Agreement (Calibrated Mean: {calibrated_threshold:.4f})"),
        ("semantic_agreement_p90", f"Semantic Class Agreement (Calibrated P90: {p90_threshold:.4f})"),
        ("baseline_agreement_p90", f"Baseline Class Agreement (Calibrated P90: {p90_threshold:.4f})"),
        ("semantic_agreement_p95", f"Semantic Class Agreement (Calibrated P95: {p95_threshold:.4f})"),
        ("baseline_agreement_p95", f"Baseline Class Agreement (Calibrated P95: {p95_threshold:.4f})"),
    ]

    for col, display_name in metrics_to_summarize:
        valid_series = df_eval[col].dropna()
        n_valid = len(valid_series)
        mean_val = float(valid_series.mean()) if n_valid > 0 else np.nan
        std_val = float(valid_series.std()) if n_valid > 1 else (0.0 if n_valid == 1 else np.nan)
        med_val = float(valid_series.median()) if n_valid > 0 else np.nan

        summary_rows.append(
            {
                "metric_key": col,
                "metric_name": display_name,
                "mean": mean_val,
                "std": std_val,
                "median": med_val,
                "valid_tracks": n_valid,
                "total_tracks": len(df_eval),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = system_output_dir / f"{system_key}_eval_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False, float_format="%.4f")
    logger.info(f"Saved aggregated summary to: [bold green]{summary_csv_path}[/bold green]")

    # Print summary table in Rich
    table = Table(title=f"Evaluation Summary: {system_key.upper()} (Macro-averaged over {len(df_eval)} tracks)")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean ± Std", style="bold green", justify="right")
    table.add_column("Median", style="magenta", justify="right")
    table.add_column("Valid Tracks", style="yellow", justify="right")

    for _, r in df_summary.iterrows():
        mean_std = f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r['mean']) else "N/A"
        med = f"{r['median']:.4f}" if pd.notna(r['median']) else "N/A"
        valid = f"{int(r['valid_tracks'])} / {int(r['total_tracks'])}"
        table.add_row(r["metric_name"], mean_std, med, valid)

    console.print(table)

    return df_eval, df_summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ2: CLAP Intra-Track Caption-Audio Semantic Grounding Evaluation.",
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
        help="Path to eval/ directory (default: parent dir of this script or current eval dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for output files (default: eval/rq2_intra_track/output).",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="laion/clap-htsat-fused",
        help="CLAP checkpoint identifier (HuggingFace).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for deterministic fusion sampling.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        help="Optional list of track names to evaluate (default: all 50 tracks in eval_processed).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of audio/text embeddings and M_CLAP matrices, bypassing cache.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve eval directory
    script_dir = Path(__file__).resolve().parent
    if args.eval_dir:
        eval_dir = Path(args.eval_dir).resolve()
    elif (script_dir / "eval_processed").is_dir():
        eval_dir = script_dir
    elif (script_dir.parent / "eval" / "eval_processed").is_dir():
        eval_dir = script_dir.parent / "eval"
    else:
        eval_dir = script_dir

    eval_processed_dir = eval_dir / "eval_processed"
    if not eval_processed_dir.is_dir():
        console.print(f"[bold red]Error: eval_processed directory not found at: {eval_processed_dir}[/bold red]")
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir).resolve()
    else:
        base_output_dir = eval_dir / "rq2_intra_track" / "output"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve systems to evaluate
    selected_systems = args.systems
    if len(selected_systems) == 1 and selected_systems[0].lower() == "all":
        selected_systems = list(SYSTEM_REGISTRY.keys())

    invalid_systems = [s for s in selected_systems if s not in SYSTEM_REGISTRY]
    if invalid_systems:
        console.print(f"[bold red]Error: Unknown systems specified: {invalid_systems}[/bold red]")
        console.print(f"Available systems: {list(SYSTEM_REGISTRY.keys())}")
        sys.exit(1)

    # Setup primary logger
    logger = setup_logger()
    logger.info("Initializing RQ2 Intra-Track CLAP Semantic Grounding Pipeline...")
    logger.info(f"CLAP Checkpoint: [bold green]{args.model_checkpoint}[/bold green]")
    logger.info(f"Random Seed: [bold cyan]{args.seed}[/bold cyan]")
    logger.info(f"Target Systems: [bold yellow]{selected_systems}[/bold yellow]")

    # Collect 50 tracks
    tracks = collect_tracks(eval_processed_dir, target_tracks=args.tracks)
    logger.info(f"Successfully collected [bold cyan]{len(tracks)}[/bold cyan] audio tracks from {eval_processed_dir}")
    if len(tracks) == 0:
        logger.error("No tracks found to evaluate. Exiting.")
        sys.exit(1)

    # Initialize CLAP Model & Embedder once
    embedder = ClapEmbedder(model_checkpoint=args.model_checkpoint, seed=args.seed, logger=logger)

    # Process each selected system independently
    for sys_key in selected_systems:
        adapter = SYSTEM_REGISTRY[sys_key]
        system_output_dir = base_output_dir / sys_key

        # System-specific logger with file handler
        sys_logger = setup_logger(output_dir=system_output_dir, system_name=sys_key)

        process_system(
            system_key=sys_key,
            adapter=adapter,
            tracks=tracks,
            embedder=embedder,
            eval_dir=eval_dir,
            system_output_dir=system_output_dir,
            force_recompute=args.force_recompute,
            logger=sys_logger,
        )

    logger.info(f"\n[bold green]All evaluations completed successfully![/bold green]")
    logger.info(f"Output files stored under: [bold cyan]{base_output_dir}[/bold cyan]")


if __name__ == "__main__":
    main()
