# Semantic Music Structure Analysis (Semantic MSA)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent, multi-agent audio intelligence framework for **Semantic Music Structure Analysis (Semantic MSA)**. 

Traditional Music Information Retrieval (MIR) relies on purely acoustic boundary detection or monolithic models that lack semantic grounding. **Semantic MSA (`tesi_msa`)** introduces an explainable, bottom-up approach: raw audio is separated, tracked, and discretized into musical tokens (16-beat bars), enriched with physical DSP telemetry, and interpreted by a coordinated team of specialized Large Language Model (LLM) agents to perform macro-structural segmentation, functional labeling, and cross-modal semantic captioning.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
  - [1. Physical Audio Telemetry Engine](#1-physical-audio-telemetry-engine)
  - [2. Multi-Agent Orchestration](#2-multi-agent-orchestration)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [Analyzing a Track (Feature Extraction)](#analyzing-a-track-feature-extraction)
  - [Running the Multi-Agent Pipeline](#running-the-multi-agent-pipeline)
  - [Batch Processing](#batch-processing)
- [Evaluation Suite](#-evaluation-suite)
  - [RQ1: Structural Segmentation](#rq1-structural-segmentation)
  - [RQ2: Cross-Modal Semantic Retrieval (CLAP)](#rq2-cross-modal-semantic-retrieval-clap)
- [Configuration & Providers](#-configuration--providers)
- [Citation & Thesis](#-citation--thesis)

---

## 🌟 Key Features

- **Bottom-Up Token Discretization**: Segments tracks into metric 16-beat musical blocks aligned to a strict 4/4 meter.
- **Physical Telemetry Extraction**:
  - Downbeat & Beat Tracking via **BeatNet**
  - Structural transition / Cue-point detection via **Cue-DETR**
  - Source separation with **Demucs v4** (vocals vs. instrumental)
  - Vocal Activity Detection (VAD) with adaptive **K-Means energy clustering**
  - Word-level lyrical transcription and time-alignment using **WhisperX**
- **Domain-Specialized Multi-Agent Pipeline**:
  - **Cultural Researcher**: Web-grounded contextual search for artist identity, cultural origins, and musicology.
  - **Acoustic Profiler**: Analytical interpretation of instrumentation, key, tempo, and energy dynamics.
  - **Token Captioner**: Generates high-density, timestamp-bounded semantic captions for every 16-beat slice.
  - **Semantic Segmenter**: Synthesizes token streams into coherent macro-structural boundaries with explanations using the **SongFormer 7-class taxonomy** (`intro`, `verse`, `chorus`, `bridge`, `inst`, `outro`, `silence`).
- **Baselines & Evaluation**:
  - Direct Audio multimodal LLM baseline (Gemini audio-native).
  - Comparative benchmarks against **SongFormer** (state-of-the-art MSA) and **FUTGA-7B** (zero-shot audio captioning).
  - Cross-modal evaluation via **CLAP (Contrastive Language-Audio Pretraining)** for text-to-audio and audio-to-text retrieval.

---

## 🏗 System Architecture

```
Raw Audio (.wav / .mp3)
   │
   ├─► Demucs v4 (Source Separation: Vocals / Instrumental)
   ├─► BeatNet (Downbeat Grid & BPM Tracking)
   ├─► Cue-DETR (Structural Boundary Priors)
   ├─► WhisperX (Lyrical Transcription & Word Alignment)
   └─► Vocal Energy VAD (K-Means Clustering)
                 │
                 ▼
       ┌───────────────────┐
       │ 16-Beat Telemetry │
       │   Token Stream    │
       └─────────┬─────────┘
                 │
     ┌───────────┴──────────────────────────────┐
     │ Multi-Agent LLM Orchestration Engine    │
     │                                          │
     │  ┌────────────────┐  ┌────────────────┐  │
     │  │   Researcher   │  │    Profiler    │  │
     │  │ (Web Grounded) │  │  (DSP Bounds)  │  │
     │  └───────┬────────┘  └───────┬────────┘  │
     │          └───────────┬───────┘           │
     │                      ▼                   │
     │             ┌────────────────┐           │
     │             │ Token Captioner│           │
     │             └────────┬───────┘           │
     │                      ▼                   │
     │             ┌────────────────┐           │
     │             │   Segmenter    │           │
     │             └────────┬───────┘           │
     └──────────────────────┼───────────────────┘
                            ▼
     ┌──────────────────────────────────────────┐
     │               final.json                 │
     │  • Macro Structural Boundaries           │
     │  • Explainable Segment Rationales        │
     │  • Dense 16-Beat Cross-Modal Captions    │
     └──────────────────────────────────────────┘
```

### 1. Physical Audio Telemetry Engine
Before any LLM invocation, audio is converted into structured telemetry:
- **`beatnet_engine/`**: Tracks exact bar-downbeats to establish the 16-beat token boundaries.
- **`cue_detr/`**: Predicts transition points and mixes them into segment candidates.
- **`whisper_engine/`**: Transcribes lyrics with exact sub-second timestamps.
- **`telemetry.py`**: Serializes audio signals into token representations combining RMS vocal density, spectral contrast, percussion energy, and lyrical content.

### 2. Multi-Agent Orchestration
The agents communicate hierarchically:
1. **Researcher Agent**: Searches the web using Google Search grounding for release context, label, era, and regional style.
2. **Profiler Agent**: Validates key, tempo, and instrumental textures against the physical telemetry.
3. **Captioner Agent**: Operates across the token stream to produce concise, dense semantic descriptions (≤20 words per token).
4. **Segmenter Agent**: Fuses cultural metadata, track profile, and the token captions to produce the final macro-structural segmentation with full explainability.

---

## 📁 Repository Structure

```
.
├── pyproject.toml              # UV workspace root configuration
├── semantic_msa/               # Core framework package
│   ├── pyproject.toml
│   └── src/semantic_msa/
│       ├── adapters/           # Demucs, Cue-DETR, WhisperX adapters
│       ├── agents/             # Multi-agent implementations & pipeline runner
│       │   ├── core/           # LLM factory, schemas, rate limiter, synthesizer
│       │   ├── direct_audio.py # Multimodal raw-audio baseline agent
│       │   └── pipeline.py     # AgentsPipeline orchestrator
│       ├── config/             # YAML configurations
│       └── main.py             # Feature extraction CLI
├── eval/                       # Benchmark & Evaluation Suite
│   ├── agents_inference/       # Batch experiment runner and YAML configs
│   ├── dataset_loaders/        # SongFormBench HarmonixSet loader & split tagger
│   ├── futga_inference/        # FUTGA baseline conversion and Kaggle inference
│   ├── eval_clap_intra_track.py# Intra-track CLAP cross-modal evaluation
│   ├── eval_clap_many_to_many.py# Many-to-many cross-track CLAP pool retrieval
│   ├── generate_postprocess_json.py # Baseline post-processing & repair
│   ├── plot_rq2_*.py           # Publication-ready CMC & confusion matrix plots
│   ├── to_msa_format.py        # Exporter to standard SongFormer MSA format
│   └── eval_artifacts.tar.gz   # Compressed reproduction bundle (7.4 MB)
```

---

## 🚀 Installation & Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) as its package and workspace manager.

### 1. Clone the repository
```bash
git clone https://github.com/ApuMbett/tesi_msa.git
cd tesi_msa
```

### 2. Install Dependencies
```bash
# Create and synchronize workspace virtual environment
uv sync
```

> **Important Note on WhisperX**:
> Upstream WhisperX requirements enforce conflicting PyTorch index URLs (`--extra-index-url cu128`). To install WhisperX safely in your environment:
> ```bash
> uv pip install git+https://github.com/m-bain/whisperx.git@2cfd7b7c5c7bba144954364db747319b50e8232b
> ```
> Run commands with `uv run --no-sync` when executing to prevent package resolution conflicts.

### 3. Environment Variables
Create a `.env` file in `semantic_msa/`:
```bash
cp semantic_msa/.env.example semantic_msa/.env   # or create manually
```
Add your provider API keys:
```env
GEMINI_API_KEY="your-google-gemini-key"
# Optional secondary providers:
OPENROUTER_API_KEY="your-openrouter-key"
GROQ_API_KEY="your-groq-key"
HF_TOKEN="your-huggingface-token"
```

---

## 💻 Usage Guide

### Analyzing a Track (Feature Extraction)
Extract physical telemetry, stems, beat grids, and transcribe lyrics into `data/processed/<track_name>/LLM.json`:

```bash
uv run --no-sync analyze-track --audio path/to/song.wav --output-dir data/processed/
```

### Running the Multi-Agent Pipeline
Run the multi-agent pipeline on an extracted track to produce `final.json`:

```bash
uv run --no-sync run-agents \
  --track data/processed/<track_name>/LLM.json \
  --config semantic_msa/src/semantic_msa/config/pipeline.yaml \
  --run-name evaluation_flash_v1
```

### Batch Processing
Run the agent pipeline over an entire directory of tracks:

```bash
python eval/agents_inference/run_agents_pipeline.py \
  --processed-dir eval/eval_processed \
  --config eval/agents_inference/evaluation_flash_lite.yaml \
  --run-name flash_lite_run
```

---

## 📊 Evaluation Suite

The evaluation suite benchmarks Semantic MSA against human ground truth (**HarmonixSet** via **SongFormBench**) and baseline models across two core Research Questions:

### RQ1: Structural Segmentation
Evaluates boundary precision ($F_{0.58}$, $F_{3.0}$) and functional label classification accuracy against HarmonixSet annotations:
```bash
# Convert final.json to standard MSA TXT format
python eval/to_msa_format.py --input-dir eval/eval_processed/ --run-name flash_lite_run

# Compute segment distribution and count statistics
python eval/calculate_average_segments.py
```

### RQ2: Cross-Modal Semantic Retrieval (CLAP)
Evaluates whether generated token captions capture distinctive acoustic features using CLAP embeddings:
```bash
# Intra-track retrieval (discriminating sections within the same song)
python eval/eval_clap_intra_track.py

# Many-to-many pool retrieval across the full 50-track corpus (500+ segments)
python eval/eval_clap_many_to_many.py

# Generate publication-grade CMC curves and confusion matrices
python eval/plot_rq2_many_to_many.py
```

---

## ⚙️ Configuration & Providers

The system supports multiple LLM inference providers via `LLMFactory`:
- **Native Gemini (`google-genai`)**: Recommended for speed, native Google Search grounding, and audio multimodal processing.
- **OpenRouter (`openrouter/<model>`)**: Third-party models.
- **Groq (`groq/<model>`)**: Low-latency open-source models (Llama 3, Qwen).
- **Hugging Face Router (`hf/<model>`)**: Serverless inference.

Client-side rate pacing is automatically handled via `RateLimiter` and exponential backoff retry mechanisms to respect provider RPM limits.
