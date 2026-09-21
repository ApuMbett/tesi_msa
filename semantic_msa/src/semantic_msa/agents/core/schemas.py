from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- PROFILER SCHEMAS ---

class MacroProfileDTO(BaseModel):
    inferred_genres: List[str] = Field(default_factory=list)
    energy_tier: str
    rationale: str


class VocalProfileDTO(BaseModel):
    presence_level: str
    lyrics_validity: str
    rationale: str


class DynamicTrajectoryDTO(BaseModel):
    energy_flow: str
    rationale: str


class ProfilerResponseDTO(BaseModel):
    macro_profile: MacroProfileDTO
    vocal_profile: VocalProfileDTO
    dynamic_trajectory: DynamicTrajectoryDTO


# --- RESEARCHER SCHEMAS ---

class SourceIdentityDTO(BaseModel):
    identified_track_name: str
    identified_original_artist: str
    original_cultural_genre: str


class CulturalImpactDTO(BaseModel):
    scale_of_impact: str
    zeitgeist_summary: str


class InferredRealityDTO(BaseModel):
    track_classification: str
    dominant_acoustic_style: str
    target_audience: str


class CulturalPriorDTO(BaseModel):
    source_identity: SourceIdentityDTO
    cultural_impact_and_history: CulturalImpactDTO
    inferred_reality: InferredRealityDTO
    researcher_notes: str


class ResearcherResponseDTO(BaseModel):
    cultural_prior: CulturalPriorDTO


# --- CAPTIONER SCHEMAS ---

class TokenCaptionDTO(BaseModel):
    token_index: int
    caption: str
    rationale: str = ""


class CaptionerResponseDTO(BaseModel):
    token_captions: List[TokenCaptionDTO] = Field(default_factory=list)


# --- SEGMENTER SCHEMAS ---

class SegmentDTO(BaseModel):
    segment_index: int
    segment_name: str
    token_range: List[int]
    macro_caption: str = ""
    rationale: str = ""


class SegmenterResponseDTO(BaseModel):
    segments: List[SegmentDTO] = Field(default_factory=list)


# --- FINAL JSON SYNTHESIS SCHEMAS ---

class FinalTrackIdentityDTO(BaseModel):
    name: str = ""
    artist: str = ""
    key: str = ""
    bpm: float = 0.0
    duration: str = ""
    cultural_context: Any = Field(default_factory=dict)
    acoustic_profile: Any = Field(default_factory=dict)


class FinalTokenDTO(BaseModel):
    index: int
    captioning_data: Any = Field(default_factory=dict)
    start_time: Any = 0.0
    end_time: Any = 0.0
    start_cue: Optional[Dict[str, Any]] = None
    end_cue: Optional[Dict[str, Any]] = None
    lyrics: Optional[Any] = None
    dsp_features: Dict[str, Any] = Field(default_factory=dict)


class FinalSegmentDTO(BaseModel):
    index: int
    name: str
    start_time: Any = 0.0
    end_time: Any = 0.0
    macro_caption: str = ""
    rationale: str = ""
    tokens: List[FinalTokenDTO] = Field(default_factory=list)


class FinalTrackDTO(BaseModel):
    track_identity: FinalTrackIdentityDTO
    token_size_beats: int = 16
    lyrics_profile: Dict[str, Any] = Field(default_factory=dict)
    caption_length: int = 12
    segmenter_granularity: str = "MEDIUM"
    segments: List[FinalSegmentDTO] = Field(default_factory=list)
