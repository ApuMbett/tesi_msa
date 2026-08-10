from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

HIGH_MEDIAN_SCORE = 0.8
HIGH_MIN_SCORE = 0.6
LOW_MEDIAN_SCORE = 0.5
LOW_MIN_SCORE = 0.2

class RawCuePoint(BaseModel):
    """
    Data object representing a temporal boundary for DJ mixing.
    Based on the CUE-DETR architecture (Argüello et al., 2024), which 
    treats cue point estimation as a computer vision object detection task.
    This is the initial output from the CUE-DETR model before any structural alignment.
    """
    time_ms: int = Field(alias="time_ms")  # The raw temporal position in milliseconds
    confidence_score: float = Field(alias="confidence_score")  # The confidence score of the prediction

    def json(self):
        """
        Serializes the raw cue point 
        """
        return {
          "time_ms": self.time_ms,
          "confidence_score": self.confidence_score,
        }
    
    # TODO refactor classes as dataclasses

    def LLM_representation(self) -> Dict[str, Any]:
        return {
            "time_ms": self.time_ms,
            "confidence_score": round(self.confidence_score, 2)
        }

    # @classmethod
    # def from_dict(cls, data: dict):
    #     """
    #     Deserializes a RawCuePoint from a dictionary, allowing for easy reconstruction from JSON data.
    #     """
    #     return cls(time_ms=data.get("time_ms", 0), score=data.get("confidence_score", 0.0))

class SnappedCuePoint(BaseModel):
    """
    Represents a cue point after snapping to the nearest downbeat in the beat grid.
    This class encapsulates the structural alignment process and the evaluation of the cue point's reliability based on psychoacoustic thresholds (taking bpm into account).
    """
    raw_cue_point: RawCuePoint
    snapped_time_ms: int
    displacement_error_ms: int
    quantization_error_ratio: float
    adjusted_confidence_score: float

    def json(self):
        """
        Serializes the snapped cue point, including the original raw prediction, the snapped time, the error, and the adjusted confidence score.
        """
        return {
          "raw_cue_point": self.raw_cue_point.json(),
          "snapped_time_ms": self.snapped_time_ms,
          "displacement_error_ms": self.displacement_error_ms,
          "quantization_error_ratio": self.quantization_error_ratio,
          "adjusted_confidence_score": self.adjusted_confidence_score,
        }

    def LLM_representation(self) -> Dict[str, Any]:
        return {
            "raw_cue_point": self.raw_cue_point.LLM_representation(),
            "snapped_time_ms": self.snapped_time_ms,
            "displacement_error_ms": self.displacement_error_ms,
            "quantization_error_ratio": round(self.quantization_error_ratio, 2),
            "adjusted_confidence_score": round(self.adjusted_confidence_score, 2),
        }

    # TODO refactor classes as dataclasses
    # @classmethod
    # def from_dict(cls, data: dict):
    #     """
    #     Deserializes a SnappedCuePoint from a dictionary, allowing for easy reconstruction from JSON data.
    #     """
    #     raw_data = data.get("raw_cue_point", {})
    #     raw_cue_point = RawCuePoint.from_dict(raw_data)
    #
    #     snapped_cue = cls(raw_cue_point, beat_grid=None)  # beat_grid is not needed for deserialization as we already have the snapped time and error
    #     snapped_cue.time = data.get("snapped_time_ms", 0)
    #     snapped_cue.error = data.get("displacement_error_ms", 0)
    #     snapped_cue.quantization_error_ratio = data.get("quantization_error_ratio", 1.0)
    #     snapped_cue.score = data.get("adjusted_confidence_score", 0.0)
    #
    #     return snapped_cue

    # @classmethod
    # def load_from_cache(cls, data: dict):
    #     """
    #     Quick and dirty cache loader. This bypasses __init__ to avoid running
    #     the snapping pipeline during deserialization.
    #     """
    #     raw_data = data.get("raw_cue_point", {})
    #     raw_cue_point = RawCuePoint.from_dict(raw_data)
    #
    #     snapped_cue = cls.__new__(cls)
    #     snapped_cue.raw = raw_cue_point
    #     snapped_cue.structural_penalty_weight = data.get("structural_penalty_weight")
    #     snapped_cue.downbeat_offset_to_skip = data.get("downbeat_offset_to_skip", 0)
    #     snapped_cue.time = data.get("snapped_time_ms", 0)
    #     snapped_cue.error = data.get("displacement_error_ms", 0)
    #     snapped_cue.quantization_error_ratio = data.get("quantization_error_ratio", 1.0)
    #     snapped_cue.score = data.get("adjusted_confidence_score", 0.0)
    #
    #     return snapped_cue

class Token(BaseModel):
    index: int
    start_time_ms: int
    end_time_ms: int
    is_mixable: bool
    start_cue: Optional[SnappedCuePoint] = None
    end_cue: Optional[SnappedCuePoint] = None
    caption: Optional[str] = None # TODO this will be generated later by the captioning model, it's not important for now
    dsp_features: Dict[str, Any] = Field(default_factory=dict) # TODO this will hold the dsp features for the token, such as bpm, key, energy, etc. we can compute these later using the beatnet wrapper and other tools. it's not important for now
    lyrics: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    def get_audio_segment(self):
        # TODO this will return the audio segment corresponding to the token, pydub to do this. it's not important for now
        pass

    # TODO  
    def _get_cue_data_LLM_representation(self):
        # just a placeholder. move it into cue_point.py
        pass

    def json(self):
        return {
          "index": self.index,
          "start_time_ms": self.start_time_ms,
          "end_time_ms": self.end_time_ms,
          "is_mixable": self.is_mixable,
          "start_cue": self.start_cue.json() if self.start_cue else None,
          "end_cue": self.end_cue.json() if self.end_cue else None,
          "caption": self.caption,
          "dsp_features": self.dsp_features,
          "lyrics": self.lyrics,
          "tags": self.tags
        }

    # TODO refactor classes as dataclasses
    # @classmethod
    # def from_dict(cls, data):
    #     return cls(
    #         index=data["index"],
    #         start_time_ms=data["start_time_ms"],
    #         end_time_ms=data["end_time_ms"],
    #         is_mixable=data["is_mixable"],
    #         dsp_features=data.get("dsp_features", {}),
    #         lyrics=data.get("lyrics", None),
    #         start_cue=RawCuePoint.from_dict(data["start_cue"]) if data.get("start_cue") else None,
    #         end_cue=RawCuePoint.from_dict(data["end_cue"]) if data.get("end_cue") else None
    #     )

    def LLM_representation(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "is_mixable": self.is_mixable,
            "start_cue": self.start_cue.LLM_representation() if self.start_cue else None,
            "end_cue": self.end_cue.LLM_representation() if self.end_cue else None,
            "caption": self.caption,
            "dsp_features": self._get_dsp_features_LLM_representation(),
            "lyrics": self._get_lyrics_LLM_representation() if self.lyrics else None,
            "tags": self.tags
        }

    def _get_dsp_features_LLM_representation(self) -> Dict[str, Any]:
        # deep copy to avoid mutating the original features
        dsp_features = self.dsp_features.copy()
        
        # Round top-level numeric features for readability.
        for key, value in dsp_features.items():
            if isinstance(value, float):
                dsp_features[key] = round(value, 2)

        def _vocal_features_LLM_representation(features: dict) -> dict:
            representation = features.copy()
            
            # Drop VED because it's not normalized and not very interpretable.
            representation.pop("vocal_energy_dominance", None)

            rename_map = {
                "vocal_intensity": "intensity",
                "vocal_confidence": "confidence",
                "vocal_density": "density",
            }
            for old_name, new_name in rename_map.items():
                if old_name in representation:
                    representation[new_name] = representation.pop(old_name)

            for vocal_key in ["intensity", "confidence", "density"]:
                if isinstance(representation.get(vocal_key), (float, int)):
                    representation[vocal_key] = round(float(representation[vocal_key]), 2)
            return representation

        vocal_features = dsp_features.get("vocal_features")
        if isinstance(vocal_features, dict):
            vocal_features = vocal_features.copy()
            vocal_features["start_edge"] = _vocal_features_LLM_representation(
                vocal_features.pop("start_edge_features", {})
            )
            vocal_features["end_edge"] = _vocal_features_LLM_representation(
                vocal_features.pop("end_edge_features", {})
            )
            vocal_features = _vocal_features_LLM_representation(vocal_features)
            dsp_features["vocal_features"] = vocal_features

        return dsp_features

    def _get_lyrics_LLM_representation(self) -> Optional[Dict[str, Any]]:
        if not self.lyrics:
            return None

        words = self.lyrics.get("words", [])
        if not words:
            return None

        # we preprocess lyrics for LLM token economy.
        text_lyrics = " ".join([word["word"] for word in words])
        
        # if a word spans across token boundaries, we add a special token to indicate that the word bleeds into the next token.
        last_end_ms = words[-1]["end"] * 1000
        if last_end_ms > self.end_time_ms:
            bleed_ms = int(last_end_ms - self.end_time_ms)
            text_lyrics += f" <BLEED {bleed_ms}>"

        # we save the start and the end time of the lyrics. useful because there could be instrumental padding useful for the mix
        start_ms = int(words[0]["start"] * 1000)
        end_ms = int(last_end_ms) # <-- the LLM can see that the word spans across the token boundary also by looking at the end time.

        med_score = self.lyrics.get("median_score", 0.0)
        min_score = self.lyrics.get("min_score", 0.0)

        # we add confidence scores for the lyrics
        if med_score < LOW_MEDIAN_SCORE or min_score < LOW_MIN_SCORE:
            reliability_tag = "LOW (Hallucination Risk)"
        elif med_score >= HIGH_MEDIAN_SCORE and min_score >= HIGH_MIN_SCORE:
            reliability_tag = "HIGH"
        else:
            reliability_tag = "MEDIUM"

        return {
            "text": text_lyrics,
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "reliability": reliability_tag
        }

class Track(BaseModel):
    path: str
    name: str
    author: str
    key: str
    bpm: Optional[float] = None
    duration_ms: int
    token_size_beats: int
    best_beat_phase_offset: Optional[int] = None
    mean_cue_score: Optional[float] = None
    global_vocal_features: Dict[str, Any] = Field(default_factory=dict)
    token_boundaries: List[int] = Field(default_factory=list)
    snapped_cues: List[SnappedCuePoint] = Field(default_factory=list)
    tokens: List[Token] = Field(default_factory=list)

    def json(self):
        return {
          "path": self.path,
          "name": self.name,
          "author": self.author,
          "key": self.key,
          "bpm": self.bpm,
          "duration_ms": self.duration_ms,
          "token_size_beats": self.token_size_beats,
          "best_beat_phase_offset": self.best_beat_phase_offset,
          "mean_cue_score": self.mean_cue_score,
          "vocal_features": self.global_vocal_features,
          "lyrics_profile": self.LLM_lyrics_profile,
          "token_boundaries": self.token_boundaries,
          "snapped_cues": [cue.json() for cue in self.snapped_cues],
          "tokens": [token.json() for token in self.tokens]
        }

    # TODO refactor classes as dataclasses
    # @classmethod
    # def from_dict(cls, data):
    #   track = cls(
    #       path=data["path"],
    #       name=data["name"],
    #       author=data["author"],
    #       beats_per_token=16,  # You can adjust this if needed
    #       structural_penalty_weight=0.5  # You can adjust this if needed
    #   )
    #   track.key = data.get("key", "Unknown Key")
    #   track.duration_ms = data.get("duration_ms", 0)
    #   track.best_beat_phase_offset = data.get("best_beat_phase_offset", None)
    #   track.mean_cue_score = data.get("mean_cue_score", None)
    #   track.global_vocal_features = data.get("vocal_features", {})
    #   track.token_boundaries = data.get("token_boundaries", [])
    #   track.snapped_cues = [SnappedCuePoint.from_dict(cue) for cue in data.get("snapped_cues", [])]
    #   track.tokens = [Token.from_dict(token) for token in data.get("tokens", [])]
    #   
    #   return track

    # TODO quick and dirty method, i will have to refactor this later by splitting the constructors and the pipeline execution. this is just to have something working for the prototype
    # @classmethod
    # def load_from_cache(cls, json_path: str):
    #   """
    #   Alternative Constructor: Instantiates the Track directly from full.json
    #   by bypassing the heavy __init__ pipeline loop using __new__, while
    #   preserving the exact instance attribute schema layout recursively.
    #   """
    #   # 1. Allocate space for a clean Track instance shell without running __init__
    #   obj = cls.__new__(cls)
    #   
    #   # 2. Parse the cache file into a plain dict
    #   with open(json_path, "r", encoding="utf-8") as f:
    #     cached_tree = json.load(f)
    #   
    #   # 3. Re-assign the properties straight to the object to preserve interface symmetry
    #   obj.path = cached_tree.get("path")
    #   obj.name = cached_tree.get("name")
    #   obj.author = cached_tree.get("author")
    #   obj.key = cached_tree.get("key")
    #   obj.duration_ms = cached_tree.get("duration_ms")
    #   obj.best_beat_phase_offset = cached_tree.get("best_beat_phase_offset")
    #   obj.mean_cue_score = cached_tree.get("mean_cue_score")
    #   obj.token_boundaries = cached_tree.get("token_boundaries", [])
    #   obj.snapped_cues = [
    #     SnappedCuePoint.load_from_cache(cue) for cue in cached_tree.get("snapped_cues", [])
    #   ]
    #   obj.tokens = [Token.from_dict(token) for token in cached_tree.get("tokens", [])]
    #   
    #   # Safely bind remaining pipeline settings or variables
    #   vocal_features = cached_tree.get("vocal_features", {})
    #   obj.vocal_density_threshold = vocal_features.get("vocal_density_threshold")
    #   obj.vocal_confidence_steepness = vocal_features.get("vocal_confidence_steepness")
    #   obj.global_vocal_features = vocal_features
    #   obj.bpm = cached_tree.get("bpm")
    #   
    #   # 4. Return the fully hydrated instance object shell
    #   return obj

    def LLM_representation(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "author": self.author,
            "duration_ms": self.duration_ms,
            "token_size_beats": self.token_size_beats, # TODO change into "self.dsp_manager.beats_per_token", now it's not possible because of the quick and dirty load_from_cache method, refactor this later by splitting the constructors and the pipeline execution. this is just to have something working for the prototype
            # TODO global dsp features. (currently in profiler.py)
            "key": self.key,
            "bpm": self.bpm,
            "tokens": [token.LLM_representation() for token in self.tokens]
        }

    @property
    def LLM_lyrics_profile(self) -> Dict[str, Any]:
        """
        Computes global lyric tracking metrics and synthesizes a trust score
        by inspecting confidence boundaries across all text tokens.
        """
        min_scores = []
        median_scores = []
        full_text = ""

        # Safely iterate through our tokens array (handles both objects and namespaces)
        for token in self.tokens:
            if token.lyrics and token.lyrics.get("words"):
                for word in token.lyrics["words"]:
                    full_text += word["word"] + " "
                min_scores.append(token.lyrics.get("min_score", 0.0))
                median_scores.append(token.lyrics.get("median_score", 0.0))

        if not min_scores or not median_scores:
            return {
                "text": "",
                "reliability": "UNKNOWN",
                "metrics": {"avg_median": 0.0, "avg_min": 0.0}
            }

        # Compute global averages to determine our reliability threshold
        avg_min = sum(min_scores) / len(min_scores)
        avg_med = sum(median_scores) / len(median_scores)

        # Apply your system constants to establish the trust classification
        if avg_med >= HIGH_MEDIAN_SCORE and avg_min >= HIGH_MIN_SCORE:
            reliability_tag = "HIGH"
        elif avg_med < LOW_MEDIAN_SCORE or avg_min < LOW_MIN_SCORE:
            reliability_tag = "LOW (Hallucination Risk)"
        else:
            reliability_tag = "MEDIUM"

        return {
            "text": full_text.strip(),
            "reliability": reliability_tag,
            "metrics": {
                "avg_median": round(avg_med, 2),
                "avg_min": round(avg_min, 2)
            }
        }
