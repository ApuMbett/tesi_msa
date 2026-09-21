import copy
import json
from typing import Any, Dict, List, Optional


class FinalTrackIdentityDTO:
    """Represents the top-level identity metadata in final.json."""

    def __init__(self, track_data: Dict[str, Any]):
        self.name = track_data.get("name", "")
        self.artist = track_data.get("artist", "")
        self.key = track_data.get("key", "")
        self.bpm = track_data.get("bpm", 0)
        duration_ms = track_data.get("duration_ms", 0.0)
        self.duration = f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}"
        self.researcher = track_data.get("researcher", "")
        self.profiler = track_data.get("profiler", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "artist": self.artist,
            "key": self.key,
            "bpm": self.bpm,
            "duration": self.duration,
            "cultural_context": self.researcher,
            "acoustic_profile": self.profiler,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class FinalTokenDTO:
    """Represents an individual 16-beat temporal token in final.json."""

    def __init__(self, token_data: Dict[str, Any], unit: str = "seconds"):
        self.index = token_data.get("index", 0)
        self.start_time = self._to_unit(token_data.get("start_time_ms", 0.0), unit)
        self.end_time = self._to_unit(token_data.get("end_time_ms", 0.0), unit)

        self.start_cue = copy.deepcopy(token_data.get("start_cue", {}))
        self.end_cue = copy.deepcopy(token_data.get("end_cue", {}))

        # apply unit conversion to cue points if they exist
        if self.start_cue:
            self.start_cue["time"] = self._to_unit(self.start_cue.get("time_ms", 0.0), unit)
        if self.end_cue:
            self.end_cue["time"] = self._to_unit(self.end_cue.get("time_ms", 0.0), unit)

        self.lyrics = copy.deepcopy(token_data.get("lyrics", ""))
        # apply unit conversion to lyrics timestamps if they exist
        if self.lyrics and isinstance(self.lyrics, dict):
            self.lyrics["start_time"] = self._to_unit(self.lyrics.get("start_time_ms", 0.0), unit)
            self.lyrics["end_time"] = self._to_unit(self.lyrics.get("end_time_ms", 0.0), unit)

        self.dsp_features = token_data.get("dsp_features", {})
        self.caption = token_data.get("captioning_data", "")

    def _to_unit(self, value_ms: float, unit: str = "seconds") -> Any:
        """Convert the value to the specified unit."""
        if unit == "milliseconds":
            return value_ms
        elif unit == "seconds":
            return value_ms / 1000
        elif unit == "minutes":
            return f"{int(value_ms // 60000)}:{int((value_ms % 60000) // 1000):02d}"
        else:
            raise ValueError(f"Unsupported unit: {unit}")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "index": self.index,
            "captioning_data": self.caption,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        # sparse representation of cue points, only include them if they exist
        if self.start_cue:
            d["start_cue"] = self.start_cue
        if self.end_cue:
            d["end_cue"] = self.end_cue

        if self.lyrics:
            d["lyrics"] = self.lyrics

        d["dsp_features"] = self.dsp_features
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class FinalSegmentDTO:
    """Represents a macro-structural musical segment in final.json."""

    def __init__(self, segment_data: Dict[str, Any], tokens: List[FinalTokenDTO]):
        self.index = segment_data.get("segment_index", 0)
        self.name = segment_data.get("segment_name", "")
        self.standard_label = segment_data.get("standard_label", "")
        self.macro_caption = segment_data.get("macro_caption", "")
        self.rationale = segment_data.get("rationale", "")
        token_range = segment_data.get("token_range", [0, 0])
        self.tokens = [token for token in tokens if token_range[0] <= token.index <= token_range[1]]

        # derive segment start and end time based on bounded tokens
        if self.tokens:
            self.start_time = self.tokens[0].start_time
            self.end_time = self.tokens[-1].end_time
        else:
            self.start_time = "0:00"
            self.end_time = "0:00"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "standard_label": self.standard_label,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "macro_caption": self.macro_caption,
            "rationale": self.rationale,
            "tokens": [token.to_dict() for token in self.tokens],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class FinalTrackDTO:
    """Root container representing the complete synthesized final.json document."""

    def __init__(self, track_data: Dict[str, Any], unit: str = "minutes"):
        self.track_identity = FinalTrackIdentityDTO(track_data)
        self.unit = unit
        self.token_size_beats = track_data.get("token_size_beats", 0)
        self.lyrics_profile = track_data.get("lyrics_profile", {})

        captioner_info = track_data.get("captioner", {})
        if isinstance(captioner_info, dict):
            self.caption_length = captioner_info.get("CAPTION_LENGTH", 0)
        else:
            self.caption_length = 0

        segmenter_info = track_data.get("segmenter", {})
        if isinstance(segmenter_info, dict):
            self.segmenter_granularity = segmenter_info.get("GRANULARITY", "MEDIUM")
        else:
            self.segmenter_granularity = "MEDIUM"

        # build the tokens with their micro-captions
        self.tokens: List[FinalTokenDTO] = []
        captions = captioner_info.get("token_captions", []) if isinstance(captioner_info, dict) else []
        for i, token in enumerate(track_data.get("tokens", [])):
            token_copy = copy.deepcopy(token)
            if i < len(captions):
                caption = captions[i].copy()
                caption.pop("token_index", None)
                token_copy["captioning_data"] = caption
            else:
                token_copy["captioning_data"] = {}

            token_dto = FinalTokenDTO(token_copy, unit=self.unit)
            self.tokens.append(token_dto)

        # build the segments and nest their tokens inside them
        raw_segments = segmenter_info.get("segments", []) if isinstance(segmenter_info, dict) else []
        self.segments = [FinalSegmentDTO(segment, self.tokens) for segment in raw_segments]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_identity": self.track_identity.to_dict(),
            "token_size_beats": self.token_size_beats,
            "lyrics_profile": self.lyrics_profile,
            "caption_length": self.caption_length,
            "segmenter_granularity": self.segmenter_granularity,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self, indent: int = 4, *args, **kwargs) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class TrackSynthesizer:
    """
    Synthesizes the intermediate multi-agent artifacts (Profiler, Researcher,
    Captioner, Segmenter) with the underlying track telemetry into a FinalTrackDTO.
    Maintains 100% byte-for-byte equivalence with orchestrator.py.
    """

    @classmethod
    def synthesize(
        cls,
        track_data: Dict[str, Any],
        profiler_output: Dict[str, Any],
        researcher_output: Dict[str, Any],
        captioner_output: Dict[str, Any],
        segmenter_output: Dict[str, Any],
        unit: str = "minutes",
        caption_length: Optional[int] = None,
        granularity: Optional[str] = None,
    ) -> FinalTrackDTO:
        full_data = copy.deepcopy(track_data)
        full_data["profiler"] = copy.deepcopy(profiler_output)
        full_data["researcher"] = copy.deepcopy(researcher_output)
        full_data["captioner"] = copy.deepcopy(captioner_output)
        full_data["segmenter"] = copy.deepcopy(segmenter_output)

        return FinalTrackDTO(full_data, unit=unit)
