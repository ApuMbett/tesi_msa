import copy
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class CueType(str, Enum):
    START = "START"
    END = "END"
    BOTH = "BOTH"
    NONE = "NONE"


class VocalEdges(str, Enum):
    NONE = "NONE"
    START = "START"
    END = "END"
    BOTH = "BOTH"


class TelemetryEncoder:
    """
    Encodes track telemetry with exact 1-to-1 byte-level equivalence to the original
    ProfilerTokenDTO and SegmenterTokenDTO implementations.
    """

    MICRO_MATH_KEYS = [
        "ROE",
        "BTR",
        "MTR",
        "TTR",
        "SC",
        "MSF",
        "SF",
        "OCN",
        "EV",
        "HER",
    ]

    @classmethod
    def format_cue_type(cls, start_cue: Any, end_cue: Any) -> str:
        if start_cue and end_cue:
            return CueType.BOTH.value
        elif start_cue:
            return CueType.START.value
        elif end_cue:
            return CueType.END.value
        return CueType.NONE.value

    @classmethod
    def format_vocal_edges(cls, vocal_features: Dict[str, Any]) -> str:
        start_edge = vocal_features.get("start_edge", {}) or {}
        end_edge = vocal_features.get("end_edge", {}) or {}

        MIN_CONFIDENCE = 0.50
        MIN_DENSITY = 0.15
        MIN_INTENSITY = 0.0

        def is_active(vocals: dict) -> bool:
            if not isinstance(vocals, dict):
                return False
            return (
                vocals.get("confidence", 0.0) > MIN_CONFIDENCE
                and vocals.get("density", 0.0) > MIN_DENSITY
                and vocals.get("intensity", 0.0) > MIN_INTENSITY
            )

        if is_active(start_edge) and is_active(end_edge):
            return VocalEdges.BOTH.value
        elif is_active(start_edge):
            return VocalEdges.START.value
        elif is_active(end_edge):
            return VocalEdges.END.value
        return VocalEdges.NONE.value

    @classmethod
    def compute_global_metrics(cls, tokens: List[Dict[str, Any]]) -> Dict[str, float]:
        collect_values = lambda key: [
            float(token["dsp_features"][key]) for token in tokens[1:-1]
        ]
        safe_max = lambda values: round(max(values), 2)
        safe_avg = lambda values: round(sum(values) / len(values), 2)

        msf_values = collect_values("MSF")
        btr_values = collect_values("BTR")
        roe_values = collect_values("ROE")
        her_values = collect_values("HER")
        sf_values = collect_values("SF")
        ocn_values = collect_values("OCN")

        return {
            "peak_msf": safe_max(msf_values),
            "peak_btr": safe_max(btr_values),
            "peak_roe": safe_max(roe_values),
            "avg_her": safe_avg(her_values),
            "avg_sf": safe_avg(sf_values),
            "avg_btr": safe_avg(btr_values),
            "avg_ocn": safe_avg(ocn_values),
        }

    @classmethod
    def format_duration(cls, duration_ms: float) -> str:
        duration_int = int(duration_ms)
        return f"{duration_int // 60000}:{(duration_int % 60000) // 1000:02d}"

    @classmethod
    def normalize_track_data(cls, track_or_dict: Any) -> Dict[str, Any]:
        if hasattr(track_or_dict, "LLM_representation"):
            data = track_or_dict.LLM_representation()
            if "name" not in data and hasattr(track_or_dict, "name"):
                data["name"] = track_or_dict.name
            if "author" not in data and hasattr(track_or_dict, "author"):
                data["author"] = track_or_dict.author
            if hasattr(track_or_dict, "LLM_lyrics_profile"):
                data["lyrics_profile"] = track_or_dict.LLM_lyrics_profile
            return data
        elif hasattr(track_or_dict, "model_dump"):
            return track_or_dict.model_dump(by_alias=True)
        elif isinstance(track_or_dict, dict):
            return copy.deepcopy(track_or_dict)
        raise ValueError(f"Unsupported track data type: {type(track_or_dict)}")

    @classmethod
    def encode_profiler_token(cls, token: Dict[str, Any]) -> Dict[str, Any]:
        """Matches original ProfilerTokenDTO.to_dict() exactly."""
        dsp_features_raw = token.get("dsp_features", {}).copy()

        vocal_edges_val = None
        if "vocal_features" in dsp_features_raw:
            v_feat = dsp_features_raw["vocal_features"].copy()
            vocal_edges_val = cls.format_vocal_edges(v_feat)
            dsp_features_raw["vocal_edges"] = vocal_edges_val
            dsp_features_raw["vocal_features"] = v_feat
            dsp_features_raw["vocal_features"].pop("start_edge", None)
            dsp_features_raw["vocal_features"].pop("end_edge", None)

        dsp_features = {}
        if "BPM" in dsp_features_raw:
            dsp_features["BPM"] = dsp_features_raw["BPM"]

        if "vocal_features" in dsp_features_raw:
            dsp_features["vocal_features"] = dsp_features_raw["vocal_features"]

        if vocal_edges_val and vocal_edges_val != VocalEdges.NONE.value:
            dsp_features["vocal_edges"] = vocal_edges_val

        for key in cls.MICRO_MATH_KEYS:
            if key in dsp_features_raw:
                dsp_features[key] = dsp_features_raw[key]

        cue_type = cls.format_cue_type(token.get("start_cue"), token.get("end_cue"))

        d = {"index": token.get("index")}
        if cue_type != CueType.NONE.value:
            d["cue_type"] = cue_type
        if token.get("lyrics"):
            d["lyrics"] = token.get("lyrics")
        d["dsp_features"] = dsp_features
        return d

    @classmethod
    def encode_segmenter_token(cls, token: Dict[str, Any], caption: str = "") -> Dict[str, Any]:
        """Matches original SegmenterTokenDTO.to_dict() exactly."""
        old_vocal_features = token.get("dsp_features", {}).get("vocal_features", {}).copy()

        # Profiler base encoding
        base_token = cls.encode_profiler_token(token)

        # Restore original vocal_features and remove vocal_edges
        if "vocal_features" in base_token["dsp_features"]:
            base_token["dsp_features"]["vocal_features"] = old_vocal_features
            base_token["dsp_features"].pop("vocal_edges", None)

        ordered_dict = {
            "index": base_token.pop("index"),
            "caption": caption,
        }
        ordered_dict.update(base_token)
        return ordered_dict

    @classmethod
    def encode_for_profiler(cls, track_data: Any) -> str:
        """Matches original Profiler.format_track_data() exactly."""
        raw = cls.normalize_track_data(track_data)
        tokens_copy = copy.deepcopy(raw.get("tokens", []))

        formatted_duration = cls.format_duration(raw.get("duration_ms", 0))

        encoded_tokens = [cls.encode_profiler_token(t) for t in tokens_copy]

        ordered_metadata = {
            "duration": formatted_duration,
            "default_token_beats": raw.get("token_size_beats", 16),
            "key": raw.get("key"),
            "global_metrics": cls.compute_global_metrics(tokens_copy),
            "tokens": encoded_tokens,
        }
        return json.dumps(ordered_metadata, ensure_ascii=True)

    @classmethod
    def encode_for_researcher(cls, track_data: Any) -> str:
        """Matches original Researcher.format_track_data() exactly."""
        raw = cls.normalize_track_data(track_data)

        lyrics_profile = raw.get("lyrics_profile") or {}
        if not lyrics_profile.get("text") and "tokens" in raw:
            words = []
            for t in raw["tokens"]:
                lyr = t.get("lyrics")
                if isinstance(lyr, dict) and "text" in lyr:
                    words.append(lyr["text"])
                elif isinstance(lyr, str):
                    words.append(lyr)
            lyrics_text = " ".join(words).strip()
            reliability = raw.get("lyrics_reliability", "UNKNOWN")
        else:
            lyrics_text = lyrics_profile.get("text", "")
            reliability = lyrics_profile.get("reliability", "UNKNOWN")

        stripped_lyrics_profile = {
            "lyrics": lyrics_text,
            "reliability": reliability,
        }

        metadata = {
            "name": raw.get("name", ""),
            "author": raw.get("author", ""),
            "key": raw.get("key", ""),
            "bpm": raw.get("bpm", 0.0),
            "duration": cls.format_duration(raw.get("duration_ms", 0)),
            "lyrics_profile": stripped_lyrics_profile,
        }
        return json.dumps(metadata, ensure_ascii=True)

    @classmethod
    def encode_for_captioner(cls, track_data: Any) -> str:
        """Matches original Captioner.format_track_data() exactly."""
        raw = cls.normalize_track_data(track_data)
        tokens_copy = copy.deepcopy(raw.get("tokens", []))

        formatted_duration = cls.format_duration(raw.get("duration_ms", 0))
        encoded_tokens = [cls.encode_profiler_token(t) for t in tokens_copy]

        ordered_metadata = {
            "name": raw.get("name"),
            "author": raw.get("author"),
            "duration": formatted_duration,
            "default_token_beats": raw.get("token_size_beats", 16),
            "key": raw.get("key"),
            "global_metrics": cls.compute_global_metrics(tokens_copy),
            "tokens": encoded_tokens,
        }
        return json.dumps(ordered_metadata, ensure_ascii=True)

    @classmethod
    def encode_for_segmenter(
        cls,
        track_data: Any,
        token_captions: List[Dict[str, Any]],
    ) -> str:
        """Matches original Segmenter.format_track_data() exactly."""
        raw = cls.normalize_track_data(track_data)
        tokens_copy = copy.deepcopy(raw.get("tokens", []))

        # Build caption map
        caption_map = {}
        for i, c in enumerate(token_captions):
            idx = c.get("token_index", i)
            caption_map[idx] = c.get("caption", "")

        formatted_duration = cls.format_duration(raw.get("duration_ms", 0))
        encoded_tokens = [
            cls.encode_segmenter_token(t, caption=caption_map.get(t.get("index", i), ""))
            for i, t in enumerate(tokens_copy)
        ]

        ordered_metadata = {
            "name": raw.get("name"),
            "author": raw.get("author"),
            "duration": formatted_duration,
            "default_token_beats": raw.get("token_size_beats", 16),
            "key": raw.get("key"),
            "global_metrics": cls.compute_global_metrics(tokens_copy),
            "tokens": encoded_tokens,
        }
        return json.dumps(ordered_metadata, ensure_ascii=True)

    @classmethod
    def format_profiler_context(cls, profiler_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Strips 'rationale' from profiler output to match original format_profiler_data()."""
        data = copy.deepcopy(profiler_dict)
        macro = data.get("macro_profile", {})
        vocal = data.get("vocal_profile", {})
        dyn = data.get("dynamic_trajectory", {})

        macro.pop("rationale", None)
        vocal.pop("rationale", None)
        dyn.pop("rationale", None)

        return {
            "macro_profile": macro,
            "vocal_profile": vocal,
            "dynamic_trajectory": dyn,
        }

    @classmethod
    def format_researcher_context(cls, researcher_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Strips 'researcher_notes' to match original format for captioner and segmenter."""
        data = copy.deepcopy(researcher_dict)
        data.pop("researcher_notes", None)
        return data
