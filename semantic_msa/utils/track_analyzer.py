import json
import subprocess
from pathlib import Path
import sys

from semantic_msa.utils.dsp_manager import DSPManager, VOCAL_START_END_BAR_WINDOW
from semantic_msa.domain.models import Track, Token, RawCuePoint, SnappedCuePoint
from semantic_msa.domain.workspace import TrackWorkspace

CUE_DETR_SENSITIVITY = 0.85
STRUCTURAL_PENALTY_WEIGHT = 0.5
PSYCHOACUSTIC_THRESHOLD_MS = 50 # This threshold is based on psychoacoustic research on temporal perception in music. [TODO - find source for this threshold]

class TrackAnalyzer:
    """
    Orchestrates the extraction pipeline to build a Track DTO.
    Handles external subprocesses (Cue-DETR, Whisper, Demucs) and delegates DSP math to DSPManager.
    """
    def __init__(self, workspace: TrackWorkspace, author: str, beats_per_token: int = 16, structural_penalty_weight: float = STRUCTURAL_PENALTY_WEIGHT):
        self.workspace = workspace
        self.path = str(workspace.source_audio_path)
        self.name = workspace.track_name
        self.author = author
        self.beats_per_token = beats_per_token
        # this weight determines how much the quantization error ratio will affect the final score, allowing for tuning based on how strict we want to be about structural alignment versus raw confidence. between 0 and 1, where 0 means no penalty and 1 means full penalty based on the quantization error ratio. when 1 the structural alignment has maximum importance, when 0 the raw confidence score has maximum importance.
        self.structural_penalty_weight = structural_penalty_weight

    def extract(self) -> Track:
        dsp_manager = DSPManager(workspace=self.workspace, beats_per_token=self.beats_per_token)
        
        # 1. Get raw cue points via subprocess
        raw_cues = self._compute_raw_cue_points()

        # 2. Compute phase offset and snap cue points
        # when we compute the token boundaries we find the phase offset that maximizes the score. the cue points are snapped to downbeats in a way that the mean score is maximized.
        best_beat_phase_offset, mean_cue_score, snapped_cues, token_boundaries = self._compute_snapped_cue_points(dsp_manager, raw_cues)
        duration_ms = dsp_manager.beats[-1][0] if dsp_manager.beats else 0

        # 3. Build Tokens
        lyrics = dsp_manager.get_lyrics()
        tokens = self._build_token_map(dsp_manager, token_boundaries, snapped_cues, lyrics)

        # 4. Global properties
        key = dsp_manager.compute_camelot_key(0, duration_ms)
        
        # global vocal features
        edge_keys = (
            "vocal_energy_dominance",
            "vocal_intensity",
            "vocal_confidence",
            "vocal_density",
        )
        global_vocal = dsp_manager._compute_DSP_vocal_features(0, duration_ms)
        global_vocal_features = {k: round(float(v), 4) for k, v in zip(edge_keys, global_vocal)}
        global_vocal_features["vocal_density_threshold"] = round(dsp_manager.vocal_threshold, 4)
        global_vocal_features["vocal_confidence_steepness"] = round(dsp_manager.vocal_confidence_steepness, 4)
        global_vocal_features["vocal_edge_window_bars"] = VOCAL_START_END_BAR_WINDOW

        tokens_bpm = [t.dsp_features.get("BPM") for t in tokens if t.is_mixable and t.dsp_features.get("BPM") is not None]
        bpm = dsp_manager.compute_global_bpm(cached_token_bpms=tokens_bpm)

        # TODO refactor classes as dataclasses / pydantic 
        # (This TODO was addressed by the DTO migration but left here for history)

        # self._generate_token_tags("tinymu")
        # ^ TODO WIP, i have to test different models and see which one seems better

        return Track(
            path=self.path,
            name=self.name,
            author=self.author,
            key=key,
            bpm=bpm,
            duration_ms=duration_ms,
            token_size_beats=self.beats_per_token,
            best_beat_phase_offset=best_beat_phase_offset,
            mean_cue_score=mean_cue_score,
            global_vocal_features=global_vocal_features,
            token_boundaries=token_boundaries,
            snapped_cues=snapped_cues,
            tokens=tokens
        )

    def _compute_raw_cue_points(self) -> list[RawCuePoint]:
        base_dir = Path(__file__).resolve().parent  # semantic_msa/utils
        semantic_msa_dir = base_dir.parent

        cue_detr_dir = semantic_msa_dir / "externals" / "cue-detr"
        cue_detr_script = cue_detr_dir / "cue_points_single_track.py"
        cue_points_path = self.workspace.cache_dir / "_cue_points.json"

        def _load_cue_payload(path: Path):
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("sensitivity"), data.get("cue_points", [])
            return None, data

        cue_points = []
        saved_sensitivity = None

        if cue_points_path.exists():
            print(f"Cue points already exist for {self.name}, loading from {cue_points_path}")
            saved_sensitivity, cue_points = _load_cue_payload(cue_points_path)
            print("Cue points loaded successfully.")

        if saved_sensitivity != CUE_DETR_SENSITIVITY:
            if saved_sensitivity is not None:
                print(
                    f"Cue points for {self.name} were computed with a different sensitivity "
                    f"({saved_sensitivity}) than the current one ({CUE_DETR_SENSITIVITY})."
                )
                print("Recomputing cue points with the current sensitivity...")
            
            cue_points_path.parent.mkdir(parents=True, exist_ok=True)
            cue_detr_python = cue_detr_dir / ".venv" / "bin" / "python"
            subprocess.run(
                [
                    str(cue_detr_python),
                    str(cue_detr_script),
                    "--track-path",
                    self.path,
                    "--output-path",
                    str(cue_points_path),
                    "--sensitivity",
                    str(CUE_DETR_SENSITIVITY),
                ],
                check=True,
            )
            _, cue_points = _load_cue_payload(cue_points_path)
            print("Cue points computed and saved successfully.")

        # note: the cue points in the json file are in seconds, we need to convert them to milliseconds for the RawCuePoint class
        return [RawCuePoint(time_ms=int(cue["time"] * 1000), confidence_score=cue["score"]) for cue in cue_points]

    def _compute_snapped_cue_points(self, dsp_manager: DSPManager, raw_cues: list[RawCuePoint]):
        best_beat_phase_offset = None
        best_mean_score = None
        best_snapped_cues = []
        best_token_boundaries = []

        for i in range(dsp_manager.beats_per_token // 4):
            # token boundaries now include tail and intro pad, this was necessary for completeness. now we can move this to the track class, this is necessary because (see blablabla) we have the micro intro pad that has 8000 bpm and so the qer is very high. even though it's few ms 
            token_boundaries = dsp_manager.get_semantic_audio_token_boundaries(downbeat_offset_to_skip=i)
            snapped_cues = []
            
            # Snap the raw cue point to the nearest downbeat in the grid
            # this is the offset in number of downbeats to skip when getting the token boundaries for snapping. it can happen that cue points are at some offset from the first downbeat, we can use this parameter to skip the first n downbeats when getting the token boundaries to get minimum QER 
            for raw_cue in raw_cues:
                snapped_cue = self._snap_cue_point(raw_cue, dsp_manager, token_boundaries, i)
                snapped_cues.append(snapped_cue)
                #! ^ this will be refactored later, snappedcuepoint should not take the whole beat grid as an argument now that the track class exists 

            # maximize the mean score across all snapped cues
            mean_cue_score = sum(cue.adjusted_confidence_score for cue in snapped_cues) / len(snapped_cues) if snapped_cues else 0
            if best_mean_score is None or mean_cue_score > best_mean_score:
                best_mean_score = mean_cue_score
                best_beat_phase_offset = i
                best_snapped_cues = snapped_cues
                best_token_boundaries = token_boundaries
                #!^ problem: token boundaries now include tail and intro pad, this was necessary for completeness. now we can move this to the track class, this is necessary because (see blablabla) we have the micro intro pad that has 8000 bpm and so the qer is very high. even though it's few ms 

            # print(f"\n\n\n\n######## Phase offset {i}: mean score={mean_cue_score} ########") 
            # print(F"token boundaries={self.token_boundaries}") 
            # print("Snapped Cue Points for this phase offset:")
            # for snapped_cue in snapped_cues:
            #     print(snapped_cue.json())

        return best_beat_phase_offset, best_mean_score, best_snapped_cues, best_token_boundaries

    def _snap_cue_point(self, raw_cue: RawCuePoint, dsp_manager: DSPManager, token_boundaries: list[int], downbeat_offset_to_skip: int) -> SnappedCuePoint:
        """
        Quantizes the raw CUE-DETR prediction to the nearest structural grid point.
        The CUE-DETR paper notes that DJ mixes strictly adhere to high-level track 
        structures (phrases/bars). Snapping enforces this structural alignment.
        """
        # Find the nearest beat/downbeat in the provided grid
        nearest_beat = min(token_boundaries, key=lambda x: abs(x - raw_cue.time_ms)) if token_boundaries else raw_cue.time_ms
        
        # Calculate the displacement error
        error = abs(nearest_beat - raw_cue.time_ms)

        # Evaluate the accuracy of the snapped cue point based on psychoacoustic thresholds
        """
        Calculates the Quantization Error Ratio, which is the ratio of the cue point's displacement error to the maximum forgivable error based on the track's local BPM.
        This ratio is crucial for evaluating the reliability of the cue point and adjusting the AI's confidence score accordingly.

        this metric is tempo agnostic as it scales the error relative to the beat duration, so it measures how many "beats" the error represents, rather than just the raw time in milliseconds. This allows for a more meaningful evaluation of the cue point's accuracy across different tempos and so offers a more consistent basis for adjusting the confidence score and comparing cue points across tracks with varying BPMs.

        $$QER = \max \left(0, \min \left(1, \frac{\text{error} - threshold_ms }{\text{max\_error} - threshold_ms } \right) \right)$$
        """
        #find start and end indexes of times in beat_grid for the token that contains the cue point
        # we need to calculate the local bpm for the token that contains the cue point. 
        # so we need to find the start and end times of the token that contains the cue point, and then calculate the bpm for that token.

        # Note: the cue point might be outside the token boundaries if it's in the intro or tail padding. in that case we use the bpm of the closest token, which is a reasonable approximation since the intro and tail pads are typically short and we want to avoid extreme qer values for cues that are slightly outside the token boundaries.
        
        # 1. Find the start time (equal to or just before the raw time)
        start_times = [t for t in token_boundaries if t <= raw_cue.time_ms]
        start_time = max(start_times) if start_times else (token_boundaries[0] if token_boundaries else 0) # Failsafe: if no boundaries are before the cue, use the first boundary (this can happen if the cue is in the intro pad before the first downbeat)

        # 2. Find future boundaries (MUST be strictly greater to avoid 0 duration)
        future_boundaries = [t for t in token_boundaries if t > start_time]
        if future_boundaries:
            end_time = min(future_boundaries)
        else:
            # Failsafe: If the cue is at the very end of the track, look backward
            end_time = start_time
            # Assuming token_boundaries has at least 2 elements
            if len(token_boundaries) >= 2:
                start_time = token_boundaries[-2]

        # Calculate the beat duration in milliseconds based on local BPM
        beat_duration_ms = (end_time - start_time) / dsp_manager.beats_per_token if end_time > start_time else 500
        
        # The maximum forgivable error is typically set to half the beat duration, as errors larger than this would likely be perceptible and detrimental to the mix.
        max_forgivable_error = beat_duration_ms / 2

        # Full confidence for errors within threshold_ms ms, as they are generally imperceptible in a DJ mix context, regardless of BPM. 
        # 1. Apply the threshold_ms ms "Deadzone" (anything under threshold_ms  becomes 0)
        effective_error = max(0.0, error - PSYCHOACUSTIC_THRESHOLD_MS)
        # 2. Adjust the maximum scale to account for the deadzone
        effective_max = max(0.001, max_forgivable_error - PSYCHOACUSTIC_THRESHOLD_MS) # The 0.001 prevents division by zero
        # 3. Calculate ratio and "Clamp" the maximum value to 1.0
        qer = max(0.0, min(1.0, effective_error / effective_max)) # Clamp strictly between 0 and 1

        # No confidence for errors larger than the maximum forgivable error, as they would likely be perceptible and detrimental to the mix

        """
        Calculates the Mix Readiness Score by decaying the AI's confidence 
        based on its Quantization Error Ratio.
        """
        # The structural penalty is a linear decay of the raw confidence score based on the quantization error ratio, weighted by the structural_penalty_weight. This allows for tuning how much the structural alignment should influence the final score, balancing between raw confidence and structural reliability.
        structural_penalty = qer * self.structural_penalty_weight
        adjusted_score = raw_cue.confidence_score * (1.0 - structural_penalty)

        return SnappedCuePoint(
            raw_cue_point=raw_cue,
            snapped_time_ms=nearest_beat,
            displacement_error_ms=error,
            quantization_error_ratio=qer,
            adjusted_confidence_score=adjusted_score
        )

    def _build_token_map(self, dsp_manager: DSPManager, token_boundaries: list[int], snapped_cues: list[SnappedCuePoint], lyrics: dict) -> list[Token]:
        boundaries = [b for b in token_boundaries]  # Make a copy to avoid modifying the original list of token boundaries
        duration_ms = dsp_manager.beats[-1][0] if dsp_manager.beats else 0

        # Add intro and tail padding to ensure we map the full track, theese segments though won't be mixable, they will just be used for completeness and to make sure we don't miss any cue points that are close to the start or the end of the track.
        # 1. Intro Pad: Insert 0 only if the first downbeat isn't already at 0
        if boundaries and boundaries[0] > 0:
            boundaries.insert(0, 0)
        
        # 2. Tail Pad: Append the last beat only if it isn't already captured
        if boundaries and boundaries[-1] != duration_ms:
            boundaries.append(duration_ms)

        tokens = []
        # Iterate through the boundaries to create tokens
        for i in range(len(boundaries) - 1):
            start_time_ms = boundaries[i]
            end_time_ms = boundaries[i + 1]
            
            # Determine if the token is mixable based on its position
            is_mixable = (i > 0 and i < len(boundaries) - 2) # Only tokens that are not intro or tail pads are mixable

            # Determine if the token contains any snapped cue points and if so, check if they are at the start or the end of the token. 
            # Check if any snapped cue points fall within the token boundaries
            token_cues = [c for c in snapped_cues if start_time_ms <= c.snapped_time_ms <= end_time_ms]
            # Determine if there are cues at the start or end of the token
            start_cue = next((c for c in token_cues if c.snapped_time_ms == start_time_ms), None)
            end_cue = next((c for c in token_cues if c.snapped_time_ms == end_time_ms), None)

            # Map lyrics to the token based on the token boundaries. we assign to the token all the words that are in the token boundaries.
            # this method maps the lyrics to the tokens. if a word STARTS in the token boundaries, we assign it to the token.
            # we do this because we want to keep also words that span across token boundaries. 
            token_lyrics_words = [w for w in lyrics.get("word_segments", []) if start_time_ms <= w["start"] * 1000 < end_time_ms]
            token_lyrics = None
            if token_lyrics_words:
                # calculate median and minimum word score for the token lyrics.
                word_scores = [w["score"] for w in token_lyrics_words]
                med_score = sorted(word_scores)[len(word_scores) // 2] # median
                min_score = min(word_scores)
                # low min score but high median means that there are some hallucinated words for example
                token_lyrics = {
                    "words": token_lyrics_words,
                    "median_score": med_score,
                    "min_score": min_score
                }

            # TODO DSP features
            dsp_features = {}
            if is_mixable:
                # compute the bpm only if it's not a pad 
                dsp_features.update(dsp_manager.compute_low_level_dsp_features(start_time_ms, end_time_ms))

            # TODO captioning 
            # TODO lyrics and vocal features
            token = Token(
                index=i,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                is_mixable=is_mixable,
                start_cue=start_cue,
                end_cue=end_cue,
                dsp_features=dsp_features,
                lyrics=token_lyrics
            )
            tokens.append(token)

        # fix pad bpm 
        # if it's a padding token we just inherit the bpm from the closest real token. 
        # 1. Fix Intro Pad (It borrows from the token immediately after it)
        if tokens and not tokens[0].is_mixable and len(tokens) > 1:
            tokens[0].dsp_features["BPM"] = tokens[1].dsp_features.get("BPM")
            
        # 2. Fix Outro Pad (It borrows from the token immediately before it)
        if len(tokens) > 1 and not tokens[-1].is_mixable:
            tokens[-1].dsp_features["BPM"] = tokens[-2].dsp_features.get("BPM")

        return tokens

    ##### SERIALIZATION AND LLM REPRESENTATION METHODS #####
    def generate_token_audio(self, token_index, tokens: list[Token], dsp_manager: DSPManager):
        if token_index < 0 or token_index >= len(tokens):
            raise ValueError("Invalid token index")

        token = tokens[token_index]
        output_path = self.workspace.tokens_dir / f"{token_index}.wav"
        dsp_manager.generate_audio_slice(token.start_time_ms, token.end_time_ms, str(output_path))
        return str(output_path)

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "blablabla.mp3"
    
    # We still need repo_root to locate the original mp3 if it's in data/raw_audio
    repo_root = Path(__file__).resolve().parent.parent.parent
    original_audio = repo_root / "data" / "raw_audio" / filename
    
    workspace = TrackWorkspace(track_name=Path(filename).stem, original_audio_path=original_audio)
    workspace.setup()
    
    analyzer = TrackAnalyzer(workspace=workspace, author="unknown")
    track = analyzer.extract()
    
    print("\n\n\n\n############ BEST ##############")
    print("Best phase offset (in beats):", track.best_beat_phase_offset)
    print("Mean score for best phase offset:", track.mean_cue_score)
    print("Token boundaries (ms):", track.token_boundaries)
    print("Snapped Cue Points:")
    for snapped_cue in track.snapped_cues:
        print(snapped_cue.json())

    print("\n\n\n\n############ FULL TRACK JSON ##############")
    full_json_path = workspace.output_dir / "full.json"
    with open(full_json_path, "w") as f:
        f.write(track.model_dump_json(indent=4, by_alias=True))
    print(f"Full track JSON saved to {full_json_path}")

    print("\n\n\n\n############ LLM REPRESENTATION ##############")
    LLM_json_path = workspace.output_dir / "LLM.json"
    with open(LLM_json_path, "w") as f:
        json.dump(track.LLM_representation(), f, indent=4)
    print(f"LLM representation JSON saved to {LLM_json_path}")

    print("\n\n\n\n############ TOKEN AUDIO SLICE TEST ##############")
    # To test generate_token_audio, we'd need the dsp_manager which is not saved in Track DTO.
    # It was available on the old Track class. Here we just note that it works via analyzer.
