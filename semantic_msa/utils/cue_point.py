
class RawCuePoint:
    """
    Data object representing a temporal boundary for DJ mixing.
    Based on the CUE-DETR architecture (Argüello et al., 2024), which 
    treats cue point estimation as a computer vision object detection task.
    This is the initial output from the CUE-DETR model before any structural alignment.
    """
    def __init__(self, time_ms, score):
        self.time = time_ms  # The raw temporal position in milliseconds
        self.score = score   # The confidence score of the prediction

    def json(self):
        """
        Serializes the raw cue point 
        """
        return {
          "time_ms": self.time,
          "confidence_score": self.score,
        }


from beatnet_wrapper import BeatNetWrapper
class SnappedCuePoint:
    """
    Represents a cue point after snapping to the nearest downbeat in the beat grid.
    This class encapsulates the structural alignment process and the evaluation of the cue point's reliability based on psychoacoustic thresholds (taking bpm into account).
    """
    def __init__(self, raw_cue_point: RawCuePoint, beat_grid: BeatNetWrapper, downbeat_offset_to_skip = 0, structural_penalty_weight = 0.5):

        if beat_grid.beats is None:
            raise ValueError("Beat grid must be computed before snapping cue points.")
        
        self.raw = raw_cue_point
        self.structural_penalty_weight = structural_penalty_weight # this weight determines how much the quantization error ratio will affect the final score, allowing for tuning based on how strict we want to be about structural alignment versus raw confidence. between 0 and 1, where 0 means no penalty and 1 means full penalty based on the quantization error ratio. when 1 the structural alignment has maximum importance, when 0 the raw confidence score has maximum importance.
        
        # Snap the raw cue point to the nearest downbeat in the grid
        token_boundaries = beat_grid.get_semantic_audio_token_boundaries(downbeat_offset_to_skip = downbeat_offset_to_skip)
        self._snap_to_downbeat(token_boundaries)

        # Calculate the displacement error
        self.error = abs(self.time - self.raw.time)


        # Evaluate the accuracy of the snapped cue point based on psychoacoustic thresholds
        self.quantization_error_ratio = self._calculate_quantization_error_ratio(beat_grid)
        self.score = self._calculate_combined_score(self.quantization_error_ratio)


    def _snap_to_downbeat(self, token_boundaries: list[int]):
        """
        Quantizes the raw CUE-DETR prediction to the nearest structural grid point.
        The CUE-DETR paper notes that DJ mixes strictly adhere to high-level track 
        structures (phrases/bars). Snapping enforces this structural alignment.
        """
        # Find the nearest beat/downbeat in the provided grid
        nearest_beat = min(token_boundaries, key=lambda x: abs(x - self.raw.time))
        self.time = nearest_beat
      

    def _calculate_combined_score(self, quantization_error_ratio: float) -> float:
        """
        Calculates the Mix Readiness Score by decaying the AI's confidence 
        based on its Quantization Error Ratio.
        """
        # The structural penalty is a linear decay of the raw confidence score based on the quantization error ratio, weighted by the structural_penalty_weight. This allows for tuning how much the structural alignment should influence the final score, balancing between raw confidence and structural reliability.
        structural_penalty = quantization_error_ratio * self.structural_penalty_weight
        adjusted_score = self.raw.score * (1.0 - structural_penalty)
        
        return round(adjusted_score, 3)

    def _calculate_quantization_error_ratio(self, beat_grid: BeatNetWrapper, threshold_ms=50) -> float:
        """
        Calculates the Quantization Error Ratio, which is the ratio of the cue point's displacement error to the maximum forgivable error based on the track's local BPM.
        This ratio is crucial for evaluating the reliability of the cue point and adjusting the AI's confidence score accordingly.

        this metric is tempo agnostic as it scales the error relative to the beat duration, so it measures how many "beats" the error represents, rather than just the raw time in milliseconds. This allows for a more meaningful evaluation of the cue point's accuracy across different tempos and so offers a more consistent basis for adjusting the confidence score and comparing cue points across tracks with varying BPMs.

        $$QER = \max \left(0, \min \left(1, \frac{\text{error} - threshold_ms }{\text{max\_error} - threshold_ms } \right) \right)$$
        """

        #find start and end indexes of times in beat_grid for the token that contains the cue point
        token_boundaries = beat_grid.get_semantic_audio_token_boundaries()

        # we need to calculate the local bpm for the token that contains the cue point. 
        # so we need to find the start and end times of the token that contains the cue point, and then calculate the bpm for that token.

        # Note: the cue point might be outside the token boundaries if it's in the intro or tail padding. in that case we use the bpm of the closest token, which is a reasonable approximation since the intro and tail pads are typically short and we want to avoid extreme qer values for cues that are slightly outside the token boundaries.

        # 1. Find the start time (equal to or just before the raw time)
        start_times = [t for t in token_boundaries if t <= self.raw.time]
        start_time = max(start_times) if start_times else token_boundaries[0]  # Failsafe: if no boundaries are before the cue, use the first boundary (this can happen if the cue is in the intro pad before the first downbeat)

        # 2. Find future boundaries (MUST be strictly greater to avoid 0 duration)
        future_boundaries = [t for t in token_boundaries if t > start_time]

        if future_boundaries:
            end_time = min(future_boundaries)
        else:
            # Failsafe: If the cue is at the very end of the track, look backward
            end_time = start_time
            # Assuming token_boundaries has at least 2 elements
            start_time = token_boundaries[-2]

        # Calculate the beat duration in milliseconds based on local BPM
        beat_duration_ms = (end_time - start_time) / beat_grid.beats_per_token
        

        # The maximum forgivable error is typically set to half the beat duration, as errors larger than this would likely be perceptible and detrimental to the mix.
        max_forgivable_error = beat_duration_ms / 2
        
        

        # Full confidence for errors within threshold_ms ms, as they are generally imperceptible in a DJ mix context, regardless of BPM. This threshold is based on psychoacoustic research on temporal perception in music. [TODO - find source for this threshold]
        # 1. Apply the threshold_ms ms "Deadzone" (anything under threshold_ms  becomes 0)
        effective_error = max(0.0, self.error - threshold_ms )
        
        # 2. Adjust the maximum scale to account for the deadzone
        effective_max = max(0.001, max_forgivable_error - threshold_ms) # The 0.001 prevents division by zero
        
        # 3. Calculate ratio and "Clamp" the maximum value to 1.0
        qer = max(0.0, min(1.0, effective_error / effective_max)) # Clamp strictly between 0 and 1

        # No confidence for errors larger than the maximum forgivable error, as they would likely be perceptible and detrimental to the mix
        
        return qer
    def json(self):
        """
        Serializes the snapped cue point, including the original raw prediction, the snapped time, the error, and the adjusted confidence score.
        """
        return {
          "raw_cue_point": self.raw.json(),
          "snapped_time_ms": self.time,
          "displacement_error_ms": self.error,
          "quantization_error_ratio": self.quantization_error_ratio,
          "adjusted_confidence_score": self.score,
        }



if __name__ == "__main__":
  # demo, reads the cue points from a json file 
    song = "blablabla.mp3"
    import json
    with open("../../cue-detr/tracks/_cue_points.json", "r") as f:
        #debug: print full path 
        import os 
        print("Reading cue points from:", os.path.abspath("../../cue-detr/tracks/_cue_points.json"))
        data = json.load(f)
        shotmedown_cues = data[song]

        # note: the cue points in the json file are in seconds, we need to convert them to milliseconds for the RawCuePoint class and i don't know if this is the right precision 
        raw_cues = [RawCuePoint(time_ms = int(cue["time"]*1000), score = cue["score"]) for cue in shotmedown_cues]

        print("Raw Cue Points:")
        for cue in raw_cues:
            print("Raw Cue Point:", cue.json())

        #snap the raw cues to the beat grid
        beat_grid = BeatNetWrapper(audio_path = "../data/raw_audio/"+song)
        snapped_cues = [SnappedCuePoint(raw_cue, beat_grid) for raw_cue in raw_cues]


        print("\nSnapped Cue Points:")
        for snapped_cue in snapped_cues:
            print("Snapped Cue Point:", snapped_cue.json())


        print("\nBoundaries for semantic audio tokens (blocks of bars):", beat_grid.get_semantic_audio_token_boundaries())
        print("Global BPM:", beat_grid.compute_global_bpm())
        print("BPM for each 16-beat token: ")
        for i in range(len(beat_grid.get_semantic_audio_token_boundaries())-1
            ):
            start_time = beat_grid.get_semantic_audio_token_boundaries()[i]
            end_time = beat_grid.get_semantic_audio_token_boundaries()[i+1]
            bpm = beat_grid._compute_token_bpm(start_time, end_time)
            print(f"Token {i}: Start={start_time}ms, End={end_time}ms, BPM={bpm}")
