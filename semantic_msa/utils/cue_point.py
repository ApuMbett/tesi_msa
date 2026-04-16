
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

class SnappedCuePoint:
    """
    Represents a cue point after snapping to the nearest downbeat in the beat grid.
    This class encapsulates the structural alignment process and the evaluation of the cue point's reliability based on psychoacoustic thresholds (taking bpm into account).
    """
    def __init__(self, raw_cue_point: RawCuePoint, downbeat_grid, bpm):
        self.raw = raw_cue_point
        
        # Snap the raw cue point to the nearest downbeat in the grid
        self._snap_to_downbeat(downbeat_grid)

        # Calculate the displacement error
        self.error = abs(self.time - self.raw.time)

        # Evaluate the accuracy of the snapped cue point based on psychoacoustic thresholds
        self.quantization_error_ratio = self._calculate_quantization_error_ratio(bpm)
        self.score = self._calculate_combined_score(bpm)


    def _snap_to_downbeat(self, downbeat_grid):
        """
        Quantizes the raw CUE-DETR prediction to the nearest structural grid point.
        The CUE-DETR paper notes that DJ mixes strictly adhere to high-level track 
        structures (phrases/bars). Snapping enforces this structural alignment.
        """
        # Find the nearest beat/downbeat in the provided grid
        nearest_beat = min(beat_grid, key=lambda x: abs(x - self.raw.time))
        self.time = nearest_beat
      

    def _calculate_combined_score(self, bpm):
        """
        Calculates the Mix Readiness Score by decaying the AI's confidence 
        based on its Quantization Error Ratio.
        """

        quantization_error_ratio = self._calculate_quantization_error_ratio(bpm)
        # The AI's confidence decays linearly as the error ratio increases
        adjusted_score = self.raw.model_confidence * (1.0 - quantization_error_ratio)
        
        return round(adjusted_score, 3)

    def _calculate_quantization_error_ratio(self, bpm):
        """
        Calculates the Quantization Error Ratio, which is the ratio of the cue point's displacement error to the maximum forgivable error based on the track's BPM.
        This ratio is crucial for evaluating the reliability of the cue point and adjusting the AI's confidence score accordingly.

        $$QER = \max \left(0, \min \left(1, \frac{\text{error} - 50}{\text{max\_error} - 50} \right) \right)$$
        """
        beat_duration_ms = 60000 / bpm

        # The maximum forgivable error is typically set to half the beat duration, as errors larger than this would likely be perceptible and detrimental to the mix.
        max_forgivable_error = beat_duration_ms / 2
        
        

        # Full confidence for errors within 50ms, as they are generally imperceptible in a DJ mix context, regardless of BPM. This threshold is based on psychoacoustic research on temporal perception in music. [TODO - find source for this threshold]
        # 1. Apply the 50ms "Deadzone" (anything under 50 becomes 0)
        effective_error = max(0.0, self.error - 50)
        
        # 2. Adjust the maximum scale to account for the deadzone
        effective_max = max_forgivable_error - 50
        
        # 3. Calculate ratio and "Clamp" the maximum value to 1.0
        qer = min(1.0, effective_error / effective_max)

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
  pass