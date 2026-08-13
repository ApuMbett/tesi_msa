import sys
import os
import json
import torch
import whisperx
import numpy as np 
import librosa
from typing import Tuple
from whisperx.vads.vad import Vad

# ==========================================
# We will define the custom SyedRMSVad class here later!
# ==========================================
# vendored from Syed et al https://github.com/jaza-syed/mss-alt/blob/main/expt/01-ss/ss_infer.py#L60
# this is required to make get_speech_timestamps_rms work, which is the VAD we will inject into the WhisperX pipeline to get more accurate word-level timestamps for DJ grid alignment.
class VadOptions: 
    def __init__(self):
        self.onset = 0.1
        self.offset = 0.1
        self.min_speech_duration_ms = 0
        self.max_speech_duration_s = 30
        self.min_silence_duration_ms = 1000
        self.speech_pad_ms = 200
# vendored from Syed et al https://github.com/jaza-syed/mss-alt/blob/main/src/alt/infer.py
def get_speech_timestamps_rms(
    audio: np.ndarray,
    vad_options: VadOptions,
    window_size_samples=512,
    sampling_rate: int = 16000,
) -> Tuple[list[dict], np.ndarray]:
    """This method is used for splitting long audio of separated vocals

    Args:
      audio: One dimensional float array of separated vocals
      vad_options: Binarization options (thresholds, min/max durations).
      sampling rate: Sampling rate of the audio.
      window_size_samples: hop length of rms feature (frame length is fixed)

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """

    onset = vad_options.onset
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    speech_pad_ms = vad_options.speech_pad_ms
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    rms = np.mean(
        librosa.feature.rms(y=audio, frame_length=2048, hop_length=window_size_samples),
        axis=0,
    )
    probs = rms / np.max(rms)

    triggered = False
    speeches = []
    current_speech = {}
    offset = vad_options.offset

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    # Create speech segments using onset and offset thresholds
    # Merging to avoid minimum silence duration
    for i, speech_prob in enumerate(probs):
        if (speech_prob >= onset) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= onset) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        # Switch off if current segment longer than max_speech_duration_s
        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                # previously reached silence (< neg_thres) and is still not speech (< thres)
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                # No prior silence available: perform min-cut on the second half of the segment
                start_frame = current_speech["start"] // window_size_samples
                end_frame = i + 1  # include the current frame in the segment
                segment_scores = probs[start_frame:end_frame]
                # Only consider the second half of the current segment
                second_half_start = len(segment_scores) // 2
                search_segment = segment_scores[second_half_start:]
                # Find the minimum value in the second half
                min_val = search_segment.min()
                min_indices = np.where(search_segment == min_val)[0]
                # If there are multiple equal minimums, pick the last one
                min_index_in_half = min_indices[-1]
                # Adjust the index to match the full segment's index
                chosen_frame = start_frame + second_half_start + min_index_in_half
                # Compute the corresponding sample index for the new cut point
                min_cut_sample = window_size_samples * chosen_frame
                # End the current segment at the min-cut
                current_speech["end"] = min_cut_sample
                speeches.append(current_speech)
                # Start a new segment from the min-cut sample
                current_speech = {"start": min_cut_sample}
                triggered = True
                prev_end = next_start = temp_end = 0
                continue

        # switch off if prob goes below offset and
        # and has been for longer than min_silence_duration_ms
        if (speech_prob < offset) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            # condition to avoid cutting in very short silence
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                # Add the chunk if it's longer than min_speech_duration_ms
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    # special handling for last chunk
    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    # Add speech_pad_ms to start of first chunk and end of last chunk
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches, probs

# ==========================================
# 2. WHISPERX VAD WRAPPER
# ==========================================
class MockSegment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker

class SyedRMSVad(Vad):
    def __init__(self):
        super().__init__(vad_onset=0.500)
        self.vad_options = VadOptions()

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio_dict):
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        # Run the academic RMS math
        raw_dict_segments, _ = get_speech_timestamps_rms(
            waveform,
            vad_options=self.vad_options,
            sampling_rate=sample_rate
        )
        
        # Convert dicts (samples) to WhisperX Segments (seconds)
        whisper_segments = []
        for s in raw_dict_segments:
            start_sec = s["start"] / sample_rate
            end_sec = s["end"] / sample_rate
            whisper_segments.append(MockSegment(start_sec, end_sec))
            
        return whisper_segments

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def check_health() -> None:
    """Verifies that the WhisperX environment and model dependencies are correctly installed."""
    try:
        import whisperx
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"WhisperX dependency missing: {e}. "
            "Please ensure you have installed whisperx manually with:\n"
            "uv pip install git+https://github.com/m-bain/whisperx.git@2cfd7b7c5c7bba144954364db747319b50e8232b\n"
            "and that you are running the project using `uv run --no-sync`."
        ) from e

def transcribe_vocals(vocals_path: str) -> dict:
    # 2. Set up Device & Compute Type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # 3. Load WhisperX Model
    print(f"Loading WhisperX (large-v2) on {device}...")
    my_vad = SyedRMSVad()
    whisper_model = whisperx.load_model(
        "large-v2", 
        device, 
        compute_type=compute_type,
        vad_model=my_vad
    )
    print("Model loaded successfully.")

    # 4. Load Audio
    print(f"Loading audio: {vocals_path}")
    audio = whisperx.load_audio(vocals_path)
    
    # 5. Base Transcription
    print("Running transcription...")
    result = whisper_model.transcribe(audio, batch_size=16)
    
    # 6. Forced Alignment (Word-level timestamps)
    print("Running forced alignment for DJ grid precision...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    
    aligned_result = whisperx.align(
        result["segments"], 
        align_model, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    return aligned_result