from BeatNet.BeatNet import BeatNet
import numpy as np
from collections import Counter
import librosa
import os
import subprocess
import json
from pathlib import Path
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parent  # semantic_msa/utils
SEMANTIC_MSA_DIR = BASE_DIR.parent
REPO_ROOT = SEMANTIC_MSA_DIR.parent
DATA_DIR = REPO_ROOT / "data"
AUDIO_DIR = DATA_DIR / "raw_audio"
class DSPManager:
  """
    Wrapper around BeatNet to extract beat grid information and convert it to a format suitable for our application. This class can be initialized with either raw BeatNet output or an audio file path, and provides methods to retrieve the full beat grid, downbeats, pixel boundaries for blocks of bars, and both global and local BPM estimates. 
  """
  def __init__(self, audio_path, beats_per_token = 16):
    self.beats_per_token = beats_per_token
    self.beats = DSPManager._compute_raw_data(audio_path)

    # loading the track into memory using librosa, useful for low level DSP features computing 
    self.sr = 22050  # Standard sampling rate for audio processing
    self.y = None 
    self.y, self.sr = librosa.load(audio_path, sr=self.sr)
    self.audio_path = audio_path
    self.track_name = audio_path.split("/")[-1].split(".")[0]    
    

  @staticmethod
  def _compute_raw_data(audio_path):
    estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
    data = estimator.process(audio_path)  
    
    return DSPManager._sanitize_raw_data(data)

  @staticmethod
  def _sanitize_raw_data(raw_data):
      return [(int(time*1000), int(beat)) for time, beat in raw_data]
      # return [(time, int(beat)) for time, beat in raw_data]
  

  def get_full_grid(self):
    return self.beats
    
  def get_downbeats(self):
    return [time for time, beat in self.beats if beat == 1]


  def get_semantic_audio_token_boundaries(self, downbeat_offset_to_skip = 0):
    """
    Converts the beat grid into boundaries for semantic audio tokens, which represent blocks of bars. This method identifies the first downbeat after downbeat_offset_to_skip to ensure phase-locked phrasing and slices the grid accordingly.
    """
    # 1. Find the index of the FIRST downbeat (anchor) (where beat == 1 or 1.0) to ensure phase-locked 16-beat phrasing
    # Find the indices of ALL downbeats (where beat == 1)
    downbeat_indices = [i for i, (time, beat) in enumerate(self.beats) if beat == 1]
    
    if not downbeat_indices:
        anchor_idx = 0  # Failsafe
    else:
        # Safely get the Nth downbeat (wrap around if offset is somehow too high)
        safe_offset = min(downbeat_offset_to_skip, len(downbeat_indices) - 1)
        anchor_idx = downbeat_indices[safe_offset]  

    # 2. Slice the grid starting strictly from the Anchor
    boundaries = [time for time, beat in self.beats[anchor_idx::self.beats_per_token]]
        
    return boundaries

  
  
  def compute_global_bpm(self): 
      """Returns the median BPM of the entire track.

      This method computes the BPM for each 16-beat token and returns the most common BPM as the global estimate. This approach accounts for potential tempo variations across the track while providing a robust overall BPM estimate for mix preparation.
      """
      bpm_estimates = []
      token_boundaries = self.get_semantic_audio_token_boundaries()
      for i in range(len(token_boundaries)-1):
        start_time = token_boundaries[i]
        end_time = token_boundaries[i+1]
        bpm = self._compute_token_bpm(start_time, end_time)
        if bpm is not None:
          bpm_estimates.append(bpm)
      
      most_common = Counter(bpm_estimates).most_common(1)
      return float(most_common[0][0])

  def _compute_token_bpm(self, start_time_ms, end_time_ms):
        """Returns the BPM for a specific token defined by start_time and end_time.

            for performance reasons, this helper method computes the BPM for a token which we know is self.beats_per_token beats long so we can directly calculate the BPM using the duration of the token, without needing to count the number of beats within it. This is more efficient than the general compute_bpm method when we are specifically working with fixed-length tokens.
        """
        # find the corresponding beat indices for the start and end times
        duration_ms = end_time_ms - start_time_ms
      
        if duration_ms <= 0:
          return None
          
        bpm = (self.beats_per_token * 60000) / duration_ms
        return round(bpm, 1)

  def compute_bpm(self, start_beat_index, end_beat_index):
      """Returns the macro BPM for a specific segment by measuring total duration."""
      segment_beats = self.beats[start_beat_index:end_beat_index]
      num_beats = len(segment_beats)
      
      # We need at least 2 beats to measure an interval
      if num_beats < 2:
          return None  
          
      # Get the exact start and end times in milliseconds
      start_time_ms = segment_beats[0][0]
      end_time_ms = segment_beats[-1][0]
      duration_ms = end_time_ms - start_time_ms
      
      if duration_ms <= 0:
          return None
          
      # The number of intervals (spaces between beats) is num_beats - 1
      intervals = num_beats - 1
      
      # Formula: (intervals / duration_in_minutes)
      bpm = (intervals / (duration_ms / 60000))
      
      return round(bpm, 1) # Rounds to a clean decimal like 128.0

  # TODO: compute vocal_confidence, vocal_presence, start_BVR and end_BVR on vocal track

  def compute_low_level_dsp_features(self, start_time_ms, end_time_ms):
    # TODO this is temporary but i just need it to see if it works, i will investigate later for better features for our purposes 
    # 1. Convert milliseconds to array indices
    start_sample = int(start_time_ms * self.sr / 1000)
    end_sample = int(end_time_ms * self.sr / 1000)
    
    # 2. get the splitted vocal and instrumental tracks.
    y_vocal, y_instrumental = self._get_y_splitted()
    # 3. Slice the audio array for this specific token
    y_token = self.y[start_sample:end_sample]
    y_token_vocal = y_vocal[start_sample:end_sample]
    y_token_instrumental = y_instrumental[start_sample:end_sample]
    
    # compute "ROE", "BTR","MTR","TTR","SC","MSF","SF","OCN","EV" on original master
    S = np.abs(librosa.stft(y_token))
    rms_env = librosa.feature.rms(S=S)[0]
    total_rms_mean = np.mean(rms_env) + 1e-6
    
    onset_env = librosa.onset.onset_strength(y=y_token, sr=self.sr, S=S)
    
    freqs = librosa.fft_frequencies(sr=self.sr)
    bass_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    treb_mask = freqs >= 4000


    bass_rms = np.sqrt(np.mean(S[bass_mask, :]**2, axis=0))
    mid_rms = np.sqrt(np.mean(S[mid_mask, :]**2, axis=0))
    treb_rms = np.sqrt(np.mean(S[treb_mask, :]**2, axis=0))

    raw_bass = np.mean(bass_rms)
    raw_mid = np.mean(mid_rms)
    raw_treb = np.mean(treb_rms)
    # normalize between 0 and 1
    total_band_energy = raw_bass + raw_mid + raw_treb + 1e-6
    
    btr = raw_bass / total_band_energy
    mtr = raw_mid / total_band_energy
    ttr = raw_treb / total_band_energy

    nyquist = self.sr / 2
    sc = np.mean(librosa.feature.spectral_centroid(S=S)) / nyquist
    sf = np.mean(librosa.feature.spectral_flatness(S=S))
    msf = np.mean(np.maximum(0, np.diff(S, axis=1))**2)
    
    roe = np.mean(onset_env)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
    ocn = len(onsets)
    ev = np.var(rms_env)
    
    # Compute HER on instrumental track
    y_instrumental_harmonic, _ = self._get_instrumental_hpss()
    S_harmonic_instrumental = np.abs(librosa.stft(y_instrumental_harmonic[start_sample:end_sample]))
    her = np.mean(librosa.feature.rms(S=S_harmonic_instrumental)) / total_rms_mean

    # TODO compute vocal_confidence, vocal_presence, start_BVR and end_BVR on vocal track

    return {
        "ROE": round(float(roe), 4),
        "BTR": round(float(btr), 4),
        "MTR": round(float(mtr), 4),
        "TTR": round(float(ttr), 4),
        "SC": round(float(sc), 4),
        "MSF": round(float(msf), 4),
        "SF": round(float(sf), 4),
        "OCN": int(ocn),
        "EV": round(float(ev), 4),
        "HER": round(float(her), 4)
    }
  


  def compute_camelot_key(self, start_time_ms: float, end_time_ms: float) -> str:
    """
    Computes the musical key of a specific audio segment instrumental track and returns its Camelot Wheel equivalent.
    Assumes self.y (the audio array) and self.sr (sample rate) are already loaded in the class.
    """
    # 1. Convert milliseconds to audio samples
    start_sample = int((start_time_ms / 1000.0) * self.sr)
    end_sample = int((end_time_ms / 1000.0) * self.sr)
    
    # 2. Extract the Chromagram (Pitch Class Profile)
    # y_harmonic isolates the tonal elements from the percussive transients
    y_harmonic, _ = self._get_instrumental_hpss()
    y_harmonic = y_harmonic[start_sample:end_sample]
    if len(y_harmonic) == 0:
        return "Unknown"


    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
    
    # Sum the chroma features over time to get the dominant 12 pitch classes
    chroma_sum = np.sum(chromagram, axis=1)
    
    # 3. Define the Krumhansl-Schmuckler Key Profiles
    # These represent the statistical distribution of notes in Major and Minor scales
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # Note names corresponding to the 12 chroma bins (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # The ultimate Camelot mapping dictionary
    camelot_map = {
        'C Major': '8B', 'A Minor': '8A',
        'G Major': '9B', 'E Minor': '9A',
        'D Major': '10B', 'B Minor': '10A',
        'A Major': '11B', 'F# Minor': '11A',
        'E Major': '12B', 'C# Minor': '12A',
        'B Major': '1B', 'G# Minor': '1A',
        'F# Major': '2B', 'D# Minor': '2A',
        'C# Major': '3B', 'A# Minor': '3A',
        'G# Major': '4B', 'F Minor': '4A',
        'D# Major': '5B', 'C Minor': '5A',
        'A# Major': '6B', 'G Minor': '6A',
        'F Major': '7B', 'D Minor': '7A'
    }

    best_corr = -1.0
    best_key = ""

    # 4. Template Matching via Pearson Correlation
    for i in range(12):
        # Rotate the profiles through all 12 root notes
        rotated_maj = np.roll(maj_profile, i)
        rotated_min = np.roll(min_profile, i)

        # Correlate the audio's chromagram with the Major template
        corr_maj, _ = pearsonr(chroma_sum, rotated_maj)
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{notes[i]} Major"

        # Correlate the audio's chromagram with the Minor template
        corr_min, _ = pearsonr(chroma_sum, rotated_min)
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{notes[i]} Minor"

    # 5. Translate the standard key to Camelot
    return camelot_map.get(best_key, "Unknown")
  
  def _get_instrumental_hpss(self):
    """
    Lazy evaluation for HPSS on the Instrumental track ONLY.
    Calculates the entire track once and caches it.
    """
    if not hasattr(self, '_y_inst_harmonic') or self._y_inst_harmonic is None:
        # 1. Make sure we have the stems loaded
        _, y_inst = self._get_y_splitted()
        
        # 2. Run the heavy math on the full instrumental track
        self._y_inst_harmonic, self._y_inst_percussive = librosa.effects.hpss(y_inst)
        
    return self._y_inst_harmonic, self._y_inst_percussive

  def _get_y_splitted(self):
    # lazy evaluation for the splitted vocal and instrumental tracks, since we need them in multiple places and we don't want to run demucs multiple times. 
    if hasattr(self, '_y_vocal') and self._y_vocal is not None:
      return self._y_vocal, self._y_instrumental
    
    if not self.has_splitted_tracks():
      self.split_song_tracks()

    vocal_path = AUDIO_DIR / "htdemucs" / self.track_name / "vocals.wav"
    instrumental_path = AUDIO_DIR / "htdemucs" / self.track_name / "no_vocals.wav"
    self._y_vocal, _ = librosa.load(vocal_path, sr=self.sr)
    self._y_instrumental, _ = librosa.load(instrumental_path, sr=self.sr)
    
    # 4. Return the cached arrays
    return self._y_vocal, self._y_instrumental


  def split_song_tracks(self):
    # this method splits the song into vocal and instrumental tracks using Demucs. 
    if not self.has_splitted_tracks():
      # run demucs and save the vocal and instrumental tracks in memory for later use.
      print(f"[Demucs] Extracting vocals for {self.audio_path}...")

      command = ["demucs", "--two-stems=vocals", "-o", str(AUDIO_DIR), self.audio_path]
      subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
  
  def has_splitted_tracks(self):
    # this method checks if the song has already been splitted into vocal and instrumental tracks, to avoid doing it multiple times. 
    expected_vocal_path = AUDIO_DIR / "htdemucs" / self.track_name / "vocals.wav"
    return expected_vocal_path.exists()

  #TODO don't know if this is the right place for these methods but for now it's easier to implement them here since they are related to the DSP features of the track, we can refactor later if needed.
  def get_lyrics(self):
    # then runs RMS-VAD on the vocal track of the song and uses WhisperX to get the lyrics. 
    # The output is the timestamps of the lyrics, word by word.

    # 1. split the song into vocals and instrumental using demucs
    self.split_song_tracks()

    # 2. run WhisperX on the vocal track with custom VAD (RMS-VAD, Syed et al) to get the lyrics with timestamps. 
    # 2. Load the main Whisper into memory 
    # i run it as a subprocess because it's easier to manage the dependencies 
    output_json_path = DATA_DIR / "json_db" / self.track_name / "_whisper_output.json"
    if not output_json_path.exists():
      vocal_path = AUDIO_DIR / "htdemucs" / self.track_name / "vocals.wav"
      whisper_env_python = REPO_ROOT / "whisper_engine" / ".venv" / "bin" / "python"  # Path to the Python executable in the whisper environment
      whisper_wrapper = REPO_ROOT / "whisper_engine" / "whisper_wrapper.py"
      command = [str(whisper_env_python), str(whisper_wrapper), str(vocal_path)]
      subprocess.run(command, check=True, )
      # Read the JSON file left behind by the bridge script
      print("Transcription complete. Ingesting timestamp data...")
      if not output_json_path.exists():
          raise FileNotFoundError("WhisperX finished, but no JSON output was found!")

    with open(output_json_path, "r", encoding="utf-8") as f:
        lyrics_data = json.load(f)

    return lyrics_data

  def compute_vocal_features(self, start_time_ms, end_time_ms):
    #TODO this method calculates vocal_confidence, vocal_presence, start_BVR and end_BVR for the interval. 
    # it does this on the vocal_track of the song  

    pass
if __name__ == "__main__":
    print("started beatnet wrapper test")
    # resolve path 
    path = DATA_DIR / "raw_audio" / "shotmedown.mp3"

    bn = DSPManager(audio_path = str(path))

    print("full grid:", bn.get_full_grid())
    print("downbeats:", bn.get_downbeats())
    token_boundaries = bn.get_semantic_audio_token_boundaries()
    print("token boundaries:", token_boundaries)
    print(len(bn.get_semantic_audio_token_boundaries()))

    print("global BPM:", bn.compute_global_bpm())
    print("BPM for each 16-beat token: ")
    bpms = []
    for i in range(len(token_boundaries)-1):
        start_time = token_boundaries[i]
        end_time = token_boundaries[i+1]
        # find the corresponding beat indices for the start and end times
        start_beat_index = next(j for j, (time, beat) in enumerate(bn.get_full_grid()) if time >= start_time)
        end_beat_index = next(j for j, (time, beat) in enumerate(bn.get_full_grid()) if time >= end_time)
        bpm = bn.compute_bpm(start_beat_index, end_beat_index)
        if bpm is not None: 
          bpms.append(bpm)
        print(f"Token {i}: BPM={bpm}")
    
    print("Median BPM across tokens:", np.median(bpms))

    print("LYRICS TEST")
    bn.split_song_tracks()

    lyrics = bn.get_lyrics()
    for word in lyrics["word_segments"]:
      print(word)
