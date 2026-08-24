from transformers import DetrImageProcessor, DetrForObjectDetection
from scipy.signal import find_peaks
import torch

from matplotlib import cm
from PIL import Image
import numpy as np
import argparse
import librosa
import json
from pathlib import Path

# CONSTANTS
OVERLAP = 0.75
W_BBOX = 21
W_WIN = 355
PADDING = 266


# (removed hardcoded path builders)

def check_health() -> None:
    """Verifies that Cue-DETR dependencies are installed."""
    try:
        import torch
        from transformers import DetrImageProcessor, DetrForObjectDetection
    except ImportError as e:
        raise RuntimeError(f"Cue-DETR dependency missing: {e}") from e

def _compute_cue_points(track_path: Path, checkpoint: str, radius: int, sensitivity: float) -> list[dict[str, float]]:
    scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    # Load Model
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DetrForObjectDetection.from_pretrained(checkpoint)
    model.to(device)

    # process track
    y, sr = librosa.load(track_path)  # standard sr of 22050
    M = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048)
    M_db = librosa.power_to_db(M, ref=np.max)

    # Convert to RGB image without saving (plt.saveimage)
    arr = M_db[::-1]
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_clim(arr.min(), arr.max())
    rgba = sm.to_rgba(arr, bytes=True)
    rgb_shape = (rgba.shape[1], rgba.shape[0])
    rgba = np.require(rgba, requirements='C')
    im = Image.frombuffer("RGBA", rgb_shape, rgba, "raw", "RGBA", 0, 1)
    image = np.array(im)
    image = image[:, :, :3]

    image_w = image.shape[1]
    image_w += PADDING
    n_windows = int(np.floor(image_w / (W_WIN * (1 - OVERLAP))))
    
    images = []
    borders = []

    # Create image batch with sliding window
    for i in range(n_windows):
        l = int(np.floor(i * W_WIN * (1 - OVERLAP))) - PADDING
        r = l + W_WIN
        borders.append(l)

        # Compute image segment
        if l < 0:
            segment = image[:, :r]        
            pad = -l
            segment = np.pad(segment, ((0, 0), (pad, 0), (0, 0)), mode='linear_ramp')
        elif r > image.shape[1]:
            segment = image[:, l:]        
            pad = r - l - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, pad), (0, 0)), mode='linear_ramp')
        else:
            segment = image[:, l:r]
        
        images.append(segment)

    # Preprocess images
    encoding = image_processor.preprocess(images, do_resize=False, return_tensors='pt')
    pixel_values = encoding['pixel_values']
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
    
    # Convert to scores, labels, boxes (in pixel coordinates)
    to_pixel = [(128, 355)] * pixel_values.shape[0]
    predictions = image_processor.post_process_object_detection(outputs, 0, to_pixel)

    # Convert to box centers
    scores = []
    positions = []
    for p, l in zip(predictions, borders):
        scores.extend(p['scores'].tolist())
        # box -> cue -> spectrogram
        pos = (p['boxes'][:, 0] + p['boxes'][:, 2]) // 2 + l
        positions.extend(pos.long().tolist())

    # Order by position
    positions, scores = zip(*sorted(zip(positions, scale(scores))))

    # Find peaks and extract BOTH position and score
    peak_idx, _ = find_peaks(scores, height=sensitivity, distance=radius)
    
    cue_positions = [positions[idx] for idx in peak_idx]
    cue_times = list(librosa.frames_to_time(cue_positions))
    cue_peak_scores = [scores[idx] for idx in peak_idx]
    
    # Map into the requested list of dictionaries
    # Note: If you need `time` to be in milliseconds for your RawCuePoint class later, 
    # just change `float(t)` to `float(t * 1000)`.
    return [
        {"time": float(t), "score": float(s)} 
        for t, s in zip(cue_times, cue_peak_scores)
    ]
        

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict cue points for a single track using CUE-DETR."
    )
    parser.add_argument(
        "-t",
        "--track-path",
        type=str,
        required=True,
        help="Full path to the audio track",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Full path to the output JSON file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="disco-eth/cue-detr",
        help="Optional local path to model checkpoint",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=16,
        help="Minimum distance in bars between cue points (default = 16)",
    )
    parser.add_argument(
        "-s",
        "--sensitivity",
        type=float,
        default=0.9,
        help="Threshold value for cue points (default = 0.9)",
    )
    parser.add_argument("-p", "--print", action="store_true", help="Print cue points")
    args = parser.parse_args()

    track_path = Path(args.track_path)
    if not track_path.exists():
        raise FileNotFoundError(f"Track not found: {track_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cue_points = _compute_cue_points(
        track_path, args.checkpoint, args.radius, args.sensitivity
    )

    if args.print:
        print(json.dumps(cue_points, indent=2))

    payload = {
        "sensitivity": args.sensitivity,
        "cue_points": cue_points,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
