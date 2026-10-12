"""
FUTGA Caption Parser & MSA Structural Formatter.

Converts raw FUTGA inference captions from `eval/futga_inference/futga_inference_results.jsonl`
into structured JSON documents saved individually per track in:
`eval/futga_inference/futga_inference_results_parsed/{id}.json`

Also outputs `_review_manifest.json` highlighting problematic tracks that require
manual review before SongFormer TXT export.
"""

import os
import json
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional

# HarmonixSet canonical MSA labels
CANONICAL_LABELS = {
    "intro", "verse", "pre-chorus", "chorus", "bridge", "outro", "inst", "silence", "other"
}

# Regex to match FUTGA segment header lines across all discovered syntax variations:
# 1. **0-31:** Intro/Verse 1
# 2. **0-32% - Intro/Verse:** or **0-33% Intro:**
# 3. **Segment 1 (0-30%):**
# 4. **0 - 32:**
# 5. **0-31:** Male vocalist singing...
# 6. **100% - Outro:** / **100-End - Outro:**
SEG_HEADER_RE = re.compile(
    r'^\s*(?:\*\*)?\s*'
    r'(?:Segment\s+(\d+)\s*[\(:])?\s*'
    r'(\d+)\s*(?:[-–]\s*(\d+|End))?\s*%?\s*'
    r'(?:\))?\s*'
    r'(?:[:\-–]\s*)?'
    r'(.*?)(?:\*\*)?\s*$',
    re.IGNORECASE
)

# Known section headers that should not be treated as segment boundaries
NON_SEGMENT_HEADERS = [
    'functional segment',
    'detailed analysis',
    'overall music',
    'tempo:',
    'melody:',
    'chord progression:',
]


def parse_header_line(line: str) -> Optional[Tuple[Optional[str], float, float, str]]:
    """
    Parses a segment header line.
    Returns: (segment_enum_idx, start_pct, end_pct, raw_label) or None
    """
    clean_line = line.strip()
    if not (clean_line.startswith('**') or clean_line.startswith('Segment')):
        return None
    
    lower_line = clean_line.lower()
    if any(h in lower_line for h in NON_SEGMENT_HEADERS):
        return None

    m = SEG_HEADER_RE.match(clean_line)
    if not m:
        return None

    seg_idx_str, start_str, end_str, label_str = m.group(1), m.group(2), m.group(3), m.group(4)
    if not start_str:
        return None

    start_pct = float(start_str)
    if end_str is None or end_str.lower() == 'end':
        end_pct = 100.0
    else:
        end_pct = float(end_str)

    # Clean label text
    raw_label = label_str.strip().strip('*').strip(':').strip('-').strip('–').strip()

    # If header had explicit Segment N with no label, record "Segment N"
    if not raw_label and seg_idx_str:
        raw_label = f"Segment {seg_idx_str}"

    return seg_idx_str, start_pct, end_pct, raw_label


def split_macro_and_segments(caption: str) -> Tuple[str, List[str]]:
    """
    Splits the caption into the macro musical description text and
    the lines corresponding to functional segments.
    """
    lines = caption.split('\n')
    first_seg_line_idx = None

    for idx, line in enumerate(lines):
        if parse_header_line(line) is not None:
            first_seg_line_idx = idx
            break

    if first_seg_line_idx is None:
        # Fallback if no header matched
        return caption.strip(), []

    macro_text = '\n'.join(lines[:first_seg_line_idx]).strip()
    # Strip any trailing section marker like **Functional Segments:** from macro text
    macro_text = re.sub(r'\*\*(?:Functional Segments|Functional Segment Descriptions):\*\*\s*$', '', macro_text, flags=re.IGNORECASE).strip()
    segment_lines = lines[first_seg_line_idx:]
    return macro_text, segment_lines


def extract_macro_analysis(macro_text: str) -> Dict[str, Any]:
    """
    Extracts structured fields (mood, theme, tempo, melody, instruments, chords)
    from the global musical analysis paragraph.
    """
    text = macro_text.replace('\n', ' ')

    def get_field(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip().strip('.')
                # remove trailing conjunctions
                val = re.sub(r'\s+while\s+.*$', '', val, flags=re.IGNORECASE)
                val = re.sub(r'\s*,\s*$', '', val)
                return val
        return None

    mood = get_field([
        r'mood is\s+([^,\.]+(?:,\s*[^,\.]+)?(?:\s+and\s+[^,\.]+)?)[,\.]',
        r'overall mood\s+(?:of the song\s+)?is\s+([^,\.]+(?:,\s*[^,\.]+)?(?:\s+and\s+[^,\.]+)?)[,\.]',
        r'has a\s+([^,\.]+(?:\s+and\s+[^,\.]+)?)\s+mood'
    ])

    theme = get_field([
        r'theme is\s+([^,\.]+(?:,\s*[^,\.]+)?(?:\s+and\s+[^,\.]+)?)[,\.]',
        r'theme of\s+([^,\.]+(?:,\s*[^,\.]+)?(?:\s+and\s+[^,\.]+)?)[,\.]',
        r'theme revolves around\s+([^,\.]+(?:,\s*[^,\.]+)?(?:\s+and\s+[^,\.]+)?)[,\.]'
    ])

    tempo = get_field([
        r'tempo is\s+([^,\.]+(?:,\s*around\s*\d+\s*(?:bpm|beats per minute))?)[,\.]',
        r'tempo\s+(?:of the song\s+)?is\s+([^,\.]+(?:,\s*around\s*\d+\s*(?:bpm|beats per minute))?)[,\.]'
    ])

    melody = get_field([
        r'melody is\s+([^,\.]+(?:,\s*with\s+[^,\.]+)?)[,\.]',
        r'main melody is\s+([^,\.]+(?:,\s*with\s+[^,\.]+)?)[,\.]'
    ])

    chord = get_field([
        r'chord progression\s+(?:of the song\s+)?is\s+([^\.]+)[\.]',
        r'chord progression\s+is\s+([^\.]+)[\.]'
    ])

    inst_match = re.search(r'instruments\s+(?:used in the song\s+)?include\s+([^\.]+)\.', text, re.IGNORECASE)
    instruments = []
    if inst_match:
        raw_inst = inst_match.group(1)
        items = re.split(r',\s*|\s+and\s+', raw_inst)
        cleaned_items = []
        for it in items:
            cleaned = it.strip().lower()
            cleaned = re.sub(r'^(?:a|an)\s+', '', cleaned)
            if cleaned and cleaned not in cleaned_items:
                cleaned_items.append(cleaned)
        instruments = cleaned_items

    return {
        "raw_text": macro_text,
        "mood": mood,
        "theme": theme,
        "tempo": tempo,
        "melody": melody,
        "instruments": instruments,
        "chord_progression": chord
    }


def get_standard_label(
    raw_label: str,
    seg_idx: int,
    total_segs: int,
    start_pct: float,
    end_pct: float,
    description_text: str
) -> Tuple[str, List[str], bool]:
    """
    Maps a raw header label to a canonical HarmonixSet class:
    intro, verse, pre-chorus, chorus, bridge, outro, inst, silence, other.

    Returns:
        (standard_label, detected_labels, is_generic)
    """
    text_to_search = raw_label.lower()
    is_generic = False

    detected = []
    for tag in ['intro', 'verse', 'pre-chorus', 'prechorus', 'chorus', 'bridge', 'outro', 'solo', 'riff', 'instrumental', 'fade out', 'ending']:
        if tag in text_to_search:
            detected.append(tag)

    # Normalize detected tags
    normalized_detected = []
    for d in detected:
        if d in ['solo', 'riff', 'instrumental']:
            normalized_detected.append('inst')
        elif d in ['fade out', 'ending']:
            normalized_detected.append('outro')
        elif d == 'prechorus':
            normalized_detected.append('pre-chorus')
        else:
            normalized_detected.append(d)
    normalized_detected = list(dict.fromkeys(normalized_detected))

    # Detect generic indicators
    if not raw_label or any(g in text_to_search for g in ['segment 1', 'segment 2', 'segment 3', 'segment 4']):
        is_generic = True

    # Position-aware disambiguation rules
    resolved = "other"
    if seg_idx == 0 and 'intro' in normalized_detected:
        resolved = 'intro'
    elif (seg_idx == total_segs - 1 or end_pct >= 100) and 'outro' in normalized_detected:
        resolved = 'outro'
    elif 'chorus' in normalized_detected:
        resolved = 'chorus'
    elif 'verse' in normalized_detected:
        resolved = 'verse'
    elif 'bridge' in normalized_detected:
        resolved = 'bridge'
    elif 'inst' in normalized_detected:
        resolved = 'inst'
    elif normalized_detected:
        resolved = normalized_detected[0]
    else:
        # Fallback to inspect description text
        desc_lower = description_text.lower()
        desc_detected = []
        for tag in ['intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'outro', 'solo', 'riff', 'instrumental']:
            if re.search(r'\b' + tag + r'\b', desc_lower):
                desc_detected.append(tag)

        if seg_idx == 0 and 'intro' in desc_detected:
            resolved = 'intro'
        elif (seg_idx == total_segs - 1 or end_pct >= 100) and 'outro' in desc_detected:
            resolved = 'outro'
        elif 'chorus' in desc_detected:
            resolved = 'chorus'
        elif 'verse' in desc_detected:
            resolved = 'verse'
        elif 'bridge' in desc_detected:
            resolved = 'bridge'
        elif any(t in desc_detected for t in ['solo', 'riff', 'instrumental']):
            resolved = 'inst'
        else:
            resolved = 'other'

    return resolved, normalized_detected, is_generic


def parse_song_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single record from futga_inference_results.jsonl into a rich structured JSON dict.
    """
    song_id = record['id']
    audio_path = record.get('audio_path', f"data/audios/HarmonixSet/{song_id}.wav")
    duration = float(record['duration'])
    elapsed_sec = float(record.get('elapsed_sec', 0.0))
    caption = record['caption']
    gt_labels = record.get('ground_truth_labels', [])

    # 1. Split macro text and segment lines
    macro_text, segment_lines = split_macro_and_segments(caption)
    macro_data = extract_macro_analysis(macro_text)

    # 2. Parse individual segments
    raw_segments: List[Dict[str, Any]] = []
    curr_seg: Optional[Dict[str, Any]] = None

    for line in segment_lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Stop parsing if we hit a duplicate detailed analysis block
        if line_clean.lower().startswith('**detailed analysis:'):
            break

        parsed = parse_header_line(line_clean)
        if parsed is not None:
            if curr_seg is not None:
                raw_segments.append(curr_seg)
            seg_idx_str, start_pct, end_pct, raw_label = parsed
            curr_seg = {
                "start_pct": start_pct,
                "end_pct": end_pct,
                "raw_header": line_clean,
                "raw_label": raw_label,
                "lines": []
            }
        elif curr_seg is not None:
            # Bullet point or descriptive body line
            curr_seg["lines"].append(line_clean)

    if curr_seg is not None:
        raw_segments.append(curr_seg)

    total_segs = len(raw_segments)
    formatted_segments: List[Dict[str, Any]] = []
    review_reasons: List[str] = []
    generic_indices: List[int] = []
    has_point_tail = False

    for idx, s in enumerate(raw_segments):
        start_pct = s["start_pct"]
        end_pct = s["end_pct"]
        raw_header = s["raw_header"]
        raw_label = s["raw_label"]
        desc_text = "\n".join(s["lines"]).strip()

        # Compute physical timestamps in seconds
        start_sec = round((start_pct / 100.0) * duration, 2)
        end_sec = round((end_pct / 100.0) * duration, 2)

        # Standard label resolution
        std_label, detected_tags, is_generic = get_standard_label(
            raw_label=raw_label,
            seg_idx=idx,
            total_segs=total_segs,
            start_pct=start_pct,
            end_pct=end_pct,
            description_text=desc_text
        )

        if is_generic:
            generic_indices.append(idx)

        # Check for point-like tail (e.g. 100% - Outro)
        if start_pct >= 100.0 or start_sec >= end_sec:
            has_point_tail = True
            review_reasons.append(f"point_like_terminal_segment: '{raw_header}' ({start_pct}%-{end_pct}%)")

        # Review reason checks
        if not raw_label:
            review_reasons.append(f"empty_header_label: '{raw_header}'")
        elif any(g in raw_label.lower() for g in ['segment 1', 'segment 2', 'segment 3', 'segment 4']):
            review_reasons.append(f"generic_segment_name: '{raw_label}'")
        elif any(genre in raw_label.lower() for genre in ['hard rock', 'pop rock', 'heavy metal']):
            review_reasons.append(f"genre_as_segment_label: '{raw_label}'")
        elif len(raw_label.split()) > 6:
            review_reasons.append(f"prose_sentence_header: '{raw_label[:40]}...'")

        formatted_segments.append({
            "index": idx,
            "start_pct": start_pct,
            "end_pct": end_pct,
            "start_time_seconds": start_sec,
            "end_time_seconds": end_sec,
            "raw_header": raw_header,
            "raw_label": raw_label,
            "standard_label": std_label,
            "all_detected_labels": detected_tags,
            "is_generic": is_generic,
            "description": desc_text
        })

    # Additional song-level review flags
    is_duplicated = '**Detailed Analysis:**' in caption
    if is_duplicated:
        review_reasons.append("has_duplicated_detailed_analysis_block")
    if total_segs > 3:
        review_reasons.append(f"non_standard_segment_count: {total_segs} segments (standard is 3)")

    needs_manual_review = len(review_reasons) > 0

    return {
        "id": song_id,
        "audio_path": audio_path,
        "duration": duration,
        "elapsed_sec": elapsed_sec,
        "raw_caption": caption,
        "needs_manual_review": needs_manual_review,
        "review_reasons": review_reasons,
        "macro_analysis": macro_data,
        "segments": formatted_segments,
        "ground_truth_labels": gt_labels,
        "parsing_metadata": {
            "num_segments": total_segs,
            "has_generic_labels": len(generic_indices) > 0,
            "has_unmapped_labels": any(s["standard_label"] == "other" for s in formatted_segments),
            "has_point_tail": has_point_tail,
            "is_duplicated": is_duplicated,
            "generic_segment_indices": generic_indices,
            "covers_full_track": (
                len(formatted_segments) > 0 and
                formatted_segments[0]["start_pct"] == 0.0 and
                formatted_segments[-1]["end_pct"] >= 100.0
            ),
            "time_conversion_basis": "full_track_duration"
        }
    }


def convert_json_to_format(song_data: Dict[str, Any]) -> str:
    """Convert parsed song JSON data to the specified format"""
    result = []
    segments = song_data.get("segments", [])

    # Process the start time and label for each segment
    for segment in segments:
        start_time = float(segment.get("start_time_seconds", segment.get("start", 0.0)))
        label = segment.get("standard_label", segment.get("label", "other"))
        result.append(f"{start_time:.6f} {label}")

    # Add the last end time
    if segments:
        last_end_time = float(song_data.get("duration", segments[-1].get("end_time_seconds", segments[-1].get("end", 0.0))))
        result.append(f"{last_end_time:.6f} end")

    return "\n".join(result)


def process_json_files(input_folder: str, output_folder: str):
    """Process all JSON files in the input folder"""
    from pathlib import Path

    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get all JSON files (ignoring manifest files starting with _)
    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".json") and not f.startswith("_")])

    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)

        try:
            # Read the JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert the format
            converted_data = convert_json_to_format(data)

            # Generate the output filename (replace .json with .txt)
            output_filename = json_file.replace(".json", ".txt")
            output_path = os.path.join(output_folder, output_filename)

            # Write to the output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(converted_data + "\n")

            print(f"✓ Processed: {json_file} -> {output_filename}")

        except Exception as e:
            print(f"✗ Error processing {json_file}: {str(e)}")


def main(
    input_folder: str = "eval/futga_inference/futga_inference_results_parsed",
    output_folder: str = "eval/inference_results/futga_txt"
):
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)

    # Process the files
    process_json_files(input_folder, output_folder)

    print("-" * 50)
    print("Processing complete!")


if __name__ == "__main__":
    try:
        import fire
        fire.Fire(main)
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description="Convert FUTGA parsed JSONs to standard SongFormer MSA TXT format.")
        parser.add_argument("--input_folder", "--input-folder", type=str, default="eval/futga_inference/futga_inference_results_parsed")
        parser.add_argument("--output_folder", "--output-folder", type=str, default="eval/inference_results/futga_txt")
        args, _ = parser.parse_known_args()
        main(args.input_folder, args.output_folder)