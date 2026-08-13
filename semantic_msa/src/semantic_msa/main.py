from semantic_msa.domain.workspace import TrackWorkspace
from semantic_msa.utils.track_analyzer import TrackAnalyzer


import argparse


def main():
    parser = argparse.ArgumentParser(description="Process track name and audio path.")
    parser.add_argument("--name", type=str, required=True, help="Track name")
    parser.add_argument("--author", type=str, required=True, help="Track author")
    parser.add_argument("--audio-path", type=str, required=True, help="Track audio path")

    args = parser.parse_args()

    print("")

    from pathlib import Path
    track_ws = TrackWorkspace(args.name, Path(args.audio_path))
    extractor = TrackAnalyzer(track_ws, args.author)
    track = extractor.extract()
    print(track.LLM_representation())

if __name__ == "__main__":
   
    main()
