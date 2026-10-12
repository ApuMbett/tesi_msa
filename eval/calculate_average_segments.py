#!/usr/bin/env python3
"""
calculate_average_segments.py

Calcola e stampa le statistiche sul numero di segmenti (Media, Deviazione Standard,
Mediana, Minimo, Massimo e Totale) per ciascun modello e run di inferenza:
- Ground Truth (HarmonixSet 50 tracks)
- SongFormer (patch 50 tracks e batch iniziale 44 tracks)
- FUTGA-7B (zero-shot)
- Flash (Pipeline Agenti con gemini-3.8-flash)
- Lite (Pipeline Agenti con gemini-3.5-flash-lite)
- Max (Pipeline Agenti con gemini-3.1-pro-preview)
- Lite Direct Audio (Baseline gemini-3.5-flash-lite su audio grezzo, raw e post-processed)

Supporta esportazione opzionale in formato Markdown o CSV.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Optional


def load_ground_truth(gt_path: Path) -> Dict[str, int]:
    """Carica il numero di segmenti annotati per traccia dalla Ground Truth HarmonixSet."""
    counts = {}
    if not gt_path.is_file():
        return counts
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            labels = item.get("labels", [])
            # Esclude l'eventuale marker terminale 'end'
            segs = [l for l in labels if l.get("label", "").lower() != "end"]
            counts[item["id"]] = len(segs)
    return counts


def load_songformer_jsonl(jsonl_path: Path) -> Dict[str, int]:
    """Carica il numero di segmenti per traccia dalle inferenze SongFormer."""
    counts = {}
    if not jsonl_path.is_file():
        return counts
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            counts[item["song_id"]] = len(item.get("content", []))
    return counts


def load_txt_segments(txt_dir: Path) -> Dict[str, int]:
    """Carica il conteggio segmenti da file standard SongFormer MSA .txt (es. FUTGA o Direct Audio)."""
    counts = {}
    if not txt_dir.is_dir():
        return counts
    for txt_file in sorted(txt_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # L'ultima riga è convenzionalmente il timestamp di fine '<durata> end'
        segs = [l for l in lines if not l.lower().endswith("end")]
        counts[txt_file.stem] = len(segs)
    return counts


def load_agents_run(eval_processed_dir: Path, experiment_name: str) -> Dict[str, int]:
    """Carica il numero di segmenti dai file final.json della pipeline multi-agente."""
    counts = {}
    if not eval_processed_dir.is_dir():
        return counts

    for song_dir in sorted(eval_processed_dir.iterdir()):
        if not song_dir.is_dir():
            continue

        # Cerca prima in agents/<experiment_name>/final.json, poi a livello superiore
        candidates = [
            song_dir / "agents" / experiment_name / "final.json",
            song_dir / f"final_{experiment_name}.json",
        ]
        final_file = next((c for c in candidates if c.is_file()), None)
        if not final_file:
            continue

        try:
            with open(final_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            counts[song_dir.name] = len(data.get("segments", []))
        except Exception as e:
            print(f"Attenzione: impossibile leggere {final_file}: {e}", file=sys.stderr)

    return counts


def load_direct_audio_postprocess(eval_processed_dir: Path) -> Dict[str, int]:
    """Carica il conteggio segmenti post-elaborati del direct audio baseline."""
    counts = {}
    if not eval_processed_dir.is_dir():
        return counts

    for song_dir in sorted(eval_processed_dir.iterdir()):
        if not song_dir.is_dir():
            continue
        p = song_dir / "direct_audio_postprocess.json"
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                counts[song_dir.name] = len(data.get("segments", []))
            except Exception as e:
                print(f"Attenzione: impossibile leggere {p}: {e}", file=sys.stderr)

    return counts


def compute_statistics(counts: Dict[str, int]) -> Optional[Dict]:
    """Calcola statistiche descrittive per una serie di conteggi di segmenti."""
    vals = list(counts.values())
    n = len(vals)
    if n == 0:
        return None
    return {
        "n": n,
        "tot": sum(vals),
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if n > 1 else 0.0,
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def format_markdown_table(rows: List[Dict]) -> str:
    """Formatta i risultati in tabella Markdown."""
    lines = [
        "| Modello / Run di Inferenza | Configurazione / Sorgente | Tracce (N) | Segmenti Medi (Mean) | Dev. Std (σ) | Mediana | Min – Max | Totale Segmenti |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['desc']} | {r['n']} | **{r['mean']:.2f}** | "
            f"{r['std']:.2f} | {r['median']:.1f} | {r['min']} – {r['max']} | {r['tot']} |"
        )
    return "\n".join(lines)


def format_csv_table(rows: List[Dict]) -> str:
    """Formatta i risultati in tabella CSV."""
    header = "modello,configurazione,tracce,media_segmenti,dev_std,mediana,min,max,totale_segmenti"
    lines = [header]
    for r in rows:
        name_clean = r['name'].replace('**', '').replace('*', '').strip()
        lines.append(
            f"\"{name_clean}\",\"{r['desc']}\",{r['n']},{r['mean']:.4f},"
            f"{r['std']:.4f},{r['median']:.1f},{r['min']},{r['max']},{r['tot']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calcola il numero di segmenti medio e le statistiche di granularità per tutte le run di inferenza."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Percorso alla cartella eval (default: cartella dello script).",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "text"],
        default="markdown",
        help="Formato di output della tabella (default: markdown).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="File di output opzionale in cui salvare la tabella.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve() if args.eval_dir else Path(__file__).resolve().parent
    eval_processed = eval_dir / "eval_processed"

    # Definizione delle sorgenti dati
    sources = [
        {
            "name": "**Ground Truth** *(Riferimento)*",
            "desc": "HarmonixSet Human Annotations",
            "loader": lambda: load_ground_truth(eval_dir / "dataset_loaders" / "sampled_eval_dataset.jsonl"),
        },
        {
            "name": "**SongFormer**",
            "desc": "processed_results_patch.jsonl",
            "loader": lambda: load_songformer_jsonl(eval_dir / "inference_results" / "songformer" / "processed_results_patch.jsonl"),
        },
        {
            "name": "SongFormer *(44 tracce iniziali)*",
            "desc": "processed_results.jsonl",
            "loader": lambda: load_songformer_jsonl(eval_dir / "inference_results" / "songformer" / "processed_results.jsonl"),
        },
        {
            "name": "**FUTGA-7B**",
            "desc": "Zero-shot futga_txt / parsed",
            "loader": lambda: load_txt_segments(eval_dir / "inference_results" / "futga_txt"),
        },
        {
            "name": "**Flash**",
            "desc": "Pipeline Agenti (gemini-3.8-flash)",
            "loader": lambda: load_agents_run(eval_processed, "evaluation_flash_grounding_v1"),
        },
        {
            "name": "**Lite**",
            "desc": "Pipeline Agenti (gemini-3.5-flash-lite)",
            "loader": lambda: load_agents_run(eval_processed, "evaluation_flash_lite"),
        },
        {
            "name": "**Max**",
            "desc": "Pipeline Agenti (gemini-3.1-pro-preview)",
            "loader": lambda: load_agents_run(eval_processed, "evaluation_max_v1"),
        },
        {
            "name": "**Lite Direct Audio** *(Post-processed)*",
            "desc": "Direct Audio Baseline (direct_audio_txt)",
            "loader": lambda: load_txt_segments(eval_dir / "inference_results" / "direct_audio_txt"),
        },
        {
            "name": "*Lite Direct Audio (Raw final.json)*",
            "desc": "Direct Audio pre-merging adiacenze",
            "loader": lambda: load_agents_run(eval_processed, "evaluation_direct_audio_flash_lite"),
        },
    ]

    table_rows = []
    for src in sources:
        counts = src["loader"]()
        stats = compute_statistics(counts)
        if stats:
            table_rows.append({
                "name": src["name"],
                "desc": src["desc"],
                **stats
            })

    if args.format == "csv":
        out_text = format_csv_table(table_rows)
    elif args.format == "text":
        lines = [f"{r['name']:<38} | N={r['n']:<2} | Mean={r['mean']:<6.2f} | Std={r['std']:<5.2f} | Median={r['median']:<4.1f} | Min={r['min']:<2} | Max={r['max']:<2} | Tot={r['tot']}" for r in table_rows]
        out_text = "\n".join(lines)
    else:
        out_text = format_markdown_table(table_rows)

    print(out_text)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
        print(f"\nTabella salvata con successo in: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
