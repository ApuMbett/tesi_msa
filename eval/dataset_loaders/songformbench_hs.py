import os
from huggingface_hub import hf_hub_download
import pandas as pd
from dotenv import load_dotenv

# Construct path relative to this script's directory to reliably find .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(env_path)

DATASET_SIZE = int(os.getenv("DATASET_SIZE", 50))
IN_DOMAIN_PERCENTAGE = int(os.getenv("DATASET_IN_DOMAIN_PERCENTAGE", 50))
MIDDLE_GROUND_PERCENTAGE = int(os.getenv("DATASET_MIDDLE_GROUND_PERCENTAGE", 30))
OUT_OF_DOMAIN_PERCENTAGE = int(os.getenv("DATASET_OUT_OF_DOMAIN_PERCENTAGE", 20))
SEED = int(os.getenv("SEED", 42))

IN_DOMAIN_TARGET = int(DATASET_SIZE * IN_DOMAIN_PERCENTAGE / 100)
MIDDLE_GROUND_TARGET = int(DATASET_SIZE * MIDDLE_GROUND_PERCENTAGE / 100)
OUT_OF_DOMAIN_TARGET = int(DATASET_SIZE * OUT_OF_DOMAIN_PERCENTAGE / 100)

# ==============================================================================
# 3 Partitions of SongFormBench-HS genres
# ==============================================================================

IN_DOMAIN_GENRE = [
    # Strictly quantized 4/4 grid with standard 4-bar (16-beat) phrasing; native target environment of the DSP pipeline.
    "Dance/Electronic",
]

MIDDLE_GROUND_GENRE = [
    # Maintains standard 4/4 meter and 16-beat phrase regularity, but introduces acoustic/organic vocals and varied production styles.
    "Pop",
    # Rigid 4/4 drum machine/beat loops, though acoustic samples and dense vocal delivery add timbral complexity
    "Hip-Hop",
    # Stable 4/4 groove, with increased vocal phrasing variance (melisma, ad-libs) over steady rhythmic foundations
    "R&B",
    # Continuous, quantized 4/4 dembow rhythmic framework that directly fits fixed-token metric subdivisions
    "Reggaeton",
    # 4/4 four-on-the-floor kick patterns and steady tempos; direct structural predecessor to modern electronic house music.
    "Funk/Disco",
    # Radio-oriented, rigid 4/4 verse-chorus forms with standard tempo stability, combining guitars with clean percussive grids
    "Pop-Rock",
    # Retains a standard 4/4 grid while introducing non-synthetic timbres like guitars and acoustic drum kits
    "Alternative",
]

OUT_OF_DOMAIN_GENRE = [
    # Frequently breaks standard 4/4 meters with odd time signatures (5/4, 7/8), shifting tempos, and non-16-beat structural phrasing
    "Prog",
    # Heavy use of polyrhythms, rapid double-bass drumming, and extreme dynamic shifts that disrupt standard onset and energy detectors
    "Metal",
    # Played with live human rhythm sections subject to natural human tempo drift, loose swing, and unquantized timing
    "Rock",
    # Analog live instrumentation with loose phrasing and tempo variance, deviating from rigid digital grid snapping
    "Classic Rock",
    # Strong dynamic contrasts (quiet-loud) and live, unquantized rhythm sections with expressive timing irregularities
    "Grunge",
    # Rooted in 12-bar structures (48 beats rather than clean 16-beat tokens), shuffle/triplet swing, and expressive tempo variations
    "Blues",
    # Common usage of 2/4 and 3/4 (waltz) meters, acoustic syncopation, and non-standard phrasing outside the fixed electronic grid
    "Country",
]

# Partitioning Conditions
IN_DOMAIN_CONDITION = lambda df: df["Genre"].isin(IN_DOMAIN_GENRE) & (df["Time Signature"] == "4|4")
MIDDLE_GROUND_CONDITION = lambda df: df["Genre"].isin(MIDDLE_GROUND_GENRE) & (df["Time Signature"] == "4|4")
OUT_OF_DOMAIN_CONDITION = lambda df: (
    df["Genre"].isin(OUT_OF_DOMAIN_GENRE) | 
    (df["Time Signature"].notna() & (df["Time Signature"] != "4|4"))
)
def load_raw_songformbench():
    """Downloads and loads the raw SongFormBench.jsonl dataset from Hugging Face."""
    annotations_path = hf_hub_download(
        repo_id="ASLP-lab/SongFormBench",
        repo_type="dataset",
        filename="data/SongFormBench.jsonl",
        local_dir="./downloaded_data"
    )
    return pd.read_json(annotations_path, lines=True)


def get_enriched_hs_annotations():
    """
    Downloads and merges HarmonixSet metadata with the SongFormBench HarmonixSet split.
    """
    # 1. Load datasets
    # harmonixset_metadata.csv downloaded from https://github.com/urinieto/harmonixset/blob/main/dataset/metadata.csv
    df_hs = pd.read_csv("./downloaded_data/data/harmonixset_metadata.csv")

    df_songformbench_hs = load_raw_songformbench()

    # Filter HarmonixSet subset and sanitize ID
    df_songformbench_hs = df_songformbench_hs[df_songformbench_hs["subset"] == "HarmonixSet"]
    df_songformbench_hs["id"] = df_songformbench_hs["id"].str.replace("BHX_", "")

    # 2. Outer Join on track ID
    merged_df = pd.merge(
        df_hs, 
        df_songformbench_hs, 
        left_on="File", 
        right_on="id", 
        how="right"
    )

    # 3. Clean redundant columns
    combined_id = merged_df["File"].fillna(merged_df["id"])
    merged_df = merged_df.drop(columns=["File", "id"])
    merged_df.insert(0, "id", combined_id)
    merged_df = merged_df.drop(columns=["Duration", "subset"], errors="ignore")

    # 4. Save
    merged_df.to_csv("./SongFormBench_HS_enriched.csv", index=False)
    return merged_df


def build_final_dataset(df, seed=SEED):
    """
    Builds the evaluation set strictly from HarmonixSet according to target quotas:
      - In-Domain: 4/4 Dance/Electronic tracks.
      - Middle-Ground: 4/4 Pop, Hip-Hop, R&B, etc.
      - Out-of-Domain: ALL non-4/4 tracks guaranteed, remainder sampled from 4/4 Out-of-Domain genres.
    """
    # 1. In-Domain Pool
    in_pool = df[IN_DOMAIN_CONDITION(df)]
    if len(in_pool) < IN_DOMAIN_TARGET:
        raise ValueError(f"Requested {IN_DOMAIN_TARGET} In-Domain tracks, but only {len(in_pool)} available.")
    in_sampled = in_pool.sample(n=IN_DOMAIN_TARGET, random_state=seed).copy()
    in_sampled["split"] = "in_domain"

    # 2. Middle-Ground Pool (prevent picking tracks already in in_domain sample)
    mg_pool = df[MIDDLE_GROUND_CONDITION(df) & ~df["id"].isin(in_sampled["id"])]
    if len(mg_pool) < MIDDLE_GROUND_TARGET:
        raise ValueError(f"Requested {MIDDLE_GROUND_TARGET} Middle-Ground tracks, but only {len(mg_pool)} available.")
    mg_sampled = mg_pool.sample(n=MIDDLE_GROUND_TARGET, random_state=seed).copy()
    mg_sampled["split"] = "middle_ground"

    # 3. Out-of-Domain Pool
    # Step 3a: Extract all non-4/4 songs (guaranteed inclusion becaure there are only 5)
    non_4_4_tracks = df[df["Time Signature"].notna() & (df["Time Signature"] != "4|4")].copy()
    num_guaranteed = len(non_4_4_tracks)

    if num_guaranteed > OUT_OF_DOMAIN_TARGET:
        raise ValueError(
            f"Found {num_guaranteed} non-4/4 tracks, exceeding the Out-of-Domain target of {OUT_OF_DOMAIN_TARGET}."
        )

    # Step 3b: Sample the remainder strictly from 4/4 Out-of-Domain genres
    remaining_needed = OUT_OF_DOMAIN_TARGET - num_guaranteed
    used_ids = set(in_sampled["id"]).union(set(mg_sampled["id"])).union(set(non_4_4_tracks["id"]))

    ood_4_4_pool = df[
        df["Genre"].isin(OUT_OF_DOMAIN_GENRE) & 
        (df["Time Signature"] == "4|4") & 
        ~df["id"].isin(used_ids)
    ]

    if len(ood_4_4_pool) < remaining_needed:
        raise ValueError(
            f"Need {remaining_needed} additional 4/4 Out-of-Domain tracks, but only {len(ood_4_4_pool)} available."
        )

    ood_sampled_remainder = ood_4_4_pool.sample(n=remaining_needed, random_state=seed)

    # Combine guaranteed non-4/4 tracks with sampled 4/4 out-of-domain tracks
    ood_sampled = pd.concat([non_4_4_tracks, ood_sampled_remainder]).copy()
    ood_sampled["split"] = "out_of_domain"

    # 4. Concatenate all partitions
    final_dataset = pd.concat([in_sampled, mg_sampled, ood_sampled]).reset_index(drop=True)
    return final_dataset


def map_final_dataset(df):
    """
    Maps the sampled dataset back to the original unmodified SongFormBench format.
    """
    df_songformbench = load_raw_songformbench()
    df_songformbench["match_id"] = df_songformbench["id"].str.replace("BHX_", "")

    merge_cols = ["id"]
    if "split" in df.columns:
        merge_cols.append("split")

    mapped_df = df[merge_cols].merge(df_songformbench, left_on="id", right_on="match_id", how="left")
    mapped_df = mapped_df.drop(columns=["id_x", "match_id"]).rename(columns={"id_y": "id"})
    
    # Keep id as the first column
    cols = ["id"] + [col for col in mapped_df.columns if col != "id"]
    return mapped_df[cols]


def verify_dataset_integrity(df_original, final_df, mapped_df):
    """Runs strict sanity checks and assertions on the sampled and mapped datasets."""

    # 1. Check Exact Target Sizes
    assert len(final_df) == DATASET_SIZE, (
        f"Expected {DATASET_SIZE} songs in final_df, but got {len(final_df)}"
    )
    assert len(mapped_df) == DATASET_SIZE, (
        f"Expected {DATASET_SIZE} songs in mapped_df, but got {len(mapped_df)}"
    )

    # 2. Check Partition Targets
    in_count = IN_DOMAIN_CONDITION(final_df).sum()
    mg_count = MIDDLE_GROUND_CONDITION(final_df).sum()
    ood_count = OUT_OF_DOMAIN_CONDITION(final_df).sum()

    assert in_count == IN_DOMAIN_TARGET, (
        f"In-Domain target mismatch: {in_count}/{IN_DOMAIN_TARGET}"
    )
    assert mg_count == MIDDLE_GROUND_TARGET, (
        f"Middle-Ground target mismatch: {mg_count}/{MIDDLE_GROUND_TARGET}"
    )
    assert ood_count == OUT_OF_DOMAIN_TARGET, (
        f"Out-of-Domain target mismatch: {ood_count}/{OUT_OF_DOMAIN_TARGET}"
    )

    # 3. Check for No Duplicates
    assert final_df["id"].nunique() == DATASET_SIZE, (
        "Duplicate IDs found in final_df!"
    )
    assert mapped_df["id"].nunique() == DATASET_SIZE, (
        "Duplicate IDs found in mapped_df!"
    )

    # 4. Check Complete and Mutual Exclusivity Across Partitions
    in_mask = IN_DOMAIN_CONDITION(final_df)
    mg_mask = MIDDLE_GROUND_CONDITION(final_df)
    ood_mask = OUT_OF_DOMAIN_CONDITION(final_df)

    assert not (in_mask & mg_mask).any(), (
        "Partition overlap detected between In-Domain and Middle-Ground!"
    )
    assert not (in_mask & ood_mask).any(), (
        "Partition overlap detected between In-Domain and Out-of-Domain!"
    )
    assert not (mg_mask & ood_mask).any(), (
        "Partition overlap detected between Middle-Ground and Out-of-Domain!"
    )
    assert (in_mask | mg_mask | ood_mask).all(), (
        "Unassigned tracks found: some tracks do not belong to any partition!"
    )

    # 5. Check All 5 Non-4/4 Tracks Are Present
    original_non_4_4_ids = set(
        df_original[
            df_original["Time Signature"].notna()
            & (df_original["Time Signature"] != "4|4")
        ]["id"]
    )
    sampled_non_4_4_ids = set(
        final_df[
            final_df["Time Signature"].notna()
            & (final_df["Time Signature"] != "4|4")
        ]["id"]
    )

    assert original_non_4_4_ids == sampled_non_4_4_ids, (
        f"Not all non-4/4 tracks were captured! Missing IDs: {original_non_4_4_ids - sampled_non_4_4_ids}"
    )
    assert len(sampled_non_4_4_ids) == 5, (
        f"Expected exactly 5 non-4/4 tracks, got {len(sampled_non_4_4_ids)}"
    )

    # 6. Check Id Alignment Between Sampled and Mapped Datasets
    # SongFormBench IDs carry the prefix 'BHX_'
    expected_mapped_ids = ["BHX_" + str(track_id) for track_id in final_df["id"]]
    assert list(mapped_df["id"]) == expected_mapped_ids, (
        "Mismatch or row order misalignment between final_df and mapped_df!"
    )

    # 7. Check Mapped Dataset Integrity (No NaN columns from failed merges)
    assert not mapped_df["id"].isna().any(), (
        "Null IDs detected in mapped_df!"
    )
    assert not mapped_df["labels"].isna().any(), (
        "Failed merge: missing 'labels' in mapped_df!"
    )

    # 8. Check Split Labels
    assert "split" in final_df.columns and not final_df["split"].isna().any(), (
        "Missing or null 'split' column in final_df!"
    )
    assert "split" in mapped_df.columns and not mapped_df["split"].isna().any(), (
        "Missing or null 'split' column in mapped_df!"
    )

    print("All assertions passed successfully!")


def get_MSA_labels(df):

    for _, row in df.iterrows():
        hf_hub_download(
            repo_id="ASLP-lab/SongFormBench",
            repo_type="dataset",
            filename=row["label_path"],
            local_dir="./downloaded_data/sampled_labels"
        )

if __name__ == "__main__":
    enriched_path = "./SongFormBench_HS_enriched.csv"
    if not os.path.exists(enriched_path):
        df = get_enriched_hs_annotations()
    else:
        df = pd.read_csv(enriched_path)

    print("==================================================")
    print(f"Total Available Enriched Tracks: {len(df)}")
    print(f"In-Domain Pool: {IN_DOMAIN_CONDITION(df).sum()} available (Target: {IN_DOMAIN_TARGET})")
    print(f"Middle-Ground Pool: {MIDDLE_GROUND_CONDITION(df).sum()} available (Target: {MIDDLE_GROUND_TARGET})")
    print(f"Out-of-Domain Pool: {OUT_OF_DOMAIN_CONDITION(df).sum()} available (Target: {OUT_OF_DOMAIN_TARGET})")
    print("==================================================")

    final_df = build_final_dataset(df, seed=SEED)

    # Verification checks
    print(f"\nSampled Dataset Size: {len(final_df)}")
    print(f" - In-Domain: {IN_DOMAIN_CONDITION(final_df).sum()}/{IN_DOMAIN_TARGET}")
    print(f" - Middle-Ground: {MIDDLE_GROUND_CONDITION(final_df).sum()}/{MIDDLE_GROUND_TARGET}")
    print(f" - Out-of-Domain: {OUT_OF_DOMAIN_CONDITION(final_df).sum()}/{OUT_OF_DOMAIN_TARGET}")

    # Verify all 5 non-4/4 tracks are present
    non_4_4_in_sample = final_df[final_df["Time Signature"].notna() & (final_df["Time Signature"] != "4|4")]
    print(f" - Guaranteed Non-4/4 Tracks Captured: {len(non_4_4_in_sample)}/5")

    mapped_dataset = map_final_dataset(final_df)
    print(f"\nFinal Mapped Dataset Size: {len(mapped_dataset)}")
    # Run the full validation suite
    verify_dataset_integrity(df_original=df, final_df=final_df, mapped_df=mapped_dataset)
    # Save output
    mapped_dataset.to_json("./sampled_eval_dataset_tagged.jsonl", orient="records", lines=True)
    print("Saved final sampled dataset to ./sampled_eval_dataset_tagged.jsonl")

    # save enriched sampled dataset (shortcut for batch processing)
    final_df.to_csv("./sampled_eval_dataset_enriched_tagged.csv", index=False)
    print("Saved enriched sampled dataset to ./sampled_eval_dataset_enriched_tagged.csv")

    get_MSA_labels(mapped_dataset)
    
