"""Process ORB2B PAR-CLIP data from CLIPdb.

Filters the fly CLIPdb data for ORB2B peaks, intersects strand-aware with
5'UTR and 3'UTR annotations from the Ensembl GTF to exclude CDS peaks, then
maps to FlyBase gene IDs and outputs a per-gene target table.

This is the key advantage over RIP-seq: CLIP peaks can be assigned to specific
UTR types, allowing us to exclude CDS binding and test UTR-specific predictions.

Universe: all Drosophila genes that have at least one annotated 5'UTR or 3'UTR
in the Ensembl GTF.  Genes with ORB2B peaks in those UTRs are marked as targets
(has_5utr_clip / has_3utr_clip = True).

Usage:
    python scripts/processORB2CLIPseq.py \\
        --clip_file    data/CLIPseq/fly_clip.txt.gz \\
        --gtf_file     data/downloaded/Drosophila_melanogaster-GCA_000001215.4-2022_07-genes.gtf.gz \\
        --output_dir   data/CLIPseq
"""
import argparse
import os
import sys
import warnings

import gzip
import re

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Process ORB2B PAR-CLIP peaks: UTR intersection + gene-level aggregation."
)
parser.add_argument("--clip_file", type=str, required=True,
                    help="CLIPdb fly.txt.gz (all RBPs).")
parser.add_argument("--gtf_file", type=str, required=True,
                    help="Ensembl GTF annotation (used to define 5'/3'UTR boundaries).")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory for output files.")
parser.add_argument("--rbp_name", type=str, default="ORB2B",
                    help="RBP name to filter from the CLIPdb file (default: ORB2B).")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Load and filter CLIP-seq data
# ---------------------------------------------------------------------------

print(f"Loading CLIP-seq data from {args.clip_file} ...")

# CLIPdb fly.txt format (tab-delimited, no header):
# chr  start  end  clip_id  strand  rbp_name  method  cell_line  geo_acc  score
CLIP_COLS = [
    "Chromosome", "Start", "End", "clip_id",
    "Strand", "rbp_name", "method", "cell_line", "geo_acc", "score",
]

clips_raw = pd.read_csv(
    args.clip_file,
    sep="\t",
    header=None,
    names=CLIP_COLS,
    comment="#",
    dtype={"Start": int, "End": int},
)
print(f"  Total peaks (all RBPs): {len(clips_raw)}")
print(f"  RBPs present: {sorted(clips_raw['rbp_name'].unique())}")

clips = clips_raw[clips_raw["rbp_name"] == args.rbp_name].copy()
print(f"  {args.rbp_name} peaks: {len(clips)}")

if clips.empty:
    available = sorted(clips_raw["rbp_name"].unique())
    # Try case-insensitive match
    match = [r for r in available if r.lower() == args.rbp_name.lower()]
    if match:
        args.rbp_name = match[0]
        clips = clips_raw[clips_raw["rbp_name"] == args.rbp_name].copy()
        print(f"  Using RBP name '{args.rbp_name}' (case-adjusted); {len(clips)} peaks.")
    else:
        print(
            f"ERROR: RBP '{args.rbp_name}' not found. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

# Strip "chr" prefix: CLIPdb uses "chr2L", Ensembl GTF uses "2L"
clips["Chromosome"] = clips["Chromosome"].str.replace(r"^chr", "", regex=True)

# ---------------------------------------------------------------------------
# Step 2: Parse GTF – extract 5'UTR and 3'UTR features (pure pandas, no pyranges)
# ---------------------------------------------------------------------------

print(f"\nParsing GTF from {args.gtf_file} ...")

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')
_GTF_COLS = ["Chromosome", "source", "Feature", "Start", "End",
             "score_gtf", "Strand", "frame", "attributes"]

_open = gzip.open if args.gtf_file.endswith(".gz") else open


def _parse_gtf(path):
    rows = []
    with _open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in {"five_prime_utr", "three_prime_utr"}:
                continue
            attrs = dict(_ATTR_RE.findall(fields[8]))
            rows.append({
                "Chromosome":    fields[0],
                "Start":         int(fields[3]) - 1,  # GTF is 1-based → 0-based
                "End":           int(fields[4]),
                "Strand":        fields[6],
                "Feature":       feature,
                "transcript_id": attrs.get("transcript_id", ""),
                "gene_id":       attrs.get("gene_id", ""),
            })
    return pd.DataFrame(rows)


gtf_df = _parse_gtf(args.gtf_file)
print(f"  GTF features: {dict(gtf_df['Feature'].value_counts())}")

utr5_df = gtf_df[gtf_df["Feature"] == "five_prime_utr"].copy()
utr3_df = gtf_df[gtf_df["Feature"] == "three_prime_utr"].copy()
print(f"  5'UTR intervals: {len(utr5_df)}")
print(f"  3'UTR intervals: {len(utr3_df)}")

for label, df in [("5'UTR", utr5_df), ("3'UTR", utr3_df)]:
    missing = [c for c in ("transcript_id", "gene_id") if c not in df.columns]
    if missing:
        print(f"ERROR: GTF {label} table missing columns {missing}.", file=sys.stderr)
        sys.exit(1)

# Build transcript_id → gene_id map from GTF (complete, reliable)
tr_to_gene_gtf = (
    gtf_df[gtf_df["gene_id"].notna() & gtf_df["transcript_id"].notna()]
    [["transcript_id", "gene_id"]]
    .drop_duplicates()
    .set_index("transcript_id")["gene_id"]
    .to_dict()
)
print(f"  Transcript→gene mapping: {len(tr_to_gene_gtf)} entries")

# ---------------------------------------------------------------------------
# Step 3: Strand-aware interval overlap (CLIP peaks vs UTRs)
# ---------------------------------------------------------------------------


def find_clip_utr_overlaps(clips_df, utrs_df, utr_label):
    """Return DataFrame of CLIP-peak × UTR overlaps (strand-aware).

    Uses numpy broadcasting per (chromosome, strand) group for efficiency.
    Coordinates are 0-based half-open (BED convention) for both inputs.
    """
    print(f"  Computing {utr_label} overlaps ...")
    batches = []

    for (chrom, strand), utr_group in utrs_df.groupby(["Chromosome", "Strand"]):
        clip_group = clips_df[
            (clips_df["Chromosome"] == chrom) &
            (clips_df["Strand"] == strand)
        ]
        if clip_group.empty:
            continue

        c_starts = clip_group["Start"].values[:, np.newaxis]   # (n_clips, 1)
        c_ends   = clip_group["End"].values[:, np.newaxis]
        u_starts = utr_group["Start"].values[np.newaxis, :]    # (1, n_utrs)
        u_ends   = utr_group["End"].values[np.newaxis, :]

        # Standard BED half-open overlap: A.start < B.end AND A.end > B.start
        mask = (c_starts < u_ends) & (c_ends > u_starts)      # (n_clips, n_utrs)
        clip_idxs, utr_idxs = np.where(mask)

        if clip_idxs.size == 0:
            continue

        clip_subset = clip_group.iloc[clip_idxs].reset_index(drop=True)
        utr_subset  = utr_group.iloc[utr_idxs].reset_index(drop=True)

        batch = pd.DataFrame({
            "Chromosome":    chrom,
            "Strand":        strand,
            "clip_id":       clip_subset["clip_id"].values,
            "clip_start":    clip_subset["Start"].values,
            "clip_end":      clip_subset["End"].values,
            "transcript_id": utr_subset["transcript_id"].values,
            "gene_id":       utr_subset["gene_id"].values,
        })
        batches.append(batch)

    if not batches:
        return pd.DataFrame(
            columns=["Chromosome", "Strand", "clip_id",
                     "clip_start", "clip_end", "transcript_id", "gene_id"]
        )
    return pd.concat(batches, ignore_index=True)


print(f"\nFinding strand-aware overlaps for {args.rbp_name} peaks ...")
hits5 = find_clip_utr_overlaps(clips, utr5_df, "5'UTR")
hits3 = find_clip_utr_overlaps(clips, utr3_df, "3'UTR")

print(f"  5'UTR overlaps: {len(hits5)} peak×UTR pairs, "
      f"{hits5['gene_id'].nunique() if not hits5.empty else 0} unique genes")
print(f"  3'UTR overlaps: {len(hits3)} peak×UTR pairs, "
      f"{hits3['gene_id'].nunique() if not hits3.empty else 0} unique genes")

# ---------------------------------------------------------------------------
# Step 4: Aggregate to gene level
# ---------------------------------------------------------------------------

print("\nAggregating to gene level ...")


def gene_level_hits(hits_df):
    if hits_df.empty:
        return pd.DataFrame(columns=["gene_id", "n_peaks"])
    return (
        hits_df.groupby("gene_id")["clip_id"]
        .nunique()
        .reset_index()
        .rename(columns={"clip_id": "n_peaks"})
    )


genes5 = gene_level_hits(hits5).rename(columns={"n_peaks": "n_5utr_peaks"})
genes3 = gene_level_hits(hits3).rename(columns={"n_peaks": "n_3utr_peaks"})

# Universe: all genes with at least one annotated 5'UTR or 3'UTR
all_utr_genes = pd.DataFrame(
    {"gene_id": list(
        set(utr5_df["gene_id"].unique()) | set(utr3_df["gene_id"].unique())
    )}
)
print(f"  Universe (genes with ≥1 UTR annotation): {len(all_utr_genes)}")

result = all_utr_genes.copy()
result = result.merge(genes5, on="gene_id", how="left")
result = result.merge(genes3, on="gene_id", how="left")
result["n_5utr_peaks"] = result["n_5utr_peaks"].fillna(0).astype(int)
result["n_3utr_peaks"] = result["n_3utr_peaks"].fillna(0).astype(int)
result["has_5utr_clip"] = result["n_5utr_peaks"] > 0
result["has_3utr_clip"] = result["n_3utr_peaks"] > 0

result = result.rename(columns={"gene_id": "FBgn_ID"})
result = result[["FBgn_ID", "has_5utr_clip", "has_3utr_clip",
                 "n_5utr_peaks", "n_3utr_peaks"]]

# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

out_path = os.path.join(args.output_dir, "orb2b_targets.csv")
result.to_csv(out_path, index=False)

n_pos5  = result["has_5utr_clip"].sum()
n_pos3  = result["has_3utr_clip"].sum()
n_any   = (result["has_5utr_clip"] | result["has_3utr_clip"]).sum()
print(f"\nSaved {len(result)} genes → {out_path}")
print(f"  Genes with 5'UTR {args.rbp_name} peaks: {n_pos5}")
print(f"  Genes with 3'UTR {args.rbp_name} peaks: {n_pos3}")
print(f"  Genes with any UTR {args.rbp_name} peak: {n_any}")
print("Done.")
