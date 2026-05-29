"""Validate ORB2 SHAP+MAST predictions against PAR-CLIP experimental data (GSE59611).

ORB2 appears in the "Neg Both" category in the SHAP analysis, meaning its binding
motifs are associated with *decreasing* ribosome density.  Therefore this script
uses the **neg5** and **neg3** parquets (low-SHAP / negative-SHAP windows) rather
than the pos5/pos3 parquets used for FMR1.

Key advantage over RIP-seq validation: PAR-CLIP peaks can be assigned to specific
UTR types (5'UTR vs 3'UTR), so CDS-binding peaks are excluded.  This aligns
directly with the UTR-based SHAP/MAST framework used by the model.

Three gene sets are defined:
  Group A – no ORB2 MAST hit in any UTR
  Group B – ORB2 MAST hit(s) present; none overlap low-SHAP (high-negative-SHAP) regions
  Group C – ORB2 MAST hit(s) present AND at least one overlaps a low-SHAP region
             (model predicts ORB2-mediated translational suppression)

Fisher's exact test + odds ratio quantify how much SHAP enrichment (C vs A+B)
improves on the baseline MAST-only signal (B+C vs A).
Additional sub-tests: 5'UTR-only CLIP targets and 3'UTR-only CLIP targets.

Outputs
-------
  <output_dir>/orb2_clip_validation_gene_table.csv  – per-gene group labels + CLIP truth
  <output_dir>/orb2_clip_validation_stats.json      – Fisher OR, p-values, precision, recall
  <output_dir>/orb2_clip_validation_barplot.svg     – stacked bar: CLIP-target fraction per group
  <output_dir>/orb2_clip_validation_upset.svg       – UpSet / Venn diagram

Usage:
    python scripts/validateORB2withCLIPseq.py \\
        --mast_results output/mast_out/train_mast_results.txt \\
                       output/mast_out/dev_mast_results.txt \\
                       output/mast_out/test_mast_results.txt \\
        --shap_neg5    output/motifEnrichment/neg5.parquet \\
        --shap_neg3    output/motifEnrichment/neg3.parquet \\
        --clip_targets data/CLIPseq/orb2b_targets.csv \\
        --output_dir   output/ripseqValidation/orb2_clip
"""
import argparse
import gzip
import json
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Validate ORB2 SHAP+MAST predictions against PAR-CLIP data (GSE59611)."
)
parser.add_argument("--mast_results", nargs="+", required=True,
                    help="MAST result files (train/dev/test).")
parser.add_argument("--shap_neg5", type=str, required=True,
                    help="Parquet: per-region RBP hit counts in low-SHAP (negative) 5'UTR windows.")
parser.add_argument("--shap_neg3", type=str, required=True,
                    help="Parquet: per-region RBP hit counts in low-SHAP (negative) 3'UTR windows.")
parser.add_argument("--clip_targets", type=str, required=True,
                    help="CSV output from processORB2CLIPseq.py (orb2b_targets.csv).")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory for output files.")
parser.add_argument("--rbp_name", type=str, default="ORB2",
                    help="Name of the ORB2 RBP in the MAST alt_id column (default: ORB2).")
parser.add_argument("--clip_file", type=str, default=None,
                    help="CLIPdb fly.txt.gz (raw peaks). Enables position-level analysis.")
parser.add_argument("--gtf_file", type=str, default=None,
                    help="Ensembl GTF (.gtf.gz). Required for position-level analysis.")
parser.add_argument("--neg3_motif_positions", type=str, default=None,
                    help="motif_positions.txt (train/negative) from findMotifs. Used to convert "
                         "the neg3 parquet index (combined-sequence coords) to 3'UTR-local coords.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Load and combine MAST results across all splits
# ---------------------------------------------------------------------------

print("Loading MAST results ...")
mast_frames = []
for path in args.mast_results:
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.", file=sys.stderr)
        continue
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, skipfooter=1, engine="python")
    df.columns = ["sequence_name", "strand", "id", "alt_id",
                  "hit_start", "hit_end", "score", "p_value"]
    mast_frames.append(df)

if not mast_frames:
    print("ERROR: No MAST result files could be loaded.", file=sys.stderr)
    sys.exit(1)

mast = pd.concat(mast_frames, ignore_index=True)
mast["trID"] = mast["sequence_name"].str.split("_").str[0]
mast = mast.drop_duplicates(subset=["sequence_name", "alt_id", "hit_start", "hit_end"])

print(f"  {len(mast)} MAST hits across {mast['trID'].nunique()} transcripts")
print(f"  RBPs present: {sorted(mast['alt_id'].unique())[:10]} ...")

# ---------------------------------------------------------------------------
# Step 2: Build per-transcript ORB2 MAST hit flag (any UTR)
# ---------------------------------------------------------------------------

rbp = args.rbp_name

if rbp not in mast["alt_id"].values:
    matches = [x for x in mast["alt_id"].unique() if x.lower() == rbp.lower()]
    if matches:
        rbp = matches[0]
        print(f"  Using RBP name '{rbp}' (case-adjusted).")
    else:
        available = sorted(mast["alt_id"].unique())
        print(
            f"ERROR: ORB2 RBP name '{args.rbp_name}' not found in MAST results.\n"
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

orb2_mast = mast[mast["alt_id"] == rbp]
transcripts_with_orb2_mast = set(orb2_mast["trID"].unique())
print(f"  Transcripts with any ORB2 MAST hit: {len(transcripts_with_orb2_mast)}")

# ---------------------------------------------------------------------------
# Step 3: Load SHAP parquet files — low-SHAP (negative) windows
# ---------------------------------------------------------------------------

print("\nLoading neg-SHAP parquet files ...")


def transcripts_with_shap_orb2(parquet_path, rbp_name):
    """Return set of transcript IDs with >=1 ORB2 MAST hit in a low-SHAP region.

    The parquet index is 'trID_start-end'; extract trID by splitting on '_'.
    """
    if not os.path.exists(parquet_path):
        print(f"  WARNING: parquet not found: {parquet_path}", file=sys.stderr)
        return set()
    df = pd.read_parquet(parquet_path)
    if rbp_name not in df.columns:
        matches = [c for c in df.columns if c.lower() == rbp_name.lower()]
        if not matches:
            print(
                f"  WARNING: '{rbp_name}' not in columns of {parquet_path}. "
                f"Available: {list(df.columns)[:10]}",
                file=sys.stderr,
            )
            return set()
        rbp_name = matches[0]
    positive_rows = df[df[rbp_name] > 0]
    tr_ids = positive_rows.index.str.split("_").str[0]
    return set(tr_ids)


shap_orb2_5utr = transcripts_with_shap_orb2(args.shap_neg5, rbp)
shap_orb2_3utr = transcripts_with_shap_orb2(args.shap_neg3, rbp)
transcripts_with_shap_orb2_hits = shap_orb2_5utr | shap_orb2_3utr
print(f"  Transcripts with ORB2 hits in low-SHAP 5'UTR: {len(shap_orb2_5utr)}")
print(f"  Transcripts with ORB2 hits in low-SHAP 3'UTR: {len(shap_orb2_3utr)}")
print(f"  Union (Group C eligible): {len(transcripts_with_shap_orb2_hits)}")

# ---------------------------------------------------------------------------
# Step 4: Universe of all transcripts in the model
# ---------------------------------------------------------------------------

all_transcripts = set(mast["trID"].unique())
print(f"\nTotal unique transcripts in model universe: {len(all_transcripts)}")

# ---------------------------------------------------------------------------
# Step 5: Map transcripts to FlyBase gene IDs
# ---------------------------------------------------------------------------

print("\nBuilding transcript -> gene ID map ...")


def build_tr_to_gene_from_fastas():
    """Scan pipeline fasta files for FBtr + FBgn header tokens."""
    tr_to_gene = {}
    candidate_paths = (
        [os.path.join("output/data/flyUTRs", f"{s}.fasta") for s in ["train", "dev", "test"]]
        + ["output/data/utrTE.fasta"]
        + [os.path.join("output/data", f"{s}.fasta") for s in ["train", "dev", "test"]]
    )
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                if not line.startswith(">"):
                    continue
                parts = line.strip().split()
                fbtr = next((p.lstrip(">") for p in parts if p.lstrip(">").startswith("FBtr")), None)
                fbgn = next((p for p in parts if p.startswith("FBgn")), None)
                if fbtr and fbgn:
                    tr_to_gene[fbtr] = fbgn
        if tr_to_gene:
            print(f"  Built tr->gene map from {path}: {len(tr_to_gene)} entries")
            break
    return tr_to_gene


gene_map = build_tr_to_gene_from_fastas()

# ---------------------------------------------------------------------------
# Step 6: Load CLIP-seq target list
# ---------------------------------------------------------------------------

print(f"\nLoading ORB2B PAR-CLIP target list from {args.clip_targets} ...")
clip_df = pd.read_csv(args.clip_targets)
print(f"  Columns: {list(clip_df.columns)}")

required = {"FBgn_ID", "has_5utr_clip", "has_3utr_clip"}
if not required.issubset(clip_df.columns):
    print(
        f"ERROR: CLIP target file missing required columns {required}.\n"
        f"  Found: {list(clip_df.columns)}",
        file=sys.stderr,
    )
    sys.exit(1)

clip_df = clip_df.set_index("FBgn_ID")
clip_df["has_5utr_clip"] = clip_df["has_5utr_clip"].astype(bool)
clip_df["has_3utr_clip"] = clip_df["has_3utr_clip"].astype(bool)
clip_df["is_clip_target"] = clip_df["has_5utr_clip"] | clip_df["has_3utr_clip"]

clip_targets_any  = set(clip_df[clip_df["is_clip_target"]].index)
clip_targets_5utr = set(clip_df[clip_df["has_5utr_clip"]].index)
clip_targets_3utr = set(clip_df[clip_df["has_3utr_clip"]].index)

print(f"  {len(clip_df)} genes in CLIP universe (have UTR annotations)")
print(f"  ORB2B 5'UTR CLIP targets: {len(clip_targets_5utr)}")
print(f"  ORB2B 3'UTR CLIP targets: {len(clip_targets_3utr)}")
print(f"  ORB2B any-UTR CLIP targets (primary truth): {len(clip_targets_any)}")

# ---------------------------------------------------------------------------
# Step 7: Assign transcripts to groups – combined and UTR-specific
# ---------------------------------------------------------------------------

PRIORITY = {"C": 2, "B": 1, "A": 0}


def _assign_groups(label, shap_hits):
    """Return gene-level group DataFrame for a given set of SHAP-hit transcripts."""
    print(f"\nAssigning transcripts to groups A / B / C ({label}) ...")
    records = []
    for tr in all_transcripts:
        has_mast = tr in transcripts_with_orb2_mast
        has_shap = tr in shap_hits
        if not has_mast:
            group = "A"
        elif has_mast and not has_shap:
            group = "B"
        else:
            group = "C"
        records.append({"trID": tr, "geneID": gene_map.get(tr, tr), "group": group})
    tr_df = pd.DataFrame(records)
    print(f"  A (no MAST hit):         {(tr_df['group'] == 'A').sum()}")
    print(f"  B (MAST hit, no SHAP):   {(tr_df['group'] == 'B').sum()}")
    print(f"  C (MAST + SHAP overlap): {(tr_df['group'] == 'C').sum()}")
    gg = (
        tr_df.groupby("geneID")["group"]
        .apply(lambda gs: max(gs, key=lambda g: PRIORITY[g]))
        .reset_index()
        .rename(columns={"group": "gene_group"})
    )
    print(f"  Gene-level – A: {(gg['gene_group'] == 'A').sum()}, "
          f"B: {(gg['gene_group'] == 'B').sum()}, "
          f"C: {(gg['gene_group'] == 'C').sum()}")
    return gg


# Combined (either UTR)
gene_group = _assign_groups("any UTR", transcripts_with_shap_orb2_hits)
# 5'UTR-specific
gene_group_5 = _assign_groups("5'UTR only", shap_orb2_5utr)
# 3'UTR-specific
gene_group_3 = _assign_groups("3'UTR only", shap_orb2_3utr)

# ---------------------------------------------------------------------------
# Step 8: Merge with CLIP-seq truth (three overlaps: combined, 5', 3')
# ---------------------------------------------------------------------------

genes_in_clip = set(clip_df.index)


def _make_overlap(gg, target_cols):
    gg = gg.copy()
    gg["in_clip_universe"] = gg["geneID"].isin(genes_in_clip)
    ov = gg[gg["in_clip_universe"]].copy()
    for col, gene_set in target_cols.items():
        ov[col] = ov["geneID"].isin(gene_set)
    return ov


overlap = _make_overlap(
    gene_group,
    {"is_clip_target": clip_targets_any,
     "has_5utr_clip": clip_targets_5utr,
     "has_3utr_clip": clip_targets_3utr},
)
overlap5 = _make_overlap(
    gene_group_5,
    {"has_5utr_clip": clip_targets_5utr},
)
overlap3 = _make_overlap(
    gene_group_3,
    {"has_3utr_clip": clip_targets_3utr},
)

print(f"\nGenes in both model and CLIP universe: {len(overlap)}")
print(f"  Combined groups – A: {(overlap['gene_group'] == 'A').sum()}, "
      f"B: {(overlap['gene_group'] == 'B').sum()}, "
      f"C: {(overlap['gene_group'] == 'C').sum()}")
print(f"  CLIP-confirmed targets (any UTR): {overlap['is_clip_target'].sum()}")
print(f"  5'UTR-specific groups – A: {(overlap5['gene_group'] == 'A').sum()}, "
      f"B: {(overlap5['gene_group'] == 'B').sum()}, "
      f"C: {(overlap5['gene_group'] == 'C').sum()}")
print(f"  CLIP-confirmed 5'UTR targets: {overlap5['has_5utr_clip'].sum()}")
print(f"  3'UTR-specific groups – A: {(overlap3['gene_group'] == 'A').sum()}, "
      f"B: {(overlap3['gene_group'] == 'B').sum()}, "
      f"C: {(overlap3['gene_group'] == 'C').sum()}")
print(f"  CLIP-confirmed 3'UTR targets: {overlap3['has_3utr_clip'].sum()}")

# ---------------------------------------------------------------------------
# Step 9: Fisher's exact tests
# ---------------------------------------------------------------------------


def contingency_and_fisher(df, group_pos, group_neg_list, target_col, label):
    pos = df[df["gene_group"] == group_pos]
    neg = df[df["gene_group"].isin(group_neg_list)]
    a = int(pos[target_col].sum())
    b = int(len(pos) - a)
    c = int(neg[target_col].sum())
    d = int(len(neg) - c)
    table = np.array([[a, b], [c, d]])
    or_, p = fisher_exact(table, alternative="greater")
    total_targets = int(df[target_col].sum())
    prec   = a / len(pos) if len(pos) > 0 else float("nan")
    recall = a / total_targets if total_targets > 0 else float("nan")
    print(
        f"\n  Fisher {label}: OR={or_:.3f}, p={p:.4g}, "
        f"precision={prec:.3f}, recall={recall:.3f}\n"
        f"  Contingency [[{a},{b}],[{c},{d}]]"
    )
    return {
        "comparison":    label,
        "target_column": target_col,
        "group_pos":     group_pos,
        "group_neg":     "+".join(group_neg_list),
        "n_pos":         int(len(pos)),
        "n_neg":         int(len(neg)),
        "pos_clip_targets": a,
        "neg_clip_targets": c,
        "odds_ratio":    float(or_),
        "p_value":       float(p),
        "precision":     float(prec),
        "recall":        float(recall),
        "contingency":   table.tolist(),
    }


print("\nStatistical tests (primary: any UTR CLIP target):")
stats = []

# Primary: any UTR (combined grouping)
stats.append(contingency_and_fisher(
    overlap, "C", ["B"], "is_clip_target",
    "C_vs_B (SHAP-informed vs MAST-only, any-UTR CLIP)",
))

all_mast_hits = overlap[overlap["gene_group"].isin(["B", "C"])].copy()
all_mast_hits["gene_group"] = "B+C"
none_mast = overlap[overlap["gene_group"] == "A"].copy()
baseline_df = pd.concat([all_mast_hits, none_mast])
stats.append(contingency_and_fisher(
    baseline_df, "B+C", ["A"], "is_clip_target",
    "BC_vs_A (any MAST hit vs no MAST, any-UTR CLIP)",
))
stats.append(contingency_and_fisher(
    overlap, "B", ["A"], "is_clip_target",
    "B_vs_A (MAST only vs no MAST, any-UTR CLIP)",
))

# 5'UTR-specific: grouping based on 5'UTR SHAP only, truth = 5'UTR CLIP
print("\nSub-tests (5'UTR SHAP grouping vs 5'UTR CLIP truth):")
if overlap5["has_5utr_clip"].sum() > 0:
    stats.append(contingency_and_fisher(
        overlap5, "C", ["B"], "has_5utr_clip",
        "C_vs_B (5'UTR SHAP vs 5'UTR CLIP)",
    ))
else:
    print("  Skipped (no 5'UTR CLIP targets in overlap5).")

# 3'UTR-specific: grouping based on 3'UTR SHAP only, truth = 3'UTR CLIP
print("\nSub-tests (3'UTR SHAP grouping vs 3'UTR CLIP truth):")
if overlap3["has_3utr_clip"].sum() > 0:
    stats.append(contingency_and_fisher(
        overlap3, "C", ["B"], "has_3utr_clip",
        "C_vs_B (3'UTR SHAP vs 3'UTR CLIP)",
    ))
else:
    print("  Skipped (no 3'UTR CLIP targets in overlap3).")

stats_path = os.path.join(args.output_dir, "orb2_clip_validation_stats.json")
with open(stats_path, "w") as fh:
    json.dump(stats, fh, indent=2)
print(f"\nStats saved to: {stats_path}")

# ---------------------------------------------------------------------------
# Step 10: Save gene table
# ---------------------------------------------------------------------------

table_path = os.path.join(args.output_dir, "orb2_clip_validation_gene_table.csv")
gene_group.merge(
    overlap[["geneID", "is_clip_target", "has_5utr_clip", "has_3utr_clip"]],
    on="geneID", how="left",
).to_csv(table_path, index=False)
print(f"Gene table saved to: {table_path}")

# ---------------------------------------------------------------------------
# Step 11: Barplots – fraction of CLIP targets per group (any, 5'UTR, 3'UTR)
# ---------------------------------------------------------------------------

print("\nGenerating barplots ...")

group_order  = ["A", "B", "C"]
group_labels = {
    "A": "A\n(no ORB2\nMAST hit)",
    "B": "B\n(ORB2 MAST hit,\nno SHAP overlap)",
    "C": "C\n(ORB2 MAST hit +\nlow-SHAP overlap)",
}
colors = ["#7fbadb", "#f5a623", "#d0312d"]


def _make_barplot(ov_df, target_col, title, ylabel, stat_label_substr, out_path):
    fractions, totals, counts = [], [], []
    for g in group_order:
        sub = ov_df[ov_df["gene_group"] == g]
        n = len(sub)
        k = int(sub[target_col].sum())
        fractions.append(k / n if n > 0 else 0)
        totals.append(n)
        counts.append(k)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        [group_labels[g] for g in group_order],
        fractions,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(fractions) * 1.35 if max(fractions) > 0 else 0.1)

    for bar, frac, n, k in zip(bars, fractions, totals, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            frac + max(fractions) * 0.01,
            f"{k}/{n}\n({frac:.1%})",
            ha="center", va="bottom", fontsize=9,
        )

    stat = next(
        (s for s in stats if s["comparison"].startswith("C_vs_B") and
         stat_label_substr in s["comparison"]),
        None,
    )
    if stat:
        ax.text(
            0.98, 0.97,
            f"Fisher C vs B:\nOR={stat['odds_ratio']:.2f}, p={stat['p_value']:.3g}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
        )

    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"Barplot saved to: {out_path}")


# Any-UTR barplot (combined grouping vs any-UTR CLIP truth)
_make_barplot(
    overlap,
    "is_clip_target",
    "ORB2 binding prediction validated by PAR-CLIP\n"
    "(GSE59611, S2 cells, any UTR; CDS peaks excluded)",
    "Fraction of genes confirmed by PAR-CLIP (any UTR)",
    "any-UTR",
    os.path.join(args.output_dir, "orb2_clip_validation_barplot.svg"),
)

# 5'UTR barplot: grouping from 5'UTR SHAP only, truth = 5'UTR CLIP
_make_barplot(
    overlap5,
    "has_5utr_clip",
    "ORB2 5'UTR prediction validated by PAR-CLIP\n"
    "(GSE59611, S2 cells; group C = 5'UTR SHAP+MAST overlap)",
    "Fraction of genes with 5'UTR PAR-CLIP peak",
    "5'UTR SHAP vs 5'UTR CLIP",
    os.path.join(args.output_dir, "orb2_clip_validation_barplot_5utr.svg"),
)

# 3'UTR barplot: grouping from 3'UTR SHAP only, truth = 3'UTR CLIP
_make_barplot(
    overlap3,
    "has_3utr_clip",
    "ORB2 3'UTR prediction validated by PAR-CLIP\n"
    "(GSE59611, S2 cells; group C = 3'UTR SHAP+MAST overlap)",
    "Fraction of genes with 3'UTR PAR-CLIP peak",
    "3'UTR SHAP vs 3'UTR CLIP",
    os.path.join(args.output_dir, "orb2_clip_validation_barplot_3utr.svg"),
)

# ---------------------------------------------------------------------------
# Step 12: UpSet / Venn diagram
# ---------------------------------------------------------------------------

print("Generating overlap diagram ...")

plot_df = overlap.copy()

try:
    from upsetplot import from_memberships, UpSet

    genes_with_mast  = set(plot_df[plot_df["gene_group"].isin(["B", "C"])]["geneID"])
    genes_with_shap  = set(plot_df[plot_df["gene_group"] == "C"]["geneID"])
    genes_clip       = set(plot_df[plot_df["is_clip_target"]]["geneID"])

    memberships = []
    for gene in set(plot_df["geneID"]):
        m = []
        if gene in genes_with_mast:
            m.append("MAST ORB2 hit")
        if gene in genes_with_shap:
            m.append("Low-SHAP overlap")
        if gene in genes_clip:
            m.append("PAR-CLIP target (UTR)")
        memberships.append(m)

    if not any(memberships):
        raise ValueError("All memberships empty – skipping UpSet plot.")

    upset_data = from_memberships(memberships)
    upset = UpSet(upset_data, subset_size="count", show_counts=True, sort_by="cardinality")
    upset_result = upset.plot()
    if isinstance(upset_result, dict):
        upset_fig = next(iter(upset_result.values())).get_figure()
    else:
        upset_fig = upset_result
    for _ax in upset_fig.get_axes():
        for _txt in _ax.texts:
            _x, _y = _txt.get_position()
            _txt.set_position((
                float(np.asarray(_x).flat[0]),
                float(np.asarray(_y).flat[0]),
            ))
    upset_path = os.path.join(args.output_dir, "orb2_clip_validation_upset.svg")
    upset_fig.savefig(upset_path, format="svg")
    plt.close(upset_fig)
    print(f"UpSet plot saved to: {upset_path}")

except (ImportError, ValueError) as _upset_err:
    print(f"  UpSet skipped ({_upset_err}); falling back to Venn diagram.")
    genes_with_mast = set(plot_df[plot_df["gene_group"].isin(["B", "C"])]["geneID"])
    genes_with_shap = set(plot_df[plot_df["gene_group"] == "C"]["geneID"])
    genes_clip      = set(plot_df[plot_df["is_clip_target"]]["geneID"])

    fig, ax = plt.subplots(figsize=(6, 5))
    patches = [
        mpatches.Circle((0.35, 0.5), 0.28, alpha=0.4, color="#f5a623",
                         label=f"MAST ORB2 hits ({len(genes_with_mast)})"),
        mpatches.Circle((0.55, 0.5), 0.22, alpha=0.4, color="#d0312d",
                         label=f"Low-SHAP+MAST ({len(genes_with_shap)})"),
        mpatches.Circle((0.48, 0.35), 0.22, alpha=0.4, color="#7fbadb",
                         label=f"PAR-CLIP targets / UTR ({len(genes_clip)})"),
    ]
    for p in patches:
        ax.add_patch(p)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("ORB2 gene set overlaps (PAR-CLIP UTR-only)")
    venn_path = os.path.join(args.output_dir, "orb2_clip_validation_venn.svg")
    plt.savefig(venn_path, format="svg")
    plt.close()
    print(f"Venn diagram saved to: {venn_path}")

# ---------------------------------------------------------------------------
# Combined figure helper – matplotlib-native 2×3 panel figure
# ---------------------------------------------------------------------------


def _sig_bracket(ax, x1, x2, y_base, or_val, p_val, fontsize=9):
    """Draw a significance bracket above bars at integer positions x1 and x2."""
    ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    arm_h = ylim_range * 0.025
    # raise bracket well clear of the count labels (+18 % of axis height)
    y_bar = y_base + ylim_range * 0.18
    ax.plot(
        [x1, x1, x2, x2],
        [y_bar - arm_h, y_bar, y_bar, y_bar - arm_h],
        color="black", lw=1.0, clip_on=False,
    )
    p_str = (
        f"p={p_val:.1e}" if p_val < 0.001 else
        f"p={p_val:.3f}" if p_val < 0.05 else
        f"p={p_val:.2f}"
    )
    ax.text(
        (x1 + x2) / 2.0, y_bar + ylim_range * 0.015,
        f"OR={or_val:.2f}, {p_str}",
        ha="center", va="bottom", fontsize=fontsize,
    )


def _draw_gene_panel(ax, ov_df, target_col, stat_CvB, ylim,
                     show_ylabel=False, subtitle="", show_xticklabels=True):
    """Render a gene-level bar chart (groups A / B / C) onto *ax*."""
    group_keys  = ["A", "B", "C"]
    group_xlbls = ["No MAST", "Outside\nSHAP", "In SHAP"]
    bar_colors  = ["#7fbadb", "#f5a623", "#d0312d"]
    fracs, ns, ks = [], [], []
    for g in group_keys:
        sub = ov_df[ov_df["gene_group"] == g]
        n   = len(sub)
        k   = int(sub[target_col].sum())
        fracs.append(k / n if n > 0 else 0.0)
        ns.append(n)
        ks.append(k)

    bars = ax.bar(group_xlbls, fracs, color=bar_colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    for bar, frac, n, k in zip(bars, fracs, ns, ks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            frac + ylim * 0.02,
            f"{k}/{n}",
            ha="center", va="bottom", fontsize=10,
        )

    # Equal outer margin: outer gap == inter-bar gap (both 0.5 units for width=0.5)
    _bw = 0.5
    ax.set_xlim(-_bw / 2 - (1 - _bw), len(group_xlbls) - 1 + _bw / 2 + (1 - _bw))
    ax.set_ylim(0, ylim)
    ax.set_title(subtitle, fontsize=11, pad=4)
    ax.set_ylabel("Fraction of Genes" if show_ylabel else "", fontsize=12)
    ax.tick_params(labelsize=10, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)

    if stat_CvB is not None:
        _sig_bracket(ax, 1, 2, max(fracs[1], fracs[2]),
                     stat_CvB["odds_ratio"], stat_CvB["p_value"])


def _draw_pos_panel(ax, stat, ylim, show_ylabel=False, subtitle="", show_xticklabels=True):
    """Render a position-level bar chart (Outside SHAP / In SHAP) onto *ax*."""
    labels     = ["Outside\nSHAP", "In SHAP"]
    rates      = [stat["rate_other"], stat["rate_shap"]]
    ns         = [stat["n_other_hits"], stat["n_shap_hits"]]
    ks         = [stat["other_with_clip"], stat["shap_with_clip"]]
    bar_colors = ["#f5a623", "#d0312d"]

    bars = ax.bar(labels, rates, color=bar_colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    for bar, rate, n, k in zip(bars, rates, ns, ks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + ylim * 0.02,
            f"{k}/{n}",
            ha="center", va="bottom", fontsize=10,
        )

    # Equal outer margin: outer gap == inter-bar gap (both 0.5 units for width=0.5)
    _bw = 0.5
    ax.set_xlim(-_bw / 2 - (1 - _bw), len(labels) - 1 + _bw / 2 + (1 - _bw))
    ax.set_ylim(0, ylim)
    ax.set_title(subtitle, fontsize=11, pad=4)
    ax.set_ylabel("Fraction of Motif Sites" if show_ylabel else "", fontsize=12)
    ax.tick_params(labelsize=10, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)

    if stat["n_shap_hits"] > 0 and stat["n_other_hits"] > 0:
        _sig_bracket(ax, 0, 1, max(rates), stat["odds_ratio"], stat["p_value"])


def _make_combined_figure(gene_panels, pos_panels, out_path):
    """Create a unified matplotlib 3×2 (or 3×1) combined figure.

    Layout: rows = UTR regions (Any, 5', 3'); left col = gene-level,
    right col = site-level (omitted when pos_panels is None).

    Parameters
    ----------
    gene_panels : list of (ov_df, target_col, subtitle, stat_CvB)
    pos_panels  : list of (stat_dict, subtitle) or None
    out_path    : str
    """
    import matplotlib.gridspec as gridspec

    n_rows  = len(gene_panels)   # one row per UTR region
    has_pos = pos_panels is not None and len(pos_panels) > 0
    n_cols  = 2 if has_pos else 1

    # Fixed y-limits: gene col 0–1.0, position col 0–0.7
    gene_ylim = 1.40
    pos_ylim  = 0.75

    ## Auto y-limits: enough headroom for brackets above the tallest bar
    #gene_fracs = []
    #for ov_df, target_col, _, _ in gene_panels:
    #    for g in ["A", "B", "C"]:
    #        sub = ov_df[ov_df["gene_group"] == g]
    #        n   = len(sub)
    #        k   = int(sub[target_col].sum())
    #        gene_fracs.append(k / n if n > 0 else 0.0)
    #gene_ylim = max(max(gene_fracs) * 1.75, 0.20) if gene_fracs else 0.5

    #pos_ylim = 0.5
    #if has_pos:
    #    pos_rates = []
    #    for stat, _ in pos_panels:
    #        pos_rates += [stat["rate_other"], stat["rate_shap"]]
    #    pos_ylim = max(max(pos_rates) * 1.75, 0.20) if pos_rates else 0.5

    # Figure geometry (figure-fraction units)
    LEFT, RIGHT, TOP, BOT = 0.16, 0.97, 0.88, 0.17
    fig = plt.figure(figsize=(7.0 if has_pos else 3.8, 3.2 * n_rows + 1.1))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        left=LEFT, right=RIGHT,
        top=TOP,   bottom=BOT,
        hspace=0.40, wspace=0.52,
    )

    # Panel letters: left col A/B/C, right col D/E/F
    left_letters  = "abc"
    right_letters = "def"
    mid_row = n_rows // 2   # row index for the vertically-centred y-axis label

    def _label_panel(ax, letter):
        ax.text(
            -0.22, 1.08, letter,
            transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top",
        )

    for row in range(n_rows):
        is_bottom = (row == n_rows - 1)
        ov_df, target_col, subtitle, stat_CvB = gene_panels[row]

        # Left column – gene-level
        ax_gene = fig.add_subplot(gs[row, 0])
        _draw_gene_panel(
            ax_gene, ov_df, target_col, stat_CvB, gene_ylim,
            show_ylabel=(row == mid_row),
            subtitle=subtitle,
            show_xticklabels=is_bottom,
        )
        _label_panel(ax_gene, left_letters[row])

        # Right column – site-level
        if has_pos:
            ax_pos = fig.add_subplot(gs[row, 1])
            stat, _ = pos_panels[row]
            _draw_pos_panel(
                ax_pos, stat, pos_ylim,
                show_ylabel=(row == mid_row),
                subtitle=subtitle,
                show_xticklabels=is_bottom,
            )
            _label_panel(ax_pos, right_letters[row])

    # Column headers – placed just above the top row of each column
    plot_w      = RIGHT - LEFT
    each_col    = plot_w / n_cols
    col_centers = [LEFT + each_col * (c + 0.5) for c in range(n_cols)]
    col_labels  = (["Gene Level", "Position Level"] if has_pos else ["Gene Level"])
    for cx, lbl in zip(col_centers, col_labels):
        fig.text(cx, TOP + 0.03, lbl,
                 ha="center", va="bottom", fontsize=13, fontweight="bold")

    # Single vertical legend (ncol=1) so width fits a narrow column
    legend_patches = [
        mpatches.Patch(color="#7fbadb", label="No ORB2 MAST hit"),
        mpatches.Patch(color="#f5a623", label="ORB2 MAST hit (Outside SHAP)"),
        mpatches.Patch(color="#d0312d", label="ORB2 MAST hit + SHAP overlap (In SHAP)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=1,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.55, 0.0),
        title="GSE59611, S2 cells  \u00b7  CDS peaks excluded",
        title_fontsize=10,
    )

    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved to: {out_path}")


# ---------------------------------------------------------------------------
# Position-level analysis (optional – requires --clip_file and --gtf_file)
# ---------------------------------------------------------------------------

if not (args.clip_file and args.gtf_file):
    print("\nSkipping position-level analysis (pass --clip_file and --gtf_file to enable).")
    # Gene-level combined figure (3 panels)
    _stat_any = next((s for s in stats if "C_vs_B" in s["comparison"] and "any-UTR" in s["comparison"]), None)
    _stat_5   = next((s for s in stats if "C_vs_B" in s["comparison"] and "5'UTR SHAP" in s["comparison"]), None)
    _stat_3   = next((s for s in stats if "C_vs_B" in s["comparison"] and "3'UTR SHAP" in s["comparison"]), None)
    _make_combined_figure(
        gene_panels=[
            (overlap5, "has_5utr_clip",  "5\u2032 UTR", _stat_5),
            (overlap3, "has_3utr_clip",  "3\u2032 UTR", _stat_3),
            (overlap,  "is_clip_target", "Any UTR",    _stat_any),
        ],
        pos_panels=None,
        out_path=os.path.join(args.output_dir, "orb2_clip_combined_figure.svg"),
    )
    print("\nDone.")
    sys.exit(0)

print("\n" + "=" * 70)
print("POSITION-LEVEL ANALYSIS")
print("=" * 70)

# --- Load raw ORB2B CLIP peaks -------------------------------------------

print(f"\nLoading raw CLIP peaks from {args.clip_file} ...")
_CLIP_COLS = ["Chromosome", "Start", "End", "clip_id",
              "Strand", "rbp_name", "method", "cell_line", "geo_acc", "score"]
clips_raw = pd.read_csv(
    args.clip_file, sep="\t", header=None, names=_CLIP_COLS,
    comment="#", dtype={"Start": int, "End": int},
)
# Filter to ORB2B (CLIP uses "ORB2B", not "ORB2")
clips_pos = clips_raw[clips_raw["rbp_name"] == "ORB2B"].copy()
clips_pos["Chromosome"] = clips_pos["Chromosome"].str.replace(r"^chr", "", regex=True)
print(f"  ORB2B peaks: {len(clips_pos)}")

# --- Parse GTF for UTR exon structures -----------------------------------

print(f"\nParsing GTF exon structures from {args.gtf_file} ...")

_ATTR_RE_POS = re.compile(r'(\w+)\s+"([^"]+)"')


def _build_utr_exon_map(gtf_path):
    """Return {(transcript_id, utr_type): [(chrom, start_0based, end_0based, strand), ...]}
    sorted in transcript order (5'→3')."""
    exons = {}
    _open = gzip.open if gtf_path.endswith(".gz") else open
    with _open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in {"five_prime_utr", "three_prime_utr"}:
                continue
            attrs = dict(_ATTR_RE_POS.findall(fields[8]))
            tr_id = attrs.get("transcript_id", "")
            if not tr_id:
                continue
            chrom  = fields[0]
            start  = int(fields[3]) - 1   # GTF is 1-based → 0-based half-open
            end    = int(fields[4])
            strand = fields[6]
            utr_t  = "5utr" if feature == "five_prime_utr" else "3utr"
            exons.setdefault((tr_id, utr_t), []).append((chrom, start, end, strand))
    # Sort each UTR in transcript order
    sorted_exons = {}
    for key, ex_list in exons.items():
        strand = ex_list[0][3]
        if strand == "+":
            sorted_exons[key] = sorted(ex_list, key=lambda x: x[1])
        else:
            sorted_exons[key] = sorted(ex_list, key=lambda x: -x[2])
    return sorted_exons


utr_exon_map = _build_utr_exon_map(args.gtf_file)
print(f"  UTR exon structures loaded for {len(utr_exon_map)} (transcript, UTR) pairs.")

# --- Parse SHAP windows from parquets ------------------------------------

print("\nParsing SHAP windows from parquets ...")


def _parse_shap_windows(parquet_path, rbp_col, sep_end_map=None):
    """Return DataFrame with (trID, shap_start, shap_end) for windows where rbp_col > 0.

    sep_end_map: optional {trID: sepEnd} dict.  When provided, convert combined-sequence
    coordinates (as stored in the neg3 parquet index) to UTR-local coords by subtracting
    sepEnd.  Required for 3'UTR parquets; not needed for 5'UTR parquets.
    """
    if not os.path.exists(parquet_path):
        return pd.DataFrame(columns=["trID", "shap_start", "shap_end"])
    df = pd.read_parquet(parquet_path)
    if rbp_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == rbp_col.lower()]
        if not matches:
            return pd.DataFrame(columns=["trID", "shap_start", "shap_end"])
        rbp_col = matches[0]
    pos = df[df[rbp_col] > 0]
    # Convert Index to Series first; Index.str.rsplit(expand=True) returns a
    # MultiIndex in pandas, not a DataFrame, so parsed[1] would be a tuple.
    idx_ser = pd.Series(pos.index)
    parsed  = idx_ser.str.rsplit("_", n=1, expand=True)
    coords  = parsed[1].str.split("-", expand=True)
    result  = pd.DataFrame({
        "trID":       parsed[0].values,
        "shap_start": coords[0].astype(int).values,
        "shap_end":   coords[1].astype(int).values,
    })
    if sep_end_map is not None:
        # The parquet index stores combined-sequence coordinates (5'UTR+sep+3'UTR).
        # Subtract sepEnd to convert to 3'UTR-local coordinates.
        sep_e = result["trID"].map(sep_end_map)
        result["shap_start"] = result["shap_start"] - sep_e
        result["shap_end"]   = result["shap_end"]   - sep_e
        n_missing = sep_e.isna().sum()
        if n_missing:
            print(f"  WARNING: sepEnd not found for {n_missing} transcripts; those 3'UTR windows dropped.")
        result = result.dropna(subset=["shap_start", "shap_end"])
        result["shap_start"] = result["shap_start"].astype(int)
        result["shap_end"]   = result["shap_end"].astype(int)
    return result


# Build sepEnd lookup for 3'UTR coordinate adjustment
sep_end_map_3 = None
if args.neg3_motif_positions and os.path.exists(args.neg3_motif_positions):
    _mp = pd.read_csv(args.neg3_motif_positions, sep="\t")
    # Keep only rows that are 3'UTR windows (start > sepStart = separator start)
    _mp_3 = _mp[_mp["start"] > _mp["sepStart"]]
    sep_end_map_3 = _mp_3.drop_duplicates("trID").set_index("trID")["sepEnd"].to_dict()
    print(f"  Loaded sepEnd for {len(sep_end_map_3)} transcripts from {args.neg3_motif_positions}")

shap_wins_5 = _parse_shap_windows(args.shap_neg5, rbp)
shap_wins_3 = _parse_shap_windows(args.shap_neg3, rbp, sep_end_map=sep_end_map_3)
print(f"  SHAP windows with ORB2 hits: 5'UTR={len(shap_wins_5)}, 3'UTR={len(shap_wins_3)}")

# Build per-transcript lookup: {trID: [(shap_start, shap_end), ...]}
shap_lookup_5 = (
    shap_wins_5.groupby("trID")
    .apply(lambda d: list(zip(d["shap_start"], d["shap_end"])))
    .to_dict()
)
shap_lookup_3 = (
    shap_wins_3.groupby("trID")
    .apply(lambda d: list(zip(d["shap_start"], d["shap_end"])))
    .to_dict()
)

# --- Convert MAST hits to genomic intervals with SHAP flag ---------------

print("\nMapping ORB2 MAST hits to genomic coordinates ...")


def _hit_in_shap(hit_start, hit_end, windows):
    return any(hit_start < w_end and hit_end > w_start
               for w_start, w_end in windows)


def _hit_to_genomic(hit_start, hit_end, exons):
    """Convert UTR-local [hit_start, hit_end) to genomic interval(s).

    exons: [(chrom, g_start, g_end, strand)] in transcript order (5'→3').
    For + strand: ascending g_start.  For - strand: descending g_end.
    Returns list of (chrom, g_start, g_end, strand).
    """
    offset = 0
    result = []
    for chrom, g_start, g_end, strand in exons:
        ex_len   = g_end - g_start
        ov_start = max(hit_start, offset)
        ov_end   = min(hit_end,   offset + ex_len)
        if ov_start < ov_end:
            w_s = ov_start - offset
            w_e = ov_end   - offset
            if strand == "+":
                result.append((chrom, g_start + w_s, g_start + w_e, strand))
            else:
                # Minus strand: higher genomic coord = transcript start
                result.append((chrom, g_end - w_e, g_end - w_s, strand))
        offset += ex_len
        if offset >= hit_end:
            break
    return result


hit_records = []
n_no_exons = 0
for _, row in orb2_mast.iterrows():
    tr_id     = row["trID"]
    seq_name  = row["sequence_name"]
    hit_start = int(row["hit_start"])
    hit_end   = int(row["hit_end"])

    if "5utr" in seq_name:
        utr_t       = "5utr"
        shap_lookup = shap_lookup_5
    elif "3utr" in seq_name:
        utr_t       = "3utr"
        shap_lookup = shap_lookup_3
    else:
        continue

    windows    = shap_lookup.get(tr_id, [])
    in_shap_f  = _hit_in_shap(hit_start, hit_end, windows) if windows else False

    exons = utr_exon_map.get((tr_id, utr_t), [])
    if not exons:
        n_no_exons += 1
        continue

    for chrom, g_start, g_end, strand in _hit_to_genomic(hit_start, hit_end, exons):
        hit_records.append({
            "trID":        tr_id,
            "utr_type":    utr_t,
            "Chromosome":  chrom,
            "Start":       g_start,
            "End":         g_end,
            "Strand":      strand,
            "in_shap":     in_shap_f,
        })

hits_df = pd.DataFrame(hit_records)
print(f"  Total hit intervals mapped: {len(hits_df)}")
print(f"  Skipped (no GTF exon data): {n_no_exons}")
print(f"  In SHAP windows:       {hits_df['in_shap'].sum()}")
print(f"  Not in SHAP windows:   {(~hits_df['in_shap']).sum()}")

# Deduplicate at genomic level (same position from multiple isoforms)
hits_df = (
    hits_df
    .groupby(["Chromosome", "Start", "End", "Strand", "utr_type"], as_index=False)
    .agg({"in_shap": "max"})
)
print(f"  After deduplication: {len(hits_df)} unique genomic hit intervals")
print(f"  In SHAP (deduplicated):   {hits_df['in_shap'].sum()}")
print(f"  Not in SHAP (deduplicated): {(~hits_df['in_shap']).sum()}")

# --- Intersect each hit with CLIP peaks ----------------------------------

print("\nIntersecting positions with CLIP peaks ...")

# Pre-index CLIP peaks by (chrom, strand)
clip_index = {}
for (chrom, strand), grp in clips_pos.groupby(["Chromosome", "Strand"]):
    clip_index[(chrom, strand)] = (grp["Start"].values, grp["End"].values)

hits_df = hits_df.reset_index(drop=True)
has_clip = np.zeros(len(hits_df), dtype=bool)
for (chrom, strand), hit_grp in hits_df.groupby(["Chromosome", "Strand"]):
    key = (chrom, strand)
    if key not in clip_index:
        continue
    c_starts, c_ends = clip_index[key]
    h_starts = hit_grp["Start"].values[:, np.newaxis]   # (n_hits, 1)
    h_ends   = hit_grp["End"].values[:, np.newaxis]
    any_overlap = ((h_starts < c_ends) & (h_ends > c_starts)).any(axis=1)
    has_clip[hit_grp.index] = any_overlap

hits_df["has_clip"] = has_clip
print(f"  Hits with CLIP overlap: {has_clip.sum()} / {len(hits_df)} "
      f"({has_clip.mean():.1%})")

# --- Fisher's test at position level ------------------------------------

print("\nPosition-level Fisher tests ...")

pos_stats = []


def _pos_fisher(df_sub, label):
    shap_hits  = df_sub[df_sub["in_shap"]]
    other_hits = df_sub[~df_sub["in_shap"]]
    a = int(shap_hits["has_clip"].sum())
    b = int(len(shap_hits)  - a)
    c = int(other_hits["has_clip"].sum())
    d = int(len(other_hits) - c)
    table = np.array([[a, b], [c, d]])
    or_, p = fisher_exact(table, alternative="greater")
    rate_shap  = a / len(shap_hits)  if len(shap_hits)  > 0 else float("nan")
    rate_other = c / len(other_hits) if len(other_hits) > 0 else float("nan")
    print(
        f"  {label}: OR={or_:.3f}, p={p:.4g}\n"
        f"    SHAP hits:  {a}/{len(shap_hits)}  ({rate_shap:.1%}) have CLIP overlap\n"
        f"    Other hits: {c}/{len(other_hits)} ({rate_other:.1%}) have CLIP overlap\n"
        f"    Contingency [[{a},{b}],[{c},{d}]]"
    )
    return {
        "comparison":       label,
        "n_shap_hits":      int(len(shap_hits)),
        "n_other_hits":     int(len(other_hits)),
        "shap_with_clip":   a,
        "other_with_clip":  c,
        "rate_shap":        float(rate_shap),
        "rate_other":       float(rate_other),
        "odds_ratio":       float(or_),
        "p_value":          float(p),
        "contingency":      table.tolist(),
    }


_all  = _pos_fisher(hits_df,                             "all UTRs combined")
_utr5 = _pos_fisher(hits_df[hits_df["utr_type"] == "5utr"], "5'UTR only")
_utr3 = _pos_fisher(hits_df[hits_df["utr_type"] == "3utr"], "3'UTR only")

pos_stats = [_all, _utr5, _utr3]

pos_stats_path = os.path.join(args.output_dir, "orb2_clip_position_stats.json")
with open(pos_stats_path, "w") as fh:
    json.dump(pos_stats, fh, indent=2)
print(f"\nPosition stats saved to: {pos_stats_path}")

# --- Barplot: CLIP overlap rate for SHAP vs non-SHAP hits ----------------

print("\nGenerating position-level barplots ...")


def _pos_barplot(df_sub, stat, title, out_path):
    labels = ["Outside SHAP region", "In SHAP region"]
    rates  = [stat["rate_other"], stat["rate_shap"]]
    ns     = [stat["n_other_hits"], stat["n_shap_hits"]]
    ks     = [stat["other_with_clip"], stat["shap_with_clip"]]
    bar_colors = ["#f5a623", "#d0312d"]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(labels, rates, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Fraction of ORB2 motif sites with CLIP peak overlap", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(rates) * 1.40 if max(rates) > 0 else 0.1)

    for bar, rate, n, k in zip(bars, rates, ns, ks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + max(rates) * 0.01,
            f"{k}/{n}\n({rate:.1%})",
            ha="center", va="bottom", fontsize=9,
        )

    ax.text(
        0.98, 0.97,
        f"Fisher (SHAP vs other):\nOR={stat['odds_ratio']:.2f}, p={stat['p_value']:.3g}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
    )
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"  Saved: {out_path}")


_pos_barplot(
    hits_df, _all,
    "ORB2 motif sites: CLIP support by SHAP status\n"
    "(all UTRs; each bar = unique genomic motif site)",
    os.path.join(args.output_dir, "orb2_clip_position_barplot.svg"),
)
_pos_barplot(
    hits_df[hits_df["utr_type"] == "5utr"], _utr5,
    "ORB2 motif sites: CLIP support by SHAP status\n"
    "(5\u2019UTR motif sites only)",
    os.path.join(args.output_dir, "orb2_clip_position_barplot_5utr.svg"),
)
_pos_barplot(
    hits_df[hits_df["utr_type"] == "3utr"], _utr3,
    "ORB2 motif sites: CLIP support by SHAP status\n"
    "(3\u2019UTR motif sites only)",
    os.path.join(args.output_dir, "orb2_clip_position_barplot_3utr.svg"),
)

# Save hit table
hits_table_path = os.path.join(args.output_dir, "orb2_clip_position_hits.csv.gz")
hits_df.to_csv(hits_table_path, index=False)
print(f"  Hit table saved: {hits_table_path}")

# Full 6-panel combined figure (gene-level + position-level)
_stat_any = next((s for s in stats if "C_vs_B" in s["comparison"] and "any-UTR" in s["comparison"]), None)
_stat_5   = next((s for s in stats if "C_vs_B" in s["comparison"] and "5'UTR SHAP" in s["comparison"]), None)
_stat_3   = next((s for s in stats if "C_vs_B" in s["comparison"] and "3'UTR SHAP" in s["comparison"]), None)
_make_combined_figure(
    gene_panels=[
        (overlap5, "has_5utr_clip",  "5\u2032 UTR", _stat_5),
        (overlap3, "has_3utr_clip",  "3\u2032 UTR", _stat_3),
        (overlap,  "is_clip_target", "Any UTR",    _stat_any),
    ],
    pos_panels=[
        (_utr5, "5\u2032 UTR"),
        (_utr3, "3\u2032 UTR"),
        (_all,  "Any UTR"),
    ],
    out_path=os.path.join(args.output_dir, "orb2_clip_combined_figure.svg"),
)

print("\nDone.")
