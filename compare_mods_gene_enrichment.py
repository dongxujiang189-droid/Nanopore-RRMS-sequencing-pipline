#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import pybedtools
import os
from tqdm import tqdm

base_dir = "/mnt/e/Data/seq_for_human_293t2"
input_dir = os.path.join(base_dir, "mod_positions")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "mod_gene_compare")
os.makedirs(out_dir, exist_ok=True)

# Collect all TSV files
tsv_files = [f for f in os.listdir(input_dir) if f.endswith(".tsv")]

all_results = []

for tsv in tsv_files:
    sample_name = tsv.replace("mods_positions_", "").replace(".tsv", "")
    print(f"\n=== Annotating {sample_name} ===")

    mods = pybedtools.BedTool(os.path.join(input_dir, tsv))
    genes = pybedtools.BedTool(gtf_file)

    # Intersection
    intersect = mods.intersect(genes, wa=True, wb=True)
    df = intersect.to_dataframe(disable_auto_names=True, header=None)

    # Basic columns: [chr, pos, mod_type, sample, ..., gene_id, gene_name]
    df.columns = ["chr", "pos", "mod_type", "sample", "g_chr", "g_src", "g_feat", 
                  "g_start", "g_end", "g_score", "g_strand", "g_frame", "g_attr"]
    # Extract gene_id
    df["gene_id"] = df["g_attr"].str.extract(r'gene_id "([^"]+)"')

    # Aggregate counts per gene & mod_type
    summary = (
        df.groupby(["gene_id", "mod_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary["sample"] = sample_name
    summary["total"] = summary.get("5mC", 0) + summary.get("5hmC", 0)
    summary["frac_5hmC"] = summary["5hmC"] / summary["total"].replace(0, pd.NA)

    all_results.append(summary)

# Merge all samples
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(os.path.join(out_dir, "gene_mod_enrichment_all_samples.csv"), index=False)
print(f"\nSaved combined per-gene file to: {out_dir}/gene_mod_enrichment_all_samples.csv")

# Plot comparison — 5hmC fraction per gene
plt.figure(figsize=(8,6))
for samp in combined["sample"].unique():
    subset = combined[combined["sample"] == samp]["frac_5hmC"].dropna()
    plt.hist(subset, bins=50, alpha=0.5, label=samp)
plt.legend()
plt.xlabel("Fraction 5hmC (per gene)")
plt.ylabel("Number of genes")
plt.title("Distribution of 5hmC enrichment across samples")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "5hmC_gene_fraction_comparison.png"), dpi=200)
plt.close()

# Boxplot of per-gene 5hmC fractions
plt.figure(figsize=(8,6))
boxdata = [combined[combined["sample"]==s]["frac_5hmC"].dropna() for s in combined["sample"].unique()]
plt.boxplot(boxdata, labels=combined["sample"].unique(), showfliers=False)
plt.ylabel("Fraction 5hmC per gene")
plt.title("Per-gene 5hmC fraction comparison")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "5hmC_gene_fraction_boxplot.png"), dpi=200)
plt.close()

print("Generated comparison plots:")
print(f" - {out_dir}/5hmC_gene_fraction_comparison.png")
print(f" - {out_dir}/5hmC_gene_fraction_boxplot.png")
