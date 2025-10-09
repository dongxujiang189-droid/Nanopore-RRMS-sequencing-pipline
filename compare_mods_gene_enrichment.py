#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import pybedtools
from tqdm import tqdm

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2"
input_dir = os.path.join(base_dir, "mod_positions")  # BED files
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "mod_gene_compare")
os.makedirs(out_dir, exist_ok=True)

# -------------------
# Collect all BED files
# -------------------
bed_files = [f for f in os.listdir(input_dir) if f.endswith(".bed")]
if not bed_files:
    raise FileNotFoundError(f"No BED files found in {input_dir}")

# -------------------
# Process each sample
# -------------------
all_results = []

for bed_file in tqdm(bed_files, desc="Processing samples"):
    sample_name = bed_file.replace("mods_positions_", "").replace(".bed", "")
    print(f"\n=== Annotating {sample_name} ===")

    mods = pybedtools.BedTool(os.path.join(input_dir, bed_file))
    genes = pybedtools.BedTool(gtf_file)

    # Intersect BED with GTF
    intersect = mods.intersect(genes, wa=True, wb=True)
    df = intersect.to_dataframe(disable_auto_names=True, header=None)

    if df.empty:
        print(f"Warning: No intersections found for {sample_name}")
        continue

    # BED has 4 columns: chr, start, end, mod_type
    df.columns = ["chr", "start", "end", "mod_type",
                  "g_chr", "g_src", "g_feat", "g_start", "g_end",
                  "g_score", "g_strand", "g_frame", "g_attr"]

    # Extract gene_id from GTF attributes
    df["gene_id"] = df["g_attr"].str.extract(r'gene_id "([^"]+)"')
    
    # Aggregate per-gene counts for 5mC and 5hmC
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

# -------------------
# Merge all samples
# -------------------
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(os.path.join(out_dir, "gene_mod_enrichment_all_samples.csv"), index=False)
print(f"\nSaved combined per-gene file to: {out_dir}/gene_mod_enrichment_all_samples.csv")

# -------------------
# Plot comparison - histograms
# -------------------
plt.figure(figsize=(10,6))
for samp in combined["sample"].unique():
    subset = combined[combined["sample"] == samp]["frac_5hmC"].dropna()
    plt.hist(subset, bins=50, alpha=0.5, label=samp)
plt.legend()
plt.xlabel("Fraction 5hmC per gene")
plt.ylabel("Number of genes")
plt.title("Distribution of 5hmC enrichment across samples")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "5hmC_gene_fraction_comparison.png"), dpi=200)
plt.close()

# -------------------
# Plot comparison - boxplot
# -------------------
plt.figure(figsize=(10,6))
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
