#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pybedtools
from collections import defaultdict

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
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
# Process each sample individually
# -------------------
for bed_file in tqdm(bed_files, desc="Processing samples"):
    sample_name = bed_file.replace("mods_positions_", "").replace(".bed", "")
    print(f"\n=== Annotating {sample_name} ===")

    mods = pybedtools.BedTool(os.path.join(input_dir, bed_file))
    genes = pybedtools.BedTool(gtf_file)

    # Dictionary to hold per-gene counts: gene_id -> {"5mC": count, "5hmC": count}
    gene_counts = defaultdict(lambda: {"5mC": 0, "5hmC": 0})

    # Iterate intersections line by line (no full DataFrame)
    for interval in mods.intersect(genes, wa=True, wb=True):
        # BED columns: chr, start, end, mod_type
        mod_type = interval[3]

        # GTF attributes are at interval[12]
        g_attr = interval[12]
        gene_id_match = pd.Series(g_attr).str.extract(r'gene_id "([^"]+)"')
        gene_id = gene_id_match[0] if gene_id_match is not None else None

        if gene_id:
            if mod_type not in gene_counts[gene_id]:
                gene_counts[gene_id][mod_type] = 0
            gene_counts[gene_id][mod_type] += 1

    # Convert dictionary to DataFrame
    summary = pd.DataFrame([
        {"gene_id": gid,
         "5mC": counts.get("5mC", 0),
         "5hmC": counts.get("5hmC", 0)}
        for gid, counts in gene_counts.items()
    ])
    summary["sample"] = sample_name
    summary["total"] = summary["5mC"] + summary["5hmC"]
    summary["frac_5hmC"] = summary["5hmC"] / summary["total"].replace(0, pd.NA)

    # Save per-sample summary immediately
    summary_file = os.path.join(out_dir, f"{sample_name}_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Saved per-sample summary: {summary_file}")

    # Clear memory
    del mods, genes, gene_counts, summary

# -------------------
# Merge all per-sample summaries
# -------------------
import glob
all_files = glob.glob(os.path.join(out_dir, "*_summary.csv"))
combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
combined_file = os.path.join(out_dir, "gene_mod_enrichment_all_samples.csv")
combined.to_csv(combined_file, index=False)
print(f"\nSaved combined per-gene file: {combined_file}")

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
