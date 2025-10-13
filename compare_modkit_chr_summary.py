#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
data_dir = "/mnt/e/Data/seq_for_human_293t2/modkit"

# Map sample names to exact files
samples_files = {
    "barcode04": os.path.join(data_dir, "barcode04_aligned_with_mod.region_mh.stats.tsv"),
    "barcode05": os.path.join(data_dir, "barcode05_aligned_with_mod.region_mh.stats.tsv"),
    "barcode06": os.path.join(data_dir, "barcode06_aligned_with_mod.region_mh.stats.tsv"),
    "barcode07": os.path.join(data_dir, "barcode07_aligned_with_mod.region_mh.stats.tsv"),
}

chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# Assign distinct colors per sample
sample_colors = {
    "barcode04": "red",
    "barcode05": "green",
    "barcode06": "blue",
    "barcode07": "purple",
}

# ========= LOAD DATA =========
dfs = []
for bc, f in samples_files.items():
    if not os.path.exists(f):
        print(f"[WARN] missing: {f}")
        continue

    df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    if "#chrom" in df.columns:
        df = df.rename(columns={"#chrom": "chrom"})
    else:
        print(f"[ERROR] {f} missing '#chrom' column")
        continue

    if "percent_m" not in df.columns or "percent_h" not in df.columns:
        print(f"[ERROR] {f} missing 'percent_m' or 'percent_h' column")
        continue

    df = df[df["chrom"].isin(chrom_order)]
    df["sample"] = bc
    dfs.append(df)

if not dfs:
    raise SystemExit("[FATAL] No valid input files found!")

df_all = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Loaded {len(df_all):,} rows total from {len(dfs)} samples")

# ========= COMPUTE AVERAGE PER CHROMOSOME =========
summary = (
    df_all.groupby(["sample", "chrom"])
    .agg(mean_5mC=("percent_m", "mean"), mean_5hmC=("percent_h", "mean"))
    .reset_index()
)

# Ensure chromosome order
summary["chrom"] = pd.Categorical(summary["chrom"], chrom_order, ordered=True)
summary = summary.sort_values(["chrom", "sample"])

# ========= PLOT SIDE-BY-SIDE =========
plt.figure(figsize=(20, 6))
n_samples = len(samples_files)
bar_width = 0.35  # width for 5mC and 5hmC bars
x = np.arange(len(chrom_order))  # positions for chromosomes

for i, bc in enumerate(samples_files.keys()):
    sub = summary[summary["sample"] == bc].set_index("chrom").reindex(chrom_order)
    sub.fillna(0, inplace=True)  # fill missing with 0 for plotting

    # Offset positions for each sample
    offset = (i - n_samples/2) * bar_width
    # Plot 5mC
    plt.bar(
        x + offset,
        sub["mean_5mC"],
        width=bar_width / 2,
        color=sample_colors[bc],
        label=f"{bc} 5mC",
        alpha=0.8
    )
    # Plot 5hmC right next to 5mC
    plt.bar(
        x + offset + bar_width / 2,
        sub["mean_5hmC"],
        width=bar_width / 2,
        color=sample_colors[bc],
        label=f"{bc} 5hmC",
        alpha=0.4
    )

plt.xticks(x, chrom_order, rotation=45)
plt.ylabel("Average modification (%)")
plt.title("Genome-wide average 5mC (solid) and 5hmC (lighter) per chromosome across samples")
plt.ylim(0, 100)

# Deduplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), ncol=4, fontsize=10)

plt.tight_layout()

# Save figure
out_fig = os.path.join(data_dir, "modkit_genome_comparison_side_by_side.png")
plt.savefig(out_fig, dpi=300)
plt.close()

# Save summary table
out_tsv = os.path.join(data_dir, "modkit_genome_comparison_summary_side_by_side.tsv")
summary.to_csv(out_tsv, sep="\t", index=False)

print(f"[DONE] Figure saved: {out_fig}")
print(f"[DONE] Summary table saved: {out_tsv}")
