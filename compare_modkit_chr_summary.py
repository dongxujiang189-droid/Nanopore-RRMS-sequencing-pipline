#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
data_dir = "/mnt/e/Data/seq_for_human_293t2/modkit"

samples_files = {
    "barcode04": os.path.join(data_dir, "barcode04_aligned_with_mod.region_mh.stats.tsv"),
    "barcode05": os.path.join(data_dir, "barcode05_aligned_with_mod.region_mh.stats.tsv"),
    "barcode06": os.path.join(data_dir, "barcode06_aligned_with_mod.region_mh.stats.tsv"),
    "barcode07": os.path.join(data_dir, "barcode07_aligned_with_mod.region_mh.stats.tsv"),
}

chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

sample_colors = {
    "barcode04": "red",
    "barcode05": "green",
    "barcode06": "blue",
    "barcode07": "purple",
}

markers = {"barcode04": "o", "barcode05": "s", "barcode06": "^", "barcode07": "D"}

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

summary["chrom"] = pd.Categorical(summary["chrom"], chrom_order, ordered=True)
summary = summary.sort_values(["chrom", "sample"])

# ========= PLOT LINE CHART =========
plt.figure(figsize=(20, 6))
x = np.arange(len(chrom_order))

for bc in samples_files.keys():
    sub = summary[summary["sample"] == bc].set_index("chrom").reindex(chrom_order)
    sub["sample"].fillna(bc, inplace=True)  # fill sample name
    color = sample_colors[bc]

    # Plot 5mC solid line
    plt.plot(
        x,
        sub["mean_5mC"],
        marker=markers[bc],
        color=color,
        linestyle='-',
        linewidth=2,
        alpha=0.8,
        label=f"{bc} 5mC"
    )

    # Plot 5hmC dashed line
    plt.plot(
        x,
        sub["mean_5hmC"],
        marker=markers[bc],
        color=color,
        linestyle='--',
        linewidth=2,
        alpha=0.8,
        label=f"{bc} 5hmC"
    )

    # Annotate values
    for xi, row in zip(x, sub.itertuples()):
        if not pd.isna(row.mean_5mC):
            plt.text(
                xi, row.mean_5mC + 0.5, f"{row.mean_5mC:.1f}",
                color=color, fontsize=7, ha="center", va="bottom"
            )
        if not pd.isna(row.mean_5hmC):
            plt.text(
                xi, row.mean_5hmC + 0.5, f"{row.mean_5hmC:.1f}",
                color=color, fontsize=7, ha="center", va="bottom"
            )

plt.xticks(x, chrom_order, rotation=45)
plt.ylabel("Average modification (%)")
plt.title("Genome-wide 5mC (solid) and 5hmC (dashed) per chromosome across samples")
plt.ylim(0, 100)

# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), ncol=4, fontsize=10)

plt.tight_layout()

# Save figure
out_fig = os.path.join(data_dir, "modkit_genome_comparison_lines_color_samples.png")
plt.savefig(out_fig, dpi=300)
plt.close()

# Save summary table
out_t_
