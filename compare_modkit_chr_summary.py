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

# Assign unique colors per sample
sample_colors = {
    "barcode04": "#1f77b4",  # blue
    "barcode05": "#ff7f0e",  # orange
    "barcode06": "#2ca02c",  # green
    "barcode07": "#d62728",  # red
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

    # Rename #chrom to chrom
    if "#chrom" in df.columns:
        df = df.rename(columns={"#chrom": "chrom"})
    else:
        print(f"[ERROR] {f} missing '#chrom' column")
        continue

    # Ensure percent_m and percent_h exist
    if "percent_m" not in df.columns or "percent_h" not in df.columns:
        print(f"[ERROR] {f} missing 'percent_m' or 'percent_h' column")
        continue

    # Keep only desired chromosomes
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
summary["chrom_num"] = pd.Categorical(summary["chrom"], chrom_order, ordered=True)
summary = summary.sort_values(["chrom_num", "sample"])

# ========= PLOT =========
plt.figure(figsize=(18, 6))
x = np.arange(len(chrom_order))  # 0..23 for 24 chromosomes

for bc in samples_files.keys():
    sub = summary[summary["sample"] == bc]
    if sub.empty:
        continue

    # Align chromosomes to x-axis positions, fill missing with NaN
    sub = sub.set_index("chrom").reindex(chrom_order).reset_index()
    sub["sample"].fillna(bc, inplace=True)

    color = sample_colors[bc]

    # Plot 5mC line
    plt.plot(
        x,
        sub["mean_5mC"],
        marker=markers[bc],
        color=color,
        label=f"{bc} 5mC",
        linewidth=2,
        alpha=0.7
    )

    # Plot 5hmC line with dashed style
    plt.plot(
        x,
        sub["mean_5hmC"],
        marker=markers[bc],
        color=color,
        linestyle='--',
        label=f"{bc} 5hmC",
        linewidth=2,
        alpha=0.7
    )

    # Annotate points (skip NaN)
    for xi, row in zip(x, sub.itertuples()):
        if not np.isnan(row.mean_5mC):
            plt.text(
                xi,
                row.mean_5mC + 0.5,
                f"{row.mean_5mC:.1f}",
                color=color,
                fontsize=7,
                ha="center",
                va="bottom"
            )
        if not np.isnan(row.mean_5hmC):
            plt.text(
                xi,
                row.mean_5hmC + 0.5,
                f"{row.mean_5hmC:.1f}",
                color=color,
                fontsize=7,
                ha="center",
                va="bottom"
            )

plt.xticks(x, chrom_order, rotation=45)
plt.ylabel("Average modification (%)")
plt.title("Genome-wide average 5mC (solid) and 5hmC (dashed) per chromosome across samples")
plt.ylim(0, 100)
plt.legend(ncol=4, fontsize=10)
plt.tight_layout()

# Save figure
out_fig = os.path.join(data_dir, "modkit_genome_comparison_lines_color_by_sample.png")
plt.savefig(out_fig, dpi=300)
plt.close()

# Save summary table
out_tsv = os.path.join(data_dir, "modkit_genome_comparison_summary_color_by_sample.tsv")
summary.to_csv(out_tsv, sep="\t", index=False)

print(f"[DONE] Figure saved: {out_fig}")
print(f"[DONE] Summary table saved: {out_tsv}")
