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

# Only chr1, chrX, chrY
chrom_order = ["chr1", "chrX", "chrY"]

# Colors per sample
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

x = np.arange(len(chrom_order))  # positions for chromosomes

# ========= PLOT 5mC =========
plt.figure(figsize=(10, 5))
for bc in samples_files.keys():
    sub = summary[summary["sample"] == bc].set_index("chrom").reindex(chrom_order)
    sub.fillna(0.01, inplace=True)  # small value to avoid log(0)
    plt.plot(
        x, sub["mean_5mC"],
        color=sample_colors[bc],
        marker=markers[bc],
        linestyle='-',
        linewidth=2,
        alpha=0.8,
        label=bc
    )

plt.xticks(x, chrom_order)
plt.ylabel("Average 5mC (%) (log scale)")
plt.xlabel("Chromosome")
plt.title("5mC levels per sample across chr1, X, Y")
plt.yscale('log')
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.legend(ncol=2, fontsize=10)
plt.tight_layout()

out_fig_mC = os.path.join(data_dir, "modkit_5mC_log_chr1_X_Y.png")
plt.savefig(out_fig_mC, dpi=300)
plt.close()
print(f"[DONE] 5mC figure saved: {out_fig_mC}")

# ========= PLOT 5hmC =========
plt.figure(figsize=(10, 5))
for bc in samples_files.keys():
    sub = summary[summary["sample"] == bc].set_index("chrom").reindex(chrom_order)
    sub.fillna(0.01, inplace=True)
    plt.plot(
        x, sub["mean_5hmC"],
        color=sample_colors[bc],
        marker=markers[bc],
        linestyle='-',
        linewidth=2,
        alpha=0.8,
        label=bc
    )

plt.xticks(x, chrom_order)
plt.ylabel("Average 5hmC (%) (log scale)")
plt.xlabel("Chromosome")
plt.title("5hmC levels per sample across chr1, X, Y")
plt.yscale('log')
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.legend(ncol=2, fontsize=10)
plt.tight_layout()

out_fig_hmC = os.path.join(data_dir, "modkit_5hmC_log_chr1_X_Y.png")
plt.savefig(out_fig_hmC, dpi=300)
plt.close()
print(f"[DONE] 5hmC figure saved: {out_fig_hmC}")

# ========= SAVE SUMMARY =========
out_tsv = os.path.join(data_dir, "modkit_summary_chr1_X_Y.tsv")
summary.to_csv(out_tsv, sep="\t", index=False)
print(f"[DONE] Summary table saved: {out_tsv}")

