#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
data_dir = "/mnt/e/Data/seq_for_human_293t2/modkit"
samples = ["barcode04", "barcode05", "barcode06", "barcode07"]
chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

colors = {"percent_m": "red", "percent_h": "blue"}
markers = {"barcode04": "o", "barcode05": "s", "barcode06": "^", "barcode07": "D"}

# ========= LOAD DATA =========
dfs = []
for bc in samples:
    f = os.path.join(data_dir, f"{bc}_aligned_with_mod.region_mh.stats.tsv")
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

# Ensure proper chromosome order
summary["chrom_num"] = pd.Categorical(summary["chrom"], chrom_order, ordered=True)
summary = summary.sort_values(["chrom_num", "sample"])

# ========= PLOT =========
plt.figure(figsize=(18, 6))
x = np.arange(len(chrom_order))

for bc in samples:
    sub = summary[summary["sample"] == bc]
    if sub.empty:
        continue

    # Plot 5mC line with markers
    plt.plot(
        x,
        sub["mean_5mC"],
        marker=markers[bc],
        color=colors["percent_m"],
        label=f"{bc} 5mC",
        linewidth=2,
        alpha=0.7
    )
    # Plot 5hmC line with markers
    plt.plot(
        x,
        sub["mean_5hmC"],
        marker=markers[bc],
        color=colors["percent_h"],
        label=f"{bc} 5hmC",
        linewidth=2,
        alpha=0.7
    )

    # Annotate each point with value
    for i, row in sub.iterrows():
        plt.text(
            x[i],
            row["mean_5mC"] + 1,
            f"{row['mean_5mC']:.1f}",
            color=colors["percent_m"],
            fontsize=8,
            ha="center",
            va="bottom"
        )
        plt.text(
            x[i],
            row["mean_5hmC"] + 1,
            f"{row['mean_5hmC']:.1f}",
            color=colors["percent_h"],
            fontsize=8,
            ha="center",
            va="bottom"
        )

plt.xticks(x, chrom_order, rotation=45)
plt.ylabel("Average modification (%)")
plt.title("Average 5mC (red) and 5hmC (blue) levels per chromosome across samples")
plt.ylim(0, 100)
plt.legend(ncol=4, fontsize=10)
plt.tight_layout()

out_fig = os.path.join(data_dir, "modkit_chr_comparison_lines.png")
plt.savefig(out_fig, dpi=300)
plt.close()

# ========= OUTPUT SUMMARY =========
out_tsv = os.path.join(data_dir, "modkit_chr_comparison_summary.tsv")
summary.to_csv(out_tsv, sep="\t", index=False)

print(f"[DONE] Figure saved: {out_fig}")
print(f"[DONE] Summary table saved: {out_tsv}")
