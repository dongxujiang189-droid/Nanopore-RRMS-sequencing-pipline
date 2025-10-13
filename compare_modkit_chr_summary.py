#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
# Use Linux-style path for WSL
data_dir = "/mnt/e/Data/seq_for_human_293t2/modkit"
samples = ["barcode04", "barcode05", "barcode06", "barcode07"]
chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# ========= LOAD =========
dfs = []
for bc in samples:
    f = os.path.join(data_dir, f"{bc}_aligned_with_mod.region_mh.stats.tsv")
    if not os.path.exists(f):
        print(f"[WARN] missing: {f}")
        continue
    df = pd.read_csv(f, sep="\t")
    df = df[df["chrom"].isin(chrom_order)]
    df["sample"] = bc
    dfs.append(df)

if not dfs:
    raise SystemExit("[FATAL] No valid input files found!")

df_all = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Loaded {len(df_all):,} rows total")

# ========= SUMMARY =========
summary = (
    df_all.groupby(["sample", "chrom"])
    .agg(mean_5mC=("percent_m", "mean"), mean_5hmC=("percent_h", "mean"))
    .reset_index()
)

# sort by chromosome order
summary["chrom_num"] = pd.Categorical(summary["chrom"], chrom_order, ordered=True)
summary = summary.sort_values(["chrom_num", "sample"])

# ========= PLOT =========
plt.figure(figsize=(16, 6))

chroms = summary["chrom"].unique()
x = np.arange(len(chroms))
width = 0.18  # bar width

for i, bc in enumerate(samples):
    sub = summary[summary["sample"] == bc]
    if sub.empty:
        continue
    offset = (i - (len(samples)-1)/2) * (width*2)
    plt.bar(x + offset, sub["mean_5mC"], width=width, color="red", alpha=0.5, label=f"{bc} 5mC" if i == 0 else "")
    plt.bar(x + offset, sub["mean_5hmC"], width=width, color="blue", alpha=0.5, label=f"{bc} 5hmC" if i == 0 else "")

plt.xticks(x, chroms, rotation=45)
plt.ylabel("Average modification (%)")
plt.title("Average 5mC (red) and 5hmC (blue) levels per chromosome across samples")
plt.ylim(0, 100)
plt.legend(ncol=4)
plt.tight_layout()

out_fig = os.path.join(data_dir, "modkit_chr_comparison.png")
plt.savefig(out_fig, dpi=300)
plt.close()

# ========= OUTPUT TABLE =========
out_tsv = os.path.join(data_dir, "modkit_chr_comparison_summary.tsv")
summary.to_csv(out_tsv, sep="\t", index=False)

print(f"[DONE] Figure saved: {out_fig}")
print(f"[DONE] Summary table saved: {out_tsv}")
