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

sample_colors = {
    "barcode04": "red",
    "barcode05": "green",
    "barcode06": "blue",
    "barcode07": "purple",
}

# ========= LOAD MODIFICATION DATA =========
mod_all = []
for bc, f in samples_files.items():
    df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "#chrom" in df.columns:
        df = df.rename(columns={"#chrom": "chrom"})
    df["sample"] = bc
    mod_all.append(df)

mod_all = pd.concat(mod_all, ignore_index=True)

# ========= CREATE GENOME-WIDE POSITION =========
chroms = sorted(mod_all["chrom"].unique(), key=lambda x: (x.replace("chr","") if x[3:].isdigit() else x))
chrom_offsets = {}
offset = 0
chrom_sizes = {}

for chrom in chroms:
    max_pos = mod_all[mod_all["chrom"]==chrom]["end"].max()
    chrom_offsets[chrom] = offset
    chrom_sizes[chrom] = max_pos
    offset += max_pos + 1_000_000  # spacer

mod_all["genome_pos"] = mod_all.apply(lambda row: row["start"] + chrom_offsets[row["chrom"]], axis=1)

# ========= PLOT 5mC AND 5hmC =========
fig, axes = plt.subplots(2, 1, figsize=(20,8), sharex=True)

for bc in samples_files.keys():
    sub = mod_all[mod_all["sample"]==bc]
    axes[0].plot(sub["genome_pos"], sub["percent_m"], color=sample_colors[bc], alpha=0.7, label=bc)
    axes[1].plot(sub["genome_pos"], sub["percent_h"], color=sample_colors[bc], alpha=0.7, label=bc)

axes[0].set_ylabel("5mC (%)")
axes[1].set_ylabel("5hmC (%)")
axes[1].set_xlabel("Genome (linearized)")

axes[0].set_yscale("log")
axes[1].set_yscale("log")

axes[0].set_title("Genome-wide 5mC modification per sample")
axes[1].set_title("Genome-wide 5hmC modification per sample")

axes[0].grid(True, linestyle=":", alpha=0.5)
axes[1].grid(True, linestyle=":", alpha=0.5)

# Chromosome separation lines
for chrom, offset in chrom_offsets.items():
    for ax in axes:
        ax.axvline(offset, color="grey", linestyle="--", alpha=0.3)
        ax.text(offset, ax.get_ylim()[1]*0.5, chrom, rotation=90, fontsize=8, alpha=0.7)

# Legends
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=4, fontsize=9)

plt.tight_layout()
out_fig = os.path.join(data_dir, "modkit_genome_tracks.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"[DONE] Figure saved: {out_fig}")
