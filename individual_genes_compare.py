#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
data_dir = "/mnt/e/Data/seq_for_human_293t2/modkit"
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"

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

# ========= PARSE GTF =========
def parse_gtf(gtf_path):
    gtf_cols = ["chrom","source","feature","start","end","score","strand","frame","attribute"]
    gtf = pd.read_csv(gtf_path, sep="\t", comment="#", names=gtf_cols)
    genes = gtf[gtf["feature"] == "gene"].copy()
    def parse_attr(attr):
        d = {}
        for item in attr.split(";"):
            if item.strip() == "":
                continue
            key, value = item.strip().split(" ", 1)
            d[key] = value.strip('"')
        return d
    genes["gene_id"] = genes["attribute"].apply(lambda x: parse_attr(x).get("gene_id",""))
    genes = genes[["chrom","start","end","gene_id"]]
    return genes

genes = parse_gtf(gtf_file)
genes = genes.sort_values(["chrom","start"]).reset_index(drop=True)

# ========= LINEARIZE GENOME =========
chrom_offsets = {}
offset = 0
for chrom in genes["chrom"].unique():
    chrom_len = genes[genes["chrom"]==chrom]["end"].max()
    chrom_offsets[chrom] = offset
    offset += chrom_len + 1_000_000
genes["genome_pos"] = genes.apply(lambda row: row["start"]+chrom_offsets[row["chrom"]], axis=1)

# ========= LOAD MODIFICATION DATA =========
dfs = []
for bc, f in samples_files.items():
    df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "#chrom" in df.columns:
        df = df.rename(columns={"#chrom":"chrom"})
    df = df[df["chrom"].isin(genes["chrom"].unique())]
    df["sample"] = bc
    dfs.append(df)

mod_all = pd.concat(dfs, ignore_index=True)

# ========= AGGREGATE 5mC AND 5hmC PER GENE =========
def compute_gene_mods(df_mod, df_genes):
    results = []
    for _, g in df_genes.iterrows():
        # find modifications within gene region
        sub = df_mod[(df_mod["chrom"]==g["chrom"]) & (df_mod["start"]>=g["start"]) & (df_mod["end"]<=g["end"])]
        mean_m = sub["percent_m"].mean() if not sub.empty else np.nan
        mean_h = sub["percent_h"].mean() if not sub.empty else np.nan
        results.append({
            "gene_id": g["gene_id"],
            "genome_pos": g["genome_pos"],
            "mean_5mC": mean_m,
            "mean_5hmC": mean_h
        })
    return pd.DataFrame(results)

# ========= PLOT CURVES =========
plt.figure(figsize=(20,6))
for bc in samples_files.keys():
    sub_mod = mod_all[mod_all["sample"]==bc]
    df_gene_mod = compute_gene_mods(sub_mod, genes)
    df_gene_mod.fillna(0, inplace=True)  # treat missing as 0
    # 5mC curve
    plt.plot(df_gene_mod["genome_pos"], df_gene_mod["mean_5mC"],
             color=sample_colors[bc], linestyle='-', alpha=0.8, label=f"{bc} 5mC")
    # 5hmC curve
    plt.plot(df_gene_mod["genome_pos"], df_gene_mod["mean_5hmC"],
             color=sample_colors[bc], linestyle='--', alpha=0.8, label=f"{bc} 5hmC")

plt.xlabel("Genome (linearized)")
plt.ylabel("Modification (%)")
plt.title("Genome-wide 5mC (solid) and 5hmC (dashed) per gene across samples")
plt.yscale("log")
plt.grid(True, linestyle=":", alpha=0.5)

# Optional: chromosome separator lines
for chrom, offset in chrom_offsets.items():
    plt.axvline(offset, color="gray", linestyle="--", alpha=0.3)
    plt.text(offset, plt.ylim()[1]*0.5, chrom, rotation=90, fontsize=8, alpha=0.7)

# Clean legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), ncol=4, fontsize=10)

plt.tight_layout()
out_fig = os.path.join(data_dir,"modkit_genome_gene_curve.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"[DONE] Figure saved: {out_fig}")
