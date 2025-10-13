import os
import pandas as pd
import plotly.graph_objects as go

data_dir = "/mnt/e/Data/seq_for_human_293t2/modkit"
samples_files = {
    "barcode04": os.path.join(data_dir, "barcode04_aligned_with_mod.region_mh.stats.tsv"),
    "barcode05": os.path.join(data_dir, "barcode05_aligned_with_mod.region_mh.stats.tsv"),
    "barcode06": os.path.join(data_dir, "barcode06_aligned_with_mod.region_mh.stats.tsv"),
    "barcode07": os.path.join(data_dir, "barcode07_aligned_with_mod.region_mh.stats.tsv"),
}
sample_colors = {"barcode04":"red","barcode05":"green","barcode06":"blue","barcode07":"purple"}

# Load all modification files
dfs = []
for bc, f in samples_files.items():
    df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "#chrom" in df.columns:
        df = df.rename(columns={"#chrom":"chrom"})
    df["sample"] = bc
    dfs.append(df)
mod_all = pd.concat(dfs, ignore_index=True)

# Linearize genome
chroms = sorted(mod_all["chrom"].unique(), key=lambda x: (x.replace("chr","") if x[3:].isdigit() else x))
chrom_offsets = {}
offset = 0
for chrom in chroms:
    max_pos = mod_all[mod_all["chrom"]==chrom]["end"].max()
    chrom_offsets[chrom] = offset
    offset += max_pos + 1_000_000
mod_all["genome_pos"] = mod_all.apply(lambda row: row["start"] + chrom_offsets[row["chrom"]], axis=1)

# ========= PLOTLY FIGURE =========
fig = go.Figure()

for bc in samples_files.keys():
    sub = mod_all[mod_all["sample"]==bc].sort_values("genome_pos")
    # 5mC histogram-like stepped line
    fig.add_trace(go.Scatter(
        x=sub["genome_pos"],
        y=sub["percent_m"],
        mode="lines",
        line=dict(color=sample_colors[bc], width=2, shape="hv"),
        name=f"{bc} 5mC"
    ))
    # 5hmC histogram-like stepped line
    fig.add_trace(go.Scatter(
        x=sub["genome_pos"],
        y=sub["percent_h"],
        mode="lines",
        line=dict(color=sample_colors[bc], width=2, dash="dash", shape="hv"),
        name=f"{bc} 5hmC"
    ))

# Layout for genome-wide visualization
fig.update_layout(
    title="Genome-wide 5mC (solid) and 5hmC (dashed) per sample",
    xaxis_title="Genome (linearized)",
    yaxis_title="Modification (%)",
    yaxis_type="log",
    autosize=True,
    height=800,
    width=2000,
    hovermode="x unified"
)

# Add vertical lines for chromosome separators
for chrom, offset in chrom_offsets.items():
    fig.add_vline(x=offset, line=dict(color="gray", width=1, dash="dot"))
    fig.add_annotation(x=offset, y=mod_all[["percent_m","percent_h"]].max().max(),
                       text=chrom, showarrow=False, yshift=10, font=dict(size=10, color="gray"))

# Save interactive HTML
out_html = os.path.join(data_dir, "modkit_genome_tracks.html")
fig.write_html(out_html)
print(f"[DONE] Interactive figure saved: {out_html}")
