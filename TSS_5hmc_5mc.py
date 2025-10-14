#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "tss_methylation")
os.makedirs(out_dir, exist_ok=True)

flank_size = 2000  # ±2kb flanking regions
n_gene_bins = 30   # Number of bins for gene body

# -------------------
# Extract gene coordinates (TSS and TES)
# -------------------
print("Extracting genes from GTF...")
genes = []
with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        chrom = fields[0] if fields[0].startswith('chr') else 'chr' + fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        
        if strand == '+':
            tss, tes = start, end
        else:
            tss, tes = end, start
        
        genes.append({
            'chrom': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'tss': tss,
            'tes': tes,
            'length': abs(end - start)
        })

genes_df = pd.DataFrame(genes)
# Filter genes with reasonable length (>500bp)
genes_df = genes_df[genes_df['length'] > 500]
print(f"Loaded {len(genes_df)} genes")

# -------------------
# Process samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

all_profiles = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing {sample_name}...")
    
    # Load methylation data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    # Create position bins
    # Upstream: 20 bins from -2kb to TSS
    # Gene body: 30 bins from TSS to TES (0-100%)
    # Downstream: 20 bins from TES to +2kb
    n_flank_bins = 20
    
    upstream_bins = np.linspace(-flank_size, 0, n_flank_bins + 1)
    genebody_bins = np.linspace(0, 100, n_gene_bins + 1)
    downstream_bins = np.linspace(0, flank_size, n_flank_bins + 1)
    
    # Storage for binned data
    upstream_5mc = [[] for _ in range(n_flank_bins)]
    upstream_5hmc = [[] for _ in range(n_flank_bins)]
    genebody_5mc = [[] for _ in range(n_gene_bins)]
    genebody_5hmc = [[] for _ in range(n_gene_bins)]
    downstream_5mc = [[] for _ in range(n_flank_bins)]
    downstream_5hmc = [[] for _ in range(n_flank_bins)]
    
    # Process each gene
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="Genes"):
        chrom = gene['chrom']
        tss = gene['tss']
        tes = gene['tes']
        strand = gene['strand']
        
        # Get methylation data for this region
        if strand == '+':
            region_start = tss - flank_size
            region_end = tes + flank_size
        else:
            region_start = tes - flank_size
            region_end = tss + flank_size
        
        meth_region = meth_df[
            (meth_df['chrom'] == chrom) &
            (meth_df['end'] > min(region_start, region_end)) &
            (meth_df['start'] < max(region_start, region_end))
        ]
        
        if len(meth_region) == 0:
            continue
        
        for _, meth in meth_region.iterrows():
            center = (meth['start'] + meth['end']) / 2
            
            # Calculate position relative to gene
            if strand == '+':
                # Upstream region
                if center < tss:
                    dist = center - tss
                    bin_idx = np.digitize(dist, upstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        upstream_5mc[bin_idx].append(meth['percent_m'])
                        upstream_5hmc[bin_idx].append(meth['percent_h'])
                
                # Gene body
                elif tss <= center <= tes:
                    pct = 100 * (center - tss) / (tes - tss)
                    bin_idx = np.digitize(pct, genebody_bins) - 1
                    if 0 <= bin_idx < n_gene_bins:
                        genebody_5mc[bin_idx].append(meth['percent_m'])
                        genebody_5hmc[bin_idx].append(meth['percent_h'])
                
                # Downstream region
                elif center > tes:
                    dist = center - tes
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        downstream_5mc[bin_idx].append(meth['percent_m'])
                        downstream_5hmc[bin_idx].append(meth['percent_h'])
            
            else:  # Minus strand
                # Upstream (right side, higher coordinates)
                if center > tss:
                    dist = center - tss
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        upstream_5mc[n_flank_bins - 1 - bin_idx].append(meth['percent_m'])
                        upstream_5hmc[n_flank_bins - 1 - bin_idx].append(meth['percent_h'])
                
                # Gene body (reversed)
                elif tes <= center <= tss:
                    pct = 100 * (tss - center) / (tss - tes)
                    bin_idx = np.digitize(pct, genebody_bins) - 1
                    if 0 <= bin_idx < n_gene_bins:
                        genebody_5mc[bin_idx].append(meth['percent_m'])
                        genebody_5hmc[bin_idx].append(meth['percent_h'])
                
                # Downstream (left side, lower coordinates)
                elif center < tes:
                    dist = tes - center
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        downstream_5mc[n_flank_bins - 1 - bin_idx].append(meth['percent_m'])
                        downstream_5hmc[n_flank_bins - 1 - bin_idx].append(meth['percent_h'])
    
    # Calculate means
    profile_5mc = []
    profile_5hmc = []
    x_labels = []
    
    # Upstream
    for i, bin_start in enumerate(upstream_bins[:-1]):
        profile_5mc.append(np.nanmean(upstream_5mc[i]) if upstream_5mc[i] else np.nan)
        profile_5hmc.append(np.nanmean(upstream_5hmc[i]) if upstream_5hmc[i] else np.nan)
        x_labels.append(bin_start)
    
    # Gene body
    for i, pct in enumerate(genebody_bins[:-1]):
        profile_5mc.append(np.nanmean(genebody_5mc[i]) if genebody_5mc[i] else np.nan)
        profile_5hmc.append(np.nanmean(genebody_5hmc[i]) if genebody_5hmc[i] else np.nan)
        x_labels.append(f"{pct:.0f}%")
    
    # Downstream
    for i, bin_start in enumerate(downstream_bins[:-1]):
        profile_5mc.append(np.nanmean(downstream_5mc[i]) if downstream_5mc[i] else np.nan)
        profile_5hmc.append(np.nanmean(downstream_5hmc[i]) if downstream_5hmc[i] else np.nan)
        x_labels.append(f"+{bin_start:.0f}")
    
    all_profiles[sample_name] = {
        '5mC': profile_5mc,
        '5hmC': profile_5hmc,
        'x_labels': x_labels
    }

# -------------------
# Plot like Figure 4a
# -------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

x_pos = np.arange(len(x_labels))
tss_pos = n_flank_bins
tes_pos = n_flank_bins + n_gene_bins

# 5mC
for sample, data in all_profiles.items():
    ax1.plot(x_pos, data['5mC'], linewidth=2, label=sample, alpha=0.85)
ax1.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.text(tss_pos, ax1.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=10, fontweight='bold')
ax1.text(tes_pos, ax1.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=10, fontweight='bold')
ax1.set_xlabel("Genomic region (5' → 3')", fontsize=12, fontweight='bold')
ax1.set_ylabel('5mC level (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Distribution Across Genes', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, tss_pos, tss_pos + n_gene_bins//3, tss_pos + 2*n_gene_bins//3, tes_pos, len(x_labels)-1])
ax1.set_xticklabels(['-2kb', 'TSS', '33%', '66%', 'TES', '+2kb'])

# 5hmC
for sample, data in all_profiles.items():
    ax2.plot(x_pos, data['5hmC'], linewidth=2, label=sample, alpha=0.85)
ax2.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(tss_pos, ax2.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=10, fontweight='bold')
ax2.text(tes_pos, ax2.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel("Genomic region (5' → 3')", fontsize=12, fontweight='bold')
ax2.set_ylabel('5hmC level (%)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Distribution Across Genes', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, tss_pos, tss_pos + n_gene_bins//3, tss_pos + 2*n_gene_bins//3, tes_pos, len(x_labels)-1])
ax2.set_xticklabels(['-2kb', 'TSS', '33%', '66%', 'TES', '+2kb'])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gene_body_methylation_fig4a.png'), dpi=300)
plt.close()

print(f"\n✓ Figure 4a-style plot saved: {out_dir}/gene_body_methylation_fig4a.png")
