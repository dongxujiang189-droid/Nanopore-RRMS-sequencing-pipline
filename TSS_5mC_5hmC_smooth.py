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
n_gene_bins = 60   # Number of bins for gene body (increased from 30 for smoother curve)
n_flank_bins = 40  # Number of bins for flanking regions (increased from 20)

# Optional: Apply smoothing
use_smoothing = True  # Set to False to disable
smooth_window = 5     # Window size for moving average (must be odd number)

# -------------------
# Helper function for smoothing
# -------------------
def smooth_profile(data, window_size):
    """Apply moving average smoothing to profile data"""
    if window_size < 3 or window_size % 2 == 0:
        return data
    
    data_array = np.array(data)
    smoothed = np.copy(data_array)
    half_window = window_size // 2
    
    for i in range(len(data_array)):
        start = max(0, i - half_window)
        end = min(len(data_array), i + half_window + 1)
        window_data = data_array[start:end]
        smoothed[i] = np.nanmean(window_data)
    
    return smoothed.tolist()

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
    # Upstream: bins from -2kb to TSS
    # Gene body: bins from TSS to TES (0-100%)
    # Downstream: bins from TES to +2kb
    
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
    
    # Apply smoothing if enabled
    if use_smoothing:
        profile_5mc = smooth_profile(profile_5mc, smooth_window)
        profile_5hmc = smooth_profile(profile_5hmc, smooth_window)
    
    all_profiles[sample_name] = {
        '5mC': profile_5mc,
        '5hmC': profile_5hmc,
        'x_labels': x_labels
    }

# -------------------
# Calculate log2 fold change vs control
# -------------------
# Use first sample as control (or specify: control_name = "barcode07")
control_name = list(all_profiles.keys())[3]
print(f"\nCalculating log2 fold change using {control_name} as control")

log2fc_profiles = {}
for sample, data in all_profiles.items():
    control_5mc = np.array(all_profiles[control_name]['5mC'])
    control_5hmc = np.array(all_profiles[control_name]['5hmC'])
    
    sample_5mc = np.array(data['5mC'])
    sample_5hmc = np.array(data['5hmC'])
    
    # Calculate log2 fold change: log2(sample/control)
    # Add pseudocount of 1 to avoid division by zero (since values are percentages 0-100)
    pseudocount = 1.0
    
    log2fc_5mc = np.log2((sample_5mc + pseudocount) / (control_5mc + pseudocount))
    log2fc_5hmc = np.log2((sample_5hmc + pseudocount) / (control_5hmc + pseudocount))
    
    # Replace any inf or -inf with NaN
    log2fc_5mc = np.where(np.isfinite(log2fc_5mc), log2fc_5mc, np.nan)
    log2fc_5hmc = np.where(np.isfinite(log2fc_5hmc), log2fc_5hmc, np.nan)
    
    log2fc_profiles[sample] = {
        '5mC': log2fc_5mc,
        '5hmC': log2fc_5hmc
    }
    
    if sample != control_name:
        print(f"  {sample}: 5mC range [{np.nanmin(log2fc_5mc):.2f}, {np.nanmax(log2fc_5mc):.2f}], "
              f"5hmC range [{np.nanmin(log2fc_5hmc):.2f}, {np.nanmax(log2fc_5hmc):.2f}]")

# -------------------
# Plot both versions
# -------------------
x_pos = np.arange(len(x_labels))
tss_pos = n_flank_bins
tes_pos = n_flank_bins + n_gene_bins

# Regular plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

for sample, data in all_profiles.items():
    ax1.plot(x_pos, data['5mC'], linewidth=2, label=sample, alpha=0.85)
ax1.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.text(tss_pos, ax1.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=10, fontweight='bold')
ax1.text(tes_pos, ax1.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=10, fontweight='bold')
ax1.set_xlabel("Genomic region (5' → 3')", fontsize=12, fontweight='bold')
ax1.set_ylabel('5mC level (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, tss_pos, tss_pos + n_gene_bins//3, tss_pos + 2*n_gene_bins//3, tes_pos, len(x_labels)-1])
ax1.set_xticklabels(['-2kb', 'TSS', '33%', '66%', 'TES', '+2kb'])

for sample, data in all_profiles.items():
    ax2.plot(x_pos, data['5hmC'], linewidth=2, label=sample, alpha=0.85)
ax2.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(tss_pos, ax2.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=10, fontweight='bold')
ax2.text(tes_pos, ax2.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel("Genomic region (5' → 3')", fontsize=12, fontweight='bold')
ax2.set_ylabel('5hmC level (%)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, tss_pos, tss_pos + n_gene_bins//3, tss_pos + 2*n_gene_bins//3, tes_pos, len(x_labels)-1])
ax2.set_xticklabels(['-2kb', 'TSS', '33%', '66%', 'TES', '+2kb'])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gene_body_methylation_absolute.png'), dpi=300)
plt.close()

# Log2 fold change plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# 5mC log2 fold change
for sample, data in log2fc_profiles.items():
    if sample == control_name:
        continue  # Skip control (always 0)
    ax1.plot(x_pos, data['5mC'], linewidth=2.5, label=sample, alpha=0.85)

ax1.axhline(y=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='No change')
ax1.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.text(tss_pos, ax1.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=10, fontweight='bold')
ax1.text(tes_pos, ax1.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=10, fontweight='bold')
ax1.set_xlabel("Genomic region (5' → 3')", fontsize=12, fontweight='bold')
ax1.set_ylabel('5mC level\n(Log₂ fold change vs control)', fontsize=12, fontweight='bold')
ax1.set_title(f'5mC: Log₂ Fold Change vs {control_name}', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, tss_pos, tss_pos + n_gene_bins//3, tss_pos + 2*n_gene_bins//3, tes_pos, len(x_labels)-1])
ax1.set_xticklabels(['-2kb', 'TSS', '33%', '66%', 'TES', '+2kb'])
# Add horizontal shading for fold change interpretation
ax1.axhspan(-1, 1, alpha=0.1, color='gray', label='<2-fold change')

# 5hmC log2 fold change
for sample, data in log2fc_profiles.items():
    if sample == control_name:
        continue
    ax2.plot(x_pos, data['5hmC'], linewidth=2.5, label=sample, alpha=0.85)

ax2.axhline(y=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='No change')
ax2.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(tss_pos, ax2.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=10, fontweight='bold')
ax2.text(tes_pos, ax2.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel("Genomic region (5' → 3')", fontsize=12, fontweight='bold')
ax2.set_ylabel('5hmC level\n(Log₂ fold change vs control)', fontsize=12, fontweight='bold')
ax2.set_title(f'5hmC: Log₂ Fold Change vs {control_name}', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, tss_pos, tss_pos + n_gene_bins//3, tss_pos + 2*n_gene_bins//3, tes_pos, len(x_labels)-1])
ax2.set_xticklabels(['-2kb', 'TSS', '33%', '66%', 'TES', '+2kb'])
ax2.axhspan(-1, 1, alpha=0.1, color='gray', label='<2-fold change')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gene_body_methylation_log2fc.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save log2 fold change data
log2fc_data = {'distance_from_tss': x_labels}
for sample, data in log2fc_profiles.items():
    if sample != control_name:
        log2fc_data[f'{sample}_5mC_log2fc'] = data['5mC']
        log2fc_data[f'{sample}_5hmC_log2fc'] = data['5hmC']

log2fc_df = pd.DataFrame(log2fc_data)
log2fc_df.to_csv(os.path.join(out_dir, 'gene_body_log2fc_data.csv'), index=False)

print(f"\n✓ Plots saved:")
print(f"  {out_dir}/gene_body_methylation_absolute.png")
print(f"  {out_dir}/gene_body_methylation_log2fc.png")
print(f"  {out_dir}/gene_body_log2fc_data.csv")
print(f"\nSettings used:")
print(f"  Gene body bins: {n_gene_bins}")
print(f"  Flanking bins: {n_flank_bins}")
print(f"  Smoothing: {'Enabled (window={})'.format(smooth_window) if use_smoothing else 'Disabled'}")
