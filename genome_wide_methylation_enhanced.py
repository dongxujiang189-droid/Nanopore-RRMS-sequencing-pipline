#!/usr/bin/env python3
"""
Enhanced Genome-wide Methylation Visualization Tool
Comprehensive analysis and visualization of 5mC and 5hmC across entire genome
Author: Genomics Analysis Pipeline
Version: 2.1 - Modified for complete chromosome analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import glob
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Configuration
# -------------------
class Config:
    base_dir = "/mnt/e/Data/seq_for_human_293t2/"
    input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
    out_dir = os.path.join(base_dir, "genome_wide_methylation")
    
    # Resolution settings
    bin_size = 500000  # *** CHANGED: 0.5 Mb bins ***
    smoothing_sigma = 2  # Gaussian smoothing (0 = no smoothing, 3 = strong)
    
    # Chromosome settings
    chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    
    # Visualization settings
    figsize_wide = (24, 6)
    figsize_tall = (24, 10)
    dpi = 300
    
    # Statistical settings
    min_coverage = 5  # Minimum valid reads per bin
    
    # Control sample - *** CHANGED: Set to 07 sample ***
    control_sample = "07"  # Will match any sample containing "07"

config = Config()
os.makedirs(config.out_dir, exist_ok=True)

# -------------------
# Chromosome information (hg38)
# -------------------
CHROM_SIZES = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
    'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
    'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
    'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
    'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
    'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
    'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
}

# Calculate cumulative positions
cumulative_positions = {}
cumulative_pos = 0
for chrom in config.chromosomes:
    if chrom in CHROM_SIZES:
        cumulative_positions[chrom] = cumulative_pos
        cumulative_pos += CHROM_SIZES[chrom]

print(f"Genome size: {cumulative_pos:,} bp ({cumulative_pos/1e9:.2f} Gb)")

# -------------------
# Helper functions
# -------------------
def load_sample_data(sample_file):
    """Load and preprocess sample methylation data"""
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    
    df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in df.columns:
        df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    # Ensure chr prefix
    if not str(df['chrom'].iloc[0]).startswith('chr'):
        df['chrom'] = 'chr' + df['chrom'].astype(str)
    
    # Filter standard chromosomes
    df = df[df['chrom'].isin(config.chromosomes)]
    
    # Calculate genome position
    df['genome_pos'] = df.apply(
        lambda row: cumulative_positions.get(row['chrom'], 0) + (row['start'] + row['end']) / 2,
        axis=1
    )
    
    return sample_name, df

def bin_genome_data(df, bin_size):
    """Bin methylation data across genome"""
    df['bin'] = (df['genome_pos'] // bin_size).astype(int)
    
    binned = df.groupby('bin').agg({
        'percent_m': 'mean',
        'percent_h': 'mean',
        'genome_pos': 'mean',
        'chrom': 'first',
        'count_valid_m': 'sum',
        'count_valid_h': 'sum'
    }).reset_index()
    
    # Filter low coverage bins
    binned = binned[
        (binned['count_valid_m'] >= config.min_coverage) | 
        (binned['count_valid_h'] >= config.min_coverage)
    ]
    
    # Apply smoothing if configured
    if config.smoothing_sigma > 0:
        binned['percent_m_smooth'] = gaussian_filter1d(
            binned['percent_m'].fillna(binned['percent_m'].mean()), 
            sigma=config.smoothing_sigma
        )
        binned['percent_h_smooth'] = gaussian_filter1d(
            binned['percent_h'].fillna(binned['percent_h'].mean()), 
            sigma=config.smoothing_sigma
        )
    else:
        binned['percent_m_smooth'] = binned['percent_m']
        binned['percent_h_smooth'] = binned['percent_h']
    
    return binned

def get_chrom_colors():
    """Generate alternating colors for chromosomes"""
    colors = {}
    for i, chrom in enumerate(config.chromosomes):
        colors[chrom] = '#E8E8E8' if i % 2 == 0 else '#F8F8F8'
    return colors

# -------------------
# Load all samples
# -------------------
print(f"\n{'='*60}")
print("LOADING SAMPLES")
print('='*60)

sample_files = glob.glob(config.input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {config.input_pattern}")

all_samples = {}
for sample_file in tqdm(sample_files, desc="Loading samples"):
    sample_name, raw_data = load_sample_data(sample_file)
    binned_data = bin_genome_data(raw_data, config.bin_size)
    all_samples[sample_name] = binned_data
    print(f"  {sample_name}: {len(raw_data):,} sites → {len(binned_data):,} bins")

# *** CHANGED: Set control sample to match "07" ***
control_name = None
for sample_name in all_samples.keys():
    if "07" in sample_name:
        control_name = sample_name
        break

if control_name is None:
    control_name = list(all_samples.keys())[0]
    print(f"\nWARNING: No sample containing '07' found. Using first sample as control.")

print(f"\nControl sample: {control_name}")

# -------------------
# Calculate statistics
# -------------------
print(f"\n{'='*60}")
print("CALCULATING STATISTICS")
print('='*60)

summary_stats = []
for sample_name, data in all_samples.items():
    stats_dict = {
        'Sample': sample_name,
        '5mC_mean': data['percent_m'].mean(),
        '5mC_median': data['percent_m'].median(),
        '5mC_std': data['percent_m'].std(),
        '5hmC_mean': data['percent_h'].mean(),
        '5hmC_median': data['percent_h'].median(),
        '5hmC_std': data['percent_h'].std(),
        'N_bins': len(data),
        'Genome_coverage_Mb': len(data) * config.bin_size / 1e6
    }
    summary_stats.append(stats_dict)

stats_df = pd.DataFrame(summary_stats)
stats_df.to_csv(os.path.join(config.out_dir, 'genome_wide_statistics.csv'), index=False)
print(stats_df.to_string(index=False))

# -------------------
# Calculate correlations
# -------------------
if len(all_samples) > 1:
    print(f"\n{'='*60}")
    print("SAMPLE CORRELATIONS")
    print('='*60)
    
    # Create matrices for correlation
    sample_names = list(all_samples.keys())
    n_samples = len(sample_names)
    
    corr_5mc = np.zeros((n_samples, n_samples))
    corr_5hmc = np.zeros((n_samples, n_samples))
    
    for i, s1 in enumerate(sample_names):
        for j, s2 in enumerate(sample_names):
            # Merge on genome position
            merged = pd.merge(
                all_samples[s1][['genome_pos', 'percent_m', 'percent_h']],
                all_samples[s2][['genome_pos', 'percent_m', 'percent_h']],
                on='genome_pos', suffixes=('_1', '_2')
            )
            
            if len(merged) > 0:
                corr_5mc[i, j] = merged['percent_m_1'].corr(merged['percent_m_2'])
                corr_5hmc[i, j] = merged['percent_h_1'].corr(merged['percent_h_2'])
    
    # Save correlation matrices
    corr_5mc_df = pd.DataFrame(corr_5mc, index=sample_names, columns=sample_names)
    corr_5hmc_df = pd.DataFrame(corr_5hmc, index=sample_names, columns=sample_names)
    
    corr_5mc_df.to_csv(os.path.join(config.out_dir, 'correlation_5mC.csv'))
    corr_5hmc_df.to_csv(os.path.join(config.out_dir, 'correlation_5hmC.csv'))
    
    print("\n5mC Correlation Matrix:")
    print(corr_5mc_df.round(3))
    print("\n5hmC Correlation Matrix:")
    print(corr_5hmc_df.round(3))

# -------------------
# Plot 1: Genome-wide overview (all samples)
# -------------------
print(f"\n{'='*60}")
print("GENERATING PLOTS")
print('='*60)

print("1. Genome-wide overview...")
fig = plt.figure(figsize=config.figsize_tall)
gs = GridSpec(3, 1, height_ratios=[1, 1, 0.3], hspace=0.3)

chrom_colors = get_chrom_colors()
colors = plt.cm.Set2(np.linspace(0, 1, len(all_samples)))

# 5mC plot
ax1 = fig.add_subplot(gs[0])
for idx, (sample, data) in enumerate(all_samples.items()):
    ax1.plot(data['genome_pos'] / 1e9, data['percent_m_smooth'], 
             linewidth=2, alpha=0.8, label=sample, color=colors[idx])

# Add chromosome backgrounds
for chrom in config.chromosomes:
    if chrom in CHROM_SIZES and chrom in cumulative_positions:
        start = cumulative_positions[chrom] / 1e9
        width = CHROM_SIZES[chrom] / 1e9
        ax1.axvspan(start, start + width, facecolor=chrom_colors[chrom], alpha=0.3)

ax1.set_ylabel('5mC (%)', fontsize=12, fontweight='bold')
ax1.set_title('Genome-wide Methylation Profile', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.2, axis='y')
ax1.set_ylim([0, 100])
ax1.set_xlim([0, cumulative_pos / 1e9])

# 5hmC plot
ax2 = fig.add_subplot(gs[1], sharex=ax1)
for idx, (sample, data) in enumerate(all_samples.items()):
    ax2.plot(data['genome_pos'] / 1e9, data['percent_h_smooth'], 
             linewidth=2, alpha=0.8, label=sample, color=colors[idx])

for chrom in config.chromosomes:
    if chrom in CHROM_SIZES and chrom in cumulative_positions:
        start = cumulative_positions[chrom] / 1e9
        width = CHROM_SIZES[chrom] / 1e9
        ax2.axvspan(start, start + width, facecolor=chrom_colors[chrom], alpha=0.3)

ax2.set_ylabel('5hmC (%)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.2, axis='y')
ax2.set_ylim([0, max(20, np.nanmax([d['percent_h_smooth'].max() for d in all_samples.values()]) * 1.1)])
ax2.set_xlim([0, cumulative_pos / 1e9])

# Chromosome labels
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.set_ylim([0, 1])
ax3.set_xlim([0, cumulative_pos / 1e9])
ax3.axis('off')

for chrom in config.chromosomes:
    if chrom in CHROM_SIZES and chrom in cumulative_positions:
        center = (cumulative_positions[chrom] + CHROM_SIZES[chrom] / 2) / 1e9
        ax3.text(center, 0.5, chrom.replace('chr', ''), 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw chromosome bar
        start = cumulative_positions[chrom] / 1e9
        width = CHROM_SIZES[chrom] / 1e9
        rect = mpatches.Rectangle((start, 0.2), width, 0.6, 
                                  facecolor=chrom_colors[chrom], 
                                  edgecolor='black', linewidth=0.5)
        ax3.add_patch(rect)

ax3.text(cumulative_pos / 1e9 / 2, -0.3, 'Genomic Position (Gb)', 
         ha='center', fontsize=12, fontweight='bold')

plt.savefig(os.path.join(config.out_dir, 'genome_wide_overview.png'), 
            dpi=config.dpi, bbox_inches='tight')
plt.close()

# -------------------
# *** NEW: Plot 2: Combined methylation profile comparison ***
# -------------------
print("2. Combined methylation profile comparison...")
fig, axes = plt.subplots(len(config.chromosomes), 2, 
                         figsize=(20, 4*len(config.chromosomes)), 
                         sharex='col')

for chrom_idx, chrom in enumerate(tqdm(config.chromosomes, desc="Combined profiles")):
    if chrom not in CHROM_SIZES:
        continue
    
    ax_5mc = axes[chrom_idx, 0] if len(config.chromosomes) > 1 else axes[0]
    ax_5hmc = axes[chrom_idx, 1] if len(config.chromosomes) > 1 else axes[1]
    
    for idx, (sample, data) in enumerate(all_samples.items()):
        chrom_data = data[data['chrom'] == chrom]
        if len(chrom_data) == 0:
            continue
        
        pos = chrom_data['genome_pos'] - cumulative_positions[chrom]
        
        # 5mC
        ax_5mc.plot(pos / 1e6, chrom_data['percent_m_smooth'], 
                    linewidth=1.5, alpha=0.7, label=sample, color=colors[idx])
        
        # 5hmC
        ax_5hmc.plot(pos / 1e6, chrom_data['percent_h_smooth'], 
                     linewidth=1.5, alpha=0.7, label=sample, color=colors[idx])
    
    # 5mC formatting
    ax_5mc.set_ylabel(f'{chrom}\n5mC (%)', fontsize=10, fontweight='bold')
    ax_5mc.grid(True, alpha=0.3)
    ax_5mc.set_ylim([0, 100])
    if chrom_idx == 0:
        ax_5mc.set_title('5mC Methylation', fontsize=12, fontweight='bold')
        ax_5mc.legend(fontsize=8, loc='upper right', ncol=2)
    
    # 5hmC formatting
    ax_5hmc.set_ylabel(f'{chrom}\n5hmC (%)', fontsize=10, fontweight='bold')
    ax_5hmc.grid(True, alpha=0.3)
    if chrom_idx == 0:
        ax_5hmc.set_title('5hmC Methylation', fontsize=12, fontweight='bold')
        ax_5hmc.legend(fontsize=8, loc='upper right', ncol=2)
    
    # X-axis only on bottom
    if chrom_idx == len(config.chromosomes) - 1:
        ax_5mc.set_xlabel('Position (Mb)', fontsize=11, fontweight='bold')
        ax_5hmc.set_xlabel('Position (Mb)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(config.out_dir, 'combined_methylation_profiles.png'), 
            dpi=config.dpi, bbox_inches='tight')
plt.close()

# -------------------
# Plot 3: Per-chromosome detailed view (*** CHANGED: ALL chromosomes ***)
# -------------------
print("3. Per-chromosome detailed plots...")
for chrom in tqdm(config.chromosomes, desc="Individual chromosomes"):  # *** ALL chromosomes ***
    if chrom not in CHROM_SIZES:
        continue
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    for idx, (sample, data) in enumerate(all_samples.items()):
        chrom_data = data[data['chrom'] == chrom]
        if len(chrom_data) == 0:
            continue
        
        pos = chrom_data['genome_pos'] - cumulative_positions[chrom]
        ax1.plot(pos / 1e6, chrom_data['percent_m_smooth'], 
                linewidth=2, alpha=0.8, label=sample, color=colors[idx])
        ax2.plot(pos / 1e6, chrom_data['percent_h_smooth'], 
                linewidth=2, alpha=0.8, label=sample, color=colors[idx])
    
    ax1.set_ylabel('5mC (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{chrom} Methylation Profile', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    ax2.set_ylabel('5hmC (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Position (Mb)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.out_dir, f'{chrom}_methylation.png'), 
                dpi=config.dpi, bbox_inches='tight')
    plt.close()

# -------------------
# Plot 4: Log2 fold change vs control (*** CHANGED: control = 07 sample ***)
# -------------------
if len(all_samples) > 1:
    print("4. Log2 fold change plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figsize_tall, sharex=True)
    
    control_data = all_samples[control_name]
    
    for idx, (sample, data) in enumerate(all_samples.items()):
        if sample == control_name:
            continue
        
        # Merge with control
        merged = pd.merge(
            control_data[['genome_pos', 'percent_m', 'percent_h']],
            data[['genome_pos', 'percent_m', 'percent_h']],
            on='genome_pos', suffixes=('_ctrl', '_sample')
        )
        
        # Calculate log2 fold change
        log2fc_5mc = np.log2((merged['percent_m_sample'] + 1) / (merged['percent_m_ctrl'] + 1))
        log2fc_5hmc = np.log2((merged['percent_h_sample'] + 1) / (merged['percent_h_ctrl'] + 1))
        
        # Smooth
        if config.smoothing_sigma > 0:
            log2fc_5mc = gaussian_filter1d(log2fc_5mc, sigma=config.smoothing_sigma)
            log2fc_5hmc = gaussian_filter1d(log2fc_5hmc, sigma=config.smoothing_sigma)
        
        ax1.plot(merged['genome_pos'] / 1e9, log2fc_5mc, 
                linewidth=2, alpha=0.8, label=sample, color=colors[idx])
        ax2.plot(merged['genome_pos'] / 1e9, log2fc_5hmc, 
                linewidth=2, alpha=0.8, label=sample, color=colors[idx])
    
    # Add chromosome backgrounds
    for chrom in config.chromosomes:
        if chrom in CHROM_SIZES and chrom in cumulative_positions:
            start = cumulative_positions[chrom] / 1e9
            width = CHROM_SIZES[chrom] / 1e9
            ax1.axvspan(start, start + width, facecolor=chrom_colors[chrom], alpha=0.3)
            ax2.axvspan(start, start + width, facecolor=chrom_colors[chrom], alpha=0.3)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=-1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_ylabel(f'5mC Log₂FC\nvs {control_name}', fontsize=12, fontweight='bold')
    ax1.set_title('Genome-wide Log₂ Fold Change', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis='y')
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_ylabel(f'5hmC Log₂FC\nvs {control_name}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Genomic Position (Gb)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.out_dir, 'genome_wide_log2fc.png'), 
                dpi=config.dpi, bbox_inches='tight')
    plt.close()

# -------------------
# Plot 5: Correlation heatmaps
# -------------------
if len(all_samples) > 1:
    print("5. Correlation heatmaps...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(corr_5mc_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.5, vmin=0, vmax=1, square=True, ax=ax1,
                cbar_kws={'label': 'Pearson Correlation'})
    ax1.set_title('5mC Correlation', fontsize=13, fontweight='bold')
    
    sns.heatmap(corr_5hmc_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.5, vmin=0, vmax=1, square=True, ax=ax2,
                cbar_kws={'label': 'Pearson Correlation'})
    ax2.set_title('5hmC Correlation', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.out_dir, 'sample_correlations.png'), 
                dpi=config.dpi, bbox_inches='tight')
    plt.close()

# -------------------
# Export data
# -------------------
print("6. Exporting data...")
for sample, data in all_samples.items():
    export_data = data[['chrom', 'genome_pos', 'percent_m', 'percent_h', 
                        'percent_m_smooth', 'percent_h_smooth']].copy()
    export_data.to_csv(os.path.join(config.out_dir, f'{sample}_genome_wide_data.csv'), 
                       index=False)

# -------------------
# Summary
# -------------------
print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print('='*60)
print(f"\nResults saved to: {config.out_dir}/")
print("\nGenerated files:")
print("  1. genome_wide_overview.png - Main visualization")
print("  2. combined_methylation_profiles.png - *** NEW: All chromosomes side-by-side ***")
print("  3. genome_wide_log2fc.png - Fold change analysis (vs sample 07)")
print("  4. sample_correlations.png - Sample similarity")
print("  5. [chr]_methylation.png - Per-chromosome details (ALL 24 chromosomes)")
print("  6. genome_wide_statistics.csv - Summary statistics")
print("  7. [sample]_genome_wide_data.csv - Processed data")
print(f"\nSettings:")
print(f"  Bin size: {config.bin_size/1e6:.1f} Mb *** CHANGED to 1.5 Mb ***")
print(f"  Smoothing: σ={config.smoothing_sigma}")
print(f"  Samples: {len(all_samples)}")
print(f"  Control: {control_name} *** CHANGED to 07 sample ***")
print(f"  Chromosomes processed: {len(config.chromosomes)} (ALL)")
