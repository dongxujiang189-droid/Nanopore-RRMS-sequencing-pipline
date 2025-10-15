#!/usr/bin/env python3
"""
Methylation Density Analysis - All Sites
Shows full methylation distribution without thresholds
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
out_dir = os.path.join(base_dir, "methylation_density_all")
os.makedirs(out_dir, exist_ok=True)

MIN_COVERAGE = 10
DISTANCE_CUTOFF = 500  # Maximum distance between adjacent CpGs
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

print(f"Configuration: Coverage ≥{MIN_COVERAGE}, Distance ≤{DISTANCE_CUTOFF}bp\n")

# -------------------
# Analysis Function
# -------------------
def extract_all_adjacent_pairs(chrom_df, mod_type='5mC'):
    """Extract ALL adjacent CpG pairs regardless of methylation level"""
    if mod_type == '5mC':
        value_col, valid_col = 'percent_m', 'count_valid_m'
    else:
        value_col, valid_col = 'percent_h', 'count_valid_h'
    
    # Get all sites with sufficient coverage (no methylation threshold)
    valid_sites = chrom_df[chrom_df[valid_col] >= MIN_COVERAGE].copy()
    
    if len(valid_sites) < 2:
        return [], 0, 0, 0
    
    valid_sites = valid_sites.sort_values('start').reset_index(drop=True)
    valid_sites['center'] = (valid_sites['start'] + valid_sites['end']) / 2
    
    pairs = []
    distances_all = []
    
    for i in range(len(valid_sites) - 1):
        dist = valid_sites.iloc[i+1]['center'] - valid_sites.iloc[i]['center']
        distances_all.append(dist)
        
        meth1 = valid_sites.iloc[i][value_col]
        meth2 = valid_sites.iloc[i+1][value_col]
        
        if 0 < dist <= DISTANCE_CUTOFF and np.isfinite(dist) and np.isfinite(meth1) and np.isfinite(meth2):
            pairs.append({'distance': dist, 'methylation': meth1})
            pairs.append({'distance': dist, 'methylation': meth2})
    
    return pairs, len(valid_sites), len(distances_all), len(pairs)//2

def calculate_density_metrics(df):
    """Calculate quantitative metrics"""
    if len(df) == 0:
        return {}
    
    high_meth = df[df['methylation'] >= 75]
    short_dist = df[df['distance'] <= 200]
    high_density = df[(df['methylation'] >= 75) & (df['distance'] <= 200)]
    
    return {
        'median_distance': df['distance'].median(),
        'mean_distance': df['distance'].mean(),
        'median_methylation': df['methylation'].median(),
        'mean_methylation': df['methylation'].mean(),
        'n_pairs': len(df),
        'pct_high_meth': 100 * len(high_meth) / len(df),
        'pct_short_dist': 100 * len(short_dist) / len(df),
        'pct_high_density': 100 * len(high_density) / len(df)
    }

# -------------------
# Process All Samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"Found {len(sample_files)} samples\n")
all_sample_data = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"{sample_name}")
    
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    all_pairs_5mc = []
    all_pairs_5hmc = []
    total_sites_5mc = 0
    total_sites_5hmc = 0
    total_dist_5mc = 0
    total_dist_5hmc = 0
    kept_dist_5mc = 0
    kept_dist_5hmc = 0
    
    for chrom in tqdm(chromosomes, desc="Chromosomes", leave=False):
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        if len(chrom_data) < 10:
            continue
        
        pairs_5mc, sites_5mc, dist_total_5mc, dist_kept_5mc = extract_all_adjacent_pairs(chrom_data, '5mC')
        pairs_5hmc, sites_5hmc, dist_total_5hmc, dist_kept_5hmc = extract_all_adjacent_pairs(chrom_data, '5hmC')
        
        all_pairs_5mc.extend(pairs_5mc)
        all_pairs_5hmc.extend(pairs_5hmc)
        total_sites_5mc += sites_5mc
        total_sites_5hmc += sites_5hmc
        total_dist_5mc += dist_total_5mc
        total_dist_5hmc += dist_total_5hmc
        kept_dist_5mc += dist_kept_5mc
        kept_dist_5hmc += dist_kept_5hmc
    
    df_5mc = pd.DataFrame(all_pairs_5mc) if all_pairs_5mc else pd.DataFrame()
    df_5hmc = pd.DataFrame(all_pairs_5hmc) if all_pairs_5hmc else pd.DataFrame()
    
    print(f"  5mC:  {total_sites_5mc:,} sites → {kept_dist_5mc:,}/{total_dist_5mc:,} pairs ≤{DISTANCE_CUTOFF}bp ({100*kept_dist_5mc/total_dist_5mc if total_dist_5mc>0 else 0:.1f}%)")
    print(f"  5hmC: {total_sites_5hmc:,} sites → {kept_dist_5hmc:,}/{total_dist_5hmc:,} pairs ≤{DISTANCE_CUTOFF}bp ({100*kept_dist_5hmc/total_dist_5hmc if total_dist_5hmc>0 else 0:.1f}%)\n")
    
    all_sample_data[sample_name] = {
        '5mC': df_5mc,
        '5hmC': df_5hmc,
        'metrics_5mC': calculate_density_metrics(df_5mc),
        'metrics_5hmC': calculate_density_metrics(df_5hmc)
    }
    
    # -------------------
    # Individual Sample Plots
    # -------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    if len(df_5mc) > 0:
        hb1 = ax1.hexbin(df_5mc['distance'], df_5mc['methylation'], 
                       gridsize=50, cmap='Reds', mincnt=1, norm=LogNorm())
        ax1.set_xlabel('Distance to Adjacent CpG (bp)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Methylation Level (%)', fontsize=13, fontweight='bold')
        ax1.set_title(f'5mC: {sample_name}', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, DISTANCE_CUTOFF)
        ax1.set_ylim(0, 100)
        plt.colorbar(hb1, ax=ax1, label='Count (log)')
        
        m = all_sample_data[sample_name]['metrics_5mC']
        stats = f"Mean meth: {m['mean_methylation']:.1f}%\nMedian dist: {m['median_distance']:.0f}bp\nHigh-density: {m['pct_high_density']:.1f}%\nN={m['n_pairs']:,}"
        ax1.text(0.02, 0.98, stats, transform=ax1.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    if len(df_5hmc) > 0:
        hb2 = ax2.hexbin(df_5hmc['distance'], df_5hmc['methylation'], 
                       gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        ax2.set_xlabel('Distance to Adjacent CpG (bp)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Methylation Level (%)', fontsize=13, fontweight='bold')
        ax2.set_title(f'5hmC: {sample_name}', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, DISTANCE_CUTOFF)
        ax2.set_ylim(0, 100)
        plt.colorbar(hb2, ax=ax2, label='Count (log)')
        
        m = all_sample_data[sample_name]['metrics_5hmC']
        stats = f"Mean meth: {m['mean_methylation']:.1f}%\nMedian dist: {m['median_distance']:.0f}bp\nHigh-density: {m['pct_high_density']:.1f}%\nN={m['n_pairs']:,}"
        ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample_name}_density.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Cross-Sample Comparison
# -------------------
print("Generating comparison plots...")

n_samples = len(all_sample_data)

# Side-by-side 2D density comparison - 5mC
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5mC']
    if len(df) > 0:
        hb = ax.hexbin(df['distance'], df['methylation'], 
                      gridsize=40, cmap='Reds', mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
        ax.set_title(f'{sample_name}\nMean: {data["metrics_5mC"]["mean_methylation"]:.1f}%', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Distance (bp)', fontweight='bold')
        ax.set_ylabel('Methylation (%)', fontweight='bold')
        ax.set_xlim(0, DISTANCE_CUTOFF)
        ax.set_ylim(0, 100)
        plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle('5mC: Distance vs Methylation (All Sites)', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5mC_all.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5hmC
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5hmC']
    if len(df) > 0:
        hb = ax.hexbin(df['distance'], df['methylation'], 
                      gridsize=40, cmap='Blues', mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
        ax.set_title(f'{sample_name}\nMean: {data["metrics_5hmC"]["mean_methylation"]:.1f}%', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Distance (bp)', fontweight='bold')
        ax.set_ylabel('Methylation (%)', fontweight='bold')
        ax.set_xlim(0, DISTANCE_CUTOFF)
        ax.set_ylim(0, 100)
        plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle('5hmC: Distance vs Methylation (All Sites)', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5hmC_all.png'), dpi=300, bbox_inches='tight')
plt.close()

# Metrics comparison
metrics_5mc = pd.DataFrame([data['metrics_5mC'] for data in all_sample_data.values()])
metrics_5mc.insert(0, 'Sample', list(all_sample_data.keys()))
metrics_5hmc = pd.DataFrame([data['metrics_5hmC'] for data in all_sample_data.values()])
metrics_5hmc.insert(0, 'Sample', list(all_sample_data.keys()))

metrics_5mc.to_csv(os.path.join(out_dir, 'metrics_5mC.csv'), index=False)
metrics_5hmc.to_csv(os.path.join(out_dir, 'metrics_5hmC.csv'), index=False)

# Bar charts
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
samples = list(all_sample_data.keys())
colors = plt.cm.Set3(np.linspace(0, 1, len(samples)))

axes[0,0].bar(samples, metrics_5mc['mean_methylation'], color=colors, edgecolor='black')
axes[0,0].set_ylabel('Mean Methylation (%)', fontweight='bold')
axes[0,0].set_title('5mC: Mean Methylation', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(axis='y', alpha=0.3)

axes[0,1].bar(samples, metrics_5mc['median_distance'], color=colors, edgecolor='black')
axes[0,1].set_ylabel('Median Distance (bp)', fontweight='bold')
axes[0,1].set_title('5mC: Median Adjacent Distance', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(axis='y', alpha=0.3)

axes[1,0].bar(samples, metrics_5hmc['mean_methylation'], color=colors, edgecolor='black')
axes[1,0].set_ylabel('Mean Methylation (%)', fontweight='bold')
axes[1,0].set_title('5hmC: Mean Methylation', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(axis='y', alpha=0.3)

axes[1,1].bar(samples, metrics_5hmc['median_distance'], color=colors, edgecolor='black')
axes[1,1].set_ylabel('Median Distance (bp)', fontweight='bold')
axes[1,1].set_title('5hmC: Median Adjacent Distance', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'comparison_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print("COMPLETE")
print(f"{'='*70}")
print(f"\nOutput: {out_dir}/")
print("  • [sample]_density.png - Individual samples")
print("  • comparison_5mC_all.png - All 5mC samples")
print("  • comparison_5hmC_all.png - All 5hmC samples")
print("  • comparison_metrics.png - Metric comparisons")
print("  • metrics_5mC.csv, metrics_5hmC.csv\n")
