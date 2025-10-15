#!/usr/bin/env python3
"""
Methylation Analysis - Distance vs Methylation
No thresholds, all sites with sufficient coverage
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
out_dir = os.path.join(base_dir, "methylation_distance_analysis")
os.makedirs(out_dir, exist_ok=True)

MIN_COVERAGE = 10
MAX_DISTANCE = 10000  # Plot distances up to 10kb
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

print(f"Configuration: Coverage ≥{MIN_COVERAGE}, Max distance for plot: {MAX_DISTANCE}bp\n")

# -------------------
# Analysis Function
# -------------------
def extract_all_adjacent_pairs(chrom_df, mod_type='5mC'):
    """Extract ALL adjacent CpG pairs with their methylation values"""
    if mod_type == '5mC':
        value_col, valid_col = 'percent_m', 'count_valid_m'
    else:
        value_col, valid_col = 'percent_h', 'count_valid_h'
    
    # Get all sites with sufficient coverage
    valid_sites = chrom_df[chrom_df[valid_col] >= MIN_COVERAGE].copy()
    
    if len(valid_sites) < 2:
        return []
    
    valid_sites = valid_sites.sort_values('start').reset_index(drop=True)
    valid_sites['center'] = (valid_sites['start'] + valid_sites['end']) / 2
    
    pairs = []
    for i in range(len(valid_sites) - 1):
        dist = valid_sites.iloc[i+1]['center'] - valid_sites.iloc[i]['center']
        meth1 = valid_sites.iloc[i][value_col]
        meth2 = valid_sites.iloc[i+1][value_col]
        
        if dist > 0 and np.isfinite(dist) and np.isfinite(meth1) and np.isfinite(meth2):
            # Store both methylation values for this distance
            pairs.append({'distance': dist, 'methylation': meth1})
            pairs.append({'distance': dist, 'methylation': meth2})
    
    return pairs

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
    
    for chrom in tqdm(chromosomes, desc="Processing", leave=False):
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        if len(chrom_data) < 10:
            continue
        
        all_pairs_5mc.extend(extract_all_adjacent_pairs(chrom_data, '5mC'))
        all_pairs_5hmc.extend(extract_all_adjacent_pairs(chrom_data, '5hmC'))
    
    df_5mc = pd.DataFrame(all_pairs_5mc) if all_pairs_5mc else pd.DataFrame()
    df_5hmc = pd.DataFrame(all_pairs_5hmc) if all_pairs_5hmc else pd.DataFrame()
    
    print(f"  5mC:  {len(df_5mc):,} data points")
    if len(df_5mc) > 0:
        print(f"        Distance range: {df_5mc['distance'].min():.0f} - {df_5mc['distance'].max():.0f} bp (median: {df_5mc['distance'].median():.0f})")
        print(f"        Methylation range: {df_5mc['methylation'].min():.1f}% - {df_5mc['methylation'].max():.1f}% (mean: {df_5mc['methylation'].mean():.1f}%)")
    
    print(f"  5hmC: {len(df_5hmc):,} data points")
    if len(df_5hmc) > 0:
        print(f"        Distance range: {df_5hmc['distance'].min():.0f} - {df_5hmc['distance'].max():.0f} bp (median: {df_5hmc['distance'].median():.0f})")
        print(f"        Methylation range: {df_5hmc['methylation'].min():.1f}% - {df_5hmc['methylation'].max():.1f}% (mean: {df_5hmc['methylation'].mean():.1f}%)")
    print()
    
    all_sample_data[sample_name] = {
        '5mC': df_5mc,
        '5hmC': df_5hmc
    }
    
    # -------------------
    # Individual Sample Plots
    # -------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 5mC - Full range
    if len(df_5mc) > 0:
        # Full distance range
        hb1 = axes[0,0].hexbin(df_5mc['distance'], df_5mc['methylation'], 
                               gridsize=50, cmap='Reds', mincnt=1, norm=LogNorm())
        axes[0,0].set_xlabel('Distance to Adjacent CpG (bp)', fontsize=12, fontweight='bold')
        axes[0,0].set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
        axes[0,0].set_title(f'5mC: Full Range - {sample_name}', fontsize=13, fontweight='bold')
        axes[0,0].set_ylim(0, 100)
        plt.colorbar(hb1, ax=axes[0,0], label='Count (log)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Zoomed to MAX_DISTANCE
        df_zoom = df_5mc[df_5mc['distance'] <= MAX_DISTANCE]
        if len(df_zoom) > 0:
            hb2 = axes[0,1].hexbin(df_zoom['distance'], df_zoom['methylation'], 
                                   gridsize=50, cmap='Reds', mincnt=1, norm=LogNorm())
            axes[0,1].set_xlabel('Distance to Adjacent CpG (bp)', fontsize=12, fontweight='bold')
            axes[0,1].set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
            axes[0,1].set_title(f'5mC: ≤{MAX_DISTANCE}bp - {sample_name}', fontsize=13, fontweight='bold')
            axes[0,1].set_xlim(0, MAX_DISTANCE)
            axes[0,1].set_ylim(0, 100)
            plt.colorbar(hb2, ax=axes[0,1], label='Count (log)')
            axes[0,1].grid(True, alpha=0.3)
            
            stats = f"N={len(df_zoom):,}\nMean meth: {df_zoom['methylation'].mean():.1f}%\nMedian dist: {df_zoom['distance'].median():.0f}bp"
            axes[0,1].text(0.02, 0.98, stats, transform=axes[0,1].transAxes, va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
    
    # 5hmC - Full range
    if len(df_5hmc) > 0:
        hb3 = axes[1,0].hexbin(df_5hmc['distance'], df_5hmc['methylation'], 
                               gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        axes[1,0].set_xlabel('Distance to Adjacent CpG (bp)', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
        axes[1,0].set_title(f'5hmC: Full Range - {sample_name}', fontsize=13, fontweight='bold')
        axes[1,0].set_ylim(0, 100)
        plt.colorbar(hb3, ax=axes[1,0], label='Count (log)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Zoomed
        df_zoom = df_5hmc[df_5hmc['distance'] <= MAX_DISTANCE]
        if len(df_zoom) > 0:
            hb4 = axes[1,1].hexbin(df_zoom['distance'], df_zoom['methylation'], 
                                   gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
            axes[1,1].set_xlabel('Distance to Adjacent CpG (bp)', fontsize=12, fontweight='bold')
            axes[1,1].set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
            axes[1,1].set_title(f'5hmC: ≤{MAX_DISTANCE}bp - {sample_name}', fontsize=13, fontweight='bold')
            axes[1,1].set_xlim(0, MAX_DISTANCE)
            axes[1,1].set_ylim(0, 100)
            plt.colorbar(hb4, ax=axes[1,1], label='Count (log)')
            axes[1,1].grid(True, alpha=0.3)
            
            stats = f"N={len(df_zoom):,}\nMean meth: {df_zoom['methylation'].mean():.1f}%\nMedian dist: {df_zoom['distance'].median():.0f}bp"
            axes[1,1].text(0.02, 0.98, stats, transform=axes[1,1].transAxes, va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample_name}_distance_vs_methylation.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Cross-Sample Comparison
# -------------------
print("Generating comparison plots...")

n_samples = len(all_sample_data)

# 5mC comparison (zoomed view)
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5mC']
    if len(df) > 0:
        df_plot = df[df['distance'] <= MAX_DISTANCE]
        if len(df_plot) > 0:
            hb = ax.hexbin(df_plot['distance'], df_plot['methylation'], 
                          gridsize=40, cmap='Reds', mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
            ax.set_title(f'{sample_name}\nMean: {df_plot["methylation"].mean():.1f}%', 
                        fontweight='bold', fontsize=11)
            ax.set_xlabel('Distance (bp)', fontweight='bold')
            ax.set_ylabel('Methylation (%)', fontweight='bold')
            ax.set_xlim(0, MAX_DISTANCE)
            ax.set_ylim(0, 100)
            plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle(f'5mC: Distance vs Methylation (≤{MAX_DISTANCE}bp)', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5mC.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5hmC comparison
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5hmC']
    if len(df) > 0:
        df_plot = df[df['distance'] <= MAX_DISTANCE]
        if len(df_plot) > 0:
            hb = ax.hexbin(df_plot['distance'], df_plot['methylation'], 
                          gridsize=40, cmap='Blues', mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
            ax.set_title(f'{sample_name}\nMean: {df_plot["methylation"].mean():.1f}%', 
                        fontweight='bold', fontsize=11)
            ax.set_xlabel('Distance (bp)', fontweight='bold')
            ax.set_ylabel('Methylation (%)', fontweight='bold')
            ax.set_xlim(0, MAX_DISTANCE)
            ax.set_ylim(0, 100)
            plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle(f'5hmC: Distance vs Methylation (≤{MAX_DISTANCE}bp)', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5hmC.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save summary statistics
summary = []
for sample_name, data in all_sample_data.items():
    for mod_type in ['5mC', '5hmC']:
        df = data[mod_type]
        if len(df) > 0:
            df_zoom = df[df['distance'] <= MAX_DISTANCE]
            summary.append({
                'Sample': sample_name,
                'Modification': mod_type,
                'Total_pairs': len(df),
                'Mean_methylation_%': df['methylation'].mean(),
                'Median_distance_bp': df['distance'].median(),
                f'Pairs_<{MAX_DISTANCE}bp': len(df_zoom),
                f'Mean_meth_<{MAX_DISTANCE}bp_%': df_zoom['methylation'].mean() if len(df_zoom) > 0 else np.nan
            })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(out_dir, 'summary_statistics.csv'), index=False)

print(f"\n{'='*70}")
print("COMPLETE")
print(f"{'='*70}")
print(f"\nOutput: {out_dir}/")
print("  • [sample]_distance_vs_methylation.png")
print("  • comparison_5mC.png")
print("  • comparison_5hmC.png")
print("  • summary_statistics.csv\n")
