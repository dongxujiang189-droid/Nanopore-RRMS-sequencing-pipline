#!/usr/bin/env python3
"""
Methylation Density Comparison Analysis
Adjacent CpG distance with 500bp cutoff + cross-sample comparison
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
out_dir = os.path.join(base_dir, "methylation_density_2d")
os.makedirs(out_dir, exist_ok=True)

MIN_COVERAGE = 10
DISTANCE_CUTOFF = 500
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
THRESHOLD_METHOD = 'mean'

# -------------------
# Calculate Thresholds
# -------------------
print("Calculating thresholds...")
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

all_5mc_values = []
all_5hmc_values = []

for sample_file in tqdm(sample_files, desc="Loading"):
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    all_5mc_values.extend(meth_df[meth_df['count_valid_m'] >= MIN_COVERAGE]['percent_m'].dropna().tolist())
    all_5hmc_values.extend(meth_df[meth_df['count_valid_h'] >= MIN_COVERAGE]['percent_h'].dropna().tolist())

THRESHOLD_5MC = np.mean(all_5mc_values)
THRESHOLD_5HMC = np.mean(all_5hmc_values)
print(f"Thresholds: 5mC={THRESHOLD_5MC:.2f}%, 5hmC={THRESHOLD_5HMC:.2f}%\n")

# -------------------
# Analysis Function
# -------------------
def extract_adjacent_pairs_per_chromosome(chrom_df, mod_type='5mC', threshold=None):
    if mod_type == '5mC':
        value_col, valid_col = 'percent_m', 'count_valid_m'
        use_threshold = threshold if threshold is not None else THRESHOLD_5MC
    else:
        value_col, valid_col = 'percent_h', 'count_valid_h'
        use_threshold = threshold if threshold is not None else THRESHOLD_5HMC
    
    methylated = chrom_df[
        (chrom_df[value_col] >= use_threshold) &
        (chrom_df[valid_col] >= MIN_COVERAGE)
    ].copy()
    
    if len(methylated) < 2:
        return [], 0, 0, 0  # pairs, n_sites, n_distances_total, n_distances_kept
    
    methylated = methylated.sort_values('start').reset_index(drop=True)
    methylated['center'] = (methylated['start'] + methylated['end']) / 2
    
    pairs = []
    distances_all = []
    for i in range(len(methylated) - 1):
        dist = methylated.iloc[i+1]['center'] - methylated.iloc[i]['center']
        distances_all.append(dist)
        meth1 = methylated.iloc[i][value_col]
        meth2 = methylated.iloc[i+1][value_col]
        
        if 0 < dist <= DISTANCE_CUTOFF and np.isfinite(dist) and np.isfinite(meth1) and np.isfinite(meth2):
            pairs.append({'distance': dist, 'methylation': meth1})
            pairs.append({'distance': dist, 'methylation': meth2})
    
    return pairs, len(methylated), len(distances_all), len(pairs)//2

def calculate_density_metrics(df, mod_type='5mC'):
    """Calculate quantitative density metrics"""
    if len(df) == 0:
        return {}
    
    # Define high-density region: short distance + high methylation
    high_meth = df[df['methylation'] >= 75]
    short_dist = df[df['distance'] <= 200]
    high_density = df[(df['methylation'] >= 75) & (df['distance'] <= 200)]
    
    metrics = {
        'median_distance': df['distance'].median(),
        'mean_distance': df['distance'].mean(),
        'median_methylation': df['methylation'].median(),
        'mean_methylation': df['methylation'].mean(),
        'n_pairs': len(df),
        'pct_high_meth': 100 * len(high_meth) / len(df),
        'pct_short_dist': 100 * len(short_dist) / len(df),
        'pct_high_density': 100 * len(high_density) / len(df),  # Key metric
        'density_score': (high_density['methylation'].mean() if len(high_density) > 0 else 0) * len(high_density) / len(df)
    }
    return metrics

# -------------------
# Process All Samples
# -------------------
print("Processing samples...")
all_sample_data = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\n{sample_name}")
    
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
        
        pairs_5mc, sites_5mc, dist_total_5mc, dist_kept_5mc = extract_adjacent_pairs_per_chromosome(chrom_data, mod_type='5mC')
        pairs_5hmc, sites_5hmc, dist_total_5hmc, dist_kept_5hmc = extract_adjacent_pairs_per_chromosome(chrom_data, mod_type='5hmC')
        
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
    
    # Diagnostic output
    print(f"  5mC: {total_sites_5mc:,} methylated sites (≥{THRESHOLD_5MC:.1f}%)")
    print(f"       {total_dist_5mc:,} total adjacent pairs")
    print(f"       {kept_dist_5mc:,} pairs within {DISTANCE_CUTOFF}bp ({100*kept_dist_5mc/total_dist_5mc if total_dist_5mc > 0 else 0:.1f}%)")
    print(f"       {len(df_5mc):,} final data points")
    
    print(f"  5hmC: {total_sites_5hmc:,} methylated sites (≥{THRESHOLD_5HMC:.1f}%)")
    print(f"        {total_dist_5hmc:,} total adjacent pairs")
    print(f"        {kept_dist_5hmc:,} pairs within {DISTANCE_CUTOFF}bp ({100*kept_dist_5hmc/total_dist_5hmc if total_dist_5hmc > 0 else 0:.1f}%)")
    print(f"        {len(df_5hmc):,} final data points")
    
    all_sample_data[sample_name] = {
        '5mC': df_5mc,
        '5hmC': df_5hmc,
        'metrics_5mC': calculate_density_metrics(df_5mc, '5mC'),
        'metrics_5hmC': calculate_density_metrics(df_5hmc, '5hmC')
    }
    
    # -------------------
    # Individual Sample Plots
    # -------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    if len(df_5mc) > 0:
        hb1 = ax1.hexbin(df_5mc['distance'], df_5mc['methylation'], 
                       gridsize=50, cmap='Reds', mincnt=1, norm=LogNorm())
        ax1.set_xlabel('Distance to Adjacent CpG (bp)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Methylation (%)', fontsize=13, fontweight='bold')
        ax1.set_title(f'5mC: {sample_name}', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, DISTANCE_CUTOFF)
        plt.colorbar(hb1, ax=ax1, label='Count (log)')
        
        m = all_sample_data[sample_name]['metrics_5mC']
        stats = f"High-density: {m['pct_high_density']:.1f}%\nMed dist: {m['median_distance']:.0f}bp\nMean meth: {m['mean_methylation']:.1f}%\nN={m['n_pairs']:,}"
        ax1.text(0.02, 0.98, stats, transform=ax1.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    if len(df_5hmc) > 0:
        hb2 = ax2.hexbin(df_5hmc['distance'], df_5hmc['methylation'], 
                       gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        ax2.set_xlabel('Distance to Adjacent CpG (bp)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Methylation (%)', fontsize=13, fontweight='bold')
        ax2.set_title(f'5hmC: {sample_name}', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, DISTANCE_CUTOFF)
        plt.colorbar(hb2, ax=ax2, label='Count (log)')
        
        m = all_sample_data[sample_name]['metrics_5hmC']
        stats = f"High-density: {m['pct_high_density']:.1f}%\nMed dist: {m['median_distance']:.0f}bp\nMean meth: {m['mean_methylation']:.1f}%\nN={m['n_pairs']:,}"
        ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample_name}_density.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Cross-Sample Comparison Plots
# -------------------
print("\nGenerating comparison plots...")

# 1. Side-by-side 2D density comparison
n_samples = len(all_sample_data)
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5mC']
    if len(df) > 0:
        hb = ax.hexbin(df['distance'], df['methylation'], 
                      gridsize=40, cmap='Reds', mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
        ax.set_title(f'{sample_name}\nHD: {data["metrics_5mC"]["pct_high_density"]:.1f}%', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Distance (bp)', fontweight='bold')
        ax.set_ylabel('Methylation (%)', fontweight='bold')
        ax.set_xlim(0, DISTANCE_CUTOFF)
        ax.set_ylim(0, 100)
        plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle('5mC Density Comparison (High-Density = ≥75% meth & ≤200bp dist)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5mC_all_samples.png'), dpi=300, bbox_inches='tight')
plt.close()

# Same for 5hmC
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5hmC']
    if len(df) > 0:
        hb = ax.hexbin(df['distance'], df['methylation'], 
                      gridsize=40, cmap='Blues', mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
        ax.set_title(f'{sample_name}\nHD: {data["metrics_5hmC"]["pct_high_density"]:.1f}%', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Distance (bp)', fontweight='bold')
        ax.set_ylabel('Methylation (%)', fontweight='bold')
        ax.set_xlim(0, DISTANCE_CUTOFF)
        ax.set_ylim(0, 100)
        plt.colorbar(hb, ax=ax, label='Count')

fig.suptitle('5hmC Density Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5hmC_all_samples.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Quantitative metrics comparison
metrics_5mc = pd.DataFrame([data['metrics_5mC'] for data in all_sample_data.values()])
metrics_5mc.insert(0, 'Sample', list(all_sample_data.keys()))

metrics_5hmc = pd.DataFrame([data['metrics_5hmC'] for data in all_sample_data.values()])
metrics_5hmc.insert(0, 'Sample', list(all_sample_data.keys()))

# Save metrics
metrics_5mc.to_csv(os.path.join(out_dir, 'metrics_5mC.csv'), index=False)
metrics_5hmc.to_csv(os.path.join(out_dir, 'metrics_5hmC.csv'), index=False)

# 3. Bar chart comparison of key metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

samples = list(all_sample_data.keys())
colors = plt.cm.Set3(np.linspace(0, 1, len(samples)))

# 5mC metrics
axes[0,0].bar(samples, metrics_5mc['pct_high_density'], color=colors, edgecolor='black')
axes[0,0].set_ylabel('% High-Density Pairs', fontweight='bold')
axes[0,0].set_title('5mC: High-Density % (≥75% meth, ≤200bp)', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(axis='y', alpha=0.3)

axes[0,1].bar(samples, metrics_5mc['mean_methylation'], color=colors, edgecolor='black')
axes[0,1].set_ylabel('Mean Methylation (%)', fontweight='bold')
axes[0,1].set_title('5mC: Mean Methylation', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(axis='y', alpha=0.3)

axes[0,2].bar(samples, metrics_5mc['median_distance'], color=colors, edgecolor='black')
axes[0,2].set_ylabel('Median Distance (bp)', fontweight='bold')
axes[0,2].set_title('5mC: Median Adjacent Distance', fontweight='bold')
axes[0,2].tick_params(axis='x', rotation=45)
axes[0,2].grid(axis='y', alpha=0.3)

# 5hmC metrics
axes[1,0].bar(samples, metrics_5hmc['pct_high_density'], color=colors, edgecolor='black')
axes[1,0].set_ylabel('% High-Density Pairs', fontweight='bold')
axes[1,0].set_title('5hmC: High-Density %', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(axis='y', alpha=0.3)

axes[1,1].bar(samples, metrics_5hmc['mean_methylation'], color=colors, edgecolor='black')
axes[1,1].set_ylabel('Mean Methylation (%)', fontweight='bold')
axes[1,1].set_title('5hmC: Mean Methylation', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(axis='y', alpha=0.3)

axes[1,2].bar(samples, metrics_5hmc['median_distance'], color=colors, edgecolor='black')
axes[1,2].set_ylabel('Median Distance (bp)', fontweight='bold')
axes[1,2].set_title('5hmC: Median Adjacent Distance', fontweight='bold')
axes[1,2].tick_params(axis='x', rotation=45)
axes[1,2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'comparison_metrics_barplots.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Heatmap of normalized metrics
metrics_norm = metrics_5mc[['pct_high_density', 'mean_methylation', 'median_distance', 'pct_high_meth']].copy()
metrics_norm = (metrics_norm - metrics_norm.min()) / (metrics_norm.max() - metrics_norm.min())
metrics_norm.insert(0, 'Sample', samples)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 5mC heatmap
data_5mc = metrics_5mc[['pct_high_density', 'mean_methylation', 'median_distance', 'pct_high_meth']].T
sns.heatmap(data_5mc, annot=True, fmt='.1f', cmap='Reds', ax=ax1, 
            xticklabels=samples, yticklabels=['High-Density %', 'Mean Meth %', 'Median Dist (bp)', 'High Meth %'],
            cbar_kws={'label': 'Value'})
ax1.set_title('5mC Metrics Heatmap', fontweight='bold', fontsize=13)

# 5hmC heatmap
data_5hmc = metrics_5hmc[['pct_high_density', 'mean_methylation', 'median_distance', 'pct_high_meth']].T
sns.heatmap(data_5hmc, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
            xticklabels=samples, yticklabels=['High-Density %', 'Mean Meth %', 'Median Dist (bp)', 'High Meth %'],
            cbar_kws={'label': 'Value'})
ax2.set_title('5hmC Metrics Heatmap', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'comparison_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"\nOutput: {out_dir}/")
print("  Individual: [sample]_density.png")
print("  Comparison: comparison_5mC_all_samples.png")
print("  Comparison: comparison_5hmC_all_samples.png")
print("  Comparison: comparison_metrics_barplots.png")
print("  Comparison: comparison_metrics_heatmap.png")
print("  Data: metrics_5mC.csv, metrics_5hmC.csv")
print("\nKey metric: High-Density % = pairs with ≥75% methylation AND ≤200bp distance")
