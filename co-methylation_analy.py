#!/usr/bin/env python3
"""
Methylation Distance Analysis - Research-Based Visualization
Based on literature: correlation decays within 400bp-2kb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
out_dir = os.path.join(base_dir, "methylation_analysis_optimized")
os.makedirs(out_dir, exist_ok=True)

MIN_COVERAGE = 10
DISTANCE_BINS = [0, 100, 500, 2000, 10000]  # Based on literature
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

print(f"Coverage threshold: ≥{MIN_COVERAGE}x\n")

# Extract adjacent pairs
def extract_pairs(chrom_df, mod_type='5mC'):
    value_col = 'percent_m' if mod_type == '5mC' else 'percent_h'
    valid_col = 'count_valid_m' if mod_type == '5mC' else 'count_valid_h'
    
    valid = chrom_df[chrom_df[valid_col] >= MIN_COVERAGE].copy()
    if len(valid) < 2:
        return []
    
    valid = valid.sort_values('start').reset_index(drop=True)
    valid['center'] = (valid['start'] + valid['end']) / 2
    
    pairs = []
    for i in range(len(valid) - 1):
        dist = valid.iloc[i+1]['center'] - valid.iloc[i]['center']
        meth1, meth2 = valid.iloc[i][value_col], valid.iloc[i+1][value_col]
        
        if dist > 0 and np.isfinite([dist, meth1, meth2]).all():
            pairs.append({'distance': dist, 'meth1': meth1, 'meth2': meth2})
    
    return pairs

# Process samples
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"Processing {len(sample_files)} samples\n")
all_data = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"{sample_name}")
    
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    pairs_5mc, pairs_5hmc = [], []
    for chrom in tqdm(chromosomes, desc="Chrom", leave=False):
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        if len(chrom_data) >= 10:
            pairs_5mc.extend(extract_pairs(chrom_data, '5mC'))
            pairs_5hmc.extend(extract_pairs(chrom_data, '5hmC'))
    
    df_5mc = pd.DataFrame(pairs_5mc) if pairs_5mc else pd.DataFrame()
    df_5hmc = pd.DataFrame(pairs_5hmc) if pairs_5hmc else pd.DataFrame()
    
    all_data[sample_name] = {'5mC': df_5mc, '5hmC': df_5hmc}
    
    if len(df_5mc) > 0:
        print(f"  5mC:  {len(df_5mc):,} pairs | Distance: {df_5mc['distance'].median():.0f}bp (median)")
    if len(df_5hmc) > 0:
        print(f"  5hmC: {len(df_5hmc):,} pairs | Distance: {df_5hmc['distance'].median():.0f}bp (median)")
    print()
    
    # Individual sample visualization
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f'{sample_name} - Methylation Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 5mC Analysis
    if len(df_5mc) > 0:
        # Distance distribution histogram
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(df_5mc['distance'], bins=100, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.axvline(df_5mc['distance'].median(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Median: {df_5mc["distance"].median():.0f}bp')
        ax1.set_xlabel('Distance to Adjacent CpG (bp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('5mC: Inter-CpG Distance Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Zoomed histograms
        for idx, (min_d, max_d) in enumerate([(0, 100), (100, 500), (500, 2000)]):
            ax = fig.add_subplot(gs[1, idx])
            subset = df_5mc[(df_5mc['distance'] > min_d) & (df_5mc['distance'] <= max_d)]
            if len(subset) > 0:
                ax.hist(subset['distance'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Distance (bp)', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'5mC: {min_d}-{max_d}bp (n={len(subset):,})', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        # 2D density plots
        for idx, (min_d, max_d, title) in enumerate([(0, 500, '≤500bp'), (0, 2000, '≤2kb'), (0, 10000, '≤10kb')]):
            ax = fig.add_subplot(gs[2, idx])
            subset = df_5mc[df_5mc['distance'] <= max_d]
            if len(subset) > 10:
                # Average methylation for each distance pair
                subset['meth_avg'] = (subset['meth1'] + subset['meth2']) / 2
                hb = ax.hexbin(subset['distance'], subset['meth_avg'], gridsize=40, 
                              cmap='Reds', mincnt=1, norm=LogNorm())
                ax.set_xlabel('Distance (bp)', fontweight='bold')
                ax.set_ylabel('Methylation (%)', fontweight='bold')
                ax.set_title(f'5mC: {title}', fontweight='bold')
                ax.set_xlim(min_d, max_d)
                ax.set_ylim(0, 100)
                plt.colorbar(hb, ax=ax, label='Count')
    
    # 5hmC Analysis
    if len(df_5hmc) > 0:
        for idx, (min_d, max_d, title) in enumerate([(0, 500, '≤500bp'), (0, 2000, '≤2kb'), (0, 10000, '≤10kb')]):
            ax = fig.add_subplot(gs[3, idx])
            subset = df_5hmc[df_5hmc['distance'] <= max_d]
            if len(subset) > 10:
                subset['meth_avg'] = (subset['meth1'] + subset['meth2']) / 2
                hb = ax.hexbin(subset['distance'], subset['meth_avg'], gridsize=40,
                              cmap='Blues', mincnt=1, norm=LogNorm())
                ax.set_xlabel('Distance (bp)', fontweight='bold')
                ax.set_ylabel('Methylation (%)', fontweight='bold')
                ax.set_title(f'5hmC: {title}', fontweight='bold')
                ax.set_xlim(min_d, max_d)
                ax.set_ylim(0, 100)
                plt.colorbar(hb, ax=ax, label='Count')
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_complete.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Cross-sample comparison
print("Generating comparisons...")

# Distance distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
colors = plt.cm.Set3(np.linspace(0, 1, len(all_data)))

for idx, (sample_name, data) in enumerate(all_data.items()):
    if len(data['5mC']) > 0:
        axes[0,0].hist(data['5mC']['distance'], bins=50, alpha=0.5, label=sample_name,
                      color=colors[idx], edgecolor='black')
        
        subset = data['5mC'][data['5mC']['distance'] <= 2000]
        if len(subset) > 0:
            axes[0,1].hist(subset['distance'], bins=50, alpha=0.5, label=sample_name,
                          color=colors[idx], edgecolor='black')
    
    if len(data['5hmC']) > 0:
        axes[1,0].hist(data['5hmC']['distance'], bins=50, alpha=0.5, label=sample_name,
                      color=colors[idx], edgecolor='black')
        
        subset = data['5hmC'][data['5hmC']['distance'] <= 2000]
        if len(subset) > 0:
            axes[1,1].hist(subset['distance'], bins=50, alpha=0.5, label=sample_name,
                          color=colors[idx], edgecolor='black')

axes[0,0].set_title('5mC: Full Range', fontweight='bold', fontsize=13)
axes[0,1].set_title('5mC: ≤2kb', fontweight='bold', fontsize=13)
axes[1,0].set_title('5hmC: Full Range', fontweight='bold', fontsize=13)
axes[1,1].set_title('5hmC: ≤2kb', fontweight='bold', fontsize=13)

for ax in axes.flat:
    ax.set_xlabel('Distance (bp)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'comparison_histograms.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2D density comparison (≤2kb)
for mod_type in ['5mC', '5hmC']:
    n_samples = len(all_data)
    fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
    gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, (sample_name, data) in enumerate(all_data.items()):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        df = data[mod_type]
        if len(df) > 0:
            subset = df[df['distance'] <= 2000].copy()
            if len(subset) > 10:
                subset['meth_avg'] = (subset['meth1'] + subset['meth2']) / 2
                hb = ax.hexbin(subset['distance'], subset['meth_avg'], gridsize=40,
                              cmap='Reds' if mod_type=='5mC' else 'Blues',
                              mincnt=1, norm=LogNorm(), vmin=1, vmax=500)
                ax.set_title(f'{sample_name}\nn={len(subset):,}', fontweight='bold', fontsize=11)
                ax.set_xlabel('Distance (bp)', fontweight='bold')
                ax.set_ylabel('Methylation (%)', fontweight='bold')
                ax.set_xlim(0, 2000)
                ax.set_ylim(0, 100)
                plt.colorbar(hb, ax=ax, label='Count')
    
    fig.suptitle(f'{mod_type}: Distance vs Methylation (≤2kb)', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(out_dir, f'comparison_2d_{mod_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Summary statistics
summary = []
for sample_name, data in all_data.items():
    for mod_type in ['5mC', '5hmC']:
        df = data[mod_type]
        if len(df) > 0:
            for min_d, max_d in [(0, 100), (100, 500), (500, 2000), (2000, 10000)]:
                subset = df[(df['distance'] > min_d) & (df['distance'] <= max_d)]
                if len(subset) > 0:
                    subset['meth_avg'] = (subset['meth1'] + subset['meth2']) / 2
                    summary.append({
                        'Sample': sample_name,
                        'Modification': mod_type,
                        'Distance_Range': f'{min_d}-{max_d}bp',
                        'N_Pairs': len(subset),
                        'Mean_Methylation_%': subset['meth_avg'].mean(),
                        'Median_Distance_bp': subset['distance'].median()
                    })

pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'summary_by_distance.csv'), index=False)

print(f"\n{'='*70}")
print("COMPLETE")
print(f"{'='*70}")
print(f"\nOutput: {out_dir}/")
print("  • [sample]_complete.png - Full analysis per sample")
print("  • comparison_histograms.png - Distance distributions")
print("  • comparison_2d_5mC.png, comparison_2d_5hmC.png")
print("  • summary_by_distance.csv\n")
