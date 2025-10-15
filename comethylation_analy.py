#!/usr/bin/env python3
"""
Methylation Density and Distance Analysis
Adjacent CpG distance analysis with 500bp cutoff
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

# Parameters
MIN_COVERAGE = 10
DISTANCE_CUTOFF = 500  # Only include adjacent sites within 500bp
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

# Threshold for defining "methylated" sites
THRESHOLD_METHOD = 'mean'
# FIXED_THRESHOLD_5MC = 50.0  # Uncomment to use fixed threshold
# FIXED_THRESHOLD_5HMC = 10.0

# -------------------
# Calculate Global Thresholds
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
    
    valid_5mc = meth_df[meth_df['count_valid_m'] >= MIN_COVERAGE]['percent_m'].dropna()
    valid_5hmc = meth_df[meth_df['count_valid_h'] >= MIN_COVERAGE]['percent_h'].dropna()
    
    all_5mc_values.extend(valid_5mc.tolist())
    all_5hmc_values.extend(valid_5hmc.tolist())

all_5mc_values = np.array(all_5mc_values)
all_5hmc_values = np.array(all_5hmc_values)

if 'FIXED_THRESHOLD_5MC' in locals():
    THRESHOLD_5MC = FIXED_THRESHOLD_5MC
    THRESHOLD_5HMC = FIXED_THRESHOLD_5HMC
elif THRESHOLD_METHOD == 'mean':
    THRESHOLD_5MC = all_5mc_values.mean()
    THRESHOLD_5HMC = all_5hmc_values.mean()
elif THRESHOLD_METHOD == 'median':
    THRESHOLD_5MC = np.median(all_5mc_values)
    THRESHOLD_5HMC = np.median(all_5hmc_values)
else:
    THRESHOLD_5MC = all_5mc_values.mean()
    THRESHOLD_5HMC = all_5hmc_values.mean()

print(f"Thresholds: 5mC={THRESHOLD_5MC:.2f}%, 5hmC={THRESHOLD_5HMC:.2f}%")
print(f"Distance cutoff: {DISTANCE_CUTOFF}bp\n")

# -------------------
# Analysis Function
# -------------------
def extract_adjacent_pairs_per_chromosome(chrom_df, mod_type='5mC', threshold=None):
    """
    For each chromosome:
    1. Number methylated sites as 1, 2, 3, ...
    2. Calculate distance between consecutive sites
    3. Keep only distances ≤ 500bp
    4. Return pairs of (distance, methylation_value)
    """
    if mod_type == '5mC':
        value_col = 'percent_m'
        valid_col = 'count_valid_m'
        use_threshold = threshold if threshold is not None else THRESHOLD_5MC
    else:
        value_col = 'percent_h'
        valid_col = 'count_valid_h'
        use_threshold = threshold if threshold is not None else THRESHOLD_5HMC
    
    # Filter for methylated sites above threshold
    methylated = chrom_df[
        (chrom_df[value_col] >= use_threshold) &
        (chrom_df[valid_col] >= MIN_COVERAGE)
    ].copy()
    
    if len(methylated) < 2:
        return []
    
    # Sort by position and number them 1, 2, 3, ...
    methylated = methylated.sort_values('start').reset_index(drop=True)
    methylated['center'] = (methylated['start'] + methylated['end']) / 2
    methylated['site_number'] = range(1, len(methylated) + 1)
    
    pairs = []
    # Calculate distance between consecutive numbered sites
    for i in range(len(methylated) - 1):
        dist = methylated.iloc[i+1]['center'] - methylated.iloc[i]['center']
        meth1 = methylated.iloc[i][value_col]
        meth2 = methylated.iloc[i+1][value_col]
        
        # Only keep if distance ≤ 500bp
        if 0 < dist <= DISTANCE_CUTOFF and np.isfinite(dist) and np.isfinite(meth1) and np.isfinite(meth2):
            # Store both methylation values for this distance
            pairs.append({'distance': dist, 'methylation': meth1})
            pairs.append({'distance': dist, 'methylation': meth2})
    
    return pairs

# -------------------
# Process All Samples
# -------------------
print("Processing samples...")

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\n{sample_name}")
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    # Collect pairs per chromosome
    all_pairs_5mc = []
    all_pairs_5hmc = []
    
    for chrom in tqdm(chromosomes, desc="Chromosomes", leave=False):
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        if len(chrom_data) < 10:
            continue
        
        pairs_5mc = extract_adjacent_pairs_per_chromosome(chrom_data, mod_type='5mC')
        pairs_5hmc = extract_adjacent_pairs_per_chromosome(chrom_data, mod_type='5hmC')
        
        all_pairs_5mc.extend(pairs_5mc)
        all_pairs_5hmc.extend(pairs_5hmc)
    
    print(f"  5mC: {len(all_pairs_5mc):,} pairs | 5hmC: {len(all_pairs_5hmc):,} pairs")
    
    # -------------------
    # Create 2D Density Plots
    # -------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 5mC Plot
    if len(all_pairs_5mc) > 0:
        df_5mc = pd.DataFrame(all_pairs_5mc)
        
        hb1 = ax1.hexbin(df_5mc['distance'], df_5mc['methylation'], 
                       gridsize=50, cmap='Reds', mincnt=1, norm=LogNorm())
        ax1.set_xlabel('Distance to Adjacent Methylated CpG (bp)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Methylation Level (%)', fontsize=13, fontweight='bold')
        ax1.set_title(f'5mC: Distance vs Methylation Density\n{sample_name}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlim(0, DISTANCE_CUTOFF)
        
        cb1 = plt.colorbar(hb1, ax=ax1)
        cb1.set_label('Count (log scale)', fontsize=11, fontweight='bold')
        
        median_dist = df_5mc['distance'].median()
        median_meth = df_5mc['methylation'].median()
        mean_meth = df_5mc['methylation'].mean()
        stats_text = f'Median distance: {median_dist:.0f} bp\nMedian meth: {median_meth:.1f}%\nMean meth: {mean_meth:.1f}%\nN pairs: {len(df_5mc):,}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.grid(True, alpha=0.3)
    
    # 5hmC Plot
    if len(all_pairs_5hmc) > 0:
        df_5hmc = pd.DataFrame(all_pairs_5hmc)
        
        hb2 = ax2.hexbin(df_5hmc['distance'], df_5hmc['methylation'], 
                       gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        ax2.set_xlabel('Distance to Adjacent Methylated CpG (bp)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Methylation Level (%)', fontsize=13, fontweight='bold')
        ax2.set_title(f'5hmC: Distance vs Methylation Density\n{sample_name}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlim(0, DISTANCE_CUTOFF)
        
        cb2 = plt.colorbar(hb2, ax=ax2)
        cb2.set_label('Count (log scale)', fontsize=11, fontweight='bold')
        
        median_dist = df_5hmc['distance'].median()
        median_meth = df_5hmc['methylation'].median()
        mean_meth = df_5hmc['methylation'].mean()
        stats_text = f'Median distance: {median_dist:.0f} bp\nMedian meth: {median_meth:.1f}%\nMean meth: {mean_meth:.1f}%\nN pairs: {len(df_5hmc):,}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample_name}_adjacent_density.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # -------------------
    # Stratified Distance Histograms
    # -------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 5mC by methylation ranges
    if len(all_pairs_5mc) > 0:
        df_5mc = pd.DataFrame(all_pairs_5mc)
        
        df_high = df_5mc[df_5mc['methylation'] >= 75]
        if len(df_high) > 10:
            axes[0,0].hist(df_high['distance'], bins=50, color='darkred', alpha=0.7, edgecolor='black')
            axes[0,0].set_title(f'5mC: High Methylation (≥75%)\nn={len(df_high):,}', fontweight='bold')
            axes[0,0].set_xlabel('Distance (bp)', fontweight='bold')
            axes[0,0].set_ylabel('Frequency', fontweight='bold')
            axes[0,0].axvline(df_high['distance'].median(), color='red', linestyle='--', 
                            label=f'Median: {df_high["distance"].median():.0f} bp')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_xlim(0, DISTANCE_CUTOFF)
        
        df_low = df_5mc[df_5mc['methylation'] < 75]
        if len(df_low) > 10:
            axes[0,1].hist(df_low['distance'], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0,1].set_title(f'5mC: Lower Methylation (<75%)\nn={len(df_low):,}', fontweight='bold')
            axes[0,1].set_xlabel('Distance (bp)', fontweight='bold')
            axes[0,1].set_ylabel('Frequency', fontweight='bold')
            axes[0,1].axvline(df_low['distance'].median(), color='red', linestyle='--',
                            label=f'Median: {df_low["distance"].median():.0f} bp')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_xlim(0, DISTANCE_CUTOFF)
    
    # 5hmC by methylation ranges
    if len(all_pairs_5hmc) > 0:
        df_5hmc = pd.DataFrame(all_pairs_5hmc)
        
        median_5hmc = df_5hmc['methylation'].median()
        df_high = df_5hmc[df_5hmc['methylation'] >= median_5hmc]
        if len(df_high) > 10:
            axes[1,0].hist(df_high['distance'], bins=50, color='darkblue', alpha=0.7, edgecolor='black')
            axes[1,0].set_title(f'5hmC: High Methylation (≥{median_5hmc:.1f}%)\nn={len(df_high):,}', fontweight='bold')
            axes[1,0].set_xlabel('Distance (bp)', fontweight='bold')
            axes[1,0].set_ylabel('Frequency', fontweight='bold')
            axes[1,0].axvline(df_high['distance'].median(), color='blue', linestyle='--',
                            label=f'Median: {df_high["distance"].median():.0f} bp')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xlim(0, DISTANCE_CUTOFF)
        
        df_low = df_5hmc[df_5hmc['methylation'] < median_5hmc]
        if len(df_low) > 10:
            axes[1,1].hist(df_low['distance'], bins=50, color='lightblue', alpha=0.7, edgecolor='black')
            axes[1,1].set_title(f'5hmC: Lower Methylation (<{median_5hmc:.1f}%)\nn={len(df_low):,}', fontweight='bold')
            axes[1,1].set_xlabel('Distance (bp)', fontweight='bold')
            axes[1,1].set_ylabel('Frequency', fontweight='bold')
            axes[1,1].axvline(df_low['distance'].median(), color='blue', linestyle='--',
                            label=f'Median: {df_low["distance"].median():.0f} bp')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].set_xlim(0, DISTANCE_CUTOFF)
    
    fig.suptitle(f'{sample_name} - Distance Distribution by Methylation Level (≤{DISTANCE_CUTOFF}bp)', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample_name}_distance_stratified.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"\nOutput: {out_dir}/")
print("  • [sample]_adjacent_density.png")
print("  • [sample]_distance_stratified.png")
