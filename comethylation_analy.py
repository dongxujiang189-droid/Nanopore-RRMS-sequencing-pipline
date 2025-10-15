#!/usr/bin/env python3
"""
Co-methylation Distance Analysis - Adjusted for Nanopore Data
Analyzes spatial correlation of methylation levels
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
out_dir = os.path.join(base_dir, "comethylation_analysis")
os.makedirs(out_dir, exist_ok=True)

# Adjusted parameters for nanopore data
MIN_COVERAGE = 10      # Minimum valid reads
MAX_DISTANCE = 5000    # Maximum distance between sites (bp)
MIN_DISTANCE = 10      # Minimum distance
WINDOW_SIZE = 10000    # Window size for calculating correlations

distance_bins = [0, 50, 100, 200, 500, 1000, 2000, 5000]
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

# -------------------
# Data Exploration
# -------------------
def explore_data_distribution(meth_df, sample_name):
    """Analyze methylation distribution in the data"""
    print(f"\n{'='*70}")
    print(f"DATA DISTRIBUTION: {sample_name}")
    print('='*70)
    
    # Overall statistics
    print(f"\nTotal sites: {len(meth_df):,}")
    
    # 5mC statistics
    valid_5mc = meth_df[meth_df['count_valid_m'] >= MIN_COVERAGE]
    print(f"\n5mC Statistics (coverage ≥ {MIN_COVERAGE}):")
    print(f"  Sites: {len(valid_5mc):,}")
    print(f"  Mean: {valid_5mc['percent_m'].mean():.2f}%")
    print(f"  Median: {valid_5mc['percent_m'].median():.2f}%")
    print(f"  Std: {valid_5mc['percent_m'].std():.2f}%")
    print(f"  Min: {valid_5mc['percent_m'].min():.2f}%")
    print(f"  Max: {valid_5mc['percent_m'].max():.2f}%")
    print(f"  25th percentile: {valid_5mc['percent_m'].quantile(0.25):.2f}%")
    print(f"  75th percentile: {valid_5mc['percent_m'].quantile(0.75):.2f}%")
    
    # 5hmC statistics
    valid_5hmc = meth_df[meth_df['count_valid_h'] >= MIN_COVERAGE]
    print(f"\n5hmC Statistics (coverage ≥ {MIN_COVERAGE}):")
    print(f"  Sites: {len(valid_5hmc):,}")
    print(f"  Mean: {valid_5hmc['percent_h'].mean():.2f}%")
    print(f"  Median: {valid_5hmc['percent_h'].median():.2f}%")
    print(f"  Std: {valid_5hmc['percent_h'].std():.2f}%")
    print(f"  Min: {valid_5hmc['percent_h'].min():.2f}%")
    print(f"  Max: {valid_5hmc['percent_h'].max():.2f}%")
    print(f"  25th percentile: {valid_5hmc['percent_h'].quantile(0.25):.2f}%")
    print(f"  75th percentile: {valid_5hmc['percent_h'].quantile(0.75):.2f}%")
    
    return valid_5mc, valid_5hmc

# -------------------
# Correlation Calculation
# -------------------
def calculate_window_correlation(sites, mod_type='5mC'):
    """
    Calculate correlation between nearby sites using sliding windows
    Returns R² (squared Pearson correlation) vs distance
    """
    if mod_type == '5mC':
        value_col = 'percent_m'
    else:
        value_col = 'percent_h'
    
    # Sort by position
    sites = sites.sort_values('start').reset_index(drop=True)
    n_sites = len(sites)
    
    correlations = []
    
    # For each site, correlate with nearby sites
    for i in range(n_sites):
        site_i = sites.iloc[i]
        pos_i = (site_i['start'] + site_i['end']) / 2
        val_i = site_i[value_col]
        
        if pd.isna(val_i):
            continue
        
        # Collect nearby sites
        nearby_vals = []
        nearby_dists = []
        
        for j in range(i+1, min(i+100, n_sites)):  # Look at next 100 sites
            site_j = sites.iloc[j]
            pos_j = (site_j['start'] + site_j['end']) / 2
            distance = abs(pos_j - pos_i)
            
            if distance > MAX_DISTANCE:
                break
            
            if distance < MIN_DISTANCE:
                continue
            
            val_j = site_j[value_col]
            if pd.notna(val_j):
                nearby_vals.append(val_j)
                nearby_dists.append(distance)
        
        # Calculate correlation with each nearby site
        for k, (val_j, dist) in enumerate(zip(nearby_vals, nearby_dists)):
            correlations.append({
                'distance': dist,
                'val1': val_i,
                'val2': val_j,
                'diff': abs(val_i - val_j)
            })
    
    return pd.DataFrame(correlations)

def calculate_binned_correlation(corr_df, distance_bins):
    """Calculate R² for each distance bin"""
    if len(corr_df) == 0:
        return pd.DataFrame()
    
    corr_df['distance_bin'] = pd.cut(corr_df['distance'], bins=distance_bins, 
                                      include_lowest=True)
    
    results = []
    for bin_label in corr_df['distance_bin'].cat.categories:
        bin_data = corr_df[corr_df['distance_bin'] == bin_label]
        
        if len(bin_data) < 3:
            continue
        
        # Calculate Pearson R between val1 and val2
        vals1 = bin_data['val1'].values
        vals2 = bin_data['val2'].values
        
        if len(vals1) > 2 and np.std(vals1) > 0 and np.std(vals2) > 0:
            r, pval = pearsonr(vals1, vals2)
            r2 = r**2
        else:
            r2 = 0
        
        results.append({
            'distance_bin': str(bin_label),
            'mean_distance': bin_data['distance'].mean(),
            'r2': r2,
            'n_pairs': len(bin_data),
            'mean_diff': bin_data['diff'].mean()
        })
    
    return pd.DataFrame(results)

# -------------------
# Process Samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"{'='*70}")
print("CO-METHYLATION DISTANCE ANALYSIS")
print('='*70)
print(f"\nParameters:")
print(f"  Minimum coverage: {MIN_COVERAGE} reads")
print(f"  Distance range: {MIN_DISTANCE} - {MAX_DISTANCE} bp")
print(f"\nProcessing {len(sample_files)} samples...")

all_results = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    # Explore data distribution
    valid_5mc, valid_5hmc = explore_data_distribution(meth_df, sample_name)
    
    # Process each chromosome
    print(f"\nCalculating correlations...")
    
    all_corr_5mc = []
    all_corr_5hmc = []
    
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        chrom_5mc = valid_5mc[valid_5mc['chrom'] == chrom]
        chrom_5hmc = valid_5hmc[valid_5hmc['chrom'] == chrom]
        
        if len(chrom_5mc) >= 10:
            corr_5mc = calculate_window_correlation(chrom_5mc, mod_type='5mC')
            if len(corr_5mc) > 0:
                corr_5mc['chrom'] = chrom
                all_corr_5mc.append(corr_5mc)
        
        if len(chrom_5hmc) >= 10:
            corr_5hmc = calculate_window_correlation(chrom_5hmc, mod_type='5hmC')
            if len(corr_5hmc) > 0:
                corr_5hmc['chrom'] = chrom
                all_corr_5hmc.append(corr_5hmc)
    
    # Combine results
    if all_corr_5mc:
        combined_5mc = pd.concat(all_corr_5mc, ignore_index=True)
        binned_5mc = calculate_binned_correlation(combined_5mc, distance_bins)
        print(f"\n5mC: {len(combined_5mc):,} pairs analyzed")
    else:
        combined_5mc = pd.DataFrame()
        binned_5mc = pd.DataFrame()
        print(f"\n5mC: No pairs found")
    
    if all_corr_5hmc:
        combined_5hmc = pd.concat(all_corr_5hmc, ignore_index=True)
        binned_5hmc = calculate_binned_correlation(combined_5hmc, distance_bins)
        print(f"5hmC: {len(combined_5hmc):,} pairs analyzed")
    else:
        combined_5hmc = pd.DataFrame()
        binned_5hmc = pd.DataFrame()
        print(f"5hmC: No pairs found")
    
    all_results[sample_name] = {
        '5mC_pairs': combined_5mc,
        '5hmC_pairs': combined_5hmc,
        '5mC_binned': binned_5mc,
        '5hmC_binned': binned_5hmc
    }

# -------------------
# Generate Plots
# -------------------
print(f"\n{'='*70}")
print("GENERATING PLOTS")
print('='*70)

for sample_name, results in all_results.items():
    print(f"\nPlotting {sample_name}...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ========== 5mC Plots ==========
    if len(results['5mC_pairs']) > 0:
        data_5mc = results['5mC_pairs']
        binned_5mc = results['5mC_binned']
        
        # Plot 1: Scatter plot with density
        ax1 = fig.add_subplot(gs[0, 0])
        h = ax1.hexbin(data_5mc['distance'], data_5mc['diff'], 
                      gridsize=80, cmap='YlOrRd', mincnt=1,
                      extent=[0, MAX_DISTANCE, 0, data_5mc['diff'].max()])
        plt.colorbar(h, ax=ax1, label='Density')
        
        ax1.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Methylation Difference (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{sample_name} - 5mC Methylation Difference vs Distance\n(n={len(data_5mc):,} pairs)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of differences by distance
        ax2 = fig.add_subplot(gs[1, 0])
        data_5mc['distance_bin'] = pd.cut(data_5mc['distance'], bins=distance_bins)
        bp_data = []
        bp_labels = []
        
        for i, cat in enumerate(data_5mc['distance_bin'].cat.categories):
            bin_data = data_5mc[data_5mc['distance_bin'] == cat]['diff']
            if len(bin_data) > 0:
                bp_data.append(bin_data)
                bp_labels.append(f"{int(distance_bins[i])}-\n{int(distance_bins[i+1])}")
        
        if bp_data:
            bp = ax2.boxplot(bp_data, labels=bp_labels, showfliers=False, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#ff9999')
                patch.set_alpha(0.7)
        
        ax2.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Methylation Difference (%)', fontsize=12, fontweight='bold')
        ax2.set_title('5mC: Difference by Distance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: R² by distance
        ax3 = fig.add_subplot(gs[2, 0])
        if len(binned_5mc) > 0:
            ax3.plot(binned_5mc['mean_distance'], binned_5mc['r2'], 
                    'o-', linewidth=2.5, markersize=10, color='red', alpha=0.8)
            
            for _, row in binned_5mc.iterrows():
                ax3.text(row['mean_distance'], row['r2'] + 0.02, 
                        f"n={int(row['n_pairs'])}", 
                        ha='center', fontsize=8, alpha=0.7)
        
        ax3.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('R² (Correlation)', fontsize=12, fontweight='bold')
        ax3.set_title('5mC: Correlation vs Distance', fontsize=12, fontweight='bold')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
    
    # ========== 5hmC Plots ==========
    if len(results['5hmC_pairs']) > 0:
        data_5hmc = results['5hmC_pairs']
        binned_5hmc = results['5hmC_binned']
        
        # Plot 4: Scatter plot
        ax4 = fig.add_subplot(gs[0, 1])
        h = ax4.hexbin(data_5hmc['distance'], data_5hmc['diff'], 
                      gridsize=80, cmap='YlOrRd', mincnt=1,
                      extent=[0, MAX_DISTANCE, 0, data_5hmc['diff'].max()])
        plt.colorbar(h, ax=ax4, label='Density')
        
        ax4.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Methylation Difference (%)', fontsize=12, fontweight='bold')
        ax4.set_title(f'{sample_name} - 5hmC Methylation Difference vs Distance\n(n={len(data_5hmc):,} pairs)', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        data_5hmc['distance_bin'] = pd.cut(data_5hmc['distance'], bins=distance_bins)
        bp_data = []
        bp_labels = []
        
        for i, cat in enumerate(data_5hmc['distance_bin'].cat.categories):
            bin_data = data_5hmc[data_5hmc['distance_bin'] == cat]['diff']
            if len(bin_data) > 0:
                bp_data.append(bin_data)
                bp_labels.append(f"{int(distance_bins[i])}-\n{int(distance_bins[i+1])}")
        
        if bp_data:
            bp = ax5.boxplot(bp_data, labels=bp_labels, showfliers=False, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#9999ff')
                patch.set_alpha(0.7)
        
        ax5.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Methylation Difference (%)', fontsize=12, fontweight='bold')
        ax5.set_title('5hmC: Difference by Distance', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: R² by distance
        ax6 = fig.add_subplot(gs[2, 1])
        if len(binned_5hmc) > 0:
            ax6.plot(binned_5hmc['mean_distance'], binned_5hmc['r2'], 
                    'o-', linewidth=2.5, markersize=10, color='blue', alpha=0.8)
            
            for _, row in binned_5hmc.iterrows():
                ax6.text(row['mean_distance'], row['r2'] + 0.02, 
                        f"n={int(row['n_pairs'])}", 
                        ha='center', fontsize=8, alpha=0.7)
        
        ax6.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('R² (Correlation)', fontsize=12, fontweight='bold')
        ax6.set_title('5hmC: Correlation vs Distance', fontsize=12, fontweight='bold')
        ax6.set_xscale('log')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_comethylation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Sample Comparison
# -------------------
print("\nGenerating comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

for idx, (sample_name, results) in enumerate(all_results.items()):
    # 5mC
    if len(results['5mC_binned']) > 0:
        data = results['5mC_binned']
        ax1.plot(data['mean_distance'], data['r2'], 
                'o-', linewidth=2.5, markersize=8, 
                label=sample_name, color=colors[idx], alpha=0.8)
    
    # 5hmC
    if len(results['5hmC_binned']) > 0:
        data = results['5hmC_binned']
        ax2.plot(data['mean_distance'], data['r2'], 
                'o-', linewidth=2.5, markersize=8,
                label=sample_name, color=colors[idx], alpha=0.8)

ax1.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
ax1.set_title('5mC Correlation vs Distance', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_ylim([0, 1])

ax2.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Correlation vs Distance', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sample_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print('='*70)
print(f"\nResults saved to: {out_dir}/")
print(f"  - [sample]_comethylation.png")
print(f"  - sample_comparison.png")
