#!/usr/bin/env python3
"""
Co-methylation Distance Analysis (Similar to Figure 1A from PMC11463401)
Analyzes correlation between nearby CpG sites and visualizes R² vs distance
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

# Parameters
max_distance = 5000  # Maximum distance between CpG sites (bp)
min_distance = 10    # Minimum distance (bp)
distance_bins = [0, 50, 100, 200, 500, 1000, 2000, 5000]  # Distance categories
min_coverage = 10    # Minimum valid reads for a site
min_sites_per_window = 5  # Minimum CpG sites in a window for correlation

chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

# -------------------
# Helper functions
# -------------------
def calculate_pairwise_correlation(sites_df, mod_type='5mC'):
    """Calculate R² between all pairs of CpG sites within max_distance"""
    correlations = []
    
    if mod_type == '5mC':
        value_col = 'percent_m'
    else:
        value_col = 'percent_h'
    
    n_sites = len(sites_df)
    
    for i in range(n_sites):
        site1 = sites_df.iloc[i]
        pos1 = (site1['start'] + site1['end']) / 2
        
        # Only look at sites within max_distance
        for j in range(i+1, n_sites):
            site2 = sites_df.iloc[j]
            pos2 = (site2['start'] + site2['end']) / 2
            
            distance = abs(pos2 - pos1)
            
            if distance > max_distance:
                break  # Sites are sorted, so we can stop
            
            if distance < min_distance:
                continue
            
            # Calculate correlation
            val1 = site1[value_col]
            val2 = site2[value_col]
            
            if pd.notna(val1) and pd.notna(val2):
                # Simple R² calculation
                # In real analysis, you'd need multiple reads to calculate correlation
                # Here we use the percentage as a proxy
                correlations.append({
                    'distance': distance,
                    'r2': 0.5,  # Placeholder - need read-level data for true R²
                    'val1': val1,
                    'val2': val2
                })
    
    return pd.DataFrame(correlations)

def calculate_window_correlations(chrom_data, window_size=1000, mod_type='5mC'):
    """Calculate correlations within sliding windows"""
    if mod_type == '5mC':
        value_col = 'percent_m'
    else:
        value_col = 'percent_h'
    
    results = []
    
    # Sort by position
    chrom_data = chrom_data.sort_values('start')
    positions = (chrom_data['start'] + chrom_data['end']) / 2
    values = chrom_data[value_col].values
    
    n = len(chrom_data)
    
    for i in range(n):
        pos_i = positions.iloc[i]
        val_i = values[i]
        
        if pd.isna(val_i):
            continue
        
        # Find sites within window
        for j in range(i+1, n):
            pos_j = positions.iloc[j]
            val_j = values[j]
            
            distance = abs(pos_j - pos_i)
            
            if distance > max_distance:
                break
            
            if distance < min_distance or pd.isna(val_j):
                continue
            
            # Calculate correlation (simplified - using value similarity as proxy)
            # True R² would require multiple reads showing methylation patterns
            diff = abs(val_i - val_j)
            similarity = 1 - (diff / 100)  # Normalized similarity
            r2_proxy = max(0, similarity)
            
            results.append({
                'distance': distance,
                'r2': r2_proxy,
                'chrom': chrom_data.iloc[i]['chrom']
            })
    
    return pd.DataFrame(results)

# -------------------
# Load and process samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"{'='*60}")
print("CO-METHYLATION DISTANCE ANALYSIS")
print('='*60)
print(f"\nProcessing {len(sample_files)} samples...")

all_correlations = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\n{sample_name}:")
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    # Filter by coverage
    meth_df = meth_df[
        (meth_df['count_valid_m'] >= min_coverage) |
        (meth_df['count_valid_h'] >= min_coverage)
    ]
    
    print(f"  Sites passing filter: {len(meth_df):,}")
    
    # Calculate correlations for each modification type
    correlations_5mc = []
    correlations_5hmc = []
    
    for chrom in tqdm(chromosomes, desc="  Chromosomes"):
        chrom_data = meth_df[meth_df['chrom'] == chrom].copy()
        
        if len(chrom_data) < min_sites_per_window:
            continue
        
        # 5mC correlations
        corr_5mc = calculate_window_correlations(chrom_data, mod_type='5mC')
        if len(corr_5mc) > 0:
            correlations_5mc.append(corr_5mc)
        
        # 5hmC correlations
        corr_5hmc = calculate_window_correlations(chrom_data, mod_type='5hmC')
        if len(corr_5hmc) > 0:
            correlations_5hmc.append(corr_5hmc)
    
    if correlations_5mc:
        all_5mc = pd.concat(correlations_5mc, ignore_index=True)
        print(f"  5mC pairs analyzed: {len(all_5mc):,}")
    else:
        all_5mc = pd.DataFrame()
    
    if correlations_5hmc:
        all_5hmc = pd.concat(correlations_5hmc, ignore_index=True)
        print(f"  5hmC pairs analyzed: {len(all_5hmc):,}")
    else:
        all_5hmc = pd.DataFrame()
    
    all_correlations[sample_name] = {
        '5mC': all_5mc,
        '5hmC': all_5hmc
    }

# -------------------
# Plot 1: R² vs Distance (like Figure 1A)
# -------------------
print(f"\n{'='*60}")
print("GENERATING PLOTS")
print('='*60)

for sample_name, data_dict in all_correlations.items():
    print(f"Plotting {sample_name}...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 5mC scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    if len(data_dict['5mC']) > 0:
        data_5mc = data_dict['5mC']
        # Create density plot
        h = ax1.hexbin(data_5mc['distance'], data_5mc['r2'], 
                      gridsize=50, cmap='YlOrRd', mincnt=1,
                      extent=[0, max_distance, 0, 1])
        plt.colorbar(h, ax=ax1, label='Density')
    
    ax1.set_xlabel('Distance (bp)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('R² (Co-methylation)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{sample_name} - 5mC Co-methylation', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, max_distance])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # 5hmC scatter plot
    ax2 = fig.add_subplot(gs[0, 1])
    if len(data_dict['5hmC']) > 0:
        data_5hmc = data_dict['5hmC']
        h = ax2.hexbin(data_5hmc['distance'], data_5hmc['r2'], 
                      gridsize=50, cmap='YlOrRd', mincnt=1,
                      extent=[0, max_distance, 0, 1])
        plt.colorbar(h, ax=ax2, label='Density')
    
    ax2.set_xlabel('Distance (bp)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R² (Co-methylation)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{sample_name} - 5hmC Co-methylation', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, max_distance])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 5mC distribution by distance bin
    ax3 = fig.add_subplot(gs[1, 0])
    if len(data_dict['5mC']) > 0:
        data_5mc['distance_bin'] = pd.cut(data_5mc['distance'], bins=distance_bins)
        bp_data = [data_5mc[data_5mc['distance_bin'] == bin]['r2'].dropna() 
                   for bin in data_5mc['distance_bin'].cat.categories]
        bp_labels = [f"{int(distance_bins[i])}-{int(distance_bins[i+1])}" 
                     for i in range(len(distance_bins)-1)]
        
        ax3.boxplot([d for d in bp_data if len(d) > 0], 
                   labels=[bp_labels[i] for i, d in enumerate(bp_data) if len(d) > 0],
                   showfliers=False)
        ax3.set_xlabel('Distance (bp)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax3.set_title('5mC R² by Distance', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 5hmC distribution by distance bin
    ax4 = fig.add_subplot(gs[1, 1])
    if len(data_dict['5hmC']) > 0:
        data_5hmc['distance_bin'] = pd.cut(data_5hmc['distance'], bins=distance_bins)
        bp_data = [data_5hmc[data_5hmc['distance_bin'] == bin]['r2'].dropna() 
                   for bin in data_5hmc['distance_bin'].cat.categories]
        
        ax4.boxplot([d for d in bp_data if len(d) > 0], 
                   labels=[bp_labels[i] for i, d in enumerate(bp_data) if len(d) > 0],
                   showfliers=False)
        ax4.set_xlabel('Distance (bp)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax4.set_title('5hmC R² by Distance', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_comethylation_distance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Plot 2: Sample comparison
# -------------------
print("\nGenerating comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.Set2(np.linspace(0, 1, len(all_correlations)))

# 5mC comparison
for idx, (sample_name, data_dict) in enumerate(all_correlations.items()):
    if len(data_dict['5mC']) > 0:
        data = data_dict['5mC']
        # Calculate mean R² for distance bins
        data['distance_bin'] = pd.cut(data['distance'], bins=distance_bins)
        mean_r2 = data.groupby('distance_bin')['r2'].mean()
        bin_centers = [(distance_bins[i] + distance_bins[i+1])/2 
                       for i in range(len(distance_bins)-1)]
        
        ax1.plot(bin_centers, mean_r2.values, 'o-', linewidth=2, 
                label=sample_name, color=colors[idx], markersize=6, alpha=0.8)

ax1.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean R² (Co-methylation)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Co-methylation vs Distance', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# 5hmC comparison
for idx, (sample_name, data_dict) in enumerate(all_correlations.items()):
    if len(data_dict['5hmC']) > 0:
        data = data_dict['5hmC']
        data['distance_bin'] = pd.cut(data['distance'], bins=distance_bins)
        mean_r2 = data.groupby('distance_bin')['r2'].mean()
        
        ax2.plot(bin_centers, mean_r2.values, 'o-', linewidth=2, 
                label=sample_name, color=colors[idx], markersize=6, alpha=0.8)

ax2.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean R² (Co-methylation)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Co-methylation vs Distance', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sample_comparison_comethylation.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Save statistics
# -------------------
print("\nCalculating statistics...")

stats_list = []
for sample_name, data_dict in all_correlations.items():
    for mod_type in ['5mC', '5hmC']:
        data = data_dict[mod_type]
        if len(data) > 0:
            # Short-range (≤200bp)
            short_range = data[data['distance'] <= 200]
            # Long-range (≥1000bp)
            long_range = data[data['distance'] >= 1000]
            
            stats_list.append({
                'Sample': sample_name,
                'Modification': mod_type,
                'Total_pairs': len(data),
                'Short_range_pairs': len(short_range),
                'Long_range_pairs': len(long_range),
                'Pct_long_range': 100 * len(long_range) / len(data) if len(data) > 0 else 0,
                'Mean_R2_short': short_range['r2'].mean() if len(short_range) > 0 else np.nan,
                'Mean_R2_long': long_range['r2'].mean() if len(long_range) > 0 else np.nan,
                'Overall_mean_R2': data['r2'].mean()
            })

stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(os.path.join(out_dir, 'comethylation_statistics.csv'), index=False)

print("\n" + "="*60)
print("Co-methylation Statistics:")
print(stats_df.to_string(index=False))
print("="*60)

print(f"\n✓ Analysis complete!")
print(f"\nResults saved to: {out_dir}/")
print(f"  - [sample]_comethylation_distance.png - Per-sample analysis")
print(f"  - sample_comparison_comethylation.png - Cross-sample comparison")
print(f"  - comethylation_statistics.csv - Summary statistics")
print(f"\nNote: This is a simplified R² calculation based on methylation similarity.")
print(f"True co-methylation R² requires read-level methylation patterns.")
