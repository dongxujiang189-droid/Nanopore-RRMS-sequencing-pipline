#!/usr/bin/env python3
"""
Methylation Density and Distance Analysis
Calculates distances between methylated CpG sites and visualizes methylation density
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
out_dir = os.path.join(base_dir, "methylation_density")
os.makedirs(out_dir, exist_ok=True)

# Parameters
MIN_COVERAGE = 10           # Minimum valid reads
METHYLATION_THRESHOLD = 50  # Consider site "methylated" if >50%
DENSITY_WINDOW = 100000     # 100kb windows for density calculation
MAX_DISTANCE_PLOT = 10000   # Maximum distance to plot (bp)

chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

# -------------------
# Analysis Functions
# -------------------
def calculate_inter_site_distances(sites_df, mod_type='5mC'):
    """
    Calculate distances between consecutive methylated CpG sites
    """
    if mod_type == '5mC':
        value_col = 'percent_m'
        valid_col = 'count_valid_m'
    else:
        value_col = 'percent_h'
        valid_col = 'count_valid_h'
    
    # Filter for methylated sites (above threshold and good coverage)
    methylated = sites_df[
        (sites_df[value_col] >= METHYLATION_THRESHOLD) &
        (sites_df[valid_col] >= MIN_COVERAGE)
    ].copy()
    
    if len(methylated) < 2:
        return pd.DataFrame(), methylated
    
    # Sort by position
    methylated = methylated.sort_values('start').reset_index(drop=True)
    methylated['center'] = (methylated['start'] + methylated['end']) / 2
    
    # Calculate distance to next methylated site
    distances = []
    for i in range(len(methylated) - 1):
        dist = methylated.iloc[i+1]['center'] - methylated.iloc[i]['center']
        distances.append({
            'distance': dist,
            'chrom': methylated.iloc[i]['chrom'],
            'pos': methylated.iloc[i]['center'],
            'methylation1': methylated.iloc[i][value_col],
            'methylation2': methylated.iloc[i+1][value_col]
        })
    
    return pd.DataFrame(distances), methylated

def calculate_methylation_density(sites_df, mod_type='5mC', window_size=DENSITY_WINDOW):
    """
    Calculate methylation density (number of methylated sites per window)
    """
    if mod_type == '5mC':
        value_col = 'percent_m'
        valid_col = 'count_valid_m'
    else:
        value_col = 'percent_h'
        valid_col = 'count_valid_h'
    
    # All valid sites
    valid_sites = sites_df[sites_df[valid_col] >= MIN_COVERAGE].copy()
    valid_sites['center'] = (valid_sites['start'] + valid_sites['end']) / 2
    
    # Methylated sites
    methylated = sites_df[
        (sites_df[value_col] >= METHYLATION_THRESHOLD) &
        (sites_df[valid_col] >= MIN_COVERAGE)
    ].copy()
    methylated['center'] = (methylated['start'] + methylated['end']) / 2
    
    # Create windows across chromosome
    chrom_start = valid_sites['start'].min()
    chrom_end = valid_sites['end'].max()
    
    windows = []
    for win_start in range(int(chrom_start), int(chrom_end), window_size):
        win_end = win_start + window_size
        
        # Count sites in window
        total_sites = len(valid_sites[
            (valid_sites['center'] >= win_start) &
            (valid_sites['center'] < win_end)
        ])
        
        methylated_sites = len(methylated[
            (methylated['center'] >= win_start) &
            (methylated['center'] < win_end)
        ])
        
        if total_sites > 0:
            windows.append({
                'window_start': win_start,
                'window_center': (win_start + win_end) / 2,
                'total_sites': total_sites,
                'methylated_sites': methylated_sites,
                'density': methylated_sites / window_size * 1e6,  # Sites per Mb
                'pct_methylated': 100 * methylated_sites / total_sites if total_sites > 0 else 0
            })
    
    return pd.DataFrame(windows)

# -------------------
# Load and Process Samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"{'='*70}")
print("METHYLATION DENSITY AND DISTANCE ANALYSIS")
print('='*70)
print(f"\nParameters:")
print(f"  Methylation threshold: ≥{METHYLATION_THRESHOLD}%")
print(f"  Minimum coverage: {MIN_COVERAGE} reads")
print(f"  Density window: {DENSITY_WINDOW/1000:.0f} kb")
print(f"\nProcessing {len(sample_files)} samples...\n")

all_results = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\n{'='*70}")
    print(f"Sample: {sample_name}")
    print('='*70)
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    print(f"Total sites: {len(meth_df):,}")
    
    # Analyze 5mC
    print("\n5mC Analysis:")
    sites_5mc = meth_df[meth_df['count_valid_m'] >= MIN_COVERAGE]
    methylated_5mc = len(meth_df[
        (meth_df['percent_m'] >= METHYLATION_THRESHOLD) &
        (meth_df['count_valid_m'] >= MIN_COVERAGE)
    ])
    print(f"  Valid sites: {len(sites_5mc):,}")
    print(f"  Methylated sites (≥{METHYLATION_THRESHOLD}%): {methylated_5mc:,} ({100*methylated_5mc/len(sites_5mc):.1f}%)")
    print(f"  Mean methylation: {sites_5mc['percent_m'].mean():.2f}%")
    
    # Analyze 5hmC
    print("\n5hmC Analysis:")
    sites_5hmc = meth_df[meth_df['count_valid_h'] >= MIN_COVERAGE]
    methylated_5hmc = len(meth_df[
        (meth_df['percent_h'] >= METHYLATION_THRESHOLD) &
        (meth_df['count_valid_h'] >= MIN_COVERAGE)
    ])
    print(f"  Valid sites: {len(sites_5hmc):,}")
    print(f"  Methylated sites (≥{METHYLATION_THRESHOLD}%): {methylated_5hmc:,} ({100*methylated_5hmc/len(sites_5hmc):.1f}%)")
    print(f"  Mean methylation: {sites_5hmc['percent_h'].mean():.2f}%")
    
    # Calculate distances and density for each chromosome
    print("\nProcessing chromosomes...")
    
    all_distances_5mc = []
    all_distances_5hmc = []
    all_density_5mc = []
    all_density_5hmc = []
    all_methylated_sites_5mc = []
    all_methylated_sites_5hmc = []
    
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        
        if len(chrom_data) < 10:
            continue
        
        # 5mC distances
        dist_5mc, meth_sites_5mc = calculate_inter_site_distances(chrom_data, mod_type='5mC')
        if len(dist_5mc) > 0:
            all_distances_5mc.append(dist_5mc)
            meth_sites_5mc['chrom'] = chrom
            all_methylated_sites_5mc.append(meth_sites_5mc)
        
        # 5hmC distances
        dist_5hmc, meth_sites_5hmc = calculate_inter_site_distances(chrom_data, mod_type='5hmC')
        if len(dist_5hmc) > 0:
            all_distances_5hmc.append(dist_5hmc)
            meth_sites_5hmc['chrom'] = chrom
            all_methylated_sites_5hmc.append(meth_sites_5hmc)
        
        # 5mC density
        dens_5mc = calculate_methylation_density(chrom_data, mod_type='5mC')
        if len(dens_5mc) > 0:
            dens_5mc['chrom'] = chrom
            all_density_5mc.append(dens_5mc)
        
        # 5hmC density
        dens_5hmc = calculate_methylation_density(chrom_data, mod_type='5hmC')
        if len(dens_5hmc) > 0:
            dens_5hmc['chrom'] = chrom
            all_density_5hmc.append(dens_5hmc)
    
    # Combine results
    results = {
        'distances_5mc': pd.concat(all_distances_5mc) if all_distances_5mc else pd.DataFrame(),
        'distances_5hmc': pd.concat(all_distances_5hmc) if all_distances_5hmc else pd.DataFrame(),
        'density_5mc': pd.concat(all_density_5mc) if all_density_5mc else pd.DataFrame(),
        'density_5hmc': pd.concat(all_density_5hmc) if all_density_5hmc else pd.DataFrame(),
        'methylated_sites_5mc': pd.concat(all_methylated_sites_5mc) if all_methylated_sites_5mc else pd.DataFrame(),
        'methylated_sites_5hmc': pd.concat(all_methylated_sites_5hmc) if all_methylated_sites_5hmc else pd.DataFrame()
    }
    
    print(f"\nResults:")
    print(f"  5mC distances: {len(results['distances_5mc']):,}")
    print(f"  5hmC distances: {len(results['distances_5hmc']):,}")
    print(f"  5mC density windows: {len(results['density_5mc']):,}")
    print(f"  5hmC density windows: {len(results['density_5hmc']):,}")
    
    all_results[sample_name] = results

# -------------------
# Generate Plots
# -------------------
print(f"\n{'='*70}")
print("GENERATING PLOTS")
print('='*70)

for sample_name, results in all_results.items():
    print(f"\nPlotting {sample_name}...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== 5mC Plots ==========
    # Plot 1: Distance distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if len(results['distances_5mc']) > 0:
        distances = results['distances_5mc']['distance']
        distances_plot = distances[distances <= MAX_DISTANCE_PLOT]
        
        ax1.hist(distances_plot, bins=100, color='#ff6b6b', alpha=0.7, edgecolor='black')
        ax1.axvline(distances_plot.median(), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {distances_plot.median():.0f} bp')
        ax1.set_xlabel('Distance to Next Methylated Site (bp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(f'{sample_name} - 5mC Inter-site Distances\n(n={len(distances):,} intervals)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Density across genome
    ax2 = fig.add_subplot(gs[1, 0])
    if len(results['density_5mc']) > 0:
        for chrom in chromosomes[:5]:  # First 5 chromosomes
            chrom_dens = results['density_5mc'][results['density_5mc']['chrom'] == chrom]
            if len(chrom_dens) > 0:
                ax2.plot(chrom_dens['window_center']/1e6, chrom_dens['density'], 
                        alpha=0.7, linewidth=1.5, label=chrom)
        
        ax2.set_xlabel('Position (Mb)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Methylation Density\n(sites per Mb)', fontsize=12, fontweight='bold')
        ax2.set_title('5mC Density Along Chromosomes', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distance distribution
    ax3 = fig.add_subplot(gs[2, 0])
    if len(results['distances_5mc']) > 0:
        distances = results['distances_5mc']['distance']
        distances_sorted = np.sort(distances[distances <= MAX_DISTANCE_PLOT])
        cumulative = np.arange(1, len(distances_sorted) + 1) / len(distances_sorted)
        
        ax3.plot(distances_sorted, cumulative, linewidth=2, color='#ff6b6b')
        ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(distances_sorted[int(len(distances_sorted)*0.5)], 
                   color='red', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax3.set_title('5mC: Cumulative Distance Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
    
    # ========== 5hmC Plots ==========
    # Plot 4: Distance distribution
    ax4 = fig.add_subplot(gs[0, 1])
    if len(results['distances_5hmc']) > 0:
        distances = results['distances_5hmc']['distance']
        distances_plot = distances[distances <= MAX_DISTANCE_PLOT]
        
        ax4.hist(distances_plot, bins=100, color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax4.axvline(distances_plot.median(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {distances_plot.median():.0f} bp')
        ax4.set_xlabel('Distance to Next Methylated Site (bp)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title(f'{sample_name} - 5hmC Inter-site Distances\n(n={len(distances):,} intervals)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Density across genome
    ax5 = fig.add_subplot(gs[1, 1])
    if len(results['density_5hmc']) > 0:
        for chrom in chromosomes[:5]:
            chrom_dens = results['density_5hmc'][results['density_5hmc']['chrom'] == chrom]
            if len(chrom_dens) > 0:
                ax5.plot(chrom_dens['window_center']/1e6, chrom_dens['density'], 
                        alpha=0.7, linewidth=1.5, label=chrom)
        
        ax5.set_xlabel('Position (Mb)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Methylation Density\n(sites per Mb)', fontsize=12, fontweight='bold')
        ax5.set_title('5hmC Density Along Chromosomes', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9, ncol=2)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative distance distribution
    ax6 = fig.add_subplot(gs[2, 1])
    if len(results['distances_5hmc']) > 0:
        distances = results['distances_5hmc']['distance']
        distances_sorted = np.sort(distances[distances <= MAX_DISTANCE_PLOT])
        cumulative = np.arange(1, len(distances_sorted) + 1) / len(distances_sorted)
        
        ax6.plot(distances_sorted, cumulative, linewidth=2, color='#4ecdc4')
        ax6.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(distances_sorted[int(len(distances_sorted)*0.5)], 
                   color='blue', linestyle='--', alpha=0.5)
        
        ax6.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax6.set_title('5hmC: Cumulative Distance Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_methylation_density.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Sample Comparison Plot
# -------------------
print("\nGenerating comparison plot...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

# 5mC distance distributions
for idx, (sample_name, results) in enumerate(all_results.items()):
    if len(results['distances_5mc']) > 0:
        distances = results['distances_5mc']['distance']
        distances = distances[distances <= MAX_DISTANCE_PLOT]
        ax1.hist(distances, bins=50, alpha=0.5, label=sample_name, 
                color=colors[idx], edgecolor='black')

ax1.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('5mC: Inter-site Distance Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# 5hmC distance distributions
for idx, (sample_name, results) in enumerate(all_results.items()):
    if len(results['distances_5hmc']) > 0:
        distances = results['distances_5hmc']['distance']
        distances = distances[distances <= MAX_DISTANCE_PLOT]
        ax2.hist(distances, bins=50, alpha=0.5, label=sample_name, 
                color=colors[idx], edgecolor='black')

ax2.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('5hmC: Inter-site Distance Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 5mC median distances
medians_5mc = []
samples = []
for sample_name, results in all_results.items():
    if len(results['distances_5mc']) > 0:
        samples.append(sample_name)
        medians_5mc.append(results['distances_5mc']['distance'].median())

if medians_5mc:
    ax3.bar(range(len(samples)), medians_5mc, color=colors[:len(samples)], 
           alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(samples)))
    ax3.set_xticklabels(samples, rotation=45, ha='right')
    ax3.set_ylabel('Median Distance (bp)', fontsize=12, fontweight='bold')
    ax3.set_title('5mC: Median Inter-site Distance', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

# 5hmC median distances
medians_5hmc = []
samples = []
for sample_name, results in all_results.items():
    if len(results['distances_5hmc']) > 0:
        samples.append(sample_name)
        medians_5hmc.append(results['distances_5hmc']['distance'].median())

if medians_5hmc:
    ax4.bar(range(len(samples)), medians_5hmc, color=colors[:len(samples)], 
           alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(samples)))
    ax4.set_xticklabels(samples, rotation=45, ha='right')
    ax4.set_ylabel('Median Distance (bp)', fontsize=12, fontweight='bold')
    ax4.set_title('5hmC: Median Inter-site Distance', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sample_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Save Statistics
# -------------------
stats_list = []
for sample_name, results in all_results.items():
    if len(results['distances_5mc']) > 0:
        dist_5mc = results['distances_5mc']['distance']
        stats_list.append({
            'Sample': sample_name,
            'Modification': '5mC',
            'N_methylated_sites': len(results['methylated_sites_5mc']),
            'N_distances': len(dist_5mc),
            'Median_distance_bp': dist_5mc.median(),
            'Mean_distance_bp': dist_5mc.mean(),
            'Min_distance_bp': dist_5mc.min(),
            'Max_distance_bp': dist_5mc.max()
        })
    
    if len(results['distances_5hmc']) > 0:
        dist_5hmc = results['distances_5hmc']['distance']
        stats_list.append({
            'Sample': sample_name,
            'Modification': '5hmC',
            'N_methylated_sites': len(results['methylated_sites_5hmc']),
            'N_distances': len(dist_5hmc),
            'Median_distance_bp': dist_5hmc.median(),
            'Mean_distance_bp': dist_5hmc.mean(),
            'Min_distance_bp': dist_5hmc.min(),
            'Max_distance_bp': dist_5hmc.max()
        })

stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(os.path.join(out_dir, 'methylation_distance_stats.csv'), index=False)

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(stats_df.to_string(index=False))
print("="*70)

print(f"\n✓ Analysis Complete!")
print(f"\nResults saved to: {out_dir}/")
print(f"  - [sample]_methylation_density.png")
print(f"  - sample_comparison.png")
print(f"  - methylation_distance_stats.csv")
