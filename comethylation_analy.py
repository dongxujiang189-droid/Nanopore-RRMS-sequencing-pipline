#!/usr/bin/env python3
"""
Methylation Density and Distance Analysis - Two-Pass Approach
Pass 1: Calculate mean methylation across all samples to determine thresholds
Pass 2: Apply thresholds consistently to all samples for analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
out_dir = os.path.join(base_dir, "methylation_density")
os.makedirs(out_dir, exist_ok=True)

# Parameters
MIN_COVERAGE = 10
DENSITY_WINDOW = 100000
MAX_DISTANCE_PLOT = 10000
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

# Threshold calculation method
THRESHOLD_METHOD = 'mean'  # Options: 'mean', 'median', 'percentile_75'
PERCENTILE = 75  # Used if THRESHOLD_METHOD = 'percentile_XX'

# -------------------
# PASS 1: Calculate Global Statistics
# -------------------
print(f"{'='*70}")
print("PASS 1: CALCULATING GLOBAL METHYLATION STATISTICS")
print('='*70)

sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"\nFound {len(sample_files)} samples")
print("Loading data to calculate thresholds...\n")

all_5mc_values = []
all_5hmc_values = []
sample_stats = []

for sample_file in tqdm(sample_files, desc="Loading samples"):
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    # Filter by coverage
    valid_5mc = meth_df[meth_df['count_valid_m'] >= MIN_COVERAGE]['percent_m'].dropna()
    valid_5hmc = meth_df[meth_df['count_valid_h'] >= MIN_COVERAGE]['percent_h'].dropna()
    
    # Store values
    all_5mc_values.extend(valid_5mc.tolist())
    all_5hmc_values.extend(valid_5hmc.tolist())
    
    # Store per-sample stats
    sample_stats.append({
        'Sample': sample_name,
        '5mC_mean': valid_5mc.mean(),
        '5mC_median': valid_5mc.median(),
        '5mC_std': valid_5mc.std(),
        '5hmC_mean': valid_5hmc.mean(),
        '5hmC_median': valid_5hmc.median(),
        '5hmC_std': valid_5hmc.std(),
        'N_sites_5mC': len(valid_5mc),
        'N_sites_5hmC': len(valid_5hmc)
    })

# Calculate global statistics
all_5mc_values = np.array(all_5mc_values)
all_5hmc_values = np.array(all_5hmc_values)

print(f"\n{'='*70}")
print("GLOBAL METHYLATION STATISTICS")
print('='*70)
print(f"\nTotal sites analyzed:")
print(f"  5mC: {len(all_5mc_values):,}")
print(f"  5hmC: {len(all_5hmc_values):,}")

print(f"\n5mC Statistics (all samples):")
print(f"  Mean: {all_5mc_values.mean():.2f}%")
print(f"  Median: {np.median(all_5mc_values):.2f}%")
print(f"  Std Dev: {all_5mc_values.std():.2f}%")
print(f"  25th percentile: {np.percentile(all_5mc_values, 25):.2f}%")
print(f"  75th percentile: {np.percentile(all_5mc_values, 75):.2f}%")

print(f"\n5hmC Statistics (all samples):")
print(f"  Mean: {all_5hmc_values.mean():.2f}%")
print(f"  Median: {np.median(all_5hmc_values):.2f}%")
print(f"  Std Dev: {all_5hmc_values.std():.2f}%")
print(f"  25th percentile: {np.percentile(all_5hmc_values, 25):.2f}%")
print(f"  75th percentile: {np.percentile(all_5hmc_values, 75):.2f}%")

# Determine thresholds based on method
if THRESHOLD_METHOD == 'mean':
    THRESHOLD_5MC = all_5mc_values.mean()
    THRESHOLD_5HMC = all_5hmc_values.mean()
    method_desc = "Mean"
elif THRESHOLD_METHOD == 'median':
    THRESHOLD_5MC = np.median(all_5mc_values)
    THRESHOLD_5HMC = np.median(all_5hmc_values)
    method_desc = "Median"
elif THRESHOLD_METHOD.startswith('percentile'):
    THRESHOLD_5MC = np.percentile(all_5mc_values, PERCENTILE)
    THRESHOLD_5HMC = np.percentile(all_5hmc_values, PERCENTILE)
    method_desc = f"{PERCENTILE}th Percentile"
else:
    THRESHOLD_5MC = all_5mc_values.mean()
    THRESHOLD_5HMC = all_5hmc_values.mean()
    method_desc = "Mean (default)"

print(f"\n{'='*70}")
print(f"THRESHOLDS DETERMINED ({method_desc} Method)")
print('='*70)
print(f"  5mC threshold: {THRESHOLD_5MC:.2f}%")
print(f"  5hmC threshold: {THRESHOLD_5HMC:.2f}%")
print(f"\nSites above threshold:")
print(f"  5mC: {(all_5mc_values >= THRESHOLD_5MC).sum():,} ({100*(all_5mc_values >= THRESHOLD_5MC).sum()/len(all_5mc_values):.1f}%)")
print(f"  5hmC: {(all_5hmc_values >= THRESHOLD_5HMC).sum():,} ({100*(all_5hmc_values >= THRESHOLD_5HMC).sum()/len(all_5hmc_values):.1f}%)")

# Save sample statistics
stats_df = pd.DataFrame(sample_stats)
stats_df.to_csv(os.path.join(out_dir, 'sample_methylation_statistics.csv'), index=False)
print(f"\n✓ Sample statistics saved to: sample_methylation_statistics.csv")

# -------------------
# Analysis Functions
# -------------------
def calculate_inter_site_distances(sites_df, mod_type='5mC', threshold=None):
    """Calculate distances between consecutive methylated CpG sites"""
    if mod_type == '5mC':
        value_col = 'percent_m'
        valid_col = 'count_valid_m'
        use_threshold = threshold if threshold is not None else THRESHOLD_5MC
    else:
        value_col = 'percent_h'
        valid_col = 'count_valid_h'
        use_threshold = threshold if threshold is not None else THRESHOLD_5HMC
    
    # Filter for methylated sites
    methylated = sites_df[
        (sites_df[value_col] >= use_threshold) &
        (sites_df[valid_col] >= MIN_COVERAGE)
    ].copy()
    
    if len(methylated) < 2:
        return pd.DataFrame(), methylated
    
    methylated = methylated.sort_values('start').reset_index(drop=True)
    methylated['center'] = (methylated['start'] + methylated['end']) / 2
    
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

def calculate_methylation_density(sites_df, mod_type='5mC', threshold=None):
    """Calculate methylation density per window"""
    if mod_type == '5mC':
        value_col = 'percent_m'
        valid_col = 'count_valid_m'
        use_threshold = threshold if threshold is not None else THRESHOLD_5MC
    else:
        value_col = 'percent_h'
        valid_col = 'count_valid_h'
        use_threshold = threshold if threshold is not None else THRESHOLD_5HMC
    
    valid_sites = sites_df[sites_df[valid_col] >= MIN_COVERAGE].copy()
    if len(valid_sites) == 0:
        return pd.DataFrame()
    
    valid_sites['center'] = (valid_sites['start'] + valid_sites['end']) / 2
    methylated = valid_sites[valid_sites[value_col] >= use_threshold].copy()
    
    chrom_start = valid_sites['start'].min()
    chrom_end = valid_sites['end'].max()
    
    windows = []
    for win_start in range(int(chrom_start), int(chrom_end), DENSITY_WINDOW):
        win_end = win_start + DENSITY_WINDOW
        
        total = len(valid_sites[(valid_sites['center'] >= win_start) & (valid_sites['center'] < win_end)])
        meth = len(methylated[(methylated['center'] >= win_start) & (methylated['center'] < win_end)])
        
        if total > 0:
            windows.append({
                'window_start': win_start,
                'window_center': (win_start + win_end) / 2,
                'total_sites': total,
                'methylated_sites': meth,
                'density': meth / DENSITY_WINDOW * 1e6,
                'pct_methylated': 100 * meth / total if total > 0 else 0
            })
    
    return pd.DataFrame(windows)

# -------------------
# PASS 2: Process All Samples with Determined Thresholds
# -------------------
print(f"\n{'='*70}")
print("PASS 2: ANALYZING SAMPLES WITH DETERMINED THRESHOLDS")
print('='*70)

all_results = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\n{'-'*70}")
    print(f"Sample: {sample_name}")
    print('-'*70)
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    
    # Count methylated sites
    sites_5mc = meth_df[meth_df['count_valid_m'] >= MIN_COVERAGE]
    methylated_5mc = len(meth_df[
        (meth_df['percent_m'] >= THRESHOLD_5MC) &
        (meth_df['count_valid_m'] >= MIN_COVERAGE)
    ])
    
    sites_5hmc = meth_df[meth_df['count_valid_h'] >= MIN_COVERAGE]
    methylated_5hmc = len(meth_df[
        (meth_df['percent_h'] >= THRESHOLD_5HMC) &
        (meth_df['count_valid_h'] >= MIN_COVERAGE)
    ])
    
    print(f"5mC: {methylated_5mc:,} / {len(sites_5mc):,} sites ({100*methylated_5mc/len(sites_5mc):.1f}%) above {THRESHOLD_5MC:.2f}%")
    print(f"5hmC: {methylated_5hmc:,} / {len(sites_5hmc):,} sites ({100*methylated_5hmc/len(sites_5hmc):.1f}%) above {THRESHOLD_5HMC:.2f}%")
    
    # Process chromosomes
    all_dist_5mc = []
    all_dist_5hmc = []
    all_dens_5mc = []
    all_dens_5hmc = []
    all_meth_5mc = []
    all_meth_5hmc = []
    
    for chrom in tqdm(chromosomes, desc="Chromosomes", leave=False):
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        if len(chrom_data) < 10:
            continue
        
        # 5mC
        dist_5mc, meth_sites_5mc = calculate_inter_site_distances(chrom_data, mod_type='5mC')
        if len(dist_5mc) > 0:
            all_dist_5mc.append(dist_5mc)
            meth_sites_5mc['chrom'] = chrom
            all_meth_5mc.append(meth_sites_5mc)
        
        dens_5mc = calculate_methylation_density(chrom_data, mod_type='5mC')
        if len(dens_5mc) > 0:
            dens_5mc['chrom'] = chrom
            all_dens_5mc.append(dens_5mc)
        
        # 5hmC
        dist_5hmc, meth_sites_5hmc = calculate_inter_site_distances(chrom_data, mod_type='5hmC')
        if len(dist_5hmc) > 0:
            all_dist_5hmc.append(dist_5hmc)
            meth_sites_5hmc['chrom'] = chrom
            all_meth_5hmc.append(meth_sites_5hmc)
        
        dens_5hmc = calculate_methylation_density(chrom_data, mod_type='5hmC')
        if len(dens_5hmc) > 0:
            dens_5hmc['chrom'] = chrom
            all_dens_5hmc.append(dens_5hmc)
    
    results = {
        'distances_5mc': pd.concat(all_dist_5mc) if all_dist_5mc else pd.DataFrame(),
        'distances_5hmc': pd.concat(all_dist_5hmc) if all_dist_5hmc else pd.DataFrame(),
        'density_5mc': pd.concat(all_dens_5mc) if all_dens_5mc else pd.DataFrame(),
        'density_5hmc': pd.concat(all_dens_5hmc) if all_dens_5hmc else pd.DataFrame(),
        'methylated_sites_5mc': pd.concat(all_meth_5mc) if all_meth_5mc else pd.DataFrame(),
        'methylated_sites_5hmc': pd.concat(all_meth_5hmc) if all_meth_5hmc else pd.DataFrame()
    }
    
    print(f"Results: 5mC={len(results['distances_5mc']):,} distances, 5hmC={len(results['distances_5hmc']):,} distances")
    
    all_results[sample_name] = results

# -------------------
# Generate Plots
# -------------------
print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS")
print('='*70)

for sample_name, results in all_results.items():
    print(f"Plotting {sample_name}...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add title with thresholds
    fig.suptitle(f'{sample_name} - Methylation Density Analysis\n5mC threshold: {THRESHOLD_5MC:.2f}% | 5hmC threshold: {THRESHOLD_5HMC:.2f}%', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 5mC Plots
    if len(results['distances_5mc']) > 0:
        distances = results['distances_5mc']['distance']
        distances_plot = distances[distances <= MAX_DISTANCE_PLOT]
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(distances_plot, bins=100, color='#ff6b6b', alpha=0.7, edgecolor='black')
        ax1.axvline(distances_plot.median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {distances_plot.median():.0f} bp')
        ax1.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(f'5mC Inter-site Distances (n={len(distances):,})', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = fig.add_subplot(gs[1, 0])
        for chrom in chromosomes[:5]:
            chrom_dens = results['density_5mc'][results['density_5mc']['chrom'] == chrom]
            if len(chrom_dens) > 0:
                ax2.plot(chrom_dens['window_center']/1e6, chrom_dens['density'], 
                        alpha=0.7, linewidth=1.5, label=chrom)
        ax2.set_xlabel('Position (Mb)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Density (sites/Mb)', fontsize=12, fontweight='bold')
        ax2.set_title('5mC Density Along Chromosomes', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[2, 0])
        distances_sorted = np.sort(distances_plot)
        cumulative = np.arange(1, len(distances_sorted) + 1) / len(distances_sorted)
        ax3.plot(distances_sorted, cumulative, linewidth=2, color='#ff6b6b')
        ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax3.set_title('5mC Cumulative Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
    
    # 5hmC Plots
    if len(results['distances_5hmc']) > 0:
        distances = results['distances_5hmc']['distance']
        distances_plot = distances[distances <= MAX_DISTANCE_PLOT]
        
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.hist(distances_plot, bins=100, color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax4.axvline(distances_plot.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {distances_plot.median():.0f} bp')
        ax4.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title(f'5hmC Inter-site Distances (n={len(distances):,})', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        ax5 = fig.add_subplot(gs[1, 1])
        for chrom in chromosomes[:5]:
            chrom_dens = results['density_5hmc'][results['density_5hmc']['chrom'] == chrom]
            if len(chrom_dens) > 0:
                ax5.plot(chrom_dens['window_center']/1e6, chrom_dens['density'], 
                        alpha=0.7, linewidth=1.5, label=chrom)
        ax5.set_xlabel('Position (Mb)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Density (sites/Mb)', fontsize=12, fontweight='bold')
        ax5.set_title('5hmC Density Along Chromosomes', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9, ncol=2)
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        distances_sorted = np.sort(distances_plot)
        cumulative = np.arange(1, len(distances_sorted) + 1) / len(distances_sorted)
        ax6.plot(distances_sorted, cumulative, linewidth=2, color='#4ecdc4')
        ax6.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax6.set_title('5hmC Cumulative Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_methylation_density.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# Comparison plot
print("\nGenerating comparison plot...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

for idx, (sample_name, results) in enumerate(all_results.items()):
    if len(results['distances_5mc']) > 0:
        distances = results['distances_5mc']['distance']
        distances = distances[distances <= MAX_DISTANCE_PLOT]
        ax1.hist(distances, bins=50, alpha=0.5, label=sample_name, color=colors[idx], edgecolor='black')
    
    if len(results['distances_5hmc']) > 0:
        distances = results['distances_5hmc']['distance']
        distances = distances[distances <= MAX_DISTANCE_PLOT]
        ax2.hist(distances, bins=50, alpha=0.5, label=sample_name, color=colors[idx], edgecolor='black')

ax1.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title(f'5mC Distance Comparison (threshold: {THRESHOLD_5MC:.2f}%)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

ax2.set_xlabel('Distance (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title(f'5hmC Distance Comparison (threshold: {THRESHOLD_5HMC:.2f}%)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Median distances
medians_5mc = [results['distances_5mc']['distance'].median() 
               for results in all_results.values() if len(results['distances_5mc']) > 0]
medians_5hmc = [results['distances_5hmc']['distance'].median() 
                for results in all_results.values() if len(results['distances_5hmc']) > 0]
samples = list(all_results.keys())

if medians_5mc:
    ax3.bar(range(len(samples)), medians_5mc, color=colors[:len(samples)], alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(samples)))
    ax3.set_xticklabels(samples, rotation=45, ha='right')
    ax3.set_ylabel('Median Distance (bp)', fontsize=12, fontweight='bold')
    ax3.set_title('5mC Median Inter-site Distance', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

if medians_5hmc:
    ax4.bar(range(len(samples)), medians_5hmc, color=colors[:len(samples)], alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(samples)))
    ax4.set_xticklabels(samples, rotation=45, ha='right')
    ax4.set_ylabel('Median Distance (bp)', fontsize=12, fontweight='bold')
    ax4.set_title('5hmC Median Inter-site Distance', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sample_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Summary
summary_list = []
for sample_name, results in all_results.items():
    if len(results['distances_5mc']) > 0:
        dist_5mc = results['distances_5mc']['distance']
        summary_list.append({
            'Sample': sample_name,
            'Modification': '5mC',
            'Threshold_%': THRESHOLD_5MC,
            'N_methylated_sites': len(results['methylated_sites_5mc']),
            'N_distances': len(dist_5mc),
            'Median_distance_bp': dist_5mc.median(),
            'Mean_distance_bp': dist_5mc.mean()
        })
    
    if len(results['distances_5hmc']) > 0:
        dist_5hmc = results['distances_5hmc']['distance']
        summary_list.append({
            'Sample': sample_name,
            'Modification': '5hmC',
            'Threshold_%': THRESHOLD_5HMC,
            'N_methylated_sites': len(results['methylated_sites_5hmc']),
            'N_distances': len(dist_5hmc),
            'Median_distance_bp': dist_5hmc.median(),
            'Mean_distance_bp': dist_5hmc.mean()
        })

summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(os.path.join(out_dir, 'methylation_distance_summary.csv'), index=False)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\n✓ Results saved to: {out_dir}/")
print(f"  - sample_methylation_statistics.csv")
print(f"  - methylation_distance_summary.csv")
print(f"  - [sample]_methylation_density.png")
print(f"  - sample_comparison.png")
print(f"\nThresholds used:")
print(f"  5mC: {THRESHOLD_5MC:.2f}%")
print(f"  5hmC: {THRESHOLD_5HMC:.2f}%")
