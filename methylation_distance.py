#!/usr/bin/env python3
"""
Methylation Analysis - Distance vs Methylation
Professional density visualization with 2D histograms and contours
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
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
            pairs.append({'distance': dist, 'methylation': meth1})
            pairs.append({'distance': dist, 'methylation': meth2})
    
    return pairs

def plot_density_2d(ax, x, y, title, xlabel, ylabel, cmap='Reds', bins=50):
    """Create professional 2D histogram with contours"""
    # 2D histogram
    h = ax.hist2d(x, y, bins=bins, cmap=cmap, norm=LogNorm(), cmin=1)
    
    # Add contour lines for density
    if len(x) > 100:
        try:
            # Create density estimation for contours
            xedges = np.linspace(x.min(), x.max(), 30)
            yedges = np.linspace(0, 100, 30)
            H, xedges, yedges = np.histogram2d(x, y, bins=[xedges, yedges])
            H = H.T
            
            # Smooth contours
            X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, 
                              (yedges[:-1] + yedges[1:]) / 2)
            ax.contour(X, Y, H, levels=5, colors='black', alpha=0.3, linewidths=0.5)
        except:
            pass
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    return h[3]  # Return colorbar mappable

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
        print(f"        Distance: {df_5mc['distance'].min():.0f}-{df_5mc['distance'].max():.0f}bp (median: {df_5mc['distance'].median():.0f})")
        print(f"        Methylation: {df_5mc['methylation'].min():.1f}-{df_5mc['methylation'].max():.1f}% (mean: {df_5mc['methylation'].mean():.1f}%)")
    
    print(f"  5hmC: {len(df_5hmc):,} data points")
    if len(df_5hmc) > 0:
        print(f"        Distance: {df_5hmc['distance'].min():.0f}-{df_5hmc['distance'].max():.0f}bp (median: {df_5hmc['distance'].median():.0f})")
        print(f"        Methylation: {df_5hmc['methylation'].min():.1f}-{df_5hmc['methylation'].max():.1f}% (mean: {df_5hmc['methylation'].mean():.1f}%)")
    print()
    
    all_sample_data[sample_name] = {'5mC': df_5mc, '5hmC': df_5hmc}
    
    # -------------------
    # Individual Sample Plots
    # -------------------
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, 
                  height_ratios=[1, 1, 0.3], width_ratios=[1, 1, 0.05])
    
    # 5mC plots
    if len(df_5mc) > 0:
        # Full range
        ax1 = fig.add_subplot(gs[0, 0])
        h1 = plot_density_2d(ax1, df_5mc['distance'], df_5mc['methylation'],
                            f'5mC: Full Range\n{sample_name}',
                            'Distance to Adjacent CpG (bp)', 'Methylation (%)',
                            cmap='Reds', bins=60)
        ax1.set_ylim(0, 100)
        plt.colorbar(h1, ax=ax1, label='Count (log scale)')
        
        # Zoomed range
        df_zoom = df_5mc[df_5mc['distance'] <= MAX_DISTANCE]
        if len(df_zoom) > 0:
            ax2 = fig.add_subplot(gs[0, 1])
            h2 = plot_density_2d(ax2, df_zoom['distance'], df_zoom['methylation'],
                                f'5mC: ≤{MAX_DISTANCE}bp\n{sample_name}',
                                'Distance to Adjacent CpG (bp)', 'Methylation (%)',
                                cmap='Reds', bins=50)
            ax2.set_xlim(0, MAX_DISTANCE)
            ax2.set_ylim(0, 100)
            plt.colorbar(h2, ax=ax2, label='Count (log scale)')
            
            stats = f"N = {len(df_zoom):,}\nMean: {df_zoom['methylation'].mean():.1f}%\nMedian dist: {df_zoom['distance'].median():.0f}bp"
            ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), 
                    fontsize=10, family='monospace')
    
    # 5hmC plots
    if len(df_5hmc) > 0:
        ax3 = fig.add_subplot(gs[1, 0])
        h3 = plot_density_2d(ax3, df_5hmc['distance'], df_5hmc['methylation'],
                            f'5hmC: Full Range\n{sample_name}',
                            'Distance to Adjacent CpG (bp)', 'Methylation (%)',
                            cmap='Blues', bins=60)
        ax3.set_ylim(0, 100)
        plt.colorbar(h3, ax=ax3, label='Count (log scale)')
        
        df_zoom = df_5hmc[df_5hmc['distance'] <= MAX_DISTANCE]
        if len(df_zoom) > 0:
            ax4 = fig.add_subplot(gs[1, 1])
            h4 = plot_density_2d(ax4, df_zoom['distance'], df_zoom['methylation'],
                                f'5hmC: ≤{MAX_DISTANCE}bp\n{sample_name}',
                                'Distance to Adjacent CpG (bp)', 'Methylation (%)',
                                cmap='Blues', bins=50)
            ax4.set_xlim(0, MAX_DISTANCE)
            ax4.set_ylim(0, 100)
            plt.colorbar(h4, ax=ax4, label='Count (log scale)')
            
            stats = f"N = {len(df_zoom):,}\nMean: {df_zoom['methylation'].mean():.1f}%\nMedian dist: {df_zoom['distance'].median():.0f}bp"
            ax4.text(0.02, 0.98, stats, transform=ax4.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), 
                    fontsize=10, family='monospace')
    
    # Marginal distributions
    if len(df_5mc) > 0:
        df_zoom = df_5mc[df_5mc['distance'] <= MAX_DISTANCE]
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(df_zoom['distance'], bins=50, color='crimson', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Distance (bp)', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('5mC Distance Distribution', fontweight='bold', fontsize=11)
        ax5.grid(True, alpha=0.2)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(df_zoom['methylation'], bins=30, color='crimson', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Methylation (%)', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('5mC Methylation Distribution', fontweight='bold', fontsize=11)
        ax6.grid(True, alpha=0.2)
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_density_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Cross-Sample Comparison
# -------------------
print("Generating comparison plots...")

n_samples = len(all_sample_data)

# 5mC comparison
fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.3, wspace=0.3)

for idx, (sample_name, data) in enumerate(all_sample_data.items()):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    df = data['5mC']
    if len(df) > 0:
        df_plot = df[df['distance'] <= MAX_DISTANCE]
        if len(df_plot) > 0:
            h = plot_density_2d(ax, df_plot['distance'], df_plot['methylation'],
                               f'{sample_name}\nMean: {df_plot["methylation"].mean():.1f}%',
                               'Distance (bp)', 'Methylation (%)',
                               cmap='Reds', bins=40)
            ax.set_xlim(0, MAX_DISTANCE)
            ax.set_ylim(0, 100)
            plt.colorbar(h, ax=ax, label='Count')

fig.suptitle(f'5mC: Distance vs Methylation Density (≤{MAX_DISTANCE}bp)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5mC_density.png'), dpi=300, bbox_inches='tight')
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
            h = plot_density_2d(ax, df_plot['distance'], df_plot['methylation'],
                               f'{sample_name}\nMean: {df_plot["methylation"].mean():.1f}%',
                               'Distance (bp)', 'Methylation (%)',
                               cmap='Blues', bins=40)
            ax.set_xlim(0, MAX_DISTANCE)
            ax.set_ylim(0, 100)
            plt.colorbar(h, ax=ax, label='Count')

fig.suptitle(f'5hmC: Distance vs Methylation Density (≤{MAX_DISTANCE}bp)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig(os.path.join(out_dir, 'comparison_5hmC_density.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save summary
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

pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'summary_statistics.csv'), index=False)

print(f"\n{'='*70}")
print("COMPLETE")
print(f"{'='*70}")
print(f"\nOutput: {out_dir}/")
print("  • [sample]_density_analysis.png")
print("  • comparison_5mC_density.png")
print("  • comparison_5hmC_density.png")
print("  • summary_statistics.csv\n")
