#!/usr/bin/env python3
"""
Window-Based Methylation Pattern Analysis
Divide genome into 500bp windows and analyze CpG position vs methylation
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

# Configuration
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
out_dir = os.path.join(base_dir, "methylation_window_analysis")
os.makedirs(out_dir, exist_ok=True)

MIN_COVERAGE = 10
WINDOW_SIZE = 500  # bp
MIN_CPGS_PER_WINDOW = 3  # Minimum CpGs required in window
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

print(f"Window size: {WINDOW_SIZE}bp | Min CpGs/window: {MIN_CPGS_PER_WINDOW} | Coverage: ≥{MIN_COVERAGE}x\n")

def extract_window_patterns(meth_df, mod_type='5mC'):
    """Extract CpG positions and methylation within 500bp windows"""
    value_col = 'percent_m' if mod_type == '5mC' else 'percent_h'
    valid_col = 'count_valid_m' if mod_type == '5mC' else 'count_valid_h'
    
    # Filter valid sites
    valid = meth_df[meth_df[valid_col] >= MIN_COVERAGE].copy()
    if len(valid) == 0:
        return []
    
    valid['center'] = (valid['start'] + valid['end']) / 2
    
    windows = []
    for chrom in valid['chrom'].unique():
        chrom_data = valid[valid['chrom'] == chrom].sort_values('center')
        
        if len(chrom_data) < MIN_CPGS_PER_WINDOW:
            continue
        
        # Create 500bp windows
        chrom_start = int(chrom_data['center'].min())
        chrom_end = int(chrom_data['center'].max())
        
        for win_start in range(chrom_start, chrom_end, WINDOW_SIZE):
            win_end = win_start + WINDOW_SIZE
            
            # Get CpGs in this window
            cpgs = chrom_data[(chrom_data['center'] >= win_start) & 
                             (chrom_data['center'] < win_end)]
            
            if len(cpgs) >= MIN_CPGS_PER_WINDOW:
                for _, cpg in cpgs.iterrows():
                    # Position relative to window start (0-500)
                    rel_pos = cpg['center'] - win_start
                    windows.append({
                        'chrom': chrom,
                        'window_start': win_start,
                        'position_in_window': rel_pos,
                        'methylation': cpg[value_col],
                        'abs_position': cpg['center']
                    })
    
    return windows

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
    
    print("  Extracting 5mC patterns...")
    windows_5mc = extract_window_patterns(meth_df, '5mC')
    df_5mc = pd.DataFrame(windows_5mc) if windows_5mc else pd.DataFrame()
    
    print("  Extracting 5hmC patterns...")
    windows_5hmc = extract_window_patterns(meth_df, '5hmC')
    df_5hmc = pd.DataFrame(windows_5hmc) if windows_5hmc else pd.DataFrame()
    
    if len(df_5mc) > 0:
        n_windows = df_5mc.groupby('window_start').size().shape[0]
        print(f"  5mC:  {len(df_5mc):,} CpGs in {n_windows:,} windows")
    if len(df_5hmc) > 0:
        n_windows = df_5hmc.groupby('window_start').size().shape[0]
        print(f"  5hmC: {len(df_5hmc):,} CpGs in {n_windows:,} windows")
    print()
    
    all_data[sample_name] = {'5mC': df_5mc, '5hmC': df_5hmc}
    
    # Individual sample visualization
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(f'{sample_name} - CpG Position vs Methylation in {WINDOW_SIZE}bp Windows', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    # 5mC analysis
    if len(df_5mc) > 0:
        # 2D density: position vs methylation
        ax1 = fig.add_subplot(gs[0, 0])
        hb1 = ax1.hexbin(df_5mc['position_in_window'], df_5mc['methylation'],
                        gridsize=50, cmap='Reds', mincnt=1, norm=LogNorm())
        ax1.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
        ax1.set_title('5mC: All Windows Overlaid', fontsize=13, fontweight='bold')
        ax1.set_xlim(0, WINDOW_SIZE)
        ax1.set_ylim(0, 100)
        plt.colorbar(hb1, ax=ax1, label='CpG Count (log)')
        ax1.grid(True, alpha=0.3)
        
        # Average methylation profile across windows
        ax2 = fig.add_subplot(gs[0, 1])
        bins = np.linspace(0, WINDOW_SIZE, 21)
        df_5mc['pos_bin'] = pd.cut(df_5mc['position_in_window'], bins=bins, labels=bins[:-1])
        profile = df_5mc.groupby('pos_bin')['methylation'].agg(['mean', 'std', 'count']).reset_index()
        profile['pos_bin'] = profile['pos_bin'].astype(float)
        
        ax2.plot(profile['pos_bin'], profile['mean'], 'o-', color='darkred', linewidth=2, markersize=6)
        ax2.fill_between(profile['pos_bin'], 
                         profile['mean'] - profile['std'],
                         profile['mean'] + profile['std'],
                         alpha=0.3, color='red')
        ax2.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Methylation (%)', fontsize=12, fontweight='bold')
        ax2.set_title('5mC: Average Profile ± SD', fontsize=13, fontweight='bold')
        ax2.set_xlim(0, WINDOW_SIZE)
        ax2.grid(True, alpha=0.3)
        
        # Example windows with high CpG density
        ax3 = fig.add_subplot(gs[1, :])
        window_counts = df_5mc.groupby('window_start').size().sort_values(ascending=False)
        top_windows = window_counts.head(10).index
        
        for i, win_start in enumerate(top_windows):
            win_data = df_5mc[df_5mc['window_start'] == win_start]
            chrom = win_data['chrom'].iloc[0]
            ax3.scatter(win_data['position_in_window'], win_data['methylation'],
                       alpha=0.6, s=50, label=f'{chrom}:{win_start}')
        
        ax3.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
        ax3.set_title('5mC: Top 10 CpG-Dense Windows', fontsize=13, fontweight='bold')
        ax3.set_xlim(0, WINDOW_SIZE)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
    
    # 5hmC analysis
    if len(df_5hmc) > 0:
        ax4 = fig.add_subplot(gs[2, 0])
        hb2 = ax4.hexbin(df_5hmc['position_in_window'], df_5hmc['methylation'],
                        gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        ax4.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Methylation (%)', fontsize=12, fontweight='bold')
        ax4.set_title('5hmC: All Windows Overlaid', fontsize=13, fontweight='bold')
        ax4.set_xlim(0, WINDOW_SIZE)
        ax4.set_ylim(0, 100)
        plt.colorbar(hb2, ax=ax4, label='CpG Count (log)')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, 1])
        bins = np.linspace(0, WINDOW_SIZE, 21)
        df_5hmc['pos_bin'] = pd.cut(df_5hmc['position_in_window'], bins=bins, labels=bins[:-1])
        profile = df_5hmc.groupby('pos_bin')['methylation'].agg(['mean', 'std', 'count']).reset_index()
        profile['pos_bin'] = profile['pos_bin'].astype(float)
        
        ax5.plot(profile['pos_bin'], profile['mean'], 'o-', color='darkblue', linewidth=2, markersize=6)
        ax5.fill_between(profile['pos_bin'],
                         profile['mean'] - profile['std'],
                         profile['mean'] + profile['std'],
                         alpha=0.3, color='blue')
        ax5.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Mean Methylation (%)', fontsize=12, fontweight='bold')
        ax5.set_title('5hmC: Average Profile ± SD', fontsize=13, fontweight='bold')
        ax5.set_xlim(0, WINDOW_SIZE)
        ax5.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, f'{sample_name}_window_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Cross-sample comparison
print("Generating cross-sample comparisons...")

for mod_type in ['5mC', '5hmC']:
    n_samples = len(all_data)
    fig = plt.figure(figsize=(9*min(n_samples, 3), 7*((n_samples+2)//3)))
    gs = GridSpec((n_samples+2)//3, min(n_samples, 3), figure=fig, hspace=0.35, wspace=0.3)
    
    for idx, (sample_name, data) in enumerate(all_data.items()):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        df = data[mod_type]
        if len(df) > 0:
            hb = ax.hexbin(df['position_in_window'], df['methylation'],
                          gridsize=40, cmap='Reds' if mod_type=='5mC' else 'Blues',
                          mincnt=1, norm=LogNorm(), vmin=1, vmax=1000)
            ax.set_title(f'{sample_name}\nn={len(df):,}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Position (bp)', fontweight='bold')
            ax.set_ylabel('Methylation (%)', fontweight='bold')
            ax.set_xlim(0, WINDOW_SIZE)
            ax.set_ylim(0, 100)
            plt.colorbar(hb, ax=ax, label='Count')
    
    fig.suptitle(f'{mod_type}: Position vs Methylation in {WINDOW_SIZE}bp Windows', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(out_dir, f'comparison_{mod_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Average profiles comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(all_data)))

for idx, (sample_name, data) in enumerate(all_data.items()):
    for ax, mod_type in [(ax1, '5mC'), (ax2, '5hmC')]:
        df = data[mod_type]
        if len(df) > 0:
            bins = np.linspace(0, WINDOW_SIZE, 21)
            df['pos_bin'] = pd.cut(df['position_in_window'], bins=bins, labels=bins[:-1])
            profile = df.groupby('pos_bin')['methylation'].mean().reset_index()
            profile['pos_bin'] = profile['pos_bin'].astype(float)
            
            ax.plot(profile['pos_bin'], profile['methylation'], 
                   'o-', label=sample_name, color=colors[idx], linewidth=2, markersize=5)

ax1.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Methylation (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC: Average Profiles', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, WINDOW_SIZE)

ax2.set_xlabel(f'Position in {WINDOW_SIZE}bp Window (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Methylation (%)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC: Average Profiles', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, WINDOW_SIZE)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'comparison_profiles.png'), dpi=300, bbox_inches='tight')
plt.close()

# Summary statistics
summary = []
for sample_name, data in all_data.items():
    for mod_type in ['5mC', '5hmC']:
        df = data[mod_type]
        if len(df) > 0:
            n_windows = df.groupby('window_start').size().shape[0]
            avg_cpgs_per_window = df.groupby('window_start').size().mean()
            
            summary.append({
                'Sample': sample_name,
                'Modification': mod_type,
                'Total_CpGs': len(df),
                'N_Windows': n_windows,
                'Avg_CpGs_per_Window': avg_cpgs_per_window,
                'Mean_Methylation_%': df['methylation'].mean(),
                'Std_Methylation_%': df['methylation'].std()
            })

pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'window_summary.csv'), index=False)

print(f"\n{'='*70}")
print("COMPLETE")
print(f"{'='*70}")
print(f"\nOutput: {out_dir}/")
print("  • [sample]_window_patterns.png - Per-sample analysis")
print("  • comparison_5mC.png, comparison_5hmC.png - Side-by-side")
print("  • comparison_profiles.png - Average profiles overlaid")
print("  • window_summary.csv - Statistics\n")
