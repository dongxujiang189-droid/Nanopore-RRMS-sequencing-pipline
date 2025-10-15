#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from collections import defaultdict

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "tss_methylation")
os.makedirs(out_dir, exist_ok=True)

# Parameters
window_size = 2000  # ±2kb around TSS
bin_size = 150     # 100bp bins

# -------------------
# Extract TSS coordinates
# -------------------
print("Extracting TSS from GTF...")
tss_data = []
with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        chrom = fields[0]
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        tss = start if strand == '+' else end
        
        tss_data.append({
            'chrom': chrom,
            'tss': tss,
            'strand': strand,
            'tss_start': tss - window_size,
            'tss_end': tss + window_size
        })

tss_df = pd.DataFrame(tss_data)
print(f"Loaded {len(tss_df)} genes")
print(f"Chromosomes: {sorted(tss_df['chrom'].unique())}")

# -------------------
# Process samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

print(f"\nProcessing {len(sample_files)} samples")

all_profiles = {}
all_stats = []

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\n{'='*60}")
    print(f"Sample: {sample_name}")
    
    # Load methylation data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    # Ensure chr prefix
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    print(f"Loaded {len(meth_df):,} methylation regions")
    
    # Find overlaps with TSS regions
    print("Finding TSS overlaps...")
    overlaps = []
    
    for chrom in tqdm(tss_df['chrom'].unique(), desc="Chromosomes"):
        # Get data for this chromosome
        tss_chrom = tss_df[tss_df['chrom'] == chrom]
        meth_chrom = meth_df[meth_df['chrom'] == chrom]
        
        if len(meth_chrom) == 0:
            continue
        
        # For each TSS region, find overlapping methylation sites
        for _, tss_row in tss_chrom.iterrows():
            # Find overlaps: meth_end > tss_start AND meth_start < tss_end
            overlap_mask = (
                (meth_chrom['end'] > tss_row['tss_start']) &
                (meth_chrom['start'] < tss_row['tss_end'])
            )
            
            matched = meth_chrom[overlap_mask].copy()
            if len(matched) == 0:
                continue
            
            # Calculate methylation site center position
            matched['center'] = (matched['start'] + matched['end']) / 2
            
            # Calculate distance from TSS (strand-aware)
            if tss_row['strand'] == '+':
                matched['dist_from_tss'] = matched['center'] - tss_row['tss']
            else:
                matched['dist_from_tss'] = tss_row['tss'] - matched['center']
            
            overlaps.append(matched)
    
    if len(overlaps) == 0:
        print("⚠️  No overlaps found!")
        continue
    
    # Combine all overlaps
    tss_meth = pd.concat(overlaps, ignore_index=True)
    print(f"Found {len(tss_meth):,} methylation sites near TSS")
    
    # Calculate genome-wide averages
    avg_5mc = tss_meth['percent_m'].mean()
    avg_5hmc = tss_meth['percent_h'].mean()
    print(f"Average 5mC: {avg_5mc:.2f}%")
    print(f"Average 5hmC: {avg_5hmc:.2f}%")
    
    all_stats.append({
        'Sample': sample_name,
        'N_sites': len(tss_meth),
        'Avg_5mC': avg_5mc,
        'Avg_5hmC': avg_5hmc
    })
    
    # Bin by distance from TSS
    print("Binning by distance...")
    bins = np.arange(-window_size, window_size + bin_size, bin_size)
    tss_meth['bin'] = pd.cut(tss_meth['dist_from_tss'], bins=bins, labels=False)
    
    # Calculate average methylation per bin
    bin_centers = bins[:-1] + bin_size / 2
    binned_5mc = tss_meth.groupby('bin')['percent_m'].mean()
    binned_5hmc = tss_meth.groupby('bin')['percent_h'].mean()
    
    # Fill missing bins with NaN
    profile_5mc = [binned_5mc.get(i, np.nan) for i in range(len(bin_centers))]
    profile_5hmc = [binned_5hmc.get(i, np.nan) for i in range(len(bin_centers))]
    
    all_profiles[sample_name] = {
        '5mC': profile_5mc,
        '5hmC': profile_5hmc,
        'bin_centers': bin_centers
    }

# -------------------
# Save summary statistics
# -------------------
if not all_stats:
    print("\n❌ No data processed!")
    exit(1)

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(os.path.join(out_dir, 'tss_methylation_summary.csv'), index=False)

print("\n" + "="*60)
print("TSS Methylation Summary (±2kb):")
print(stats_df.to_string(index=False))
print("="*60)

# -------------------
# Plot profiles
# -------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bin_centers = list(all_profiles.values())[0]['bin_centers']

# 5mC
for sample, data in all_profiles.items():
    ax1.plot(bin_centers, data['5mC'], linewidth=2.5, label=sample, alpha=0.85)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='TSS')
ax1.set_xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('5mC (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Distribution Around TSS', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-window_size, window_size)

# 5hmC
for sample, data in all_profiles.items():
    ax2.plot(bin_centers, data['5hmC'], linewidth=2.5, label=sample, alpha=0.85)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='TSS')
ax2.set_xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('5hmC (%)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Distribution Around TSS', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-window_size, window_size)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'tss_methylation_profiles.png'), dpi=300)
plt.close()

# -------------------
# Save detailed profile data
# -------------------
profile_data = {'distance_from_tss': bin_centers}
for sample, data in all_profiles.items():
    profile_data[f'{sample}_5mC'] = data['5mC']
    profile_data[f'{sample}_5hmC'] = data['5hmC']

profile_df = pd.DataFrame(profile_data)
profile_df.to_csv(os.path.join(out_dir, 'tss_profiles_detailed.csv'), index=False)

print(f"\n✓ Results saved:")
print(f"  {out_dir}/tss_methylation_summary.csv")
print(f"  {out_dir}/tss_methylation_profiles.png")
print(f"  {out_dir}/tss_profiles_detailed.csv")
