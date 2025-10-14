#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import glob

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "tss_methylation")
os.makedirs(out_dir, exist_ok=True)

# Parameters
window_size = 2000  # ±2kb around TSS
bin_size = 100
bins = range(-window_size, window_size + bin_size, bin_size)
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# -------------------
# Extract TSS from GTF
# -------------------
print("Extracting TSS from GTF...")
tss_list = []
with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        chrom = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        tss = start if strand == '+' else end
        
        tss_list.append({
            'chrom': chrom,
            'tss': tss,
            'strand': strand,
            'start': tss - window_size,
            'end': tss + window_size
        })

tss_df = pd.DataFrame(tss_list)
print(f"Found {len(tss_df)} genes")

# -------------------
# Process all samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found matching: {input_pattern}")

print(f"\nFound {len(sample_files)} samples")

all_profiles = {}
all_averages = []

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing {sample_name}...")
    
    # Read data
    meth_df = pd.read_csv(sample_file, sep='\t', comment='#')
    
    # Filter data within ±2kb of any TSS
    tss_regions = []
    for _, tss_row in tqdm(tss_df.iterrows(), total=len(tss_df), desc="Filtering TSS regions"):
        region = meth_df[
            (meth_df['chrom'] == tss_row['chrom']) &
            (meth_df['start'] >= tss_row['start']) &
            (meth_df['end'] <= tss_row['end'])
        ].copy()
        
        if len(region) > 0:
            # Calculate distance from TSS
            pos = (region['start'] + region['end']) / 2
            if tss_row['strand'] == '+':
                region['dist_from_tss'] = pos - tss_row['tss']
            else:
                region['dist_from_tss'] = tss_row['tss'] - pos
            
            tss_regions.append(region)
    
    tss_data = pd.concat(tss_regions, ignore_index=True) if tss_regions else pd.DataFrame()
    
    # Calculate overall averages across all TSS regions
    avg_5mc = tss_data['percent_m'].mean() if len(tss_data) > 0 else np.nan
    avg_5hmc = tss_data['percent_h'].mean() if len(tss_data) > 0 else np.nan
    
    all_averages.append({
        'Sample': sample_name,
        '5mC': avg_5mc,
        '5hmC': avg_5hmc
    })
    
    # Calculate binned profiles
    meth_profile = defaultdict(list)
    hmeth_profile = defaultdict(list)
    
    for _, row in tss_data.iterrows():
        dist = row['dist_from_tss']
        bin_idx = int((dist + window_size) / bin_size)
        
        if 0 <= bin_idx < len(bin_centers):
            if pd.notna(row['percent_m']):
                meth_profile[bin_centers[bin_idx]].append(row['percent_m'])
            if pd.notna(row['percent_h']):
                hmeth_profile[bin_centers[bin_idx]].append(row['percent_h'])
    
    meth_mean = [np.mean(meth_profile[bc]) if bc in meth_profile else np.nan 
                 for bc in bin_centers]
    hmeth_mean = [np.mean(hmeth_profile[bc]) if bc in hmeth_profile else np.nan 
                  for bc in bin_centers]
    
    all_profiles[sample_name] = {'5mC': meth_mean, '5hmC': hmeth_mean}

# -------------------
# Save average summary table
# -------------------
avg_df = pd.DataFrame(all_averages)
avg_df.to_csv(os.path.join(out_dir, 'tss_average_methylation.csv'), index=False)
print("\n" + "="*50)
print("Average methylation levels in TSS regions (±2kb):")
print(avg_df.to_string(index=False))
print("="*50)

# -------------------
# Plot combined profile comparison (-2kb to +2kb)
# -------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 5mC profile
for sample_name, profiles in all_profiles.items():
    ax1.plot(bin_centers, profiles['5mC'], linewidth=2.5, label=sample_name, alpha=0.85)
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label='TSS')
ax1.set_xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average 5mC level (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Distribution Around TSS', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-window_size, window_size)

# 5hmC profile
for sample_name, profiles in all_profiles.items():
    ax2.plot(bin_centers, profiles['5hmC'], linewidth=2.5, label=sample_name, alpha=0.85)
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label='TSS')
ax2.set_xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average 5hmC level (%)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Distribution Around TSS', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-window_size, window_size)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'tss_methylation_profiles.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Plot 5mC profile comparison
# -------------------
plt.figure(figsize=(10, 6))
for sample_name, profiles in all_profiles.items():
    plt.plot(bin_centers, profiles['5mC'], linewidth=2, label=sample_name, alpha=0.8)

plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
plt.xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
plt.ylabel('5mC level (%)', fontsize=12, fontweight='bold')
plt.title('5mC Distribution Around TSS', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '5mC_tss_profile.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Plot 5hmC profile comparison
# -------------------
plt.figure(figsize=(10, 6))
for sample_name, profiles in all_profiles.items():
    plt.plot(bin_centers, profiles['5hmC'], linewidth=2, label=sample_name, alpha=0.8)

plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
plt.xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
plt.ylabel('5hmC level (%)', fontsize=12, fontweight='bold')
plt.title('5hmC Distribution Around TSS', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '5hmC_tss_profile.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Save profile data
# -------------------
output_data = {'distance_from_tss': bin_centers}
for sample_name, profiles in all_profiles.items():
    output_data[f'{sample_name}_5mC'] = profiles['5mC']
    output_data[f'{sample_name}_5hmC'] = profiles['5hmC']

profile_df = pd.DataFrame(output_data)
profile_df.to_csv(os.path.join(out_dir, 'tss_profiles_all_samples.csv'), index=False)

print(f"\n✓ Results saved to {out_dir}/")
print(f"  - tss_average_methylation.csv (summary table)")
print(f"  - tss_average_barplot.png (average comparison)")
print(f"  - 5mC_tss_profile.png (profile comparison)")
print(f"  - 5hmC_tss_profile.png (profile comparison)")
print(f"  - tss_profiles_all_samples.csv (detailed data)")
