#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import glob

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
bin_size = 100
bins = range(-window_size, window_size + bin_size, bin_size)
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# -------------------
# Helper function to normalize chromosome names
# -------------------
def normalize_chrom(chrom):
    """Remove 'chr' prefix if present"""
    if isinstance(chrom, str) and chrom.startswith('chr'):
        return chrom[3:]
    return str(chrom)

def add_chr_prefix(chrom):
    """Add 'chr' prefix if not present"""
    chrom_str = str(chrom)
    if not chrom_str.startswith('chr'):
        return 'chr' + chrom_str
    return chrom_str

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
print(f"GTF chromosome examples: {tss_df['chrom'].unique()[:5]}")

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
    print(f"\n{'='*60}")
    print(f"Processing {sample_name}...")
    
    # Read data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    print(f"Loaded {len(meth_df)} methylation sites")
    print(f"TSV chromosome examples: {meth_df['chrom'].unique()[:5]}")
    
    # Check chromosome format and standardize
    sample_has_chr = str(meth_df['chrom'].iloc[0]).startswith('chr') if len(meth_df) > 0 else False
    gtf_has_chr = str(tss_df['chrom'].iloc[0]).startswith('chr')
    
    print(f"TSV has 'chr' prefix: {sample_has_chr}")
    print(f"GTF has 'chr' prefix: {gtf_has_chr}")
    
    if sample_has_chr and not gtf_has_chr:
        print("⚠ Adding 'chr' prefix to GTF chromosomes...")
        tss_df['chrom'] = tss_df['chrom'].apply(add_chr_prefix)
        print(f"After conversion: {tss_df['chrom'].unique()[:5]}")
    elif not sample_has_chr and gtf_has_chr:
        print("⚠ Adding 'chr' prefix to TSV chromosomes...")
        meth_df['chrom'] = meth_df['chrom'].apply(add_chr_prefix)
        print(f"After conversion: {meth_df['chrom'].unique()[:5]}")
    
    # Filter data within ±2kb of any TSS - optimized approach
    print("Filtering TSS regions...")
    tss_regions = []
    matched_genes = 0
    
    # Group methylation data by chromosome for faster lookup
    meth_by_chrom = {chrom: group for chrom, group in meth_df.groupby('chrom')}
    
    for _, tss_row in tqdm(tss_df.iterrows(), total=len(tss_df), desc="Processing TSS"):
        chrom = tss_row['chrom']
        
        # Skip if chromosome not in methylation data
        if chrom not in meth_by_chrom:
            continue
        
        # Filter region
        chrom_data = meth_by_chrom[chrom]
        region = chrom_data[
            (chrom_data['start'] >= tss_row['start']) &
            (chrom_data['end'] <= tss_row['end'])
        ].copy()
        
        if len(region) > 0:
            matched_genes += 1
            # Calculate distance from TSS
            pos = (region['start'] + region['end']) / 2
            if tss_row['strand'] == '+':
                region['dist_from_tss'] = pos - tss_row['tss']
            else:
                region['dist_from_tss'] = tss_row['tss'] - pos
            
            tss_regions.append(region)
    
    print(f"Found methylation data for {matched_genes} genes")
    
    if len(tss_regions) == 0:
        print("⚠ WARNING: No TSS regions found! Check chromosome naming.")
        continue
    
    tss_data = pd.concat(tss_regions, ignore_index=True)
    print(f"Total methylation sites in TSS regions: {len(tss_data)}")
    
    # Calculate overall averages
    avg_5mc = tss_data['percent_m'].mean()
    avg_5hmc = tss_data['percent_h'].mean()
    
    print(f"Average 5mC: {avg_5mc:.2f}%")
    print(f"Average 5hmC: {avg_5hmc:.2f}%")
    
    all_averages.append({
        'Sample': sample_name,
        '5mC': avg_5mc,
        '5hmC': avg_5hmc,
        'N_sites': len(tss_data),
        'N_genes': matched_genes
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
    
    meth_mean = [np.mean(meth_profile[bc]) if bc in meth_profile and len(meth_profile[bc]) > 0 
                 else np.nan for bc in bin_centers]
    hmeth_mean = [np.mean(hmeth_profile[bc]) if bc in hmeth_profile and len(hmeth_profile[bc]) > 0 
                  else np.nan for bc in bin_centers]
    
    all_profiles[sample_name] = {'5mC': meth_mean, '5hmC': hmeth_mean}

# -------------------
# Save summary
# -------------------
if not all_averages:
    print("\n❌ ERROR: No data was processed. Check file paths and chromosome naming.")
    exit(1)

avg_df = pd.DataFrame(all_averages)
avg_df.to_csv(os.path.join(out_dir, 'tss_average_methylation.csv'), index=False)
print("\n" + "="*60)
print("Summary of TSS methylation (±2kb):")
print(avg_df.to_string(index=False))
print("="*60)

# -------------------
# Plot combined profiles
# -------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for sample_name, profiles in all_profiles.items():
    ax1.plot(bin_centers, profiles['5mC'], linewidth=2.5, label=sample_name, alpha=0.85, marker='o', markersize=3)
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
ax1.set_xlabel('Distance from TSS (bp)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average 5mC level (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Distribution Around TSS', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-window_size, window_size)

for sample_name, profiles in all_profiles.items():
    ax2.plot(bin_centers, profiles['5hmC'], linewidth=2.5, label=sample_name, alpha=0.85, marker='o', markersize=3)
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
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
# Save profile data
# -------------------
output_data = {'distance_from_tss': bin_centers}
for sample_name, profiles in all_profiles.items():
    output_data[f'{sample_name}_5mC'] = profiles['5mC']
    output_data[f'{sample_name}_5hmC'] = profiles['5hmC']

profile_df = pd.DataFrame(output_data)
profile_df.to_csv(os.path.join(out_dir, 'tss_profiles_all_samples.csv'), index=False)

print(f"\n✓ Results saved to {out_dir}/")
print(f"  - tss_average_methylation.csv")
print(f"  - tss_methylation_profiles.png")
print(f"  - tss_profiles_all_samples.csv")
