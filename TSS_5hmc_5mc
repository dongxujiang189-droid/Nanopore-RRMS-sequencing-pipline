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
input_pattern = os.path.join(base_dir, "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "tss_methylation")
os.makedirs(out_dir, exist_ok=True)

# Window parameters
window_size = 2000  # ±2kb around TSS
bin_size = 100      # 100bp bins
bins = range(-window_size, window_size + bin_size, bin_size)
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# -------------------
# Extract TSS from GTF (once)
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
        
        tss_list.append({'chrom': chrom, 'tss': tss, 'strand': strand})

tss_df = pd.DataFrame(tss_list)
print(f"Found {len(tss_df)} genes")

# -------------------
# Process all samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found matching: {input_pattern}")

print(f"\nFound {len(sample_files)} samples")

all_profiles = {}  # {sample_name: {'5mC': [...], '5hmC': [...]}}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing {sample_name}...")
    
    # Read methylation data
    meth_df = pd.read_csv(sample_file, sep='\t', comment='#')
    
    # Initialize profiles
    meth_profile = defaultdict(list)
    hmeth_profile = defaultdict(list)
    
    # Process each gene's TSS region
    for idx, tss_row in tqdm(tss_df.iterrows(), total=len(tss_df), desc=f"{sample_name}"):
        chrom = tss_row['chrom']
        tss = tss_row['tss']
        strand = tss_row['strand']
        
        # Filter data for this region
        region_data = meth_df[
            (meth_df['chrom'] == chrom) &
            (meth_df['start'] >= tss - window_size) &
            (meth_df['end'] <= tss + window_size)
        ]
        
        if len(region_data) == 0:
            continue
        
        # Bin each methylation site
        for _, meth_row in region_data.iterrows():
            pos = (meth_row['start'] + meth_row['end']) / 2
            dist = (pos - tss) if strand == '+' else (tss - pos)
            
            bin_idx = int((dist + window_size) / bin_size)
            if 0 <= bin_idx < len(bin_centers):
                if pd.notna(meth_row['percent_m']):
                    meth_profile[bin_centers[bin_idx]].append(meth_row['percent_m'])
                if pd.notna(meth_row['percent_h']):
                    hmeth_profile[bin_centers[bin_idx]].append(meth_row['percent_h'])
    
    # Calculate means
    meth_mean = [np.mean(meth_profile[bc]) if bc in meth_profile and len(meth_profile[bc]) > 0 
                 else np.nan for bc in bin_centers]
    hmeth_mean = [np.mean(hmeth_profile[bc]) if bc in hmeth_profile and len(hmeth_profile[bc]) > 0 
                  else np.nan for bc in bin_centers]
    
    all_profiles[sample_name] = {'5mC': meth_mean, '5hmC': hmeth_mean}

# -------------------
# Plot 5mC comparison
# -------------------
plt.figure(figsize=(10, 6))
for sample_name, profiles in all_profiles.items():
    plt.plot(bin_centers, profiles['5mC'], linewidth=2, label=sample_name, alpha=0.8)

plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
plt.xlabel('Distance from TSS (bp)', fontsize=12)
plt.ylabel('5mC level (%)', fontsize=12)
plt.title('5mC Distribution Around TSS', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '5mC_tss_comparison.png'), dpi=300)
plt.close()

# -------------------
# Plot 5hmC comparison
# -------------------
plt.figure(figsize=(10, 6))
for sample_name, profiles in all_profiles.items():
    plt.plot(bin_centers, profiles['5hmC'], linewidth=2, label=sample_name, alpha=0.8)

plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
plt.xlabel('Distance from TSS (bp)', fontsize=12)
plt.ylabel('5hmC level (%)', fontsize=12)
plt.title('5hmC Distribution Around TSS', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '5hmC_tss_comparison.png'), dpi=300)
plt.close()

# -------------------
# Save data
# -------------------
output_data = {'distance_from_tss': bin_centers}
for sample_name, profiles in all_profiles.items():
    output_data[f'{sample_name}_5mC'] = profiles['5mC']
    output_data[f'{sample_name}_5hmC'] = profiles['5hmC']

profile_df = pd.DataFrame(output_data)
profile_df.to_csv(os.path.join(out_dir, 'tss_profiles_all_samples.csv'), index=False)

print(f"\n✓ Results saved to {out_dir}/")
print(f"  - 5mC_tss_comparison.png")
print(f"  - 5hmC_tss_comparison.png")
print(f"  - tss_profiles_all_samples.csv")
