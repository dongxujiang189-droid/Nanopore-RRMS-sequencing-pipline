#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from collections import defaultdict
from scipy import stats
import seaborn as sns

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "feature_methylation")
os.makedirs(out_dir, exist_ok=True)

flank_size = 2000  # ±2kb for TSS/intergenic
tss_window = 2000  # Define TSS as ±2kb
n_gene_bins = 80
n_flank_bins = 80

use_smoothing = True
smooth_window = 5
control_sample_idx = 3  # Change this to select control

# -------------------
# Helper functions
# -------------------
def smooth_profile(data, window_size):
    """Apply moving average smoothing"""
    if window_size < 3 or window_size % 2 == 0:
        return data
    
    data_array = np.array(data)
    smoothed = np.copy(data_array)
    half_window = window_size // 2
    
    for i in range(len(data_array)):
        start = max(0, i - half_window)
        end = min(len(data_array), i + half_window + 1)
        window_data = data_array[start:end]
        smoothed[i] = np.nanmean(window_data)
    
    return smoothed.tolist()

def interval_overlap(start1, end1, start2, end2):
    """Check if two intervals overlap"""
    return start1 < end2 and start2 < end1

# -------------------
# Extract genomic features from GTF
# -------------------
print("Extracting genomic features from GTF...")

genes = []
exons = []
cds_regions = []
utrs = []  # Collect all UTRs first, will classify later

# First pass: collect features and parse transcript_id for UTR classification
transcript_cds = {}  # {transcript_id: {'start': min_cds, 'end': max_cds, 'chrom': ..., 'strand': ...}}
transcript_utrs = {}  # {transcript_id: [utr_regions]}

with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        chrom = fields[0] if fields[0].startswith('chr') else 'chr' + fields[0]
        feature = fields[2]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attributes = fields[8]
        
        # Parse transcript_id
        transcript_id = None
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('transcript_id'):
                transcript_id = attr.split('"')[1]
                break
        
        if feature == 'gene':
            if strand == '+':
                tss, tes = start, end
            else:
                tss, tes = end, start
            
            genes.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand,
                'tss': tss,
                'tes': tes,
                'length': abs(end - start)
            })
        
        elif feature == 'exon':
            exons.append({'chrom': chrom, 'start': start, 'end': end, 'strand': strand})
        
        elif feature == 'CDS':
            cds_regions.append({'chrom': chrom, 'start': start, 'end': end, 'strand': strand})
            
            # Track CDS boundaries per transcript
            if transcript_id:
                if transcript_id not in transcript_cds:
                    transcript_cds[transcript_id] = {
                        'start': start, 'end': end, 'chrom': chrom, 'strand': strand
                    }
                else:
                    transcript_cds[transcript_id]['start'] = min(transcript_cds[transcript_id]['start'], start)
                    transcript_cds[transcript_id]['end'] = max(transcript_cds[transcript_id]['end'], end)
        
        elif feature == 'UTR':
            if transcript_id:
                if transcript_id not in transcript_utrs:
                    transcript_utrs[transcript_id] = []
                transcript_utrs[transcript_id].append({
                    'chrom': chrom, 'start': start, 'end': end, 'strand': strand
                })

# Second pass: classify UTRs as 5' or 3' based on position relative to CDS
utrs_5 = []
utrs_3 = []

for transcript_id, utr_list in transcript_utrs.items():
    if transcript_id not in transcript_cds:
        continue  # No CDS for this transcript
    
    cds_info = transcript_cds[transcript_id]
    cds_start = cds_info['start']
    cds_end = cds_info['end']
    strand = cds_info['strand']
    
    for utr in utr_list:
        utr_center = (utr['start'] + utr['end']) / 2
        
        if strand == '+':
            # Plus strand: 5'UTR before CDS, 3'UTR after CDS
            if utr_center < cds_start:
                utrs_5.append(utr)
            elif utr_center > cds_end:
                utrs_3.append(utr)
        else:
            # Minus strand: 5'UTR after CDS, 3'UTR before CDS
            if utr_center > cds_end:
                utrs_5.append(utr)
            elif utr_center < cds_start:
                utrs_3.append(utr)

genes_df = pd.DataFrame(genes)
genes_df = genes_df[genes_df['length'] > 500]

# Convert to DataFrames for easier querying
exons_df = pd.DataFrame(exons)
utrs_5_df = pd.DataFrame(utrs_5)
utrs_3_df = pd.DataFrame(utrs_3)
cds_df = pd.DataFrame(cds_regions)

# If no UTRs found, compute them from exons - CDS
if len(utrs_5) == 0 and len(utrs_3) == 0 and len(cds_regions) > 0:
    print("Computing UTRs from CDS and exon coordinates...")
    
    for _, gene in genes_df.iterrows():
        chrom, g_start, g_end, strand = gene['chrom'], gene['start'], gene['end'], gene['strand']
        
        # Get gene's CDS regions
        gene_cds = cds_df[(cds_df['chrom'] == chrom) & 
                          (cds_df['start'] < g_end) & 
                          (cds_df['end'] > g_start)]
        
        if len(gene_cds) == 0:
            continue
        
        cds_start = gene_cds['start'].min()
        cds_end = gene_cds['end'].max()
        
        # Get gene's exons
        gene_exons = exons_df[(exons_df['chrom'] == chrom) & 
                              (exons_df['start'] < g_end) & 
                              (exons_df['end'] > g_start)]
        
        if len(gene_exons) == 0:
            continue
        
        # Compute UTRs based on strand
        if strand == '+':
            # 5'UTR: exons before CDS
            for _, exon in gene_exons.iterrows():
                if exon['end'] <= cds_start:
                    utrs_5.append({'chrom': chrom, 'start': exon['start'], 'end': exon['end'], 'strand': strand})
                elif exon['start'] < cds_start < exon['end']:
                    utrs_5.append({'chrom': chrom, 'start': exon['start'], 'end': cds_start, 'strand': strand})
            
            # 3'UTR: exons after CDS
            for _, exon in gene_exons.iterrows():
                if exon['start'] >= cds_end:
                    utrs_3.append({'chrom': chrom, 'start': exon['start'], 'end': exon['end'], 'strand': strand})
                elif exon['start'] < cds_end < exon['end']:
                    utrs_3.append({'chrom': chrom, 'start': cds_end, 'end': exon['end'], 'strand': strand})
        else:
            # Minus strand: reversed
            for _, exon in gene_exons.iterrows():
                if exon['start'] >= cds_end:
                    utrs_5.append({'chrom': chrom, 'start': exon['start'], 'end': exon['end'], 'strand': strand})
                elif exon['start'] < cds_end < exon['end']:
                    utrs_5.append({'chrom': chrom, 'start': cds_end, 'end': exon['end'], 'strand': strand})
            
            for _, exon in gene_exons.iterrows():
                if exon['end'] <= cds_start:
                    utrs_3.append({'chrom': chrom, 'start': exon['start'], 'end': exon['end'], 'strand': strand})
                elif exon['start'] < cds_start < exon['end']:
                    utrs_3.append({'chrom': chrom, 'start': exon['start'], 'end': cds_start, 'strand': strand})
    
    utrs_5_df = pd.DataFrame(utrs_5)
    utrs_3_df = pd.DataFrame(utrs_3)
    print(f"  Computed {len(utrs_5)} 5'UTRs and {len(utrs_3)} 3'UTRs")

print(f"Loaded {len(genes_df)} genes, {len(exons)} exons, {len(utrs_5)} 5'UTRs, {len(utrs_3)} 3'UTRs")

# -------------------
# Build feature index by chromosome
# -------------------
print("Building feature indices...")

def build_chr_index(df):
    """Build chromosome-based index for fast lookup"""
    index = defaultdict(list)
    for _, row in df.iterrows():
        index[row['chrom']].append((row['start'], row['end']))
    return index

exon_index = build_chr_index(exons_df) if len(exons_df) > 0 else {}
utr5_index = build_chr_index(utrs_5_df) if len(utrs_5_df) > 0 else {}
utr3_index = build_chr_index(utrs_3_df) if len(utrs_3_df) > 0 else {}

# Build intron index (gene body minus exons)
def build_intron_index(genes_df, exons_df):
    intron_index = defaultdict(list)
    for _, gene in genes_df.iterrows():
        chrom = gene['chrom']
        g_start, g_end = gene['start'], gene['end']
        
        # Get all exons in this gene
        gene_exons = exons_df[
            (exons_df['chrom'] == chrom) &
            (exons_df['start'] < g_end) &
            (exons_df['end'] > g_start)
        ].sort_values('start')
        
        if len(gene_exons) > 1:
            # Introns are between exons
            for i in range(len(gene_exons) - 1):
                intron_start = gene_exons.iloc[i]['end']
                intron_end = gene_exons.iloc[i + 1]['start']
                if intron_end > intron_start:
                    intron_index[chrom].append((intron_start, intron_end))
    
    return intron_index

intron_index = build_intron_index(genes_df, exons_df)

# Build TSS index (±2kb around TSS)
tss_index = defaultdict(list)
for _, gene in genes_df.iterrows():
    tss = gene['tss']
    tss_index[gene['chrom']].append((tss - tss_window, tss + tss_window))

# -------------------
# Process samples and classify methylation by feature
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

feature_methylation = {}  # {sample: {feature: {'5mC': [], '5hmC': []}}}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing {sample_name}...")
    
    # Load methylation data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    # Initialize storage
    feature_data = {
        'TSS': {'5mC': [], '5hmC': []},
        '5UTR': {'5mC': [], '5hmC': []},
        'Exon': {'5mC': [], '5hmC': []},
        'Intron': {'5mC': [], '5hmC': []},
        '3UTR': {'5mC': [], '5hmC': []},
        'Intergenic': {'5mC': [], '5hmC': []}
    }
    
    # Classify each methylation site
    for _, meth in tqdm(meth_df.iterrows(), total=len(meth_df), desc="Classifying sites"):
        chrom = meth['chrom']
        center = (meth['start'] + meth['end']) / 2
        percent_m = meth['percent_m']
        percent_h = meth['percent_h']
        
        classified = False
        
        # Check TSS (highest priority)
        if chrom in tss_index:
            for tss_start, tss_end in tss_index[chrom]:
                if tss_start <= center <= tss_end:
                    feature_data['TSS']['5mC'].append(percent_m)
                    feature_data['TSS']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if classified:
            continue
        
        # Check 5'UTR
        if chrom in utr5_index:
            for utr_start, utr_end in utr5_index[chrom]:
                if utr_start <= center <= utr_end:
                    feature_data['5UTR']['5mC'].append(percent_m)
                    feature_data['5UTR']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if classified:
            continue
        
        # Check 3'UTR
        if chrom in utr3_index:
            for utr_start, utr_end in utr3_index[chrom]:
                if utr_start <= center <= utr_end:
                    feature_data['3UTR']['5mC'].append(percent_m)
                    feature_data['3UTR']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if classified:
            continue
        
        # Check Exon
        if chrom in exon_index:
            for exon_start, exon_end in exon_index[chrom]:
                if exon_start <= center <= exon_end:
                    feature_data['Exon']['5mC'].append(percent_m)
                    feature_data['Exon']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if classified:
            continue
        
        # Check Intron
        if chrom in intron_index:
            for intron_start, intron_end in intron_index[chrom]:
                if intron_start <= center <= intron_end:
                    feature_data['Intron']['5mC'].append(percent_m)
                    feature_data['Intron']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if classified:
            continue
        
        # Default: Intergenic
        feature_data['Intergenic']['5mC'].append(percent_m)
        feature_data['Intergenic']['5hmC'].append(percent_h)
    
    feature_methylation[sample_name] = feature_data
    
    # Print summary
    print(f"  Sites classified:")
    for feat in feature_data:
        n_sites = len(feature_data[feat]['5mC'])
        if n_sites > 0:
            mean_5mc = np.mean(feature_data[feat]['5mC'])
            mean_5hmc = np.mean(feature_data[feat]['5hmC'])
            print(f"    {feat}: {n_sites:,} sites (5mC={mean_5mc:.1f}%, 5hmC={mean_5hmc:.1f}%)")

# -------------------
# Create comparison plots
# -------------------
print("\nGenerating plots...")

features = ['TSS', '5UTR', 'Exon', 'Intron', '3UTR', 'Intergenic']
samples = list(feature_methylation.keys())
control_name = samples[control_sample_idx]

# 1. Bar plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(features))
width = 0.8 / len(samples)

for i, sample in enumerate(samples):
    means_5mc = [np.mean(feature_methylation[sample][f]['5mC']) if feature_methylation[sample][f]['5mC'] else 0 
                 for f in features]
    means_5hmc = [np.mean(feature_methylation[sample][f]['5hmC']) if feature_methylation[sample][f]['5hmC'] else 0 
                  for f in features]
    
    ax1.bar(x + i * width, means_5mc, width, label=sample, alpha=0.8)
    ax2.bar(x + i * width, means_5hmc, width, label=sample, alpha=0.8)

ax1.set_xlabel('Genomic Feature', fontsize=12, fontweight='bold')
ax1.set_ylabel('5mC Level (%)', fontsize=12, fontweight='bold')
ax1.set_title('5mC Distribution Across Genomic Features', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width * (len(samples) - 1) / 2)
ax1.set_xticklabels(features, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.set_xlabel('Genomic Feature', fontsize=12, fontweight='bold')
ax2.set_ylabel('5hmC Level (%)', fontsize=12, fontweight='bold')
ax2.set_title('5hmC Distribution Across Genomic Features', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width * (len(samples) - 1) / 2)
ax2.set_xticklabels(features, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_methylation_barplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Create matrices
matrix_5mc = np.zeros((len(samples), len(features)))
matrix_5hmc = np.zeros((len(samples), len(features)))

for i, sample in enumerate(samples):
    for j, feature in enumerate(features):
        matrix_5mc[i, j] = np.mean(feature_methylation[sample][feature]['5mC']) if feature_methylation[sample][feature]['5mC'] else 0
        matrix_5hmc[i, j] = np.mean(feature_methylation[sample][feature]['5hmC']) if feature_methylation[sample][feature]['5hmC'] else 0

sns.heatmap(matrix_5mc, annot=True, fmt='.1f', cmap='YlOrRd', 
            xticklabels=features, yticklabels=samples, ax=ax1, cbar_kws={'label': '5mC (%)'})
ax1.set_title('5mC Levels Across Features', fontsize=13, fontweight='bold')

sns.heatmap(matrix_5hmc, annot=True, fmt='.1f', cmap='YlGnBu', 
            xticklabels=features, yticklabels=samples, ax=ax2, cbar_kws={'label': '5hmC (%)'})
ax2.set_title('5hmC Levels Across Features', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_methylation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Log2 fold change vs control
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(features))
width = 0.8 / (len(samples) - 1)

for i, sample in enumerate(samples):
    if sample == control_name:
        continue
    
    log2fc_5mc = []
    log2fc_5hmc = []
    
    for feature in features:
        control_5mc = np.mean(feature_methylation[control_name][feature]['5mC']) if feature_methylation[control_name][feature]['5mC'] else 1
        control_5hmc = np.mean(feature_methylation[control_name][feature]['5hmC']) if feature_methylation[control_name][feature]['5hmC'] else 1
        
        sample_5mc = np.mean(feature_methylation[sample][feature]['5mC']) if feature_methylation[sample][feature]['5mC'] else 1
        sample_5hmc = np.mean(feature_methylation[sample][feature]['5hmC']) if feature_methylation[sample][feature]['5hmC'] else 1
        
        log2fc_5mc.append(np.log2((sample_5mc + 1) / (control_5mc + 1)))
        log2fc_5hmc.append(np.log2((sample_5hmc + 1) / (control_5hmc + 1)))
    
    offset = i if i < list(samples).index(control_name) else i - 1
    ax1.bar(x + offset * width, log2fc_5mc, width, label=sample, alpha=0.8)
    ax2.bar(x + offset * width, log2fc_5hmc, width, label=sample, alpha=0.8)

ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Genomic Feature', fontsize=12, fontweight='bold')
ax1.set_ylabel('Log₂ Fold Change', fontsize=12, fontweight='bold')
ax1.set_title(f'5mC Log₂FC vs {control_name}', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width * (len(samples) - 2) / 2)
ax1.set_xticklabels(features, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Genomic Feature', fontsize=12, fontweight='bold')
ax2.set_ylabel('Log₂ Fold Change', fontsize=12, fontweight='bold')
ax2.set_title(f'5hmC Log₂FC vs {control_name}', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width * (len(samples) - 2) / 2)
ax2.set_xticklabels(features, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_methylation_log2fc.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Box plots for distribution comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]
    
    data_5mc = []
    data_5hmc = []
    labels = []
    
    for sample in samples:
        if len(feature_methylation[sample][feature]['5mC']) > 0:
            # Subsample if too many points
            vals_5mc = feature_methylation[sample][feature]['5mC']
            vals_5hmc = feature_methylation[sample][feature]['5hmC']
            
            if len(vals_5mc) > 10000:
                indices = np.random.choice(len(vals_5mc), 10000, replace=False)
                vals_5mc = [vals_5mc[i] for i in indices]
                vals_5hmc = [vals_5hmc[i] for i in indices]
            
            data_5mc.append(vals_5mc)
            data_5hmc.append(vals_5hmc)
            labels.append(sample)
    
    bp1 = ax.boxplot(data_5mc, positions=np.arange(len(labels)) * 2, widths=0.6,
                      patch_artist=True, showfliers=False, 
                      boxprops=dict(facecolor='salmon', alpha=0.7),
                      medianprops=dict(color='darkred', linewidth=2))
    
    bp2 = ax.boxplot(data_5hmc, positions=np.arange(len(labels)) * 2 + 0.8, widths=0.6,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor='skyblue', alpha=0.7),
                      medianprops=dict(color='darkblue', linewidth=2))
    
    ax.set_title(feature, fontsize=12, fontweight='bold')
    ax.set_ylabel('Methylation Level (%)', fontsize=10)
    ax.set_xticks(np.arange(len(labels)) * 2 + 0.4)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    if idx == 0:
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['5mC', '5hmC'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_methylation_boxplots.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Statistical testing
# -------------------
print("\nPerforming statistical tests...")

stats_results = []
for feature in features:
    for sample in samples:
        if sample == control_name:
            continue
        
        control_5mc = feature_methylation[control_name][feature]['5mC']
        sample_5mc = feature_methylation[sample][feature]['5mC']
        control_5hmc = feature_methylation[control_name][feature]['5hmC']
        sample_5hmc = feature_methylation[sample][feature]['5hmC']
        
        if len(control_5mc) > 0 and len(sample_5mc) > 0:
            # Mann-Whitney U test
            stat_5mc, p_5mc = stats.mannwhitneyu(control_5mc, sample_5mc, alternative='two-sided')
            stat_5hmc, p_5hmc = stats.mannwhitneyu(control_5hmc, sample_5hmc, alternative='two-sided')
            
            mean_ctrl_5mc = np.mean(control_5mc)
            mean_sample_5mc = np.mean(sample_5mc)
            mean_ctrl_5hmc = np.mean(control_5hmc)
            mean_sample_5hmc = np.mean(sample_5hmc)
            
            stats_results.append({
                'Feature': feature,
                'Sample': sample,
                'Control': control_name,
                'Control_5mC_mean': mean_ctrl_5mc,
                'Sample_5mC_mean': mean_sample_5mc,
                '5mC_log2FC': np.log2((mean_sample_5mc + 1) / (mean_ctrl_5mc + 1)),
                '5mC_pvalue': p_5mc,
                'Control_5hmC_mean': mean_ctrl_5hmc,
                'Sample_5hmC_mean': mean_sample_5hmc,
                '5hmC_log2FC': np.log2((mean_sample_5hmc + 1) / (mean_ctrl_5hmc + 1)),
                '5hmC_pvalue': p_5hmc
            })

stats_df = pd.DataFrame(stats_results)
stats_df.to_csv(os.path.join(out_dir, 'feature_methylation_statistics.csv'), index=False)

# Save summary data
summary_data = []
for sample in samples:
    for feature in features:
        n_sites = len(feature_methylation[sample][feature]['5mC'])
        if n_sites > 0:
            summary_data.append({
                'Sample': sample,
                'Feature': feature,
                'N_sites': n_sites,
                '5mC_mean': np.mean(feature_methylation[sample][feature]['5mC']),
                '5mC_median': np.median(feature_methylation[sample][feature]['5mC']),
                '5mC_std': np.std(feature_methylation[sample][feature]['5mC']),
                '5hmC_mean': np.mean(feature_methylation[sample][feature]['5hmC']),
                '5hmC_median': np.median(feature_methylation[sample][feature]['5hmC']),
                '5hmC_std': np.std(feature_methylation[sample][feature]['5hmC'])
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(out_dir, 'feature_methylation_summary.csv'), index=False)

print(f"\n✓ Analysis complete! Files saved in: {out_dir}/")
print(f"  - feature_methylation_barplot.png")
print(f"  - feature_methylation_heatmap.png")
print(f"  - feature_methylation_log2fc.png")
print(f"  - feature_methylation_boxplots.png")
print(f"  - feature_methylation_summary.csv")
print(f"  - feature_methylation_statistics.csv")

# Print significant results
print("\nSignificant differences (p < 0.05):")
sig_results = stats_df[(stats_df['5mC_pvalue'] < 0.05) | (stats_df['5hmC_pvalue'] < 0.05)]
if len(sig_results) > 0:
    for _, row in sig_results.iterrows():
        if row['5mC_pvalue'] < 0.05:
            print(f"  {row['Sample']} vs {row['Control']} in {row['Feature']}: 5mC log2FC={row['5mC_log2FC']:.2f} (p={row['5mC_pvalue']:.2e})")
        if row['5hmC_pvalue'] < 0.05:
            print(f"  {row['Sample']} vs {row['Control']} in {row['Feature']}: 5hmC log2FC={row['5hmC_log2FC']:.2f} (p={row['5hmC_pvalue']:.2e})")
else:
    print("  No significant differences found")
