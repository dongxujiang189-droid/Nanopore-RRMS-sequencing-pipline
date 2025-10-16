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

tss_window = 100  # TSS = ±100bp
control_sample_idx = 3  # Index of control sample

# -------------------
# Helper functions
# -------------------
def build_chr_index(df):
    """Build chromosome-based index for fast lookup"""
    index = defaultdict(list)
    for _, row in df.iterrows():
        index[row['chrom']].append((row['start'], row['end']))
    return index

# -------------------
# Extract genomic features from GTF
# -------------------
print("Extracting genomic features from GTF...")

genes = []
exons = []
cds_regions = []
utrs = []

# Parse GTF
transcript_cds = {}
transcript_utrs = {}

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
                tss = start
            else:
                tss = end
            
            genes.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand,
                'tss': tss,
                'length': abs(end - start)
            })
        
        elif feature == 'exon':
            exons.append({'chrom': chrom, 'start': start, 'end': end, 'strand': strand})
        
        elif feature == 'CDS':
            cds_regions.append({'chrom': chrom, 'start': start, 'end': end, 'strand': strand})
            
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

# Classify UTRs
utrs_5 = []
utrs_3 = []

for transcript_id, utr_list in transcript_utrs.items():
    if transcript_id not in transcript_cds:
        continue
    
    cds_info = transcript_cds[transcript_id]
    cds_start = cds_info['start']
    cds_end = cds_info['end']
    strand = cds_info['strand']
    
    for utr in utr_list:
        utr_center = (utr['start'] + utr['end']) / 2
        
        if strand == '+':
            if utr_center < cds_start:
                utrs_5.append(utr)
            elif utr_center > cds_end:
                utrs_3.append(utr)
        else:
            if utr_center > cds_end:
                utrs_5.append(utr)
            elif utr_center < cds_start:
                utrs_3.append(utr)

genes_df = pd.DataFrame(genes)
genes_df = genes_df[genes_df['length'] > 500]

exons_df = pd.DataFrame(exons)
utrs_5_df = pd.DataFrame(utrs_5)
utrs_3_df = pd.DataFrame(utrs_3)

print(f"Loaded {len(genes_df)} genes, {len(exons)} exons, {len(utrs_5)} 5'UTRs, {len(utrs_3)} 3'UTRs")

# -------------------
# Build feature indices
# -------------------
print("Building feature indices...")

exon_index = build_chr_index(exons_df) if len(exons_df) > 0 else {}
utr5_index = build_chr_index(utrs_5_df) if len(utrs_5_df) > 0 else {}
utr3_index = build_chr_index(utrs_3_df) if len(utrs_3_df) > 0 else {}

# Build intron index
def build_intron_index(genes_df, exons_df):
    intron_index = defaultdict(list)
    for _, gene in genes_df.iterrows():
        chrom = gene['chrom']
        g_start, g_end = gene['start'], gene['end']
        
        gene_exons = exons_df[
            (exons_df['chrom'] == chrom) &
            (exons_df['start'] < g_end) &
            (exons_df['end'] > g_start)
        ].sort_values('start')
        
        if len(gene_exons) > 1:
            for i in range(len(gene_exons) - 1):
                intron_start = gene_exons.iloc[i]['end']
                intron_end = gene_exons.iloc[i + 1]['start']
                if intron_end > intron_start:
                    intron_index[chrom].append((intron_start, intron_end))
    
    return intron_index

intron_index = build_intron_index(genes_df, exons_df)

# Build TSS index
tss_index = defaultdict(list)
for _, gene in genes_df.iterrows():
    tss = gene['tss']
    tss_index[gene['chrom']].append((tss - tss_window, tss + tss_window))

# -------------------
# Process samples
# -------------------
sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

feature_methylation = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing {sample_name}...")
    
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
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
        
        # Priority: TSS → 5'UTR → 3'UTR → Exon → Intron → Intergenic
        if chrom in tss_index:
            for tss_start, tss_end in tss_index[chrom]:
                if tss_start <= center <= tss_end:
                    feature_data['TSS']['5mC'].append(percent_m)
                    feature_data['TSS']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if not classified and chrom in utr5_index:
            for utr_start, utr_end in utr5_index[chrom]:
                if utr_start <= center <= utr_end:
                    feature_data['5UTR']['5mC'].append(percent_m)
                    feature_data['5UTR']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if not classified and chrom in utr3_index:
            for utr_start, utr_end in utr3_index[chrom]:
                if utr_start <= center <= utr_end:
                    feature_data['3UTR']['5mC'].append(percent_m)
                    feature_data['3UTR']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if not classified and chrom in exon_index:
            for exon_start, exon_end in exon_index[chrom]:
                if exon_start <= center <= exon_end:
                    feature_data['Exon']['5mC'].append(percent_m)
                    feature_data['Exon']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if not classified and chrom in intron_index:
            for intron_start, intron_end in intron_index[chrom]:
                if intron_start <= center <= intron_end:
                    feature_data['Intron']['5mC'].append(percent_m)
                    feature_data['Intron']['5hmC'].append(percent_h)
                    classified = True
                    break
        
        if not classified:
            feature_data['Intergenic']['5mC'].append(percent_m)
            feature_data['Intergenic']['5hmC'].append(percent_h)
    
    feature_methylation[sample_name] = feature_data
    
    print(f"  Sites classified:")
    for feat in feature_data:
        n_sites = len(feature_data[feat]['5mC'])
        if n_sites > 0:
            mean_5mc = np.mean(feature_data[feat]['5mC'])
            mean_5hmc = np.mean(feature_data[feat]['5hmC'])
            print(f"    {feat}: {n_sites:,} sites (5mC={mean_5mc:.1f}%, 5hmC={mean_5hmc:.1f}%)")

# -------------------
# Create bar plots with value labels
# -------------------
print("\nGenerating plots...")

features = ['TSS', '5UTR', 'Exon', 'Intron', '3UTR', 'Intergenic']
samples = list(feature_methylation.keys())
control_name = samples[control_sample_idx]

# Bar plot with value labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

x = np.arange(len(features))
width = 0.8 / len(samples)

for i, sample in enumerate(samples):
    means_5mc = [np.mean(feature_methylation[sample][f]['5mC']) if feature_methylation[sample][f]['5mC'] else 0 
                 for f in features]
    means_5hmc = [np.mean(feature_methylation[sample][f]['5hmC']) if feature_methylation[sample][f]['5hmC'] else 0 
                  for f in features]
    
    bars_5mc = ax1.bar(x + i * width, means_5mc, width, label=sample, alpha=0.8)
    bars_5hmc = ax2.bar(x + i * width, means_5hmc, width, label=sample, alpha=0.8)
    
    # Add value labels on bars
    for j, (bar, val) in enumerate(zip(bars_5mc, means_5mc)):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    for j, (bar, val) in enumerate(zip(bars_5hmc, means_5hmc)):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

ax1.set_xlabel('Genomic Feature', fontsize=13, fontweight='bold')
ax1.set_ylabel('5mC Level (%)', fontsize=13, fontweight='bold')
ax1.set_title('5mC Distribution Across Genomic Features', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * (len(samples) - 1) / 2)
ax1.set_xticklabels(features, rotation=0, ha='center')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max([max([np.mean(feature_methylation[s][f]['5mC']) if feature_methylation[s][f]['5mC'] else 0 
                          for f in features]) for s in samples]) * 1.15)

ax2.set_xlabel('Genomic Feature', fontsize=13, fontweight='bold')
ax2.set_ylabel('5hmC Level (%)', fontsize=13, fontweight='bold')
ax2.set_title('5hmC Distribution Across Genomic Features', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * (len(samples) - 1) / 2)
ax2.set_xticklabels(features, rotation=0, ha='center')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max([max([np.mean(feature_methylation[s][f]['5hmC']) if feature_methylation[s][f]['5hmC'] else 0 
                          for f in features]) for s in samples]) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_methylation_barplot.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Analysis complete!")
print(f"  Plot saved: {out_dir}/feature_methylation_barplot.png")
print(f"\nFeatures analyzed: TSS (±{tss_window}bp), 5'UTR, Exon, Intron, 3'UTR, Intergenic")
