#!/usr/bin/env python3
"""
Gene-level Methylation Analysis and Heatmap Generation
Calculates average methylation for each gene across the genome
Creates comprehensive heatmaps and genome-wide annotations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
input_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "gene_level_methylation")
os.makedirs(out_dir, exist_ok=True)

# Parameters
chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
min_gene_length = 500  # Minimum gene length (bp)
top_variable_genes = 1000  # Number of most variable genes for heatmap

# Chromosome lengths (hg38)
CHROM_SIZES = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
    'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
    'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
    'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
    'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
    'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
    'chr22': 50818468, 'chrX': 156040895
}

# Calculate cumulative positions
cumulative_positions = {}
cumulative_pos = 0
for chrom in chromosomes:
    if chrom in CHROM_SIZES:
        cumulative_positions[chrom] = cumulative_pos
        cumulative_pos += CHROM_SIZES[chrom]

# -------------------
# Extract gene information from GTF
# -------------------
print("="*60)
print("EXTRACTING GENE ANNOTATIONS")
print("="*60)

genes = []
with open(gtf_file, 'r') as f:
    for line in tqdm(f, desc="Reading GTF"):
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        chrom = fields[0] if fields[0].startswith('chr') else 'chr' + fields[0]
        if chrom not in chromosomes:
            continue
        
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attributes = fields[8]
        
        # Extract gene_id and gene_name
        gene_id = gene_name = None
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_id'):
                gene_id = attr.split('"')[1]
            elif attr.startswith('gene_name'):
                gene_name = attr.split('"')[1]
        
        length = end - start
        if length < min_gene_length:
            continue
        
        genes.append({
            'gene_id': gene_id,
            'gene_name': gene_name or gene_id,
            'chrom': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'length': length,
            'genome_start': cumulative_positions[chrom] + start,
            'genome_end': cumulative_positions[chrom] + end,
            'genome_center': cumulative_positions[chrom] + (start + end) / 2
        })

genes_df = pd.DataFrame(genes)
print(f"Loaded {len(genes_df):,} genes")
print(f"Chromosomes: {genes_df['chrom'].unique()}")

# -------------------
# Load methylation data
# -------------------
print(f"\n{'='*60}")
print("LOADING METHYLATION DATA")
print('='*60)

sample_files = glob.glob(input_pattern)
if not sample_files:
    raise FileNotFoundError(f"No files found: {input_pattern}")

all_sample_data = {}

for sample_file in sample_files:
    sample_name = os.path.basename(sample_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing {sample_name}...")
    
    # Load data
    meth_df = pd.read_csv(sample_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    meth_df = meth_df[meth_df['chrom'].isin(chromosomes)]
    print(f"  Loaded {len(meth_df):,} methylation sites")
    
    all_sample_data[sample_name] = meth_df

# -------------------
# Calculate average methylation per gene
# -------------------
print(f"\n{'='*60}")
print("CALCULATING GENE-LEVEL METHYLATION")
print('='*60)

gene_methylation = {}

for sample_name, meth_df in all_sample_data.items():
    print(f"\nProcessing {sample_name}...")
    
    gene_meth_list = []
    
    # Group methylation data by chromosome for faster lookup
    meth_by_chrom = {chrom: group for chrom, group in meth_df.groupby('chrom')}
    
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="Genes"):
        chrom = gene['chrom']
        
        if chrom not in meth_by_chrom:
            continue
        
        # Find overlapping methylation sites
        chrom_data = meth_by_chrom[chrom]
        overlaps = chrom_data[
            (chrom_data['end'] > gene['start']) &
            (chrom_data['start'] < gene['end'])
        ]
        
        if len(overlaps) > 0:
            avg_5mc = overlaps['percent_m'].mean()
            avg_5hmc = overlaps['percent_h'].mean()
            n_sites = len(overlaps)
        else:
            avg_5mc = np.nan
            avg_5hmc = np.nan
            n_sites = 0
        
        gene_meth_list.append({
            'gene_id': gene['gene_id'],
            'gene_name': gene['gene_name'],
            'chrom': gene['chrom'],
            'start': gene['start'],
            'end': gene['end'],
            'genome_center': gene['genome_center'],
            'avg_5mC': avg_5mc,
            'avg_5hmC': avg_5hmc,
            'n_sites': n_sites
        })
    
    gene_meth_df = pd.DataFrame(gene_meth_list)
    gene_methylation[sample_name] = gene_meth_df
    
    # Print stats
    with_data = gene_meth_df[gene_meth_df['n_sites'] > 0]
    print(f"  Genes with methylation data: {len(with_data):,} / {len(gene_meth_df):,}")
    print(f"  Mean 5mC: {with_data['avg_5mC'].mean():.2f}%")
    print(f"  Mean 5hmC: {with_data['avg_5hmC'].mean():.2f}%")

# Save gene-level data
for sample_name, gene_df in gene_methylation.items():
    gene_df.to_csv(os.path.join(out_dir, f'{sample_name}_gene_methylation.csv'), index=False)

# -------------------
# Create combined matrix for heatmap
# -------------------
print(f"\n{'='*60}")
print("CREATING METHYLATION MATRICES")
print('='*60)

sample_names = list(gene_methylation.keys())

# Merge all samples
merged = genes_df[['gene_id', 'gene_name', 'chrom', 'start', 'end', 'genome_center']].copy()

for sample_name in sample_names:
    sample_data = gene_methylation[sample_name][['gene_id', 'avg_5mC', 'avg_5hmC', 'n_sites']]
    merged = merged.merge(
        sample_data, 
        on='gene_id', 
        how='left',
        suffixes=('', f'_{sample_name}')
    )
    merged.rename(columns={
        'avg_5mC': f'{sample_name}_5mC',
        'avg_5hmC': f'{sample_name}_5hmC',
        'n_sites': f'{sample_name}_n_sites'
    }, inplace=True)

# Save combined data
merged.to_csv(os.path.join(out_dir, 'all_genes_all_samples.csv'), index=False)

# Filter genes with data in all samples
has_data_cols = [f'{s}_n_sites' for s in sample_names]
merged_filtered = merged[merged[has_data_cols].min(axis=1) > 0].copy()

print(f"Total genes: {len(merged):,}")
print(f"Genes with data in all samples: {len(merged_filtered):,}")

# -------------------
# Plot 1: Genome-wide gene methylation annotation
# -------------------
print(f"\n{'='*60}")
print("GENERATING PLOTS")
print('='*60)

print("1. Genome-wide gene methylation profile...")

fig, axes = plt.subplots(len(sample_names), 2, figsize=(24, 4*len(sample_names)), 
                         sharex=True, sharey='col')

if len(sample_names) == 1:
    axes = axes.reshape(1, -1)

for idx, sample_name in enumerate(sample_names):
    gene_df = gene_methylation[sample_name]
    gene_df_filtered = gene_df[gene_df['n_sites'] > 0]
    
    # 5mC
    ax = axes[idx, 0]
    scatter = ax.scatter(
        gene_df_filtered['genome_center'] / 1e9,
        gene_df_filtered['avg_5mC'],
        c=gene_df_filtered['avg_5mC'],
        cmap='RdYlBu_r',
        s=1,
        alpha=0.6,
        vmin=0,
        vmax=100
    )
    
    # Add chromosome boundaries
    for chrom in chromosomes:
        if chrom in cumulative_positions:
            ax.axvline(x=cumulative_positions[chrom] / 1e9, 
                      color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_ylabel(f'{sample_name}\n5mC (%)', fontsize=10, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.2)
    
    if idx == 0:
        ax.set_title('Gene-level 5mC', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='5mC (%)')
    
    # 5hmC
    ax = axes[idx, 1]
    scatter = ax.scatter(
        gene_df_filtered['genome_center'] / 1e9,
        gene_df_filtered['avg_5hmC'],
        c=gene_df_filtered['avg_5hmC'],
        cmap='RdYlBu_r',
        s=1,
        alpha=0.6,
        vmin=0,
        vmax=20
    )
    
    for chrom in chromosomes:
        if chrom in cumulative_positions:
            ax.axvline(x=cumulative_positions[chrom] / 1e9, 
                      color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax.set_ylabel(f'{sample_name}\n5hmC (%)', fontsize=10, fontweight='bold')
    ax.set_ylim([0, 20])
    ax.grid(True, alpha=0.2)
    
    if idx == 0:
        ax.set_title('Gene-level 5hmC', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='5hmC (%)')

# Add chromosome labels
axes[-1, 0].set_xlabel('Genomic Position (Gb)', fontsize=11, fontweight='bold')
axes[-1, 1].set_xlabel('Genomic Position (Gb)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'genome_wide_gene_methylation.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Plot 2: Heatmap of most variable genes
# -------------------
print("2. Heatmap of variable genes...")

# Calculate variance across samples
variance_5mc = merged_filtered[[f'{s}_5mC' for s in sample_names]].var(axis=1)
variance_5hmc = merged_filtered[[f'{s}_5hmC' for s in sample_names]].var(axis=1)

merged_filtered['variance_5mC'] = variance_5mc
merged_filtered['variance_5hmC'] = variance_5hmc

# Select top variable genes
top_var_5mc = merged_filtered.nlargest(min(top_variable_genes, len(merged_filtered)), 'variance_5mC')
top_var_5hmc = merged_filtered.nlargest(min(top_variable_genes, len(merged_filtered)), 'variance_5hmC')

# Create matrices
matrix_5mc = top_var_5mc[[f'{s}_5mC' for s in sample_names]].values
matrix_5hmc = top_var_5hmc[[f'{s}_5hmC' for s in sample_names]].values

gene_labels_5mc = top_var_5mc['gene_name'].values
gene_labels_5hmc = top_var_5hmc['gene_name'].values

# Plot heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

# 5mC heatmap
sns.heatmap(matrix_5mc.T, 
            xticklabels=gene_labels_5mc if len(gene_labels_5mc) < 100 else False,
            yticklabels=sample_names,
            cmap='RdYlBu_r',
            vmin=0, vmax=100,
            cbar_kws={'label': '5mC (%)'},
            ax=ax1)
ax1.set_title(f'5mC - Top {len(top_var_5mc)} Variable Genes', fontsize=13, fontweight='bold')
ax1.set_xlabel('Genes', fontsize=11)
ax1.set_ylabel('Samples', fontsize=11)

# 5hmC heatmap
sns.heatmap(matrix_5hmc.T, 
            xticklabels=gene_labels_5hmc if len(gene_labels_5hmc) < 100 else False,
            yticklabels=sample_names,
            cmap='RdYlBu_r',
            vmin=0, vmax=20,
            cbar_kws={'label': '5hmC (%)'},
            ax=ax2)
ax2.set_title(f'5hmC - Top {len(top_var_5hmc)} Variable Genes', fontsize=13, fontweight='bold')
ax2.set_xlabel('Genes', fontsize=11)
ax2.set_ylabel('Samples', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'variable_genes_heatmap.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Plot 3: Clustered heatmap (top 100 genes)
# -------------------
print("3. Clustered heatmap...")

top_100_5mc = merged_filtered.nlargest(100, 'variance_5mC')
top_100_5hmc = merged_filtered.nlargest(100, 'variance_5hmC')

# Create clustered heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

# 5mC clustered
data_5mc = top_100_5mc[[f'{s}_5mC' for s in sample_names]].T
data_5mc.columns = top_100_5mc['gene_name'].values
data_5mc.index = sample_names

sns.clustermap(data_5mc, 
               cmap='RdYlBu_r',
               vmin=0, vmax=100,
               figsize=(12, 10),
               cbar_kws={'label': '5mC (%)'},
               row_cluster=True,
               col_cluster=True,
               yticklabels=True,
               xticklabels=True if len(data_5mc.columns) < 50 else False)
plt.savefig(os.path.join(out_dir, '5mC_top100_clustered.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# 5hmC clustered
data_5hmc = top_100_5hmc[[f'{s}_5hmC' for s in sample_names]].T
data_5hmc.columns = top_100_5hmc['gene_name'].values
data_5hmc.index = sample_names

sns.clustermap(data_5hmc, 
               cmap='RdYlBu_r',
               vmin=0, vmax=20,
               figsize=(12, 10),
               cbar_kws={'label': '5hmC (%)'},
               row_cluster=True,
               col_cluster=True,
               yticklabels=True,
               xticklabels=True if len(data_5hmc.columns) < 50 else False)
plt.savefig(os.path.join(out_dir, '5hmC_top100_clustered.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# -------------------
# Plot 4: Per-chromosome heatmap
# -------------------
print("4. Per-chromosome gene heatmap...")

for chrom in chromosomes[:5]:  # First 5 chromosomes
    chrom_genes = merged_filtered[merged_filtered['chrom'] == chrom]
    
    if len(chrom_genes) < 10:
        continue
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # 5mC
    data_5mc = chrom_genes[[f'{s}_5mC' for s in sample_names]].T
    data_5mc.columns = chrom_genes['gene_name'].values
    
    sns.heatmap(data_5mc, 
                cmap='RdYlBu_r',
                vmin=0, vmax=100,
                cbar_kws={'label': '5mC (%)'},
                ax=ax1,
                xticklabels=False,
                yticklabels=sample_names)
    ax1.set_title(f'{chrom} - Gene-level 5mC', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Samples', fontsize=10)
    
    # 5hmC
    data_5hmc = chrom_genes[[f'{s}_5hmC' for s in sample_names]].T
    data_5hmc.columns = chrom_genes['gene_name'].values
    
    sns.heatmap(data_5hmc, 
                cmap='RdYlBu_r',
                vmin=0, vmax=20,
                cbar_kws={'label': '5hmC (%)'},
                ax=ax2,
                xticklabels=False,
                yticklabels=sample_names)
    ax2.set_title(f'{chrom} - Gene-level 5hmC', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Samples', fontsize=10)
    ax2.set_xlabel('Genes (ordered by position)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{chrom}_gene_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Save top variable genes
# -------------------
top_var_5mc.to_csv(os.path.join(out_dir, 'top_variable_genes_5mC.csv'), index=False)
top_var_5hmc.to_csv(os.path.join(out_dir, 'top_variable_genes_5hmC.csv'), index=False)

# -------------------
# Summary
# -------------------
print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print('='*60)
print(f"\nResults saved to: {out_dir}/")
print("\nGenerated files:")
print("  1. genome_wide_gene_methylation.png - Genome-wide gene annotation")
print("  2. variable_genes_heatmap.png - Most variable genes")
print("  3. 5mC_top100_clustered.png - Clustered heatmap (5mC)")
print("  4. 5hmC_top100_clustered.png - Clustered heatmap (5hmC)")
print("  5. [chr]_gene_heatmap.png - Per-chromosome heatmaps")
print("  6. all_genes_all_samples.csv - Complete gene methylation data")
print("  7. [sample]_gene_methylation.csv - Per-sample data")
print("  8. top_variable_genes_5mC.csv - Top variable genes (5mC)")
print("  9. top_variable_genes_5hmC.csv - Top variable genes (5hmC)")
print(f"\nStatistics:")
print(f"  Total genes analyzed: {len(genes_df):,}")
print(f"  Genes with data: {len(merged_filtered):,}")
print(f"  Samples: {len(sample_names)}")
