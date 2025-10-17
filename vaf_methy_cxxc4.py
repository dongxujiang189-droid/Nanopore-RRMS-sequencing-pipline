#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import pysam
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
vcf_pattern = os.path.join(base_dir, "clair3_output_variants calling", "*_clair3", "full_alignment.vcf.gz")
meth_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
cxxc4_file = os.path.join(base_dir, "cxxc4_binding", "GSM1054009_Myc-Idax_vs_empty.bed")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "multiscale_analysis")
os.makedirs(out_dir, exist_ok=True)

chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
chrom_bin_size = 1_000_000
min_vaf = 0.0001

chrom_sizes = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
}

# -------------------
# Load genes
# -------------------
print("Loading genes...")
genes = []
with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        chrom = fields[0] if fields[0].startswith('chr') else 'chr' + fields[0]
        start, end = int(fields[3]), int(fields[4])
        strand = fields[6]
        
        gene_name = None
        for attr in fields[8].split(';'):
            if 'gene_name' in attr:
                gene_name = attr.split('"')[1]
                break
        
        genes.append({
            'chrom': chrom, 'start': start, 'end': end, 'strand': strand,
            'name': gene_name or f'{chrom}:{start}-{end}',
            'length': end - start
        })

genes_df = pd.DataFrame(genes)
genes_df = genes_df[genes_df['length'] > 500]
print(f"  {len(genes_df)} genes")

# -------------------
# Load CXXC4 - 5 column BED
# -------------------
print("Loading CXXC4...")
cxxc4_df = pd.read_csv(cxxc4_file, sep='\t', header=None,
                       names=['chrom', 'start', 'end', 'name', 'score'],
                       usecols=[0, 1, 2, 3, 4])
print(f"  {len(cxxc4_df)} peaks")

# Chromosome-level CXXC4
print("Processing CXXC4 by chromosome...")
cxxc4_chrom_list = []
for chrom in tqdm(chromosomes, desc="CXXC4 bins"):
    if chrom not in chrom_sizes:
        continue
    bins = np.arange(0, chrom_sizes[chrom], chrom_bin_size)
    peaks = cxxc4_df[cxxc4_df['chrom'] == chrom]
    
    for idx, start in enumerate(bins):
        end = start + chrom_bin_size
        overlaps = peaks[(peaks['end'] > start) & (peaks['start'] < end)]
        signal = overlaps['score'].sum() if len(overlaps) > 0 else 0
        
        cxxc4_chrom_list.append({
            'chrom': chrom,
            'bin_start': start,
            'bin_end': end,
            'bin_index': idx,
            'cxxc4_signal': signal
        })

cxxc4_chrom_df = pd.DataFrame(cxxc4_chrom_list)
cxxc4_chrom_df.to_csv(os.path.join(out_dir, 'cxxc4_chromosome.csv'), index=False)

# -------------------
# Process VCF
# -------------------
vcf_files = glob.glob(vcf_pattern)
all_data = {}

for vcf_file in vcf_files:
    sample = os.path.basename(os.path.dirname(vcf_file)).replace("_clair3", "")
    print(f"\nProcessing {sample}...")
    vcf = pysam.VariantFile(vcf_file)
    
    vaf_chrom_list = []
    for chrom in tqdm(chromosomes, desc="VAF chrom"):
        if chrom not in chrom_sizes:
            continue
        bins = np.arange(0, chrom_sizes[chrom], chrom_bin_size)
        position_vaf = defaultdict(float)
        
        try:
            for rec in vcf.fetch(chrom):
                if 'RefCall' in rec.filter or ('PASS' not in rec.filter and rec.filter.keys()):
                    continue
                
                total_vaf = 0
                for samp in rec.samples.values():
                    af = samp.get('AF', None)
                    if af:
                        total_vaf = sum(af) if isinstance(af, tuple) else af
                
                if total_vaf >= min_vaf:
                    position_vaf[rec.pos] += total_vaf
        except:
            pass
        
        bin_vaf = [[] for _ in range(len(bins))]
        for pos, vaf in position_vaf.items():
            idx = int(pos // chrom_bin_size)
            if idx < len(bin_vaf):
                bin_vaf[idx].append(vaf)
        
        for idx, (start, vafs) in enumerate(zip(bins, bin_vaf)):
            vaf_chrom_list.append({
                'chrom': chrom,
                'bin_start': start,
                'bin_end': start + chrom_bin_size,
                'bin_index': idx,
                'mean_vaf': np.nanmean(vafs) if vafs else np.nan,
                'n_variants': len(vafs)
            })
    
    vaf_genes = []
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="  VAF genes"):
        try:
            vafs = []
            for rec in vcf.fetch(gene['chrom'], gene['start'], gene['end']):
                if 'RefCall' in rec.filter:
                    continue
                for samp in rec.samples.values():
                    af = samp.get('AF', None)
                    if af:
                        v = sum(af) if isinstance(af, tuple) else af
                        if v >= min_vaf:
                            vafs.append(v)
            
            vaf_genes.append({
                'gene': gene['name'],
                'chrom': gene['chrom'],
                'mean_vaf': np.nanmean(vafs) if vafs else np.nan,
                'n_variants': len(vafs)
            })
        except:
            vaf_genes.append({
                'gene': gene['name'],
                'chrom': gene['chrom'],
                'mean_vaf': np.nan,
                'n_variants': 0
            })
    
    vcf.close()
    
    all_data[sample] = {
        'vaf_chrom': pd.DataFrame(vaf_chrom_list),
        'vaf_genes': pd.DataFrame(vaf_genes)
    }
    
    all_data[sample]['vaf_chrom'].to_csv(os.path.join(out_dir, f'{sample}_vaf_chrom.csv'), index=False)
    all_data[sample]['vaf_genes'].to_csv(os.path.join(out_dir, f'{sample}_vaf_genes.csv'), index=False)

# -------------------
# Process methylation
# -------------------
meth_files = glob.glob(meth_pattern)

for meth_file in meth_files:
    sample = os.path.basename(meth_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    if sample not in all_data:
        all_data[sample] = {}
    
    print(f"\nMethylation {sample}...")
    meth_df = pd.read_csv(meth_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    mc_chrom_list, hmc_chrom_list = [], []
    for chrom in tqdm(chromosomes, desc="Meth chrom"):
        if chrom not in chrom_sizes:
            continue
        bins = np.arange(0, chrom_sizes[chrom], chrom_bin_size)
        mc_bins = [[] for _ in range(len(bins))]
        hmc_bins = [[] for _ in range(len(bins))]
        
        data = meth_df[meth_df['chrom'] == chrom]
        for _, row in data.iterrows():
            center = (row['start'] + row['end']) / 2
            idx = int(center // chrom_bin_size)
            if idx < len(mc_bins):
                mc_bins[idx].append(row['percent_m'])
                hmc_bins[idx].append(row['percent_h'])
        
        for idx, start in enumerate(bins):
            mc_chrom_list.append({
                'chrom': chrom,
                'bin_start': start,
                'bin_end': start + chrom_bin_size,
                'bin_index': idx,
                'mean_5mc': np.nanmean(mc_bins[idx]) if mc_bins[idx] else np.nan
            })
            hmc_chrom_list.append({
                'chrom': chrom,
                'bin_start': start,
                'bin_end': start + chrom_bin_size,
                'bin_index': idx,
                'mean_5hmc': np.nanmean(hmc_bins[idx]) if hmc_bins[idx] else np.nan
            })
    
    meth_genes = []
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="  Meth genes"):
        gene_meth = meth_df[
            (meth_df['chrom'] == gene['chrom']) &
            (meth_df['end'] > gene['start']) &
            (meth_df['start'] < gene['end'])
        ]
        
        meth_genes.append({
            'gene': gene['name'],
            'chrom': gene['chrom'],
            'mean_5mc': gene_meth['percent_m'].mean() if len(gene_meth) > 0 else np.nan,
            'mean_5hmc': gene_meth['percent_h'].mean() if len(gene_meth) > 0 else np.nan
        })
    
    all_data[sample]['mc_chrom'] = pd.DataFrame(mc_chrom_list)
    all_data[sample]['hmc_chrom'] = pd.DataFrame(hmc_chrom_list)
    all_data[sample]['meth_genes'] = pd.DataFrame(meth_genes)
    
    all_data[sample]['mc_chrom'].to_csv(os.path.join(out_dir, f'{sample}_5mc_chrom.csv'), index=False)
    all_data[sample]['hmc_chrom'].to_csv(os.path.join(out_dir, f'{sample}_5hmc_chrom.csv'), index=False)
    all_data[sample]['meth_genes'].to_csv(os.path.join(out_dir, f'{sample}_meth_genes.csv'), index=False)

# Gene-level CXXC4
print("\nCXXC4 at genes...")
cxxc4_genes = []
for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df)):
    peaks = cxxc4_df[
        (cxxc4_df['chrom'] == gene['chrom']) &
        (cxxc4_df['end'] > gene['start']) &
        (cxxc4_df['start'] < gene['end'])
    ]
    
    cxxc4_genes.append({
        'gene': gene['name'],
        'chrom': gene['chrom'],
        'cxxc4_signal': peaks['score'].sum() if len(peaks) > 0 else 0
    })

cxxc4_genes_df = pd.DataFrame(cxxc4_genes)
cxxc4_genes_df.to_csv(os.path.join(out_dir, 'cxxc4_genes.csv'), index=False)

# -------------------
# REVISED: Chromosome-level plots - All samples side by side
# -------------------
print("\nPlotting chromosomes (multi-sample comparison)...")
samples = [s for s in all_data.keys() if 'vaf_chrom' in all_data[s] and 'mc_chrom' in all_data[s]]
n_samples = len(samples)

if n_samples > 0:
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        if chrom not in chrom_sizes:
            continue
        
        # Check if any sample has data for this chromosome
        has_data = False
        for sample in samples:
            vaf_data = all_data[sample]['vaf_chrom'][all_data[sample]['vaf_chrom']['chrom'] == chrom]
            if len(vaf_data) > 0:
                has_data = True
                break
        
        if not has_data:
            continue
        
        # Create figure: 4 metrics x n_samples
        fig, axes = plt.subplots(4, n_samples, figsize=(6*n_samples, 12), 
                                sharex=True, sharey='row')
        
        # Handle single sample case
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for col, sample in enumerate(samples):
            vaf_data = all_data[sample]['vaf_chrom'][all_data[sample]['vaf_chrom']['chrom'] == chrom]
            mc_data = all_data[sample]['mc_chrom'][all_data[sample]['mc_chrom']['chrom'] == chrom]
            hmc_data = all_data[sample]['hmc_chrom'][all_data[sample]['hmc_chrom']['chrom'] == chrom]
            cxxc4_data = cxxc4_chrom_df[cxxc4_chrom_df['chrom'] == chrom]
            
            if len(vaf_data) == 0:
                continue
            
            bins = vaf_data['bin_start'].values / 1e6
            vaf = vaf_data['mean_vaf'].values
            mc = mc_data['mean_5mc'].values
            hmc = hmc_data['mean_5hmc'].values
            cxxc4 = cxxc4_data['cxxc4_signal'].values
            
            # VAF
            axes[0, col].fill_between(bins, gaussian_filter1d(np.nan_to_num(vaf), 1.5), 
                                     alpha=0.5, color='#e74c3c')
            axes[0, col].plot(bins, gaussian_filter1d(np.nan_to_num(vaf), 1.5), 
                             linewidth=2, color='#c0392b')
            axes[0, col].set_title(f'{sample}', fontsize=12, fontweight='bold')
            axes[0, col].grid(alpha=0.2)
            if col == 0:
                axes[0, col].set_ylabel('VAF', fontweight='bold')
            
            # 5mC
            axes[1, col].fill_between(bins, gaussian_filter1d(np.nan_to_num(mc), 1.5), 
                                     alpha=0.5, color='#3498db')
            axes[1, col].plot(bins, gaussian_filter1d(np.nan_to_num(mc), 1.5), 
                             linewidth=2, color='#2980b9')
            axes[1, col].grid(alpha=0.2)
            if col == 0:
                axes[1, col].set_ylabel('5mC (%)', fontweight='bold')
            
            # 5hmC
            axes[2, col].fill_between(bins, gaussian_filter1d(np.nan_to_num(hmc), 1.5), 
                                     alpha=0.5, color='#2ecc71')
            axes[2, col].plot(bins, gaussian_filter1d(np.nan_to_num(hmc), 1.5), 
                             linewidth=2, color='#27ae60')
            axes[2, col].grid(alpha=0.2)
            if col == 0:
                axes[2, col].set_ylabel('5hmC (%)', fontweight='bold')
            
            # CXXC4
            axes[3, col].fill_between(bins, gaussian_filter1d(cxxc4, 1.5), 
                                     alpha=0.5, color='#9b59b6')
            axes[3, col].plot(bins, gaussian_filter1d(cxxc4, 1.5), 
                             linewidth=2, color='#8e44ad')
            axes[3, col].set_xlabel('Position (Mb)', fontweight='bold')
            axes[3, col].grid(alpha=0.2)
            if col == 0:
                axes[3, col].set_ylabel('CXXC4', fontweight='bold')
        
        plt.suptitle(f'{chrom} - Multi-Sample Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'ALL_SAMPLES_{chrom}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# -------------------
# REVISED: Gene-level analysis - Side by side comparison
# -------------------
print("\nPlotting gene-level comparisons...")

# First save individual integrated files
for sample in samples:
    gene_data = all_data[sample]['vaf_genes'].merge(
        all_data[sample]['meth_genes'], on=['gene', 'chrom']
    ).merge(cxxc4_genes_df, on=['gene', 'chrom'])
    gene_data.to_csv(os.path.join(out_dir, f'{sample}_gene_integrated.csv'), index=False)

# Create side-by-side comparison plots
fig, axes = plt.subplots(6, n_samples, figsize=(6*n_samples, 24))
if n_samples == 1:
    axes = axes.reshape(-1, 1)

for col, sample in enumerate(samples):
    gene_data = all_data[sample]['vaf_genes'].merge(
        all_data[sample]['meth_genes'], on=['gene', 'chrom']
    ).merge(cxxc4_genes_df, on=['gene', 'chrom'])
    
    df = gene_data.dropna(subset=['mean_vaf', 'mean_5mc', 'mean_5hmc'])
    
    if len(df) == 0:
        print(f"  Warning: No valid data for {sample}, skipping")
        continue
    
    # VAF vs 5mC
    axes[0, col].hexbin(df['mean_vaf'], df['mean_5mc'], gridsize=40, cmap='Blues', mincnt=1)
    axes[0, col].set_title(f'{sample}', fontweight='bold')
    axes[0, col].set_xlabel('VAF', fontweight='bold')
    if col == 0:
        axes[0, col].set_ylabel('5mC', fontweight='bold')
    
    # VAF vs 5hmC
    axes[1, col].hexbin(df['mean_vaf'], df['mean_5hmc'], gridsize=40, cmap='Greens', mincnt=1)
    axes[1, col].set_xlabel('VAF', fontweight='bold')
    if col == 0:
        axes[1, col].set_ylabel('5hmC', fontweight='bold')
    
    # VAF vs CXXC4
    axes[2, col].hexbin(df['mean_vaf'], df['cxxc4_signal'], gridsize=40, cmap='Purples', mincnt=1)
    axes[2, col].set_xlabel('VAF', fontweight='bold')
    if col == 0:
        axes[2, col].set_ylabel('CXXC4', fontweight='bold')
    
    # 5mC vs 5hmC
    axes[3, col].hexbin(df['mean_5mc'], df['mean_5hmc'], gridsize=40, cmap='viridis', mincnt=1)
    axes[3, col].set_xlabel('5mC', fontweight='bold')
    if col == 0:
        axes[3, col].set_ylabel('5hmC', fontweight='bold')
    
    # 5mC vs CXXC4
    axes[4, col].hexbin(df['mean_5mc'], df['cxxc4_signal'], gridsize=40, cmap='plasma', mincnt=1)
    axes[4, col].set_xlabel('5mC', fontweight='bold')
    if col == 0:
        axes[4, col].set_ylabel('CXXC4', fontweight='bold')
    
    # 5hmC vs CXXC4
    axes[5, col].hexbin(df['mean_5hmc'], df['cxxc4_signal'], gridsize=40, cmap='magma', mincnt=1)
    axes[5, col].set_xlabel('5hmC', fontweight='bold')
    if col == 0:
        axes[5, col].set_ylabel('CXXC4', fontweight='bold')

plt.suptitle('Gene-level Multi-Sample Comparison', fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'ALL_SAMPLES_gene_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Complete: {out_dir}/")
