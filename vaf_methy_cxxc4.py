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

chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
chrom_bin_size = 1_000_000
gene_flank = 2000
min_vaf = 0.0001

chrom_sizes = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895
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
        
        # Extract gene name
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
# Load CXXC4
# -------------------
print("Loading CXXC4...")
cxxc4_df = pd.read_csv(cxxc4_file, sep='\t', header=None, 
                       names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue'],
                       usecols=[0,1,2,6])
if not cxxc4_df['chrom'].iloc[0].startswith('chr'):
    cxxc4_df['chrom'] = 'chr' + cxxc4_df['chrom'].astype(str)
print(f"  {len(cxxc4_df)} peaks")

# Chromosome-level CXXC4
cxxc4_chrom = {}
for chrom in chromosomes:
    bins = np.arange(0, chrom_sizes[chrom], chrom_bin_size)
    signals = np.zeros(len(bins))
    peaks = cxxc4_df[cxxc4_df['chrom'] == chrom]
    
    for idx, start in enumerate(bins):
        end = start + chrom_bin_size
        overlaps = peaks[(peaks['end'] > start) & (peaks['start'] < end)]
        if len(overlaps) > 0:
            signals[idx] = overlaps['signalValue'].sum()
    
    cxxc4_chrom[chrom] = signals

pd.DataFrame(cxxc4_chrom).to_csv(os.path.join(out_dir, 'cxxc4_chromosome.csv'), index=False)

# -------------------
# Process VCF
# -------------------
vcf_files = glob.glob(vcf_pattern)
all_data = {}

for vcf_file in vcf_files:
    sample = os.path.basename(os.path.dirname(vcf_file)).replace("_clair3", "")
    print(f"\nProcessing {sample}...")
    vcf = pysam.VariantFile(vcf_file)
    
    # Chromosome-level VAF
    vaf_chrom = {}
    for chrom in chromosomes:
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
        
        vaf_chrom[chrom] = [np.nanmean(b) if b else np.nan for b in bin_vaf]
    
    # Gene-level VAF
    vaf_genes = []
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="  Genes"):
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
    
    all_data[sample] = {'vaf_chrom': vaf_chrom, 'vaf_genes': pd.DataFrame(vaf_genes)}
    
    pd.DataFrame(vaf_chrom).to_csv(os.path.join(out_dir, f'{sample}_vaf_chrom.csv'), index=False)
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
    
    # Chromosome-level
    mc_chrom, hmc_chrom = {}, {}
    for chrom in chromosomes:
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
        
        mc_chrom[chrom] = [np.nanmean(b) if b else np.nan for b in mc_bins]
        hmc_chrom[chrom] = [np.nanmean(b) if b else np.nan for b in hmc_bins]
    
    # Gene-level
    meth_genes = []
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="  Genes"):
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
    
    all_data[sample]['mc_chrom'] = mc_chrom
    all_data[sample]['hmc_chrom'] = hmc_chrom
    all_data[sample]['meth_genes'] = pd.DataFrame(meth_genes)
    
    pd.DataFrame(mc_chrom).to_csv(os.path.join(out_dir, f'{sample}_5mc_chrom.csv'), index=False)
    pd.DataFrame(hmc_chrom).to_csv(os.path.join(out_dir, f'{sample}_5hmc_chrom.csv'), index=False)
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
        'cxxc4_signal': peaks['signalValue'].sum() if len(peaks) > 0 else 0
    })

cxxc4_genes_df = pd.DataFrame(cxxc4_genes)
cxxc4_genes_df.to_csv(os.path.join(out_dir, 'cxxc4_genes.csv'), index=False)

# -------------------
# Integrated plots: chromosome-level
# -------------------
print("\nPlotting chromosome-level...")
samples = [s for s in all_data.keys() if 'vaf_chrom' in all_data[s] and 'mc_chrom' in all_data[s]]

for sample in samples:
    for chrom in tqdm(chromosomes, desc=sample):
        bins = np.arange(0, chrom_sizes[chrom], chrom_bin_size) / 1e6
        
        vaf = np.array(all_data[sample]['vaf_chrom'].get(chrom, []))
        mc = np.array(all_data[sample]['mc_chrom'].get(chrom, []))
        hmc = np.array(all_data[sample]['hmc_chrom'].get(chrom, []))
        cxxc4 = np.array(cxxc4_chrom.get(chrom, []))
        
        if len(vaf) == 0:
            continue
        
        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
        
        axes[0].fill_between(bins[:len(vaf)], gaussian_filter1d(np.nan_to_num(vaf), 1.5), alpha=0.5, color='#e74c3c')
        axes[0].plot(bins[:len(vaf)], gaussian_filter1d(np.nan_to_num(vaf), 1.5), linewidth=2, color='#c0392b')
        axes[0].set_ylabel('VAF', fontweight='bold')
        axes[0].set_title(f'{sample} - {chrom} (Chromosome-level)', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.2)
        
        axes[1].fill_between(bins[:len(mc)], gaussian_filter1d(np.nan_to_num(mc), 1.5), alpha=0.5, color='#3498db')
        axes[1].plot(bins[:len(mc)], gaussian_filter1d(np.nan_to_num(mc), 1.5), linewidth=2, color='#2980b9')
        axes[1].set_ylabel('5mC (%)', fontweight='bold')
        axes[1].grid(alpha=0.2)
        
        axes[2].fill_between(bins[:len(hmc)], gaussian_filter1d(np.nan_to_num(hmc), 1.5), alpha=0.5, color='#2ecc71')
        axes[2].plot(bins[:len(hmc)], gaussian_filter1d(np.nan_to_num(hmc), 1.5), linewidth=2, color='#27ae60')
        axes[2].set_ylabel('5hmC (%)', fontweight='bold')
        axes[2].grid(alpha=0.2)
        
        axes[3].fill_between(bins[:len(cxxc4)], gaussian_filter1d(cxxc4, 1.5), alpha=0.5, color='#9b59b6')
        axes[3].plot(bins[:len(cxxc4)], gaussian_filter1d(cxxc4, 1.5), linewidth=2, color='#8e44ad')
        axes[3].set_ylabel('CXXC4', fontweight='bold')
        axes[3].set_xlabel('Position (Mb)', fontweight='bold')
        axes[3].grid(alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{sample}_{chrom}_chromosome.png'), dpi=300, bbox_inches='tight')
        plt.close()

# -------------------
# Gene-level comparisons
# -------------------
print("\nGene-level analysis...")

for sample in samples:
    # Merge gene data
    gene_data = all_data[sample]['vaf_genes'].merge(
        all_data[sample]['meth_genes'], on=['gene', 'chrom']
    ).merge(cxxc4_genes_df, on=['gene', 'chrom'])
    
    gene_data = gene_data.dropna(subset=['mean_vaf', 'mean_5mc', 'mean_5hmc'])
    gene_data.to_csv(os.path.join(out_dir, f'{sample}_gene_integrated.csv'), index=False)
    
    # Gene-level scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].hexbin(gene_data['mean_vaf'], gene_data['mean_5mc'], gridsize=40, cmap='Blues', mincnt=1)
    axes[0,0].set_xlabel('VAF', fontweight='bold')
    axes[0,0].set_ylabel('5mC', fontweight='bold')
    axes[0,0].set_title('VAF vs 5mC (Gene-level)')
    
    axes[0,1].hexbin(gene_data['mean_vaf'], gene_data['mean_5hmc'], gridsize=40, cmap='Greens', mincnt=1)
    axes[0,1].set_xlabel('VAF', fontweight='bold')
    axes[0,1].set_ylabel('5hmC', fontweight='bold')
    axes[0,1].set_title('VAF vs 5hmC')
    
    axes[0,2].hexbin(gene_data['mean_vaf'], gene_data['cxxc4_signal'], gridsize=40, cmap='Purples', mincnt=1)
    axes[0,2].set_xlabel('VAF', fontweight='bold')
    axes[0,2].set_ylabel('CXXC4', fontweight='bold')
    axes[0,2].set_title('VAF vs CXXC4')
    
    axes[1,0].hexbin(gene_data['mean_5mc'], gene_data['mean_5hmc'], gridsize=40, cmap='viridis', mincnt=1)
    axes[1,0].set_xlabel('5mC', fontweight='bold')
    axes[1,0].set_ylabel('5hmC', fontweight='bold')
    axes[1,0].set_title('5mC vs 5hmC')
    
    axes[1,1].hexbin(gene_data['mean_5mc'], gene_data['cxxc4_signal'], gridsize=40, cmap='plasma', mincnt=1)
    axes[1,1].set_xlabel('5mC', fontweight='bold')
    axes[1,1].set_ylabel('CXXC4', fontweight='bold')
    axes[1,1].set_title('5mC vs CXXC4')
    
    axes[1,2].hexbin(gene_data['mean_5hmc'], gene_data['cxxc4_signal'], gridsize=40, cmap='magma', mincnt=1)
    axes[1,2].set_xlabel('5hmC', fontweight='bold')
    axes[1,2].set_ylabel('CXXC4', fontweight='bold')
    axes[1,2].set_title('5hmC vs CXXC4')
    
    plt.suptitle(f'{sample}: Gene-level correlations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample}_gene_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Multi-scale comparison
# -------------------
print("\nMulti-scale comparisons...")

for sample in samples:
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    
    # Chromosome-level distributions
    all_vaf_c = np.concatenate([all_data[sample]['vaf_chrom'][c] for c in chromosomes if c in all_data[sample]['vaf_chrom']])
    all_mc_c = np.concatenate([all_data[sample]['mc_chrom'][c] for c in chromosomes])
    all_hmc_c = np.concatenate([all_data[sample]['hmc_chrom'][c] for c in chromosomes])
    all_cxxc4_c = np.concatenate([cxxc4_chrom[c] for c in chromosomes])
    
    axes[0,0].hist(all_vaf_c[~np.isnan(all_vaf_c)], bins=50, color='#e74c3c', alpha=0.7)
    axes[0,0].set_title('VAF - Chromosome', fontweight='bold')
    
    axes[1,0].hist(all_mc_c[~np.isnan(all_mc_c)], bins=50, color='#3498db', alpha=0.7)
    axes[1,0].set_title('5mC - Chromosome', fontweight='bold')
    
    axes[2,0].hist(all_hmc_c[~np.isnan(all_hmc_c)], bins=50, color='#2ecc71', alpha=0.7)
    axes[2,0].set_title('5hmC - Chromosome', fontweight='bold')
    
    axes[3,0].hist(all_cxxc4_c[all_cxxc4_c > 0], bins=50, color='#9b59b6', alpha=0.7)
    axes[3,0].set_title('CXXC4 - Chromosome', fontweight='bold')
    
    # Gene-level distributions
    gene_data = all_data[sample]['vaf_genes'].merge(
        all_data[sample]['meth_genes'], on=['gene', 'chrom']
    ).merge(cxxc4_genes_df, on=['gene', 'chrom'])
    
    axes[0,1].hist(gene_data['mean_vaf'].dropna(), bins=50, color='#e74c3c', alpha=0.7)
    axes[0,1].set_title('VAF - Gene', fontweight='bold')
    
    axes[1,1].hist(gene_data['mean_5mc'].dropna(), bins=50, color='#3498db', alpha=0.7)
    axes[1,1].set_title('5mC - Gene', fontweight='bold')
    
    axes[2,1].hist(gene_data['mean_5hmc'].dropna(), bins=50, color='#2ecc71', alpha=0.7)
    axes[2,1].set_title('5hmC - Gene', fontweight='bold')
    
    axes[3,1].hist(gene_data['cxxc4_signal'][gene_data['cxxc4_signal'] > 0], bins=50, color='#9b59b6', alpha=0.7)
    axes[3,1].set_title('CXXC4 - Gene', fontweight='bold')
    
    plt.suptitle(f'{sample}: Chromosome vs Gene-level distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample}_multiscale.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n✓ Complete: {out_dir}/")
