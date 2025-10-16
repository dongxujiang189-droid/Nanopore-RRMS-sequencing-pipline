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
out_dir = os.path.join(base_dir, "comprehensive_analysis")
os.makedirs(out_dir, exist_ok=True)

chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
bin_size = 1_000_000
min_vaf = 0.0001  # Changed to 0.0001
smooth_sigma = 1.5

chrom_sizes = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895
}

# -------------------
# Load CXXC4
# -------------------
print("Loading CXXC4 data...")
cxxc4_df = pd.read_csv(cxxc4_file, sep='\t', header=None, 
                       names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue'],
                       usecols=[0,1,2,6])

if not cxxc4_df['chrom'].iloc[0].startswith('chr'):
    cxxc4_df['chrom'] = 'chr' + cxxc4_df['chrom'].astype(str)

cxxc4_data = {}
for chrom in chromosomes:
    if chrom not in chrom_sizes:
        continue
    
    bins = np.arange(0, chrom_sizes[chrom], bin_size)
    bin_signals = np.zeros(len(bins))
    chrom_peaks = cxxc4_df[cxxc4_df['chrom'] == chrom]
    
    for idx, bin_start in enumerate(bins):
        bin_end = bin_start + bin_size
        overlapping = chrom_peaks[
            (chrom_peaks['end'] > bin_start) & 
            (chrom_peaks['start'] < bin_end)
        ]
        if len(overlapping) > 0:
            bin_signals[idx] = overlapping['signalValue'].sum()
    
    cxxc4_data[chrom] = bin_signals

print(f"  Loaded {len(cxxc4_df)} peaks")
pd.DataFrame(cxxc4_data).to_csv(os.path.join(out_dir, 'cxxc4_binned.csv'), index=False)

# -------------------
# Process VCF - Improved multiallelic handling
# -------------------
vcf_files = glob.glob(vcf_pattern)
all_vaf_data = {}
all_variant_counts = {}

for vcf_file in vcf_files:
    sample_name = os.path.basename(os.path.dirname(vcf_file)).replace("_clair3", "")
    print(f"\nProcessing variants: {sample_name}")
    
    vaf_by_chrom = {}
    count_by_chrom = {}
    vcf = pysam.VariantFile(vcf_file)
    
    for chrom in chromosomes:
        if chrom not in chrom_sizes:
            continue
        
        bins = np.arange(0, chrom_sizes[chrom], bin_size)
        # Store position-level VAF sums
        position_vaf = defaultdict(float)
        
        try:
            for record in vcf.fetch(chrom):
                if 'RefCall' in record.filter or ('PASS' not in record.filter and record.filter.keys()):
                    continue
                
                pos = record.pos
                total_vaf = 0
                
                for sample in record.samples.values():
                    af = sample.get('AF', None)
                    if af is not None:
                        # Handle multiallelic: AF can be tuple
                        if isinstance(af, tuple):
                            total_vaf = sum([a for a in af if a >= min_vaf])
                        else:
                            if af >= min_vaf:
                                total_vaf = af
                
                if total_vaf > 0:
                    position_vaf[pos] += total_vaf
        except:
            pass
        
        # Bin the aggregated position VAFs
        vaf_per_bin = [[] for _ in range(len(bins))]
        count_per_bin = np.zeros(len(bins), dtype=int)
        
        for pos, vaf_sum in position_vaf.items():
            bin_idx = int(pos // bin_size)
            if bin_idx < len(vaf_per_bin):
                vaf_per_bin[bin_idx].append(vaf_sum)
                count_per_bin[bin_idx] += 1
        
        vaf_by_chrom[chrom] = [np.nanmean(bin_vaf) if bin_vaf else np.nan 
                                for bin_vaf in vaf_per_bin]
        count_by_chrom[chrom] = count_per_bin.tolist()
    
    vcf.close()
    all_vaf_data[sample_name] = vaf_by_chrom
    all_variant_counts[sample_name] = count_by_chrom
    
    pd.DataFrame(vaf_by_chrom).to_csv(
        os.path.join(out_dir, f'{sample_name}_vaf.csv'), index=False)
    pd.DataFrame(count_by_chrom).to_csv(
        os.path.join(out_dir, f'{sample_name}_counts.csv'), index=False)

# -------------------
# Process methylation
# -------------------
meth_files = glob.glob(meth_pattern)
all_meth_data = {}

for meth_file in meth_files:
    sample_name = os.path.basename(meth_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing methylation: {sample_name}")
    
    meth_df = pd.read_csv(meth_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    meth_5mc_by_chrom = {}
    meth_5hmc_by_chrom = {}
    
    for chrom in chromosomes:
        if chrom not in chrom_sizes:
            continue
        
        bins = np.arange(0, chrom_sizes[chrom], bin_size)
        mc_per_bin = [[] for _ in range(len(bins))]
        hmc_per_bin = [[] for _ in range(len(bins))]
        chrom_data = meth_df[meth_df['chrom'] == chrom]
        
        for _, row in chrom_data.iterrows():
            center = (row['start'] + row['end']) / 2
            bin_idx = int(center // bin_size)
            if bin_idx < len(mc_per_bin):
                mc_per_bin[bin_idx].append(row['percent_m'])
                hmc_per_bin[bin_idx].append(row['percent_h'])
        
        meth_5mc_by_chrom[chrom] = [np.nanmean(b) if b else np.nan for b in mc_per_bin]
        meth_5hmc_by_chrom[chrom] = [np.nanmean(b) if b else np.nan for b in hmc_per_bin]
    
    all_meth_data[sample_name] = {'5mC': meth_5mc_by_chrom, '5hmC': meth_5hmc_by_chrom}
    
    pd.DataFrame(meth_5mc_by_chrom).to_csv(os.path.join(out_dir, f'{sample_name}_5mC.csv'), index=False)
    pd.DataFrame(meth_5hmc_by_chrom).to_csv(os.path.join(out_dir, f'{sample_name}_5hmC.csv'), index=False)

# -------------------
# 4-panel plots per chromosome
# -------------------
common_samples = set(all_vaf_data.keys()) & set(all_meth_data.keys())
print(f"\n\nCreating plots for {len(common_samples)} samples")

for sample in common_samples:
    for chrom in tqdm(chromosomes, desc=sample):
        if chrom not in chrom_sizes:
            continue
        
        bins = np.arange(0, chrom_sizes[chrom], bin_size) / 1e6
        vaf = np.array(all_vaf_data[sample].get(chrom, []))
        mc = np.array(all_meth_data[sample]['5mC'].get(chrom, []))
        hmc = np.array(all_meth_data[sample]['5hmC'].get(chrom, []))
        cxxc4 = np.array(cxxc4_data.get(chrom, []))
        
        if len(vaf) == 0 or len(mc) == 0:
            continue
        
        vaf_smooth = gaussian_filter1d(np.nan_to_num(vaf), sigma=smooth_sigma)
        mc_smooth = gaussian_filter1d(np.nan_to_num(mc), sigma=smooth_sigma)
        hmc_smooth = gaussian_filter1d(np.nan_to_num(hmc), sigma=smooth_sigma)
        cxxc4_smooth = gaussian_filter1d(cxxc4[:len(bins)], sigma=smooth_sigma)
        
        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
        
        axes[0].fill_between(bins[:len(vaf)], vaf_smooth, alpha=0.5, color='#e74c3c')
        axes[0].plot(bins[:len(vaf)], vaf_smooth, linewidth=2, color='#c0392b')
        axes[0].set_ylabel('VAF', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{sample} - {chrom}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.2)
        
        axes[1].fill_between(bins[:len(mc)], mc_smooth, alpha=0.5, color='#3498db')
        axes[1].plot(bins[:len(mc)], mc_smooth, linewidth=2, color='#2980b9')
        axes[1].set_ylabel('5mC (%)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.2)
        
        axes[2].fill_between(bins[:len(hmc)], hmc_smooth, alpha=0.5, color='#2ecc71')
        axes[2].plot(bins[:len(hmc)], hmc_smooth, linewidth=2, color='#27ae60')
        axes[2].set_ylabel('5hmC (%)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.2)
        
        axes[3].fill_between(bins[:len(cxxc4_smooth)], cxxc4_smooth, alpha=0.5, color='#9b59b6')
        axes[3].plot(bins[:len(cxxc4_smooth)], cxxc4_smooth, linewidth=2, color='#8e44ad')
        axes[3].set_ylabel('CXXC4', fontsize=12, fontweight='bold')
        axes[3].set_xlabel(f'Position (Mb)', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{sample}_{chrom}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# -------------------
# Genome-wide heatmaps
# -------------------
for sample in common_samples:
    chrom_list = [c for c in chromosomes if c in chrom_sizes]
    max_bins = max([len(all_vaf_data[sample].get(c, [])) for c in chrom_list])
    
    vaf_matrix = np.full((len(chrom_list), max_bins), np.nan)
    mc_matrix = np.full((len(chrom_list), max_bins), np.nan)
    hmc_matrix = np.full((len(chrom_list), max_bins), np.nan)
    cxxc4_matrix = np.full((len(chrom_list), max_bins), np.nan)
    
    for i, chrom in enumerate(chrom_list):
        vaf = all_vaf_data[sample].get(chrom, [])
        mc = all_meth_data[sample]['5mC'].get(chrom, [])
        hmc = all_meth_data[sample]['5hmC'].get(chrom, [])
        cxxc4 = cxxc4_data.get(chrom, [])
        
        vaf_matrix[i, :len(vaf)] = vaf
        mc_matrix[i, :len(mc)] = mc
        hmc_matrix[i, :len(hmc)] = hmc
        cxxc4_matrix[i, :len(cxxc4)] = cxxc4
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 14))
    
    sns.heatmap(vaf_matrix, ax=axes[0], cmap='Reds', yticklabels=chrom_list, cbar_kws={'label': 'VAF'})
    axes[0].set_title(f'{sample}: VAF', fontsize=14, fontweight='bold')
    
    sns.heatmap(mc_matrix, ax=axes[1], cmap='Blues', yticklabels=chrom_list, cbar_kws={'label': '5mC'})
    axes[1].set_title('5mC', fontsize=14, fontweight='bold')
    
    sns.heatmap(hmc_matrix, ax=axes[2], cmap='Greens', yticklabels=chrom_list, cbar_kws={'label': '5hmC'})
    axes[2].set_title('5hmC', fontsize=14, fontweight='bold')
    
    sns.heatmap(cxxc4_matrix, ax=axes[3], cmap='Purples', yticklabels=chrom_list, cbar_kws={'label': 'CXXC4'})
    axes[3].set_title('CXXC4', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Genomic position (Mb)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample}_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------
# Correlations
# -------------------
print("\nCalculating correlations...")
correlation_results = []

for sample in common_samples:
    for chrom in chromosomes:
        if chrom not in chrom_sizes:
            continue
        
        vaf = np.array(all_vaf_data[sample].get(chrom, []))
        mc = np.array(all_meth_data[sample]['5mC'].get(chrom, []))
        hmc = np.array(all_meth_data[sample]['5hmC'].get(chrom, []))
        cxxc4 = np.array(cxxc4_data.get(chrom, []))
        
        min_len = min(len(vaf), len(mc), len(hmc), len(cxxc4))
        if min_len == 0:
            continue
        
        vaf, mc, hmc, cxxc4 = vaf[:min_len], mc[:min_len], hmc[:min_len], cxxc4[:min_len]
        mask = ~(np.isnan(vaf) | np.isnan(mc) | np.isnan(hmc))
        
        if mask.sum() < 3:
            continue
        
        vaf_c, mc_c, hmc_c, cxxc4_c = vaf[mask], mc[mask], hmc[mask], cxxc4[mask]
        
        result = {'sample': sample, 'chrom': chrom, 'n_bins': mask.sum()}
        
        try:
            result['corr_VAF_5mC'], result['p_VAF_5mC'] = stats.pearsonr(vaf_c, mc_c)
            result['corr_VAF_5hmC'], result['p_VAF_5hmC'] = stats.pearsonr(vaf_c, hmc_c)
            result['corr_VAF_CXXC4'], result['p_VAF_CXXC4'] = stats.pearsonr(vaf_c, cxxc4_c)
            result['corr_5mC_5hmC'], result['p_5mC_5hmC'] = stats.pearsonr(mc_c, hmc_c)
            result['corr_5mC_CXXC4'], result['p_5mC_CXXC4'] = stats.pearsonr(mc_c, cxxc4_c)
            result['corr_5hmC_CXXC4'], result['p_5hmC_CXXC4'] = stats.pearsonr(hmc_c, cxxc4_c)
        except:
            pass
        
        correlation_results.append(result)

corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv(os.path.join(out_dir, 'correlations.csv'), index=False)

summary = corr_df.groupby('sample')[
    ['corr_VAF_5mC', 'corr_VAF_5hmC', 'corr_VAF_CXXC4', 'corr_5mC_5hmC', 'corr_5mC_CXXC4', 'corr_5hmC_CXXC4']
].mean()
summary.to_csv(os.path.join(out_dir, 'correlation_summary.csv'))
print("\nCorrelation summary:")
print(summary)

# -------------------
# Scatter plots
# -------------------
for sample in common_samples:
    all_vaf, all_mc, all_hmc, all_cxxc4 = [], [], [], []
    
    for chrom in chromosomes:
        vaf = np.array(all_vaf_data[sample].get(chrom, []))
        mc = np.array(all_meth_data[sample]['5mC'].get(chrom, []))
        hmc = np.array(all_meth_data[sample]['5hmC'].get(chrom, []))
        cxxc4 = np.array(cxxc4_data.get(chrom, []))
        
        min_len = min(len(vaf), len(mc), len(hmc), len(cxxc4))
        if min_len > 0:
            all_vaf.extend(vaf[:min_len])
            all_mc.extend(mc[:min_len])
            all_hmc.extend(hmc[:min_len])
            all_cxxc4.extend(cxxc4[:min_len])
    
    df = pd.DataFrame({'VAF': all_vaf, '5mC': all_mc, '5hmC': all_hmc, 'CXXC4': all_cxxc4}).dropna()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].hexbin(df['VAF'], df['5mC'], gridsize=50, cmap='Blues', mincnt=1)
    axes[0,0].set_xlabel('VAF', fontweight='bold')
    axes[0,0].set_ylabel('5mC', fontweight='bold')
    
    axes[0,1].hexbin(df['VAF'], df['5hmC'], gridsize=50, cmap='Greens', mincnt=1)
    axes[0,1].set_xlabel('VAF', fontweight='bold')
    axes[0,1].set_ylabel('5hmC', fontweight='bold')
    
    axes[0,2].hexbin(df['VAF'], df['CXXC4'], gridsize=50, cmap='Purples', mincnt=1)
    axes[0,2].set_xlabel('VAF', fontweight='bold')
    axes[0,2].set_ylabel('CXXC4', fontweight='bold')
    
    axes[1,0].hexbin(df['5mC'], df['5hmC'], gridsize=50, cmap='viridis', mincnt=1)
    axes[1,0].set_xlabel('5mC', fontweight='bold')
    axes[1,0].set_ylabel('5hmC', fontweight='bold')
    
    axes[1,1].hexbin(df['5mC'], df['CXXC4'], gridsize=50, cmap='plasma', mincnt=1)
    axes[1,1].set_xlabel('5mC', fontweight='bold')
    axes[1,1].set_ylabel('CXXC4', fontweight='bold')
    
    axes[1,2].hexbin(df['5hmC'], df['CXXC4'], gridsize=50, cmap='magma', mincnt=1)
    axes[1,2].set_xlabel('5hmC', fontweight='bold')
    axes[1,2].set_ylabel('CXXC4', fontweight='bold')
    
    plt.suptitle(f'{sample}: Genome-wide correlations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample}_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n✓ Complete. Output: {out_dir}/")
