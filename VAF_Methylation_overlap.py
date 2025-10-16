#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pysam

# -------------------
# Configuration
# -------------------
base_dir = "/mnt/e/Data/seq_for_human_293t2/"
vcf_pattern = os.path.join(base_dir, "clair3_output_variants calling", "*_clair3", "full_alignment.vcf.gz")
meth_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "integrated_vaf_methylation")
os.makedirs(out_dir, exist_ok=True)

flank_size = 2000
n_gene_bins = 80
n_flank_bins = 80
use_smoothing = True
smooth_window = 5
min_vaf = 0.1  # Filter variants with VAF >= 10%

# -------------------
# Helper functions
# -------------------
def smooth_profile(data, window_size):
    if window_size < 3 or window_size % 2 == 0:
        return data
    data_array = np.array(data)
    smoothed = np.copy(data_array)
    half_window = window_size // 2
    for i in range(len(data_array)):
        start = max(0, i - half_window)
        end = min(len(data_array), i + half_window + 1)
        smoothed[i] = np.nanmean(data_array[start:end])
    return smoothed.tolist()

def extract_sample_name(filepath):
    """Extract barcode name from filepath"""
    basename = os.path.basename(os.path.dirname(filepath))
    return basename.replace("_clair3", "")

# -------------------
# Load genes
# -------------------
print("Loading genes from GTF...")
genes = []
with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        chrom = fields[0] if fields[0].startswith('chr') else 'chr' + fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        
        if strand == '+':
            tss, tes = start, end
        else:
            tss, tes = end, start
        
        genes.append({
            'chrom': chrom, 'start': start, 'end': end,
            'strand': strand, 'tss': tss, 'tes': tes,
            'length': abs(end - start)
        })

genes_df = pd.DataFrame(genes)
genes_df = genes_df[genes_df['length'] > 500]
print(f"Loaded {len(genes_df)} genes")

# -------------------
# Process VCF files - Extract VAF
# -------------------
vcf_files = glob.glob(vcf_pattern)
if not vcf_files:
    raise FileNotFoundError(f"No VCF files found: {vcf_pattern}")

vaf_profiles = {}

for vcf_file in vcf_files:
    sample_name = extract_sample_name(vcf_file)
    print(f"\nProcessing variants for {sample_name}...")
    
    # Load variants with VAF
    variants = []
    vcf = pysam.VariantFile(vcf_file)
    
    for record in vcf:
        # Skip reference calls
        if 'RefCall' in record.filter:
            continue
        if record.filter.keys() and 'PASS' not in record.filter:
            continue
            
        # Extract VAF
        for sample in record.samples.values():
            af = sample.get('AF', None)
            if af is not None:
                if isinstance(af, tuple):
                    af = af[0]
                if af >= min_vaf:
                    variants.append({
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'vaf': af
                    })
    
    vcf.close()
    variants_df = pd.DataFrame(variants)
    print(f"  Found {len(variants_df)} variants (VAF >= {min_vaf})")
    
    # Bin variants similar to methylation
    upstream_vaf = [[] for _ in range(n_flank_bins)]
    genebody_vaf = [[] for _ in range(n_gene_bins)]
    downstream_vaf = [[] for _ in range(n_flank_bins)]
    
    upstream_bins = np.linspace(-flank_size, 0, n_flank_bins + 1)
    genebody_bins = np.linspace(0, 100, n_gene_bins + 1)
    downstream_bins = np.linspace(0, flank_size, n_flank_bins + 1)
    
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="Binning variants"):
        chrom = gene['chrom']
        tss = gene['tss']
        tes = gene['tes']
        strand = gene['strand']
        
        if strand == '+':
            region_start = tss - flank_size
            region_end = tes + flank_size
        else:
            region_start = tes - flank_size
            region_end = tss + flank_size
        
        gene_variants = variants_df[
            (variants_df['chrom'] == chrom) &
            (variants_df['pos'] >= min(region_start, region_end)) &
            (variants_df['pos'] <= max(region_start, region_end))
        ]
        
        for _, var in gene_variants.iterrows():
            pos = var['pos']
            vaf = var['vaf']
            
            if strand == '+':
                if pos < tss:
                    dist = pos - tss
                    bin_idx = np.digitize(dist, upstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        upstream_vaf[bin_idx].append(vaf)
                elif tss <= pos <= tes:
                    pct = 100 * (pos - tss) / (tes - tss)
                    bin_idx = np.digitize(pct, genebody_bins) - 1
                    if 0 <= bin_idx < n_gene_bins:
                        genebody_vaf[bin_idx].append(vaf)
                elif pos > tes:
                    dist = pos - tes
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        downstream_vaf[bin_idx].append(vaf)
            else:
                if pos > tss:
                    dist = pos - tss
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        upstream_vaf[n_flank_bins - 1 - bin_idx].append(vaf)
                elif tes <= pos <= tss:
                    pct = 100 * (tss - pos) / (tss - tes)
                    bin_idx = np.digitize(pct, genebody_bins) - 1
                    if 0 <= bin_idx < n_gene_bins:
                        genebody_vaf[bin_idx].append(vaf)
                elif pos < tes:
                    dist = tes - pos
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        downstream_vaf[n_flank_bins - 1 - bin_idx].append(vaf)
    
    # Calculate mean VAF per bin
    profile_vaf = []
    for i in range(n_flank_bins):
        profile_vaf.append(np.nanmean(upstream_vaf[i]) if upstream_vaf[i] else np.nan)
    for i in range(n_gene_bins):
        profile_vaf.append(np.nanmean(genebody_vaf[i]) if genebody_vaf[i] else np.nan)
    for i in range(n_flank_bins):
        profile_vaf.append(np.nanmean(downstream_vaf[i]) if downstream_vaf[i] else np.nan)
    
    if use_smoothing:
        profile_vaf = smooth_profile(profile_vaf, smooth_window)
    
    vaf_profiles[sample_name] = profile_vaf

# -------------------
# Process methylation files
# -------------------
meth_files = glob.glob(meth_pattern)
meth_profiles = {}

for meth_file in meth_files:
    sample_name = os.path.basename(meth_file).replace("_aligned_with_mod.region_mh.stats.tsv", "")
    print(f"\nProcessing methylation for {sample_name}...")
    
    meth_df = pd.read_csv(meth_file, sep='\t')
    if '#chrom' in meth_df.columns:
        meth_df.rename(columns={'#chrom': 'chrom'}, inplace=True)
    if not str(meth_df['chrom'].iloc[0]).startswith('chr'):
        meth_df['chrom'] = 'chr' + meth_df['chrom'].astype(str)
    
    upstream_bins = np.linspace(-flank_size, 0, n_flank_bins + 1)
    genebody_bins = np.linspace(0, 100, n_gene_bins + 1)
    downstream_bins = np.linspace(0, flank_size, n_flank_bins + 1)
    
    upstream_5mc = [[] for _ in range(n_flank_bins)]
    upstream_5hmc = [[] for _ in range(n_flank_bins)]
    genebody_5mc = [[] for _ in range(n_gene_bins)]
    genebody_5hmc = [[] for _ in range(n_gene_bins)]
    downstream_5mc = [[] for _ in range(n_flank_bins)]
    downstream_5hmc = [[] for _ in range(n_flank_bins)]
    
    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df), desc="Binning methylation"):
        chrom = gene['chrom']
        tss = gene['tss']
        tes = gene['tes']
        strand = gene['strand']
        
        if strand == '+':
            region_start = tss - flank_size
            region_end = tes + flank_size
        else:
            region_start = tes - flank_size
            region_end = tss + flank_size
        
        meth_region = meth_df[
            (meth_df['chrom'] == chrom) &
            (meth_df['end'] > min(region_start, region_end)) &
            (meth_df['start'] < max(region_start, region_end))
        ]
        
        for _, meth in meth_region.iterrows():
            center = (meth['start'] + meth['end']) / 2
            
            if strand == '+':
                if center < tss:
                    dist = center - tss
                    bin_idx = np.digitize(dist, upstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        upstream_5mc[bin_idx].append(meth['percent_m'])
                        upstream_5hmc[bin_idx].append(meth['percent_h'])
                elif tss <= center <= tes:
                    pct = 100 * (center - tss) / (tes - tss)
                    bin_idx = np.digitize(pct, genebody_bins) - 1
                    if 0 <= bin_idx < n_gene_bins:
                        genebody_5mc[bin_idx].append(meth['percent_m'])
                        genebody_5hmc[bin_idx].append(meth['percent_h'])
                elif center > tes:
                    dist = center - tes
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        downstream_5mc[bin_idx].append(meth['percent_m'])
                        downstream_5hmc[bin_idx].append(meth['percent_h'])
            else:
                if center > tss:
                    dist = center - tss
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        upstream_5mc[n_flank_bins - 1 - bin_idx].append(meth['percent_m'])
                        upstream_5hmc[n_flank_bins - 1 - bin_idx].append(meth['percent_h'])
                elif tes <= center <= tss:
                    pct = 100 * (tss - center) / (tss - tes)
                    bin_idx = np.digitize(pct, genebody_bins) - 1
                    if 0 <= bin_idx < n_gene_bins:
                        genebody_5mc[bin_idx].append(meth['percent_m'])
                        genebody_5hmc[bin_idx].append(meth['percent_h'])
                elif center < tes:
                    dist = tes - center
                    bin_idx = np.digitize(dist, downstream_bins) - 1
                    if 0 <= bin_idx < n_flank_bins:
                        downstream_5mc[n_flank_bins - 1 - bin_idx].append(meth['percent_m'])
                        downstream_5hmc[n_flank_bins - 1 - bin_idx].append(meth['percent_h'])
    
    profile_5mc = []
    profile_5hmc = []
    for i in range(n_flank_bins):
        profile_5mc.append(np.nanmean(upstream_5mc[i]) if upstream_5mc[i] else np.nan)
        profile_5hmc.append(np.nanmean(upstream_5hmc[i]) if upstream_5hmc[i] else np.nan)
    for i in range(n_gene_bins):
        profile_5mc.append(np.nanmean(genebody_5mc[i]) if genebody_5mc[i] else np.nan)
        profile_5hmc.append(np.nanmean(genebody_5hmc[i]) if genebody_5hmc[i] else np.nan)
    for i in range(n_flank_bins):
        profile_5mc.append(np.nanmean(downstream_5mc[i]) if downstream_5mc[i] else np.nan)
        profile_5hmc.append(np.nanmean(downstream_5hmc[i]) if downstream_5hmc[i] else np.nan)
    
    if use_smoothing:
        profile_5mc = smooth_profile(profile_5mc, smooth_window)
        profile_5hmc = smooth_profile(profile_5hmc, smooth_window)
    
    meth_profiles[sample_name] = {
        '5mC': profile_5mc,
        '5hmC': profile_5hmc
    }

# -------------------
# Plot integrated profiles
# -------------------
x_pos = np.arange(n_flank_bins + n_gene_bins + n_flank_bins)
tss_pos = n_flank_bins
tes_pos = n_flank_bins + n_gene_bins

# Match samples between VAF and methylation
common_samples = set(vaf_profiles.keys()) & set(meth_profiles.keys())
print(f"\nPlotting {len(common_samples)} samples with both VAF and methylation data")

for sample in common_samples:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # VAF
    ax1.plot(x_pos, vaf_profiles[sample], linewidth=2, color='#e74c3c', alpha=0.8)
    ax1.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_ylabel('Mean VAF', fontsize=11, fontweight='bold')
    ax1.set_title(f'{sample}: Variant Allele Frequency', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 5mC
    ax2.plot(x_pos, meth_profiles[sample]['5mC'], linewidth=2, color='#3498db', alpha=0.8)
    ax2.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_ylabel('5mC (%)', fontsize=11, fontweight='bold')
    ax2.set_title('5mC Methylation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 5hmC
    ax3.plot(x_pos, meth_profiles[sample]['5hmC'], linewidth=2, color='#2ecc71', alpha=0.8)
    ax3.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_ylabel('5hmC (%)', fontsize=11, fontweight='bold')
    ax3.set_title('5hmC Hydroxymethylation', fontsize=12, fontweight='bold')
    ax3.set_xlabel("Genomic region", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    for ax in [ax1, ax2, ax3]:
        ax.text(tss_pos, ax.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=9, fontweight='bold')
        ax.text(tes_pos, ax.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_xticks([0, tss_pos, tss_pos + n_gene_bins//2, tes_pos, len(x_pos)-1])
    ax3.set_xticklabels(['-2kb', 'TSS', 'Gene body', 'TES', '+2kb'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{sample}_integrated_profile.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Multi-sample overlay
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for sample in common_samples:
    ax1.plot(x_pos, vaf_profiles[sample], linewidth=2, label=sample, alpha=0.7)
    ax2.plot(x_pos, meth_profiles[sample]['5mC'], linewidth=2, label=sample, alpha=0.7)
    ax3.plot(x_pos, meth_profiles[sample]['5hmC'], linewidth=2, label=sample, alpha=0.7)

for ax in [ax1, ax2, ax3]:
    ax.axvline(x=tss_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=tes_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.text(tss_pos, ax.get_ylim()[1]*0.95, 'TSS', ha='center', fontsize=9, fontweight='bold')
    ax.text(tes_pos, ax.get_ylim()[1]*0.95, 'TES', ha='center', fontsize=9, fontweight='bold')

ax1.set_ylabel('Mean VAF', fontsize=11, fontweight='bold')
ax1.set_title('Variant Allele Frequency (All Samples)', fontsize=12, fontweight='bold')
ax2.set_ylabel('5mC (%)', fontsize=11, fontweight='bold')
ax2.set_title('5mC Methylation (All Samples)', fontsize=12, fontweight='bold')
ax3.set_ylabel('5hmC (%)', fontsize=11, fontweight='bold')
ax3.set_title('5hmC Hydroxymethylation (All Samples)', fontsize=12, fontweight='bold')
ax3.set_xlabel("Genomic region", fontsize=11, fontweight='bold')
ax3.set_xticks([0, tss_pos, tss_pos + n_gene_bins//2, tes_pos, len(x_pos)-1])
ax3.set_xticklabels(['-2kb', 'TSS', 'Gene body', 'TES', '+2kb'])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'all_samples_integrated.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save data
output_data = {'bin_position': range(len(x_pos))}
for sample in common_samples:
    output_data[f'{sample}_VAF'] = vaf_profiles[sample]
    output_data[f'{sample}_5mC'] = meth_profiles[sample]['5mC']
    output_data[f'{sample}_5hmC'] = meth_profiles[sample]['5hmC']

pd.DataFrame(output_data).to_csv(os.path.join(out_dir, 'integrated_profiles.csv'), index=False)

print(f"\n✓ Analysis complete. Files saved to: {out_dir}/")
print(f"  Individual profiles: {len(common_samples)} samples")
print(f"  Multi-sample overlay: all_samples_integrated.png")
print(f"  Data table: integrated_profiles.csv")
