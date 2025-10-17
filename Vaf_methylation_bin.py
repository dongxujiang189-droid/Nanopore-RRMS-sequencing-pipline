#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pysam
import seaborn as sns

# Configuration
base_dir = "/mnt/c/Data/seq_for_human_293t2/"
vcf_pattern = os.path.join(base_dir, "clair3_output_variants calling", "*_clair3", "full_alignment.vcf.gz")
meth_pattern = os.path.join(base_dir, "modkit", "*_aligned_with_mod.region_mh.stats.tsv")
gtf_file = "/mnt/c/annotations/Homo_sapiens.GRCh38.gtf"
out_dir = os.path.join(base_dir, "mutation_dotplot_analysis")
os.makedirs(out_dir, exist_ok=True)

flank_size = 2000
n_gene_bins = 20  # Fewer bins for clearer visualization
n_flank_bins = 10
min_vaf = 0.02

def extract_sample_name(filepath):
    return os.path.basename(os.path.dirname(filepath)).replace("_clair3", "")

# Load genes
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
print(f"  {len(genes_df)} genes")

# Process VCF files - keep individual variants
vcf_files = glob.glob(vcf_pattern)
all_variants = []

for vcf_file in vcf_files:
    sample = extract_sample_name(vcf_file)
    print(f"\nProcessing {sample}...")
    
    vcf = pysam.VariantFile(vcf_file)
    
    for rec in vcf:
        if 'RefCall' in rec.filter or (rec.filter.keys() and 'PASS' not in rec.filter):
            continue
        
        for samp in rec.samples.values():
            af = samp.get('AF', None)
            if af:
                af = af[0] if isinstance(af, tuple) else af
                if af >= min_vaf:
                    all_variants.append({
                        'sample': sample,
                        'chrom': rec.chrom,
                        'pos': rec.pos,
                        'vaf': af
                    })
    
    vcf.close()

variants_df = pd.DataFrame(all_variants)
print(f"\nTotal variants: {len(variants_df)}")

# Assign bins to each variant
print("\nAssigning bins...")
upstream_bins = np.linspace(-flank_size, 0, n_flank_bins + 1)
genebody_bins = np.linspace(0, 100, n_gene_bins + 1)
downstream_bins = np.linspace(0, flank_size, n_flank_bins + 1)

bin_labels = []
bin_labels += [f"Up{i+1}" for i in range(n_flank_bins)]
bin_labels += [f"Gene{i+1}" for i in range(n_gene_bins)]
bin_labels += [f"Down{i+1}" for i in range(n_flank_bins)]

variants_with_bins = []

for _, var in tqdm(variants_df.iterrows(), total=len(variants_df), desc="Binning"):
    for _, gene in genes_df.iterrows():
        if var['chrom'] != gene['chrom']:
            continue
        
        chrom, tss, tes, strand = gene['chrom'], gene['tss'], gene['tes'], gene['strand']
        pos, vaf = var['pos'], var['vaf']
        
        if strand == '+':
            region_start, region_end = tss - flank_size, tes + flank_size
        else:
            region_start, region_end = tes - flank_size, tss + flank_size
        
        if not (min(region_start, region_end) <= pos <= max(region_start, region_end)):
            continue
        
        bin_idx = None
        
        if strand == '+':
            if pos < tss:
                dist = pos - tss
                idx = np.digitize(dist, upstream_bins) - 1
                if 0 <= idx < n_flank_bins:
                    bin_idx = idx
            elif tss <= pos <= tes:
                pct = 100 * (pos - tss) / (tes - tss) if tes != tss else 50
                idx = np.digitize(pct, genebody_bins) - 1
                if 0 <= idx < n_gene_bins:
                    bin_idx = n_flank_bins + idx
            else:
                dist = pos - tes
                idx = np.digitize(dist, downstream_bins) - 1
                if 0 <= idx < n_flank_bins:
                    bin_idx = n_flank_bins + n_gene_bins + idx
        else:
            if pos > tss:
                dist = pos - tss
                idx = np.digitize(dist, downstream_bins) - 1
                if 0 <= idx < n_flank_bins:
                    bin_idx = n_flank_bins - 1 - idx
            elif tes <= pos <= tss:
                pct = 100 * (tss - pos) / (tss - tes) if tss != tes else 50
                idx = np.digitize(pct, genebody_bins) - 1
                if 0 <= idx < n_gene_bins:
                    bin_idx = n_flank_bins + idx
            else:
                dist = tes - pos
                idx = np.digitize(dist, downstream_bins) - 1
                if 0 <= idx < n_flank_bins:
                    bin_idx = n_flank_bins + n_gene_bins + (n_flank_bins - 1 - idx)
        
        if bin_idx is not None:
            variants_with_bins.append({
                'sample': var['sample'],
                'bin_idx': bin_idx,
                'bin_label': bin_labels[bin_idx],
                'vaf': vaf
            })

binned_df = pd.DataFrame(variants_with_bins)
binned_df.to_csv(os.path.join(out_dir, 'variants_binned.csv'), index=False)
print(f"  Binned variants: {len(binned_df)}")

# Plot 1: Violin plot - distribution by sample and bin
print("\nCreating violin plot...")
samples = sorted(binned_df['sample'].unique())
total_bins = n_flank_bins + n_gene_bins + n_flank_bins

fig, ax = plt.subplots(figsize=(20, 8))

positions = []
data_groups = []
colors_list = []
sample_colors = plt.cm.Set3(np.linspace(0, 1, len(samples)))

for i, sample in enumerate(samples):
    for bin_idx in range(total_bins):
        data = binned_df[(binned_df['sample'] == sample) & 
                        (binned_df['bin_idx'] == bin_idx)]['vaf'].values
        if len(data) > 0:
            positions.append(bin_idx * (len(samples) + 1) + i)
            data_groups.append(data)
            colors_list.append(sample_colors[i])

parts = ax.violinplot(data_groups, positions=positions, widths=0.8, 
                      showmeans=True, showmedians=False)

for pc, color in zip(parts['bodies'], colors_list):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

# Mark TSS and TES
tss_pos = n_flank_bins * (len(samples) + 1)
tes_pos = (n_flank_bins + n_gene_bins) * (len(samples) + 1)
ax.axvline(tss_pos, color='black', linestyle='--', linewidth=2, alpha=0.7, label='TSS')
ax.axvline(tes_pos, color='black', linestyle='--', linewidth=2, alpha=0.7, label='TES')

ax.set_ylabel('VAF', fontweight='bold', fontsize=12)
ax.set_xlabel('Genomic Region', fontweight='bold', fontsize=12)
ax.set_title('Mutation Distribution Across Genomic Regions', fontweight='bold', fontsize=14)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=sample_colors[i], alpha=0.7, label=s) 
                  for i, s in enumerate(samples)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_xticks([i * (len(samples) + 1) + len(samples)/2 for i in range(total_bins)])
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mutation_violin_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Strip plot with jitter - individual dots
print("\nCreating strip plot...")
fig, ax = plt.subplots(figsize=(20, 8))

for i, sample in enumerate(samples):
    sample_data = binned_df[binned_df['sample'] == sample]
    x_jitter = sample_data['bin_idx'].values * (len(samples) + 1) + i + \
               np.random.normal(0, 0.15, len(sample_data))
    ax.scatter(x_jitter, sample_data['vaf'].values, 
              color=sample_colors[i], alpha=0.3, s=20, label=sample)

ax.axvline(tss_pos, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(tes_pos, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.text(tss_pos, ax.get_ylim()[1]*0.95, 'TSS', ha='center', fontweight='bold')
ax.text(tes_pos, ax.get_ylim()[1]*0.95, 'TES', ha='center', fontweight='bold')

ax.set_ylabel('VAF', fontweight='bold', fontsize=12)
ax.set_xlabel('Genomic Region', fontweight='bold', fontsize=12)
ax.set_title('Individual Mutations Across Genomic Regions', fontweight='bold', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_xticks([i * (len(samples) + 1) + len(samples)/2 for i in range(total_bins)])
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mutation_strip_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Box plot comparison
print("\nCreating box plot...")
fig, ax = plt.subplots(figsize=(20, 8))

box_data = []
box_positions = []
box_labels_plot = []

for bin_idx in range(total_bins):
    for i, sample in enumerate(samples):
        data = binned_df[(binned_df['sample'] == sample) & 
                        (binned_df['bin_idx'] == bin_idx)]['vaf'].values
        if len(data) > 0:
            box_data.append(data)
            box_positions.append(bin_idx * (len(samples) + 1) + i)
            box_labels_plot.append(sample)

bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True,
               showfliers=False)

for patch, pos in zip(bp['boxes'], box_positions):
    sample_idx = int((pos % (len(samples) + 1)))
    patch.set_facecolor(sample_colors[sample_idx])
    patch.set_alpha(0.7)

ax.axvline(tss_pos, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(tes_pos, color='black', linestyle='--', linewidth=2, alpha=0.7)

ax.set_ylabel('VAF', fontweight='bold', fontsize=12)
ax.set_xlabel('Genomic Region', fontweight='bold', fontsize=12)
ax.set_title('Mutation VAF Distribution (Box Plot)', fontweight='bold', fontsize=14)

legend_elements = [Patch(facecolor=sample_colors[i], alpha=0.7, label=s) 
                  for i, s in enumerate(samples)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_xticks([i * (len(samples) + 1) + len(samples)/2 for i in range(total_bins)])
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mutation_boxplot_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
for sample in samples:
    sample_data = binned_df[binned_df['sample'] == sample]
    print(f"\n{sample}:")
    print(f"  Total mutations: {len(sample_data)}")
    print(f"  Mean VAF: {sample_data['vaf'].mean():.4f}")
    print(f"  Median VAF: {sample_data['vaf'].median():.4f}")
    print(f"  VAF range: {sample_data['vaf'].min():.4f} - {sample_data['vaf'].max():.4f}")

print(f"\n✓ Complete: {out_dir}/")
