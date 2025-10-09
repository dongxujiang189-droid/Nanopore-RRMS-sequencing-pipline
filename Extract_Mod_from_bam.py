#!/usr/bin/env python3
import pysam
import pandas as pd
from tqdm import tqdm
import os
import pybedtools

base_dir = "/mnt/e/Data/seq_for_human_293t2"
samples = ["barcode04", "barcode05", "barcode06", "barcode07"]

output_dir = os.path.join(base_dir, "mod_positions")
os.makedirs(output_dir, exist_ok=True)

gtf_file = "/mnt/e/annotations/Homo_sapiens.GRCh38.gtf"  # make sure GTF exists

all_gene_enrichment = []

for sample in samples:
    bam_file = os.path.join(base_dir, sample, f"{sample}_aligned_with_mod.bam")
    print(f"\n=== Processing {sample} ===")
    bam = pysam.AlignmentFile(bam_file, "rb")
    records = []

    # Extract modifications
    for read in tqdm(bam.fetch(until_eof=True)):
        if not read.has_tag("MM") or not read.has_tag("ML"):
            continue

        mm = read.get_tag("MM")
        chrom = bam.get_reference_name(read.reference_id)
        start = read.reference_start

        # Skip invalid coordinates
        if chrom is None or start is None or start < 0:
            continue

        if "C+m?" in mm:
            records.append([chrom, start, start + 1, "5mC"])
        if "C+h?" in mm:
            records.append([chrom, start, start + 1, "5hmC"])

    bam.close()

    if not records:
        print(f"No valid modification records found for {sample}")
        continue

    # Save BED file
    bed_df = pd.DataFrame(records, columns=["chr", "start", "end", "mod_type"])
    bed_file = os.path.join(output_dir, f"mods_positions_{sample}.bed")
    bed_df.to_csv(bed_file, sep="\t", header=False, index=False)
    print(f"Saved BED: {bed_file} ({len(bed_df)} records)")

    # Intersect with GTF to assign mods to genes
    mods = pybedtools.BedTool(bed_file)
    genes = pybedtools.BedTool(gtf_file)

    intersect = mods.intersect(genes, wa=True, wb=True)

    # Convert intersect result to DataFrame
    cols = ["chr", "start", "end", "mod_type",
            "gene_chr", "source", "feature", "gene_start", "gene_end",
            "score", "strand", "frame", "attributes"]
    intersect_df = pd.read_csv(intersect.fn, sep="\t", names=cols, header=None)

    # Extract gene_id from attributes
    intersect_df["gene_id"] = intersect_df["attributes"].str.extract('gene_id "([^"]+)"')

    # Compute per-gene enrichment
    gene_summary = intersect_df.groupby(["gene_id", "mod_type"]).size().unstack(fill_value=0)
    gene_summary["sample"] = sample
    all_gene_enrichment.append(gene_summary.reset_index())

# Combine all samples
all_gene_df = pd.concat(all_gene_enrichment, axis=0)
all_gene_df.to_csv(os.path.join(output_dir, "per_gene_mods_all_samples.tsv"), sep="\t", index=False)
print("Saved combined per-gene modification table.")


