#!/usr/bin/env python3
import pysam
import os

# --- Settings ---
bam_path = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned.sorted.bam"  # Update to your BAM file
output_bed = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_modifications.bed" 

# --- Open BAM file ---
bamfile = pysam.AlignmentFile(bam_path, "rb")

# --- Open output BED-like file ---
with open(output_bed, "w") as out:
    for read in bamfile.fetch():
        # Check if read has modified base tags (Mm/Ml)
        tags = dict(read.get_tags())
        if "Mm" in tags and "Ml" in tags:
            mm_tag = tags["Mm"]
            ml_tag = tags["Ml"]
            chrom = bamfile.get_reference_name(read.reference_id)
            start = read.reference_start
            # Write a simple BED-like line: chrom, start, end, read_name, modification_info
            out.write(f"{chrom}\t{start}\t{read.reference_end}\t{read.query_name}\t{mm_tag}|{ml_tag}\n")

bamfile.close()
print(f"[INFO] Methylation BED-like file saved to: {output_bed}")
