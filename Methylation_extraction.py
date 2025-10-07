#!/usr/bin/env python3
import pysam
import os
import sys

# --- Settings ---
bam_path = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned.sorted.bam"  # Path to your BAM
output_bed = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_modifications.bed"  # Output BED-like file

# --- Check if BAM exists ---
if not os.path.isfile(bam_path):
    print(f"[ERROR] BAM file not found: {bam_path}")
    sys.exit(1)

# --- Open BAM file ---
try:
    bamfile = pysam.AlignmentFile(bam_path, "rb")
except Exception as e:
    print(f"[ERROR] Could not open BAM file: {e}")
    sys.exit(1)

# --- Open output BED-like file ---
with open(output_bed, "w") as out:
    for read in bamfile.fetch():
        # Convert tags to dictionary
        tags = dict(read.get_tags())
        # Only process reads with Mm (modification string) and Ml (modification probabilities)
        if "Mm" in tags and "Ml" in tags:
            mm_tag = tags["Mm"]
            ml_tag = tags["Ml"]
            chrom = bamfile.get_reference_name(read.reference_id)
            start = read.reference_start
            end = read.reference_end
            # BED-like line: chrom, start, end, read_name, modification_info
            out.write(f"{chrom}\t{start}\t{end}\t{read.query_name}\t{mm_tag}|{ml_tag}\n")

bamfile.close()
print(f"[INFO] Methylation BED-like file saved to: {output_bed}")
