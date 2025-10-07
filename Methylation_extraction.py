#!/bin/bash

# --- 1. Set paths ---
BAM_PATH="/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned.sorted.bam"
OUTPUT_TSV="/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_methylation.tsv"
OUTPUT_HTML="/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_methylation.html"

# --- 2. Extract 5mC/5hmC from BAM ---
python3 << EOF
import pysam

bamfile = pysam.AlignmentFile("$BAM_PATH", "rb")
out = open("$OUTPUT_TSV", "w")
out.write("chrom\tpos\tstrand\tmod_type\tprob\n")

for read in bamfile.fetch():
    if read.has_tag("Mm") and read.has_tag("Ml"):
        mm_tag = read.get_tag("Mm")
        ml_tag = list(read.get_tag("Ml"))
        chrom = read.reference_name
        strand = "-" if read.is_reverse else "+"
        mods = mm_tag.split(";")
        idx = 0
        for mod in mods:
            if not mod: continue
            mod_type, positions = mod.split(",",1)
            positions = positions.split(",")
            for p in positions:
                p_int = int(p) + read.reference_start
                out.write(f"{chrom}\t{p_int}\t{strand}\t{mod_type}\t{ml_tag[idx]}\n")
                idx += 1

out.close()
bamfile.close()
EOF

# --- 3.
