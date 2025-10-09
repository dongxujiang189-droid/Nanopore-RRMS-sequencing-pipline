#!/usr/bin/env python3
import pysam
import pandas as pd
from tqdm import tqdm
import os

base_dir = "/mnt/e/Data/seq_for_human_293t2"
samples = ["barcode04", "barcode05", "barcode06", "barcode07"]

output_dir = os.path.join(base_dir, "mod_positions")
os.makedirs(output_dir, exist_ok=True)

for sample in samples:
    bam_file = os.path.join(base_dir, sample, f"{sample}_aligned_with_mod.bam")
    print(f"\n=== Processing {sample} ===")
    bam = pysam.AlignmentFile(bam_file, "rb")
    records = []

    for read in tqdm(bam.fetch(until_eof=True)):
        if not read.has_tag("MM") or not read.has_tag("ML"):
            continue
        mm = read.get_tag("MM")
        # Quick check for presence of 5mC / 5hmC indicators
        if ("C+m?" not in mm) and ("C+h?" not in mm):
            continue

        chrom = bam.get_reference_name(read.reference_id)
        start = read.reference_start
        if "C+m?" in mm:
            records.append([chrom, start, "5mC", sample])
        if "C+h?" in mm:
            records.append([chrom, start, "5hmC", sample])

    bam.close()
    df = pd.DataFrame(records, columns=["chr", "pos", "mod_type", "sample"])
    out_tsv = os.path.join(output_dir, f"mods_positions_{sample}.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved: {out_tsv} ({len(df)} records)")
