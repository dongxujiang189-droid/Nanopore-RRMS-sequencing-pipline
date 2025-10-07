import pysam

unaligned_mod_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_merged.bam"
aligned_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned.sorted.bam"
output_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned_with_mod.bam"

# Open unaligned BAM with check_sq=False
unaligned = pysam.AlignmentFile(unaligned_mod_bam, "rb", check_sq=False)
aligned = pysam.AlignmentFile(aligned_bam, "rb")
out = pysam.AlignmentFile(output_bam, "wb", template=aligned)

# Iterate and merge (example: copying MM/ML tags from unaligned to aligned)
for a in aligned:
    # Find corresponding read in unaligned BAM
    try:
        u = unaligned.fetch(a.query_name).__next__()
        # Copy modification tags if present
        if u.has_tag("MM"):
            a.set_tag("MM", u.get_tag("MM"))
        if u.has_tag("ML"):
            a.set_tag("ML", u.get_tag("ML"))
    except StopIteration:
        pass
    out.write(a)

unaligned.close()
aligned.close()
out.close()
