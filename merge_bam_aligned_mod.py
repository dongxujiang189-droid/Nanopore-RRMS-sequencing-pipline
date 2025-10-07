import pysam

# Paths to your BAM files (WSL format)
unaligned_mod_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_merged.bam"
aligned_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned.sorted.bam"
output_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned_with_mod.bam"

# Open BAM files
unaligned = pysam.AlignmentFile(unaligned_mod_bam, "rb")
aligned = pysam.AlignmentFile(aligned_bam, "rb")
out = pysam.AlignmentFile(output_bam, "wb", template=aligned)

# Extract modification tags from unaligned BAM
mod_tags = {}
for read in unaligned.fetch(until_eof=True):
    mm = read.get_tag("MM") if read.has_tag("MM") else None
    ml = read.get_tag("ML") if read.has_tag("ML") else None
    if mm and ml:
        mod_tags[read.query_name] = (mm, ml)

# Add mod tags to aligned BAM
for read in aligned.fetch(until_eof=True):
    if read.query_name in mod_tags:
        mm, ml = mod_tags[read.query_name]
        read.set_tag("MM", mm)
        read.set_tag("ML", ml)
    out.write(read)

# Close BAMs
unaligned.close()
aligned.close()
out.close()

print("Finished writing aligned BAM with modifications!")
