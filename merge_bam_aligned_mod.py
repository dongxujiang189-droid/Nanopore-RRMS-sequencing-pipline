import pysam

unaligned_mod_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_merged.bam"
aligned_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned.sorted.bam"
output_bam = "/mnt/e/Data/seq_for_human_293t2/barcode04/barcode04_aligned_with_mod.bam"

unaligned = pysam.AlignmentFile(unaligned_mod_bam, "rb", check_sq=False)
aligned = pysam.AlignmentFile(aligned_bam, "rb")
out = pysam.AlignmentFile(output_bam, "wb", template=aligned)

# Build a dictionary of unaligned reads
unaligned_dict = {}
for u in unaligned:
    unaligned_dict[u.query_name] = u

# Merge MM/ML tags
for a in aligned:
    u = unaligned_dict.get(a.query_name)
    if u:
        if u.has_tag("MM"):
            a.set_tag("MM", u.get_tag("MM"))
        if u.has_tag("ML"):
            a.set_tag("ML", u.get_tag("ML"))
    out.write(a)

unaligned.close()
aligned.close()
out.close()
