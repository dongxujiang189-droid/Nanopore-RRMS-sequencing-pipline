import pysam

# Paths
unaligned_mod_bam = "/mnt/e/Data/seq_for_human_293t2/barcode07/barcode07_merged.bam"
aligned_bam = "/mnt/e/Data/seq_for_human_293t2/barcode07/barcode07_aligned.sorted.bam"
output_bam = "/mnt/e/Data/seq_for_human_293t2/barcode07/barcode07_aligned_with_mod.bam"

# Open BAM files
unaligned = pysam.AlignmentFile(unaligned_mod_bam, "rb", check_sq=False)
aligned = pysam.AlignmentFile(aligned_bam, "rb")
out = pysam.AlignmentFile(output_bam, "wb", template=aligned)

# Build a dictionary of MM/ML tags only (smaller memory footprint)
unaligned_tags = {}
for u in unaligned.fetch(until_eof=True):
    tags = {}
    if u.has_tag("MM"):
        tags["MM"] = u.get_tag("MM")
    if u.has_tag("ML"):
        tags["ML"] = u.get_tag("ML")
    if tags:
        unaligned_tags[u.query_name] = tags

print(f"Loaded modification tags for {len(unaligned_tags)} reads.")

# Merge MM/ML tags into aligned BAM
for a in aligned.fetch(until_eof=True):
    tags = unaligned_tags.get(a.query_name)
    if tags:
        for tag_name, tag_value in tags.items():
            a.set_tag(tag_name, tag_value)
    out.write(a)

unaligned.close()
aligned.close()
out.close()

print(f"Finished merging modifications. Output saved to {output_bam}")
