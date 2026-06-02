#!/bin/bash
# Self-update simulator usage example for DNATerra workflow
#   - Paired-end: fastp merge + bwa mem + samtools + statistics
#   - Single-end:  bwa mem + samtools + statistics
# Usage: run from dnaterra/input_dir, e.g.:
#   cd /path/to/dnaterra/input_dir && bash self_update_simulator_usage_example.sh

set -e

# ============================================================================
# User inputs
# ============================================================================
# Paired-end inputs
PE_REF="/path/to/reference.fasta"
PE_R1="/path/to/sample_R1.fastq"
PE_R2="/path/to/sample_R2.fastq"

# Single-end inputs
SE_REF="/path/to/reference.fasta"
SE_READS="/path/to/sample.fastq"

# Output directories
PE_OUT="/path/to/output/paired_end"
SE_OUT="/path/to/output/single_end"

# ============================================================================
# Threads
# ============================================================================
THREADS=8

mkdir -p "$PE_OUT" "$SE_OUT"

# ============================================================================
# Paired-end example: fastp merge + bwa mem + samtools + statistics
# Skip if PE_R1 or PE_R2 is empty
# ============================================================================
if [ -n "$PE_R1" ] && [ -n "$PE_R2" ] && [ -f "$PE_R1" ] && [ -f "$PE_R2" ]; then
    echo "=== Paired-end ==="

    bwa index "$PE_REF"

    fastp \
        -i "$PE_R1" \
        -I "$PE_R2" \
        --merge \
        --merged_out "$PE_OUT/merged.fastq" \
        --html "$PE_OUT/fastp.html" \
        --json "$PE_OUT/fastp.json" \
        -w $THREADS

    bwa mem -t $THREADS "$PE_REF" "$PE_OUT/merged.fastq" \
        | samtools view -Sb - \
        | samtools sort -o "$PE_OUT/aligned_sorted.bam"

    samtools index "$PE_OUT/aligned_sorted.bam"

    python3 sequencing_statistics.py \
        --bam "$PE_OUT/aligned_sorted.bam" \
        --ref "$PE_REF" \
        --name "paired_end_sample" \
        --output "$PE_OUT"

    echo "Paired-end done"
else
    echo "Skipping paired-end (PE_R1 or PE_R2 not set or not found)"
fi

# ============================================================================
# Single-end example: bwa mem + samtools + statistics
# Skip if SE_READS is empty
# ============================================================================
if [ -n "$SE_READS" ] && [ -f "$SE_READS" ]; then
    echo "=== Single-end ==="

    bwa index "$SE_REF"

    bwa mem -t $THREADS "$SE_REF" "$SE_READS" \
        | samtools view -Sb - \
        | samtools sort -o "$SE_OUT/aligned_sorted.bam"

    samtools index "$SE_OUT/aligned_sorted.bam"

    python3 sequencing_statistics.py \
        --bam "$SE_OUT/aligned_sorted.bam" \
        --ref "$SE_REF" \
        --name "single_end_sample" \
        --output "$SE_OUT"

    echo "Single-end done"
else
    echo "Skipping single-end (SE_READS not set or not found)"
fi

echo "All done!"
