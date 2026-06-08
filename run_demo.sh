#!/bin/bash
# =============================================================================
# DNATerra demo: end-to-end DNA storage pipeline
# encoding -> DNATerra simulation -> IEC correction -> decode -> verify
# =============================================================================

set -e
cd "$(dirname "$0")"

DEMO_DIR="demo/demo1"
INPUT_TAR="$DEMO_DIR/input_file.tar.gz"
ENCODED_FASTA="$DEMO_DIR/input_file.tar.gz.fasta"
SIM_OUT="$DEMO_DIR/sequencing_data"
SIM_FASTA="$SIM_OUT/output/input_file.tar.gz_merged.fasta"
CLUSTER_FASTA="$DEMO_DIR/sequencing_data_clustered.fasta"
IEC_OUT="$DEMO_DIR/iec_corrected_sequences.txt"
DECODED_TAR="$DEMO_DIR/output_file/input_file.tar.gz"
WORKERS=24

echo "============================================"
echo "DNATerra demo: encoding + sim + IEC + decode"
echo "============================================"

mkdir -p "$DEMO_DIR/output_file"

echo ""
echo "[setup] Packing input files..."
tar -czvf "$INPUT_TAR" -C "$DEMO_DIR" input_file
SIZE=$(stat --format="%s" "$INPUT_TAR")
PADDED_SIZE=$(( (SIZE + 511) / 512 * 512 ))
if [ "$SIZE" -ne "$PADDED_SIZE" ]; then
    echo "[setup] Padding $INPUT_TAR from $SIZE to $PADDED_SIZE bytes"
    truncate -s "$PADDED_SIZE" "$INPUT_TAR"
    SIZE=$PADDED_SIZE
fi
ORIG_MD5=$(md5sum "$INPUT_TAR" | awk '{print $1}')
echo "      input: $INPUT_TAR ($SIZE bytes)"

echo ""
echo "[setup] Building DNA Fountain Cython extensions..."
cd demo/dna_fountain
python3 setup.py build_ext --inplace
cd ../..

echo ""
echo "[1/6] DNA fountain encoding..."
time python3 demo/dna_fountain/encode.py \
    --file_in "$INPUT_TAR" \
    --size 32 -m 4 --gc 0.10 --rs 10 --delta 0.5 \
    --c_dist 0.8 --alpha 0.36 \
    --out "$ENCODED_FASTA"

SEQ_LEN=$(sed -n '2p' "$ENCODED_FASTA" | tr -d '\n' | wc -c)
echo "      encoded: $ENCODED_FASTA ($(grep -c '^>' "$ENCODED_FASTA") reads, len=$SEQ_LEN)"

echo ""
echo "[2/6] DNATerra simulation..."
time python main.py \
    -i "$ENCODED_FASTA" \
    -o "$SIM_OUT" \
    --method PCR_75c_Twist_GCall \
    --target-read-depth 6 \
    --drop-rate 0.0027 \
    --dist normal \
    --cv 0.350958 \
    --error-rate 10.270 0.048 0.633 \
    --num-workers "$WORKERS" \
    --chunk-size 100000 \
    --use-kmer y \
    --shuffle n \
    --merge-files y \
    --stats y \
    --random-seed 42 \
    --timestamp-suffix n

echo "      simulated: $SIM_FASTA"

echo ""
echo "[3/6] cd-hit clustering..."
time cd-hit-est \
    -i "$SIM_FASTA" \
    -o "$CLUSTER_FASTA" \
    -c 0.92 -n 8 -T "$WORKERS" -M 6000 -d 0
echo "      clusters: $(grep -c '^>' "$CLUSTER_FASTA") reads"

echo ""
echo "[4/6] IEC correction..."
time python demo/iec/iec_correction_cdhitest.py \
    -f "$SIM_FASTA" \
    -c "$CLUSTER_FASTA.clstr" \
    -l "$SEQ_LEN" \
    -o "$IEC_OUT" \
    -s 2 -ws 6 -cl 5 -ct 3 -t "$WORKERS" \
    --batch-clusters 1000 \
    --hamming-threshold 5 \
    --z-threshold 5
echo "      corrected: $IEC_OUT"

echo ""
echo "[5/6] Fountain decoding..."
CHUNKS=$(python3 -c "import math; S=$SIZE; print(math.ceil(S/32))")
time python demo/dna_fountain/decode.py \
    -f "$IEC_OUT" \
    -n "$CHUNKS" \
    --rs 10 --gc 0.10 -m 4 --delta 0.5 --c_dist 0.8 \
    --out "$DECODED_TAR"
DECODED_MD5=$(md5sum "$DECODED_TAR" | awk '{print $1}')
echo "      decoded: $DECODED_TAR"

echo ""
echo "[6/6] Verification..."
if [ "$ORIG_MD5" = "$DECODED_MD5" ]; then
    echo "      PASS - MD5 match: $ORIG_MD5"
else
    echo "      FAIL - original: $ORIG_MD5, decoded: $DECODED_MD5"
    exit 1
fi

echo ""
echo "============================================"
echo "Demo complete. Output in: $DEMO_DIR/"
echo "============================================"
