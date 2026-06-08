#!/bin/bash
# =============================================================================
# DNATerra demo — end-to-end DNA storage pipeline
# encoding → DNATerra simulation → IEC correction → decode → verify
#
# Requirements:
#   pip install -e .
#   conda install -c bioconda seqkit cd-hit -y
#   pip install -e ".[fountain]"       # reedsolo + cython
# =============================================================================

set -e
cd "$(dirname "$0")"

DEMO_DIR="demo1"
WORKERS=4

echo "============================================"
echo "DNATerra demo: encoding + sim + IEC + decode"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Generate test input
# ---------------------------------------------------------------------------
echo ""
echo "[1/7] Generating test input..."
mkdir -p "$DEMO_DIR/input_file"
dd if=/dev/urandom of="$DEMO_DIR/input_file/test.bin" bs=512 count=10 2>/dev/null
tar -czvf "$DEMO_DIR/input_file.tar.gz" -C "$DEMO_DIR" input_file
SIZE=$(stat -c%s "$DEMO_DIR/input_file.tar.gz")
ALIGNED=$(( (SIZE + 511) / 512 * 512 ))
truncate -s "$ALIGNED" "$DEMO_DIR/input_file.tar.gz"
echo "      input: $DEMO_DIR/input_file.tar.gz ($(stat -c%s $DEMO_DIR/input_file.tar.gz) bytes)"

# ---------------------------------------------------------------------------
# 2. DNA fountain encoding
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] DNA fountain encoding..."
cd demo/dna_fountain
python setup.py build_ext --inplace 2>&1 | tail -n1
cd ../..
ORIG_MD5=$(md5sum "$DEMO_DIR/input_file.tar.gz" | awk '{print $1}')
python demo/dna_fountain/encode.py --file_in "$DEMO_DIR/input_file.tar.gz" \
    --size 32 -m 4 --gc 0.10 --rs 10 --delta 0.5 \
    --c_dist 0.8 --alpha 0.36 --out "$DEMO_DIR/encoded.fasta"
SEQ_LEN=$(sed -n '2p' "$DEMO_DIR/encoded.fasta" | tr -d '\n' | wc -c)
echo "      encoded: $DEMO_DIR/encoded.fasta ($(grep -c '^>' $DEMO_DIR/encoded.fasta) reads, len=$SEQ_LEN)"

# ---------------------------------------------------------------------------
# 3. DNATerra simulation
# ---------------------------------------------------------------------------
echo ""
echo "[3/7] DNATerra simulation (depth=6, PCR_75c_Twist_GCall)..."
python main.py \
    -i "$DEMO_DIR/encoded.fasta" \
    -o "$DEMO_DIR/sequencing_data" \
    --method PCR_75c_Twist_GCall \
    --target-read-depth 6 --drop-rate 0.0027 \
    --dist normal --cv 0.351 --error-rate 10.27 0.048 0.633 \
    --num-workers $WORKERS --chunk-size 100000 \
    --shuffle n --merge-files y --stats y \
    --random-seed 42

SIM_FASTA=$(ls "$DEMO_DIR/sequencing_data/output/"*.fasta 2>/dev/null | head -1)
echo "      simulated: $SIM_FASTA"

# ---------------------------------------------------------------------------
# 4. cd-hit clustering
# ---------------------------------------------------------------------------
echo ""
echo "[4/7] cd-hit clustering..."
cd-hit-est -i "$SIM_FASTA" \
    -o "$DEMO_DIR/clustered.fasta" \
    -c 0.92 -n 8 -T $WORKERS -M 6000 -d 0
echo "      clusters: $(grep -c '^>' $DEMO_DIR/clustered.fasta) reads"

# ---------------------------------------------------------------------------
# 5. IEC correction
# ---------------------------------------------------------------------------
echo ""
echo "[5/7] IEC correction..."
python demo/iec/iec_correction_cdhitest.py \
    -f "$SIM_FASTA" \
    -c "$DEMO_DIR/clustered.fasta.clstr" \
    -l $SEQ_LEN \
    -o "$DEMO_DIR/iec_corrected.txt" \
    -s 2 -ws 6 -cl 5 -ct 3 -t $WORKERS \
    --hamming-threshold 5 --z-threshold 5 \
    --batch-clusters 1000
echo "      corrected: $DEMO_DIR/iec_corrected.txt"

# ---------------------------------------------------------------------------
# 6. Fountain decode
# ---------------------------------------------------------------------------
echo ""
echo "[6/7] Fountain decoding..."
CHUNKS=$(python3 -c "import math; print(math.ceil($SIZE/32))")
python demo/dna_fountain/decode.py \
    -f "$DEMO_DIR/iec_corrected.txt" \
    -n $CHUNKS --rs 10 --gc 0.10 -m 4 --delta 0.5 --c_dist 0.8 \
    --out "$DEMO_DIR/output_decoded.tar.gz"
DECODED_MD5=$(md5sum "$DEMO_DIR/output_decoded.tar.gz" | awk '{print $1}')
echo "      decoded: $DEMO_DIR/output_decoded.tar.gz"

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
echo ""
echo "[7/7] Verification..."
if [ "$ORIG_MD5" = "$DECODED_MD5" ]; then
    echo "      PASS — MD5 match: $ORIG_MD5"
else
    echo "      FAIL — original: $ORIG_MD5, decoded: $DECODED_MD5"
    exit 1
fi

echo ""
echo "============================================"
echo "Demo complete. Output in: $DEMO_DIR/"
echo "============================================"
