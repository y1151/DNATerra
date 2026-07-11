#!/bin/bash
# Example DNATerra simulation commands.
#
# Usage:
#   bash run_simulation.sh
#   bash run_simulation.sh normal
#   bash run_simulation.sh simple
#
# If no mode is provided, both normal and simple modes are run.

set -e
cd "$(dirname "$0")"

MODE="${1:-all}"

# Dynamically list available modes from this script
AVAILABLE_MODES=$(grep -E '^\s+[a-zA-Z_]+)\)[[:space:]]*$' "$0" | sed 's/)//g' | tr '\n' '|' | sed 's/|$//')
AVAILABLE_MODES="${AVAILABLE_MODES%|all}"  # remove leading 'all|' since it's the default

case "$MODE" in
    all)
        bash "$0" normal
        echo ""
        echo "========================================"
        echo ""
        bash "$0" simple
        ;;

    normal)
        echo ""
        echo ">>> Running: NORMAL mode"
        echo ""
        python main.py \
            -i input_dir/seq_n50000_l150.fasta \
            -o output_dir \
            --method PCR_15c_Twist_GCall \
            --target-read-depth 13.4 \
            --drop-rate 0.000083 \
            --dist gamma \
            --cv 0.2855 \
            --error-rate 3.260 0.038 0.370 \
            --num-workers 24 \
            --chunk-size 100000 \
            --use-kmer y \
            --shuffle y \
            --merge-files y \
            --stats y \
            --random-seed 42
        ;;

    simple)
        echo ""
        echo ">>> Running: SIMPLE mode"
        echo ""
        python main.py \
            --input input_dir/seq_n50_l150.fasta \
            --output output_dir/simple_mode \
            --simple_mode true \
            --ref_copy input_dir/ref_copy.txt \
            --read_error input_dir/read_error.txt \
            --num-workers 24 \
            --chunk-size 100000 \
            --seq-length 150 \
            --shuffle y \
            --merge-files y
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: ${AVAILABLE_MODES:-normal simple}"
        exit 1
        ;;
esac
