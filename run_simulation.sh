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

case "$MODE" in
    all)
        "$0" normal
        "$0" simple
        ;;

    normal)
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
        echo "Usage: bash run_simulation.sh [normal|simple]"
        exit 1
        ;;
esac
