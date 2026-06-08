#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DNA Sequence Error Simulator - Parallel Version - Main Entry"""

import sys
import os
import importlib.util
from pathlib import Path
import logging
import time
import numpy as np

# Configure logging system (must run before importing other modules)
def setup_logging(level=logging.WARNING):
    """Configure logging system"""
    # Override time module converter to use local timezone
    logging.Formatter.converter = time.localtime

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    return root_logger

# Initialize logging
logger = setup_logging()

# Detect whether running as module or direct script
# __name__ == "__main__" handles multiprocessing spawn case (spawn uses runpy.run_path,
# which sets __package__ but __name__ becomes "__mp_main__", so we still take the absolute import branch)
if __package__ is None or __name__ in ("__main__", "__mp_main__"):
    # Direct script execution (python main.py) or runpy.run_path, use absolute imports
    # Add project parent to Python path. If the source directory is not named
    # "dnaterra" (for example GitHub zip extracts to DNATerra-main), register the
    # current directory as the dnaterra package for this process.
    _package_root = Path(__file__).resolve().parent
    _project_root = _package_root.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    if importlib.util.find_spec("dnaterra") is None:
        _package_init = _package_root / "__init__.py"
        _spec = importlib.util.spec_from_file_location(
            "dnaterra",
            _package_init,
            submodule_search_locations=[str(_package_root)],
        )
        _module = importlib.util.module_from_spec(_spec)
        sys.modules["dnaterra"] = _module
        _spec.loader.exec_module(_module)
    
    from dnaterra import (
        # Configuration
        SynthesisConfig,
        get_synthesis_method_display_name,
        get_synthesis_method_short_name,
        load_synthesis_config,
        resize_error_rates_spline,
        build_substitution_lookup_from_matrix,
        # Coverage
        count_sequences_with_seqkit,
        calculate_depth_distribution_params,
        split_refs_into_chunks_streaming,
        write_ref_count_and_read_to_ref_tsv,
        generate_shuffle_mapping,
        write_shuffle_chunks_parallel,
        # Errors
        load_error_rates_from_config,
        compute_error_counts_per_position,
        sample_errors_from_bucket,
        compute_bucket_metadata,
        cleanup_orphaned_shared_memory,
        cleanup_orphaned_temp_dirs,
        # Mutations
        apply_substitutions,
        sample_substitutions_kmer,
        apply_substitutions_kmer,
        apply_indels_and_write,
        preallocate_output_file,
        # Utilities
        get_timestamp_string,
        get_merge_filename,
        detect_fasta_sequence_length,
        preview_fasta_file,
        run_batch_tests,
        process_error_rate_input,
        build_summary_item,
        build_print_output,
        validate_fasta_format_and_length,
        # Main workflow
        error_simulator_worker,
        parallel_simulate_errors,
        # Simple mode
        simple_mode_worker,
        parallel_simulate_errors_simple_mode,
        # Constants
        DEFAULT_CHUNK_SIZE,
        DEFAULT_TOTAL_CPUS,
        SEQKIT_MAX_THREADS,
        DEFAULT_DEPTH_DISTRIBUTION,
    )
else:
    # Import as package, use relative imports
    from . import (
        # Configuration
        SynthesisConfig,
        get_synthesis_method_display_name,
        get_synthesis_method_short_name,
        load_synthesis_config,
        resize_error_rates_spline,
        build_substitution_lookup_from_matrix,
        # Coverage
        count_sequences_with_seqkit,
        calculate_depth_distribution_params,
        split_refs_into_chunks_streaming,
        write_ref_count_and_read_to_ref_tsv,
        generate_shuffle_mapping,
        write_shuffle_chunks_parallel,
        # Errors
        load_error_rates_from_config,
        compute_error_counts_per_position,
        sample_errors_from_bucket,
        compute_bucket_metadata,
        cleanup_orphaned_shared_memory,
        cleanup_orphaned_temp_dirs,
        # Mutations
        apply_substitutions,
        sample_substitutions_kmer,
        apply_substitutions_kmer,
        apply_indels_and_write,
        preallocate_output_file,
        # Utilities
        get_timestamp_string,
        get_merge_filename,
        detect_fasta_sequence_length,
        preview_fasta_file,
        run_batch_tests,
        process_error_rate_input,
        build_summary_item,
        build_print_output,
        validate_fasta_format_and_length,
        # Main workflow
        error_simulator_worker,
        parallel_simulate_errors,
        # Simple mode
        simple_mode_worker,
        parallel_simulate_errors_simple_mode,
        # Constants
        DEFAULT_CHUNK_SIZE,
        DEFAULT_TOTAL_CPUS,
        SEQKIT_MAX_THREADS,
        DEFAULT_DEPTH_DISTRIBUTION,
    )

    # Get logger
logger = logging.getLogger(__name__)


def main():
    """Main function - supports single-file mode and batch-test mode"""
    import argparse
    import platform

    # ====== Environment Detection ======
    if platform.system() == 'Windows':
        print("=" * 80)
        print("WARNING: This program only supports Linux/macOS, not Windows.")
        print("Running on Windows may cause crashes or incorrect results.")
        print("Please use Linux or macOS to run this program.")
        print("=" * 80)

    # ====== Program Title ======
    print("=" * 80)
    print("DNATerra: a computational prelude to large-scale DNA storage")
    print("=" * 80)
    print()

    parser = argparse.ArgumentParser(
        description='DNA Sequence Error Simulator - Parallel Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Synthesis Methods:
  This tool supports three DNA synthesis methods:
  
  inkjet          Inkjet-based DNA synthesis (Twist, 155bp)
                  Inkjet-based DNA synthesis: high-throughput chemical synthesis that delivers reagents to specific chip locations via jet deposition
  
  electro        Electrochemical DNA synthesis (GenScript, 130bp)
                Electrochemical DNA synthesis: high-throughput chemical synthesis that triggers localized deprotection at specific chip locations via microelectrodes
  
  photo          Photochemical DNA synthesis
                Photochemical DNA synthesis: high-throughput chemical synthesis that triggers localized deprotection at specific chip locations via photomasks or DMD controlling specific-wavelength light

Sequencing Method:
  This tool uses Illumina sequencing

Run Modes:
  1. Single-file mode: only --input is required (input FASTA); -o is optional, specifies output directory (defaults to output_dir when not specified)
  2. Batch-test mode: omit --input, automatically processes test files in test_sequences directory

Examples:
  # Single-file mode - basic usage (output to default output_dir)
  python main.py --input test.fasta
  
  # Single-file mode - specify output directory
  python main.py --input test.fasta -o my_output
  
  # Single-file mode - specify synthesis method and number of reads
  python main.py --input test.fasta -o my_output --method inkjet --target-num-reads 540000
  
  # Single-file mode - specify worker count and chunk size
  python main.py --input test.fasta -o my_output --num-workers 64 --chunk-size 3000000

  # Batch-test mode - use default parameters (220.10x read depth)
  python main.py
  
  # Batch-test mode - specify synthesis method and read depth
  python main.py --method electro --target-read-depth 100.0
        '''
    )
    
    # Input/Output Arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input FASTA file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (must be a directory; defaults to output_dir under current directory if not specified). Output filename is auto-generated based on input'
    )
    
    # Synthesis Method Arguments
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='inkjet',
        help='DNA synthesis method identifier (used to find corresponding .npz config file in input_dir, e.g., inkjet, electro, photo, PCR_15c_Genscript_GCall, etc.)'
    )
    
    # Performance Arguments
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_TOTAL_CPUS,
        help=f'Number of available CPUs (default: {DEFAULT_TOTAL_CPUS})'
    )
    parser.add_argument(
        '--cpu',
        type=int,
        default=None,
        help=f'Total CPU count, overrides --num-workers and its default (default: {DEFAULT_TOTAL_CPUS})'
    )
    
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=None,
        help=f'Chunk size (weighted number of sequences, auto-calculated by default, recommended: {DEFAULT_CHUNK_SIZE:,})'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--seq-length', '-l',
        type=int,
        default=None,
        help='Sequence length in bp. If specified, validated against the actual length of the input file'
    )
    
    parser.add_argument(
        '--ref-count', '-r',
        type=str,
        default=None,
        help='Number of ref sequences (must be a plain positive integer without commas). If specified, validated against the actual sequence count in the input file'
    )
    
    parser.add_argument(
        '--target-read-depth', '-d',
        type=float,
        default=None,
        help='Target read depth (average reads per ref, e.g., 220.10)'
    )

    parser.add_argument(
        '--dist',
        type=str,
        default=None,
        help='Sequencing coverage depth distribution type: lognormal, gamma, normal, weibull, exponential, poisson, uniform, nbinom, beta'
    )

    parser.add_argument(
        '--cv',
        type=float,
        default=None,
        help='Coefficient of variation for sequencing coverage depth (CV = std/mean). Not needed for Poisson and Exponential distributions'
    )

    parser.add_argument(
        '--drop-rate',
        type=float,
        default=None,
        help='Dropout rate: randomly set this proportion of refs to counts=0 (defaults to the dropout rate from the config file)'
    )

    parser.add_argument(
        '--beta-min',
        type=float,
        default=None,
        help='Minimum value for Beta distribution (only used with --dist beta)'
    )

    parser.add_argument(
        '--beta-max',
        type=float,
        default=None,
        help='Maximum value for Beta distribution (only used with --dist beta)'
    )
    
    parser.add_argument(
        '--error-rate', '-e',
        type=str,
        nargs='+',
        default=None,
        help='Error rate parameter, unit: 10^-3 per nucleotide (0.1%% error rate = 1.0). Supports two formats:\n'
             '1. Single number: total error rate (e.g., 1.0 means 0.1%%)\n'
             '2. Three numbers: substitution, insertion, deletion error rates (e.g., 0.5 0.3 0.2)'
    )

    # Kmer (multi-base preference) related arguments
    parser.add_argument(
        '--use-kmer',
        type=str,
        default='n',
        help='Enable kmer (multi-base preference) error matrix (y/n). When enabled, uses kmer matrix where full 5mer context is available, falls back to single-base preference at boundaries'
    )
    parser.add_argument(
        '--merge-files',
        type=str,
        default=None,
        help='Output mode: multiple chunks or single fasta. Pass y to enable merging: zero-copy merge of ordered/shuffled chunks into *_merged_*.fasta and *_shuffled_merged_*.fasta respectively (chunks deleted after merge); default is no merge, only chunk files are kept'
    )
    parser.add_argument(
        '--shuffle',
        type=str,
        default='n',
        help='Whether to shuffle sequences (y/n). Default n: skip shuffle mapping and shuffle merge'
    )
    
    # simple_mode (user-defined explicit mode) arguments
    parser.add_argument(
        '--simple_mode',
        type=str,
        default=None,
        help='User-defined explicit mode: true/false. When true, enables explicit amplification via ref_copy.txt and explicit errors via read_error.txt, without deriving error model from experimental conditions'
    )
    parser.add_argument(
        '--ref_copy',
        type=str,
        default=None,
        help='Path to ref_copy.txt (only effective when --simple_mode true). Each line: seq_index copy_count (1-based seq_index)'
    )
    parser.add_argument(
        '--read_error',
        type=str,
        default=None,
        help='Path to read_error.txt (only effective when --simple_mode true). Each line: seq_index pos type base (all 1-based; type=S/I/D)'
    )
    parser.add_argument(
        '--timestamp-suffix',
        type=str,
        default='n',
        help='Add timestamp suffix to output directory and merge filenames (y/n). Default n (no suffix)'
    )
    parser.add_argument(
        '--stats',
        type=str,
        default='n',
        help='Whether to output statistics files. Format: three y/n values separated by commas, controlling ref_count.tsv, read_to_ref_ordered.tsv, read_to_ref_shuffled.tsv respectively.'
             'Example: --stats y,y,n outputs the first two. Default n (no output). Skipping output avoids CIGAR/MD collection and TSV writing, improving performance'
    )
    
    args = parser.parse_args()
    
    # ====== Argument Validation ======
    # Note: --method parameter now accepts any value, used to find corresponding .npz config file in input_dir
    # If the corresponding file is not found, an error will be raised in a later step
    
    # 2. CPU count handling: if --cpu is provided, it takes priority
    if args.cpu is not None:
        args.num_workers = args.cpu
    
    # Expected parallel process count validation
    if args.num_workers < 4:
        parser.error(f"Expected number of parallel processes must be at least 4, current value: {args.num_workers}")
    if not isinstance(args.num_workers, int) or args.num_workers <= 0:
        parser.error(f"Expected number of parallel processes must be a positive integer, current value: {args.num_workers}")
    
    # 3. Chunk size validation
    if args.chunk_size is not None:
        if not isinstance(args.chunk_size, int) or args.chunk_size <= 0:
            parser.error(f"Chunk size must be a positive integer, current value: {args.chunk_size}")
    
    # 4. Target read depth validation
    if args.target_read_depth is not None:
        if args.target_read_depth <= 0:
            parser.error(f"Target read depth must be greater than 0, cannot be negative or zero, current value: {args.target_read_depth}")
    
    # 5. Distribution type and CV validation
    DIST_NO_CV = {'exponential', 'poisson'}
    if args.dist is not None:
        args.dist = args.dist.lower()
        if args.dist not in DIST_NO_CV and args.cv is None:
            parser.error(f"Distribution '{args.dist}' requires --cv parameter (coefficient of variation)")
        if args.dist in DIST_NO_CV and args.cv is not None:
            parser.error(f"Distribution '{args.dist}' does not need --cv parameter; CV is automatically determined from the mean")
        if args.target_read_depth is None:
            parser.error(f"--target-read-depth (or --target-num-reads) must be provided when --dist is specified")
        # Beta distribution validation: if one of beta-min/beta-max is provided, the other must also be provided; if neither is provided, read from DEFAULT_DEPTH_DISTRIBUTION
        if args.dist == 'beta':
            if (args.beta_min is None) != (args.beta_max is None):
                parser.error(f"Distribution 'beta' requires both --beta-min and --beta-max to be specified together, or both omitted (using default range)")
            if args.beta_min is not None and args.beta_max is not None and args.beta_min >= args.beta_max:
                parser.error(f"--beta-min ({args.beta_min}) must be less than --beta-max ({args.beta_max})")
    elif args.cv is not None:
        parser.error("--cv parameter must be used with --dist")
    
    # 6. Target total reads validation
    if args.seq_length is not None:
        if not isinstance(args.seq_length, int):
            parser.error(f"Ref sequence length must be an integer, current value: {args.seq_length}")
        if args.seq_length <= 0:
            parser.error(f"Ref sequence length must be a positive integer, cannot be 0 or negative, current value: {args.seq_length}")
        if args.seq_length > 200:
            logger.warning(f"Ref sequence length ({args.seq_length} bp) exceeds 200bp; simulation may be inaccurate.")
        elif args.seq_length < 30:
            logger.warning(f"Ref sequence length ({args.seq_length} bp) is less than 30bp; simulation may crash or produce inaccurate results.")
    
    # 7.5 Output mode: multiple chunks or single fasta (--merge-files y merges ordered+shuffled chunks into two single FASTA files)
    args.merge_files_enabled = False
    if args.merge_files is not None:
        v = str(args.merge_files).strip().lower()
        if v in ('y', 'yes'):
            args.merge_files_enabled = True
        elif v in ('n', 'no', ''):
            args.merge_files_enabled = False
        else:
            parser.error(f"--merge-files only supports y or yes to enable merge, n or no to disable merge, current value: {args.merge_files}")

    # 7.6 Sequence shuffle (--shuffle y enables shuffle mapping and shuffle merge)
    args.shuffle_enabled = False
    if args.shuffle is not None:
        v = str(args.shuffle).strip().lower()
        if v in ('y', 'yes'):
            args.shuffle_enabled = True
        elif v in ('n', 'no', ''):
            args.shuffle_enabled = False
        else:
            parser.error(f"--shuffle only supports y or n, current value: {args.shuffle}")

    # 7.7 use_kmer parameter parsing (string y/n -> bool)
    args.use_kmer_enabled = False
    if args.use_kmer is not None:
        v = str(args.use_kmer).strip().lower()
        if v in ('y', 'yes'):
            args.use_kmer_enabled = True
        elif v in ('n', 'no', ''):
            args.use_kmer_enabled = False
        else:
            parser.error(f"--use-kmer only supports y or n, current value: {args.use_kmer}")

    # 7.8 timestamp_suffix parameter parsing (string y/n -> bool)
    args.timestamp_suffix_enabled = False
    if args.timestamp_suffix is not None:
        v = str(args.timestamp_suffix).strip().lower()
        if v in ('y', 'yes'):
            args.timestamp_suffix_enabled = True
        elif v in ('n', 'no', ''):
            args.timestamp_suffix_enabled = False
        else:
            parser.error(f"--timestamp-suffix only supports y or n, current value: {args.timestamp_suffix}")

    # 7.9 stats parameter parsing (string y/n -> bool)
    args.output_stats = False
    if args.stats is not None:
        v = str(args.stats).strip().lower()
        if v in ('y', 'yes'):
            args.output_stats = True
        elif v in ('n', 'no', ''):
            args.output_stats = False
        else:
            parser.error(f"--stats only supports y or n, current value: {args.stats}")

    # 7.10 simple_mode parameter parsing (string true/false -> bool)
    args.simple_mode_enabled = False
    if args.simple_mode is not None:
        v = str(args.simple_mode).strip().lower()
        if v in ("true", "t", "1", "y", "yes"):
            args.simple_mode_enabled = True
        elif v in ("false", "f", "0", "n", "no", ""):
            args.simple_mode_enabled = False
        else:
            parser.error(f"--simple_mode only supports true/false (case-insensitive), current value: {args.simple_mode}")
    
    # 7. Ref sequence count validation (if user specified it)
    user_input_ref_count = None
    if args.ref_count is not None:
        # Must be a plain number without commas
        ref_count_str = args.ref_count.strip()
        if ',' in ref_count_str or not ref_count_str.isdigit():
            parser.error(f"Ref sequence count must be a plain number without commas, current value: {args.ref_count}")
        
        # Convert to integer and validate as positive integer
        try:
            user_input_ref_count = int(ref_count_str)
            if user_input_ref_count <= 0:
                parser.error(f"Ref sequence count must be a positive integer, cannot be 0 or negative, current value: {user_input_ref_count}")
        except ValueError:
            parser.error(f"Ref sequence count must be an integer, current value: {args.ref_count}")
    
    # Determine run mode: single-file mode only requires --input
    if args.input is not None:
        # ====== Single-file Mode ======
        # -o is optional; if specified it must be a directory (cannot be a file path)
        if args.output is not None:
            out_path = Path(args.output)
            if out_path.exists() and out_path.is_file():
                parser.error("--output must specify a directory, cannot be a file path")
        
        # Check if input file exists
        if not Path(args.input).exists():
            parser.error(f"Input file does not exist: {args.input}")

        # Set output directory
        if args.output is None:
            output_dir = Path("output_dir")
        else:
            output_dir = Path(args.output) / (f"output_{get_timestamp_string()}" if args.timestamp_suffix_enabled else "output")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = str(output_dir)

        # Check for leftover files in output directory
        if output_dir.exists():
            old_files = list(output_dir.iterdir())
            if old_files:
                import shutil
                for f in old_files:
                    try:
                        if f.is_dir():
                            shutil.rmtree(f)
                        else:
                            f.unlink()
                    except Exception as e:
                        logger.warning(f"  Failed to clean up leftover file {f}: {e}")
                remaining = list(output_dir.iterdir())
                if remaining:
                    logger.warning(f"  {len(remaining)} files/directories still cannot be cleaned: {remaining}")

        # ====== Simple mode: early jump, skip all normal-mode initialization ======
        if getattr(args, "simple_mode_enabled", False):
            import time as _time
            _start_time = _time.time()
            stats = parallel_simulate_errors_simple_mode(
                input_fasta=args.input,
                output_dir=final_output_path,
                seq_length=args.seq_length,
                chunk_size=args.chunk_size,
                random_seed=args.random_seed,
                read_id_offset=1,
                target_num_chunks=None,
                num_workers_global=args.num_workers,
                merge_files_enabled=getattr(args, 'merge_files_enabled', False),
                command_line=" ".join(sys.argv),
                ref_copy_path=args.ref_copy,
                read_error_path=args.read_error,
                timestamp_suffix=args.timestamp_suffix_enabled,
                output_stats=args.output_stats,
            )
            elapsed = _time.time() - _start_time
            return 0

        # Detect input file sequence length (based on the input file itself)
        # Note: detect_fasta_sequence_length already validates file format and length; no need to re-validate type
        detected_seq_length = detect_fasta_sequence_length(Path(args.input))
        
        # Length range warnings (warnings only, do not block execution)
        if detected_seq_length > 200:
            logger.warning(f"Ref sequence length ({detected_seq_length} bp) exceeds 200bp; simulation may be inaccurate.")
        elif detected_seq_length < 30:
            logger.warning(f"Ref sequence length ({detected_seq_length} bp) is less than 30bp; simulation may crash or produce inaccurate results.")
        
        # If user specified length, compare and validate
        if args.seq_length is not None:
            if args.seq_length != detected_seq_length:
                logger.warning(f"User-specified sequence length ({args.seq_length} bp) does not match the actual length of the input file ({detected_seq_length} bp); simulation will use the actual length of the input file.")

        # Actual sequence length used (based on input file)
        actual_seq_length = detected_seq_length
        
        # Handle target read parameters
        # If read depth or total reads not specified, use default read depth (220.10x)
        if args.target_read_depth is None:
            args.target_read_depth = 220.10
            print(f"[Default] Using 220.10x read depth")
        
        # ====== Stage 1: FASTA Validation and Config Loading ======
        script_dir = Path(__file__).parent  # Points to dnaterrasim package directory

        try:
            detected_seq_length = validate_fasta_format_and_length(args.input)
        except Exception as e:
            logger.error(f"FASTA validation failed, program terminated")
            raise

        # Data type safety check
        if detected_seq_length > 65535:
            error_msg = (
                f"Error: Sequence length ({detected_seq_length}bp) exceeds uint16 maximum (65,535bp).\n"
                f"The current code uses uint16 to store error positions, and does not support sequences longer than 65535bp."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Build CLI coverage depth parameter dict (used instead of npz file)
        cli_coverage_params = None
        if args.dist is not None:
            mean_depth = args.target_read_depth
            cli_coverage_params = {
                'mean_depth': mean_depth,
                'cv': args.cv,
                'dist': args.dist,
            }
            if args.dist == 'beta':
                method_default = DEFAULT_DEPTH_DISTRIBUTION.get(args.method, {})
                cli_coverage_params['beta_min'] = args.beta_min if args.beta_min is not None else method_default.get('beta_min')
                cli_coverage_params['beta_max'] = args.beta_max if args.beta_max is not None else method_default.get('beta_max')

        # Load synthesis config
        synthesis_config = load_synthesis_config(
            args.method,
            script_dir,
            target_seq_length=detected_seq_length,
            use_kmer=args.use_kmer_enabled,
            cli_coverage_params=cli_coverage_params,
            drop_rate=args.drop_rate,
        )
        # If user did not specify dropout rate, use the dropout_rate from config file (percentage form, already divided by 100)
        effective_drop_rate = args.drop_rate if args.drop_rate is not None else synthesis_config.get('dropout_rate', 0.0)
        if args.drop_rate is None:
            config_seq_length = synthesis_config['seq_length']
            if config_seq_length != detected_seq_length:
                print(f"  Scaled sequence length: {config_seq_length} bp (scaled from {detected_seq_length} bp)")
            seq_length = config_seq_length
        else:
            seq_length = detected_seq_length

        # ====== Stage 2: Count Sequences, Sample Distribution, Chunk Splitting ======
        seqkit_threads = min(SEQKIT_MAX_THREADS, args.num_workers - 2)
        num_ref_seqs = count_sequences_with_seqkit(args.input, num_threads=seqkit_threads)

        # Calculate sampling distribution parameters
        # Prefer using the pre-calculated distribution parameters from load_synthesis_config (when user specifies --dist via CLI)
        if synthesis_config.get('depth_dist_info') is not None:
            depth_dist_info = synthesis_config['depth_dist_info']
        elif args.dist is not None:
            # Fallback: direct calculation (theoretically should not reach here because cli_coverage_params is already passed)
            mean_depth = args.target_read_depth * (1 - effective_drop_rate)
            method_default = DEFAULT_DEPTH_DISTRIBUTION.get(args.method, {})
            depth_dist_info = calculate_depth_distribution_params(
                args.dist, mean_depth, args.cv,
                beta_min=args.beta_min if args.beta_min is not None else method_default.get('beta_min'),
                beta_max=args.beta_max if args.beta_max is not None else method_default.get('beta_max'),
            )
        else:
            # Old logic: when --dist is not specified at all, read from config file
            default_config = DEFAULT_DEPTH_DISTRIBUTION.get(args.method)
            avg_coverage = synthesis_config['depth_params']['avg_coverage']
            cv = synthesis_config['depth_params']['cv']
            if default_config is None:
                depth_dist_info = calculate_depth_distribution_params(
                    'lognormal', avg_coverage, cv=cv
                )
            else:
                dist_name = default_config['dist']
                cv_val = default_config['cv']
                if args.target_read_depth is None:
                    target_depth = avg_coverage
                elif abs(args.target_read_depth - avg_coverage) > 0.1:
                    target_depth = args.target_read_depth
                else:
                    target_depth = avg_coverage
                depth_dist_info = calculate_depth_distribution_params(
                    dist_name, target_depth, cv_val,
                    beta_min=args.beta_min if args.beta_min is not None else default_config.get('beta_min'),
                    beta_max=args.beta_max if args.beta_max is not None else default_config.get('beta_max'),
                )

        dist_name = depth_dist_info['dist_name']
        dist_params = depth_dist_info['params']

        # Calculate chunk_size
        if args.chunk_size is not None:
            chunk_size = args.chunk_size
        else:
            if int(num_ref_seqs * args.target_read_depth) < DEFAULT_CHUNK_SIZE:
                chunk_size = int(num_ref_seqs * args.target_read_depth)
            else:
                chunk_size = DEFAULT_CHUNK_SIZE

        # Generate chunk metadata
        chunk_metadata_list, split_dir, total_reads, num_nonzero = split_refs_into_chunks_streaming(
            num_ref_seqs=num_ref_seqs,
            target_read_depth=args.target_read_depth,
            dist_name=dist_name,
            dist_params=dist_params,
            random_seed=args.random_seed,
            chunk_size=chunk_size,
            split_dir=output_dir / 'split',
            num_parallel_workers=args.num_workers - 1,
            target_num_chunks=None,
            total_cpus=args.num_workers,
            drop_rate=effective_drop_rate,
            seq_length=seq_length,
        )

        # Calculate fixed width for sequence IDs
        max_reads_seq_global_id = 1 + total_reads - 1
        reads_seq_id_width = len(str(max_reads_seq_global_id))

        total_cpus = args.num_workers

        # ====== Stage 3: Error Rate Processing ======
        custom_position_rates = None
        error_rate_input_type = None
        user_input_total_error_rate = None
        user_input_sub_error_rate = None
        user_input_ins_error_rate = None
        user_input_del_error_rate = None

        if args.error_rate is not None:
            custom_position_rates, error_rate_input_type = process_error_rate_input(
                error_rate_input=args.error_rate,
                input_fasta_path=args.input,
                config_position_rates=synthesis_config['position_rates'],
                target_seq_length=seq_length
            )
            if error_rate_input_type:
                error_rate_source = error_rate_input_type
            elif custom_position_rates.get('user_input_total_error_rate') is not None:
                error_rate_source = "Custom total error rate"
            else:
                error_rate_source = "Custom three error rates"
        else:
            error_rate_source = "From file"

        # Load error rates
        if custom_position_rates is not None:
            user_input_total_error_rate = custom_position_rates.get('user_input_total_error_rate')
            user_input_sub_error_rate = custom_position_rates.get('user_input_sub_error_rate')
            user_input_ins_error_rate = custom_position_rates.get('user_input_ins_error_rate')
            user_input_del_error_rate = custom_position_rates.get('user_input_del_error_rate')
            error_rates = load_error_rates_from_config(custom_position_rates, seq_length)
        else:
            error_rates = load_error_rates_from_config(synthesis_config['position_rates'], seq_length)

        # Build substitution lookup table
        sub_choices, sub_cum_probs = build_substitution_lookup_from_matrix(
            synthesis_config['error_matrix']['substitution']
        )

        # Insertion base preference
        insertion_vector = synthesis_config['error_matrix']['insertion']
        insertion_total = np.sum(insertion_vector)
        if insertion_total > 0:
            insertion_probs = insertion_vector.astype(np.float64) / insertion_total
        else:
            insertion_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        # Extract kmer matrix
        _use_kmer = synthesis_config.get('use_kmer', False)
        _kmer_data = synthesis_config.get('kmer')
        kmer_sub = _kmer_data['substitution'] if (_use_kmer and _kmer_data is not None) else None
        kmer_ins = _kmer_data['insertion'] if (_use_kmer and _kmer_data is not None) else None

        # ====== Build Preloaded Data ======
        precomputed_data = {
            'synthesis_config': synthesis_config,
            'seq_length': seq_length,
            'num_ref_seqs': num_ref_seqs,
            'chunk_metadata_list': chunk_metadata_list,
            'total_reads': total_reads,
            'dist_name': dist_name,
            'dist_params': dist_params,
            'target_read_depth': args.target_read_depth,
            'error_rates': error_rates,
            'sub_choices': sub_choices,
            'sub_cum_probs': sub_cum_probs,
            'insertion_probs': insertion_probs,
            'kmer_sub': kmer_sub,
            'kmer_ins': kmer_ins,
            '_use_kmer': _use_kmer,
            'error_rate_source': error_rate_source,
            'user_input_total_error_rate': user_input_total_error_rate,
            'user_input_sub_error_rate': user_input_sub_error_rate,
            'user_input_ins_error_rate': user_input_ins_error_rate,
            'user_input_del_error_rate': user_input_del_error_rate,
            'total_cpus': total_cpus,
            'reads_seq_id_width': reads_seq_id_width,
            'split_dir': str(split_dir),
            'chunk_size': chunk_size,
            'output_dir': output_dir,
        }
        
        # Run single-file simulation
        try:
            # Record start time
            start_time = time.time()
            
            # Run simulation
            stats = parallel_simulate_errors(
                input_fasta=args.input,
                output_dir=final_output_path,
                synthesis_method=args.method,
                random_seed=args.random_seed,
                num_workers_global=args.num_workers,
                merge_files_enabled=getattr(args, 'merge_files_enabled', False),
                command_line=" ".join(sys.argv),
                timestamp_suffix=args.timestamp_suffix_enabled,
                shuffle_enabled=getattr(args, 'shuffle_enabled', False),
                output_stats=args.output_stats,
                precomputed_data=precomputed_data,
            )
            
            # Record end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate total bases and estimated time for 100TB
            total_reads = stats['total_reads']
            seq_length = stats['seq_length']
            total_bases = total_reads * seq_length
            hundred_tb_bases = 100 * (10 ** 12)  # 100TB = 100 * 10^12 bases
            
            if total_bases > 0:
                estimated_time_100tb = elapsed_time * (hundred_tb_bases / total_bases)
            else:
                estimated_time_100tb = 0
            
            # Build summary_item for output (using the unified helper function)
            summary_item = build_summary_item(
                stats=stats,
                file_name=Path(args.input).name,
                output_file_path=str(final_output_path),
                elapsed_time=elapsed_time,
                ref_count=None,  # Use stats['num_ref_seqs']
                total_cpus_default=args.num_workers
            )
            
            # Generate detailed output (similar to batch-test mode)
            print()
            print_output_lines = build_print_output(
                summary_item,
                Path(final_output_path),
                test_number=None,
                input_file_name=Path(args.input).name,
                elapsed_time=elapsed_time,
                total_bases=total_bases,
                estimated_time_100tb=estimated_time_100tb
            )
            
            # Output to console
            for line in print_output_lines:
                print(line)
            
            print("\nProcessing complete!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    else:
        # ====== Batch-Test Mode ======
        # If neither parameter is specified, use default reads count (based on 220.10x read depth)
        if args.target_read_depth is None:
            # In batch-test mode, default to 220.10x read depth
            
            args.target_read_depth = 220.10
        
        print("\n" + "=" * 80)
        print("DNATerra: a computational prelude to large-scale DNA storage")
        print("=" * 80)
        print(f"\nSynthesis method: {get_synthesis_method_display_name(args.method)}")
        print(f"Target read depth: {args.target_read_depth}x")
        print("  Note: Total reads per file are auto-calculated based on its sequence count")
        
        print(f"\nWorker count: {args.num_workers}")
        print(f"Chunk strategy: auto-calculated")
        print("\nWill automatically test files in test_sequences directory")
        print("Sequence length: auto-read from config file")
        
        run_batch_tests(
            synthesis_method=args.method, 
            target_read_depth=args.target_read_depth,
            target_num_chunks=None,  # Batch-test mode uses auto-calculation
            dist_name=args.dist,
            cv=args.cv,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            num_workers=args.num_workers,
            timestamp_suffix=args.timestamp_suffix_enabled,
        )


if __name__ == "__main__":
    import time
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C signal detected, cleaning up resources and exiting... (do NOT press Ctrl+C again; program will exit automatically after cleanup)")
    finally:
        import multiprocessing
        from multiprocessing import shared_memory
        # Terminate all child processes
        for p in multiprocessing.active_children():
            p.terminate()
            p.join()
        
        # Clean up shared memory potentially created by this process (prevents leftover files after Ctrl+C)
        try:
            current_pid = os.getpid()
            shm_dir = Path('/dev/shm')
            if shm_dir.exists():
                cleaned = 0
                for shm_file in shm_dir.glob(f'buffer_*_{current_pid}'):
                    try:
                        shm = shared_memory.SharedMemory(name=shm_file.name)
                        shm.close()
                        shm.unlink()
                        cleaned += 1
                    except (FileNotFoundError, OSError, PermissionError):
                        pass
                    except Exception:
                        try:
                            shm_file.unlink()
                            cleaned += 1
                        except:
                            pass
        except Exception:
            pass  # Ignore cleanup errors; do not affect program exit
