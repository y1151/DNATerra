"""dnaterra package - DNA sequence error simulator"""

import numpy as np


class _CONSTMeta(type):
    def __setattr__(cls, name, value):
        raise TypeError(f"CONST is a read-only constant namespace, modification not allowed: {name}")

    def __delattr__(cls, name):
        raise TypeError(f"CONST is a read-only constant namespace, deletion not allowed: {name}")


class CONST(metaclass=_CONSTMeta):
    BASE_ASCII = (65, 84, 67, 71)
    SUB = 0
    INS = 1
    DEL = 2
    KMER_M = 5
    KMER_N_HEAD = KMER_M - 2
    KMER_N_TAIL = KMER_M - 3
    _ASCII_TO_IDX = np.full(256, 255, dtype=np.uint8)
    _ASCII_TO_IDX[65] = 0
    _ASCII_TO_IDX[84] = 1
    _ASCII_TO_IDX[67] = 2
    _ASCII_TO_IDX[71] = 3
    _IDX_TO_ASCII = np.array([65, 84, 67, 71], dtype=np.uint8)
    DEFAULT_DEPTH_DISTRIBUTION = {
        'electro': {'dist': 'beta',    'cv': 0.9371, 'beta_min': 0, 'beta_max': 40},
        'photo':   {'dist': 'weibull', 'cv': 0.5503},
        'inkjet':  {'dist': 'normal',   'cv': 0.3233},
    }
    DEFAULT_CHUNK_SIZE = 100_000
    DEFAULT_TOTAL_CPUS = 20
    INDEL_SAFETY_FACTOR = 1.3
    SPLIT_FILE_PRELOAD_BATCH_SIZE = 20
    SPLIT_FILE_CACHE_MAX_SIZE = 50
    SPLIT_FILE_LOAD_BATCH_SIZE = 10
    HYPERGEOMETRIC_LIMIT = 1_000_000_000
    BINOMIAL_VS_HYPERGEOMETRIC_THRESHOLD = 100_000_000
    COVERAGE_BATCH_MULTIPLIER = 100
    SEQKIT_MAX_THREADS = 64
    COPY_FILE_RANGE_AVAILABLE = True
    SHUFFLE_MASTER_SEED = 42


from .config import (
    SynthesisConfig,
    get_synthesis_method_display_name,
    get_synthesis_method_short_name,
    load_synthesis_config,
    resize_error_rates_spline,
    build_substitution_lookup_from_matrix,
)

from .coverage import (
    count_sequences_with_seqkit,
    calculate_depth_distribution_params,
    split_refs_into_chunks_streaming,
    write_ref_count_and_read_to_ref_tsv,
    generate_shuffle_mapping,
    write_shuffle_chunks_parallel,
    _build_fasta_chunk_record_index,
    _save_chunk_index,
)

from .errors import (
    load_error_rates_from_config,
    compute_error_counts_per_position,
    sample_errors_from_bucket,
    compute_bucket_metadata,
    cleanup_orphaned_shared_memory,
    cleanup_orphaned_temp_dirs,
)

from .mutations import (
    apply_substitutions,
    apply_substitutions_return_bases,
    sample_substitutions_kmer,
    apply_substitutions_kmer,
    apply_substitutions_kmer_return_bases,
    apply_indels_and_write,
    preallocate_output_file,
    prepare_indels,
    prepare_indels_kmer,
)

from .utils import (
    get_timestamp_string,
    get_merge_filename,
    detect_fasta_sequence_length,
    preview_fasta_file,
    run_batch_tests,
    process_error_rate_input,
    build_summary_item,
    build_print_output,
    update_progress_counter,
    validate_fasta_format_and_length,
    parse_ref_copy_file,
    parse_read_error_file,
    apply_explicit_errors,
)

from .normal_mode import (
    error_simulator_worker,
    parallel_simulate_errors,
)

from .simple_mode import (
    simple_mode_worker,
    parallel_simulate_errors_simple_mode,
)

# Export constants directly from the CONST class to avoid callers writing CONST.DEFAULT_CHUNK_SIZE
from . import CONST
DEFAULT_CHUNK_SIZE = CONST.DEFAULT_CHUNK_SIZE
DEFAULT_TOTAL_CPUS = CONST.DEFAULT_TOTAL_CPUS
SEQKIT_MAX_THREADS = CONST.SEQKIT_MAX_THREADS
COPY_FILE_RANGE_AVAILABLE = CONST.COPY_FILE_RANGE_AVAILABLE
DEFAULT_DEPTH_DISTRIBUTION = CONST.DEFAULT_DEPTH_DISTRIBUTION
INDEL_SAFETY_FACTOR = CONST.INDEL_SAFETY_FACTOR
HYPERGEOMETRIC_LIMIT = CONST.HYPERGEOMETRIC_LIMIT
BINOMIAL_VS_HYPERGEOMETRIC_THRESHOLD = CONST.BINOMIAL_VS_HYPERGEOMETRIC_THRESHOLD
COVERAGE_BATCH_MULTIPLIER = CONST.COVERAGE_BATCH_MULTIPLIER
SHUFFLE_MASTER_SEED = CONST.SHUFFLE_MASTER_SEED
SPLIT_FILE_PRELOAD_BATCH_SIZE = CONST.SPLIT_FILE_PRELOAD_BATCH_SIZE
SPLIT_FILE_CACHE_MAX_SIZE = CONST.SPLIT_FILE_CACHE_MAX_SIZE
SPLIT_FILE_LOAD_BATCH_SIZE = CONST.SPLIT_FILE_LOAD_BATCH_SIZE

__all__ = [
    # Package
    'CONST',
    # Configuration
    'SynthesisConfig',
    'get_synthesis_method_display_name',
    'get_synthesis_method_short_name',
    'load_synthesis_config',
    'resize_error_rates_spline',
    'build_substitution_lookup_from_matrix',
    # Coverage
    'count_sequences_with_seqkit',
    'calculate_depth_distribution_params',
    'split_refs_into_chunks_streaming',
    'write_ref_count_and_read_to_ref_tsv',
    'generate_shuffle_mapping',
    'write_shuffle_chunks_parallel',
    '_build_fasta_chunk_record_index',
    '_save_chunk_index',
    # Errors
    'load_error_rates_from_config',
    'compute_error_counts_per_position',
    'sample_errors_from_bucket',
    'compute_bucket_metadata',
    'cleanup_orphaned_shared_memory',
    'cleanup_orphaned_temp_dirs',
    # Mutations
    'apply_substitutions',
    'apply_substitutions_return_bases',
    'sample_substitutions_kmer',
    'apply_substitutions_kmer',
    'apply_substitutions_kmer_return_bases',
    'apply_indels_and_write',
    'preallocate_output_file',
    'prepare_indels',
    'prepare_indels_kmer',
    # Utilities
    'get_timestamp_string',
    'get_merge_filename',
    'detect_fasta_sequence_length',
    'preview_fasta_file',
    'run_batch_tests',
    'process_error_rate_input',
    'build_summary_item',
    'build_print_output',
    'update_progress_counter',
    'validate_fasta_format_and_length',
    'parse_ref_copy_file',
    'parse_read_error_file',
    'apply_explicit_errors',
    # Main workflow
    'error_simulator_worker',
    'parallel_simulate_errors',
    # Simple mode
    'simple_mode_worker',
    'parallel_simulate_errors_simple_mode',
    # Constants
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_TOTAL_CPUS',
    'SEQKIT_MAX_THREADS',
    'COPY_FILE_RANGE_AVAILABLE',
    'DEFAULT_DEPTH_DISTRIBUTION',
    'INDEL_SAFETY_FACTOR',
    'HYPERGEOMETRIC_LIMIT',
    'BINOMIAL_VS_HYPERGEOMETRIC_THRESHOLD',
    'COVERAGE_BATCH_MULTIPLIER',
    'SHUFFLE_MASTER_SEED',
    'SPLIT_FILE_PRELOAD_BATCH_SIZE',
    'SPLIT_FILE_CACHE_MAX_SIZE',
    'SPLIT_FILE_LOAD_BATCH_SIZE',
]
