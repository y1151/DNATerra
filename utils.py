"""Utility functions module"""
import logging
import multiprocessing
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import CONST
from .config import get_synthesis_method_display_name, get_synthesis_method_short_name
from .mutations import _merge_worker

logger = logging.getLogger(__name__)


def _format_error_rate(value: float) -> str:
    """Format error rate in 10^-3 nt^-1 units.

    Args:
        value: Raw fractional error rate (e.g. 0.001 = 0.1% = 1.0 in this unit)

    Returns:
        Human-readable string in 10^-3 nt^-1 units with smart precision.
    """
    per_thousand = value * 1000
    if per_thousand >= 10:
        return f"{per_thousand:.1f}"
    elif per_thousand >= 1:
        return f"{per_thousand:.2f}"
    elif per_thousand >= 0.01:
        return f"{per_thousand:.3f}"
    else:
        return f"{per_thousand:.4g}"


def get_timestamp_string() -> str:
    """
    Generate timestamp string in YYYYMMDDHHmmss format (year-month-day-hour-minute-second, digits only)

    Returns:
        Timestamp string, e.g.: 20250104123045
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_merge_filename(basename: str, timestamp: str = None, shuffled: bool = False) -> str:
    """
    Generate merged output filename (unified function)

    Args:
        basename: Input FASTA file base name (without extension)
        timestamp: Timestamp string. If None, no timestamp suffix is added
        shuffled: Whether this is a shuffled version

    Returns:
        Merged filename, e.g.:
        - basename_merged_20250104123045.fasta (with timestamp)
        - basename_shuffled_merged_20250104123045.fasta (with timestamp)
        - basename_merged.fasta (without timestamp)
        - basename_shuffled_merged.fasta (without timestamp)
    """
    if shuffled:
        suffix_part = f"_shuffled_merged_{timestamp}" if timestamp else "_shuffled_merged"
    else:
        suffix_part = f"_merged_{timestamp}" if timestamp else "_merged"
    return f"{basename}{suffix_part}.fasta"


def encode_3mer_index(b0: int, b1: int, b2: int) -> int:
    """Encode three bases' ASCII values into a 3mer index (0..63).

    Encoding rule consistent with error_matrix_kmer.py:
        idx = c0 * 16 + c1 * 4 + c2
    where c_i = _ASCII_TO_IDX[base_ascii] (A=0, T=1, C=2, G=3).

    Used in scalar scenarios; vectorized batch scenarios directly broadcast over _ASCII_TO_IDX array,
    no need to call this function.
    """
    return int(CONST._ASCII_TO_IDX[b0]) * 16 + int(CONST._ASCII_TO_IDX[b1]) * 4 + int(CONST._ASCII_TO_IDX[b2])


def update_progress_counter(progress_counter_worker, total_chunks: int, progress_interval: int):
    """
    Update global progress counter and output progress log (if output interval is reached)

    Optimization: Use atomic operations to reduce lock contention, use try-except for log output to avoid blocking

    Args:
        progress_counter_worker: Shared progress counter (multiprocessing.Value)
        total_chunks: Total number of chunks (used for calculating progress percentage)
        progress_interval: Progress output interval (output every N chunks)

    Returns:
        bool: Whether progress log was output
    """
    if progress_counter_worker is None:
        return False

    # Quickly acquire lock to reduce lock hold time
    should_log = False
    current_value = 0
    try:
        with progress_counter_worker.get_lock():
            progress_counter_worker.value += 1
            current_value = progress_counter_worker.value
            if (current_value % progress_interval == 0) or (current_value >= total_chunks - progress_interval):
                should_log = True
    except Exception:
        return False

    # Move log output outside of lock
    if should_log:
        try:
            pass
        except Exception:
            pass

    return should_log


def _open_fasta_file_with_retry(fasta_file: str, worker_id: int, max_retries: int = 10):
    """
    Open FASTA file (with retry mechanism, no file lock)

    Multiple worker processes each hold independent file handles, reading different sequence ranges in order,
    therefore no file lock is needed. OS file buffering layer handles concurrent I/O automatically.

    Args:
        fasta_file: FASTA file path
        worker_id: Worker ID (for logging)
        max_retries: Maximum retry count

    Returns:
        file_handle: File handle

    Raises:
        IOError: Failed to open file (exceeded maximum retry count)
    """
    retry_idx = 0

    while retry_idx < max_retries:
        try:
            fasta_file_handle = open(fasta_file, 'rb', buffering=8*1024*1024)
            return fasta_file_handle

        except (IOError, OSError, PermissionError) as open_err:
            if retry_idx == 0:
                base_delay = 0.1
            elif retry_idx < 5:
                base_delay = 0.1 * (2 ** retry_idx)
            else:
                base_delay = min(0.1 * (2 ** retry_idx), 60.0)

            retry_delay = base_delay + random.uniform(0, min(base_delay * 0.2, 0.2))

            if retry_idx < 3:
                logger.warning(f"Worker{worker_id}: FASTA file open failed (retry {retry_idx+1}/{max_retries}): {open_err}, retrying in {retry_delay:.2f}s...")

            time.sleep(retry_delay)
            retry_idx += 1

    error_msg = f"Worker{worker_id}: FASTA file open failed (retried {max_retries} times): {fasta_file}"
    logger.error(error_msg)
    raise IOError(error_msg)


def _readline_with_retry(file_handle, worker_id: int, chunk_idx: int, max_retries: int = 10):
    """
    Read a line from FASTA file (with retry mechanism)

    Args:
        file_handle: File handle
        worker_id: Worker ID (for logging)
        chunk_idx: Chunk index (for logging)
        max_retries: Maximum retry count

    Returns:
        line: Read line (bytes)

    Raises:
        IOError: Read failed (exceeded maximum retry count)
    """
    retry_idx = 0

    while retry_idx < max_retries:
        try:
            line = file_handle.readline()
            return line

        except (IOError, OSError) as read_err:
            if retry_idx == 0:
                base_delay = 0.1
            elif retry_idx < 5:
                base_delay = 0.1 * (2 ** retry_idx)
            else:
                base_delay = min(0.1 * (2 ** retry_idx), 60.0)

            retry_delay = base_delay + random.uniform(0, min(base_delay * 0.2, 0.2))

            if retry_idx < 3:
                logger.warning(f"Worker{worker_id} Chunk {chunk_idx}: readline failed (retry {retry_idx+1}/{max_retries}): {read_err}, retrying in {retry_delay:.2f}s...")

            time.sleep(retry_delay)
            retry_idx += 1

    error_msg = f"Worker{worker_id} Chunk {chunk_idx}: readline failed (retried {max_retries} times)"
    logger.error(error_msg)
    raise IOError(error_msg)


def _seek_with_retry(file_handle, position: int, worker_id: int, chunk_idx: int, max_retries: int = 10):
    """
    FASTA file seek operation (with retry mechanism)

    Args:
        file_handle: File handle
        position: Seek position
        worker_id: Worker ID (for logging)
        chunk_idx: Chunk index (for logging)
        max_retries: Maximum retry count

    Returns:
        new_position: New position after seek
    """
    retry_idx = 0

    while retry_idx < max_retries:
        try:
            new_position = file_handle.seek(position)
            return new_position

        except (IOError, OSError) as seek_err:
            if retry_idx == 0:
                base_delay = 0.1
            elif retry_idx < 5:
                base_delay = 0.1 * (2 ** retry_idx)
            else:
                base_delay = min(0.1 * (2 ** retry_idx), 60.0)

            retry_delay = base_delay + random.uniform(0, min(base_delay * 0.2, 0.2))

            if retry_idx < 3:
                logger.warning(f"Worker{worker_id} Chunk {chunk_idx}: seek failed (retry {retry_idx+1}/{max_retries}): {seek_err}, retrying in {retry_delay:.2f}s...")

            time.sleep(retry_delay)
            retry_idx += 1

    error_msg = f"Worker{worker_id} Chunk {chunk_idx}: seek failed (retried {max_retries} times)"
    logger.error(error_msg)
    raise IOError(error_msg)


def _load_split_file(worker_id: int, chunk_idx: int, split_dir: Path) -> Tuple[List[int], List[int]]:
    """
    Load a single split file into memory (with retry mechanism)

    Args:
        worker_id: Worker ID (for logging)
        chunk_idx: Chunk index
        split_dir: Split file directory

    Returns:
        (ref_indices, counts): ref_indices list and counts list

    Raises:
        FileNotFoundError: Split file does not exist
        ValueError: Load failed or data incomplete
    """
    split_file = split_dir / f'chunk_{chunk_idx}_split.npy'
    max_retries = 15  # Maximum 15 retries (with exponential backoff, up to ~1 minute)
    retry_idx = 0
    import random
    import time

    while retry_idx < max_retries:
        try:
            if not split_file.exists():
                raise FileNotFoundError(f"Split file does not exist: {split_file}")

            file_size = split_file.stat().st_size
            if file_size < 128:
                raise ValueError(f"Split file size abnormal ({file_size} bytes), file may be incomplete or corrupted, retry needed")

            split_data = np.load(split_file, mmap_mode=None, allow_pickle=False)

            if split_data.ndim != 2:
                raise ValueError(f"Split file array dimension error: expected 2D array, actual {split_data.ndim}D, shape={split_data.shape}, file may be corrupted")

            if split_data.shape[0] != 2:
                raise ValueError(f"Split file array shape error: expected first dimension to be 2, actual shape={split_data.shape}, file may be corrupted")

            n_cols = split_data.shape[1]
            if n_cols == 0:
                raise ValueError(f"Split file array is empty (shape={split_data.shape}), chunk may be empty")

            dtype_size = split_data.dtype.itemsize
            expected_data_size = split_data.shape[0] * split_data.shape[1] * dtype_size
            expected_min_size = 128 + expected_data_size

            if file_size < expected_min_size:
                raise ValueError(f"Split file size mismatch: actual {file_size} bytes < expected minimum {expected_min_size} bytes (based on shape={split_data.shape}, dtype={split_data.dtype}), file may be incomplete")

            try:
                ref_indices = split_data[0, :].tolist()
                counts = split_data[1, :].tolist()
            except (ValueError, IndexError) as idx_err:
                raise ValueError(f"Failed to access split array: shape={split_data.shape}, error: {idx_err}, file may be corrupted")

            if len(ref_indices) != len(counts):
                raise ValueError(f"ref_indices and counts length mismatch (ref_indices={len(ref_indices)}, counts={len(counts)})")

            del split_data

            return ref_indices, counts

        except Exception as load_err:
            if retry_idx == 0:
                base_delay = 0.1
                retry_delay = base_delay + random.uniform(0, 0.1)
            elif retry_idx < 5:
                base_delay = 0.1 * (2 ** retry_idx)
                retry_delay = base_delay + random.uniform(0, min(base_delay * 0.2, 0.2))
            else:
                base_delay = min(0.1 * (2 ** retry_idx), 60.0)
                jitter = random.uniform(0, min(base_delay * 0.1, 5.0))
                retry_delay = base_delay + jitter

            if retry_idx < 3:
                file_info = ""
                try:
                    if split_file.exists():
                        file_size = split_file.stat().st_size
                        file_info = f", file size: {file_size} bytes"
                    else:
                        file_info = ", file does not exist"
                except:
                    pass
                logger.warning(f"Worker{worker_id} Chunk {chunk_idx}: split file load failed (retry {retry_idx+1}/{max_retries}){file_info}, continuing to retry in {retry_delay:.2f}s... Error: {load_err}")

            time.sleep(retry_delay)
            retry_idx += 1

            if retry_idx >= max_retries:
                file_info = ""
                try:
                    if split_file.exists():
                        file_size = split_file.stat().st_size
                        file_info = f", file size: {file_size} bytes"
                except:
                    pass
                error_msg = f"Worker{worker_id} Chunk {chunk_idx}: split file load failed (retried {max_retries} times, exceeded maximum retry count): {split_file}{file_info}, last error: {load_err}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    file_info = ""
    try:
        if split_file.exists():
            file_size = split_file.stat().st_size
            file_info = f", file size: {file_size} bytes"
    except:
        pass
    error_msg = f"Worker{worker_id} Chunk {chunk_idx}: split file load failed (still failed after {retry_idx} retries): {split_file}{file_info}"
    logger.error(error_msg)
    raise ValueError(error_msg)


def detect_fasta_sequence_length(fasta_path: Path) -> int:
    """
    Detect the length of the first sequence in FASTA file (optimized: only reads first few lines, not the rest)

    Skip empty/blank lines between header and sequence, return immediately after finding the first valid sequence.

    Args:
        fasta_path: FASTA file path

    Returns:
        seq_length: Length of the first sequence (bp)

    Raises:
        ValueError: If file format is incorrect or no sequence data
        FileNotFoundError: If file does not exist
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {fasta_path}")

    try:
        with open(fasta_path, 'rb', buffering=8192) as f:
            # Read first header
            header = f.readline()
            if not header:
                raise ValueError(f"FASTA file format error: file is empty: {fasta_path}")
            if not header.startswith(b'>'):
                raise ValueError(f"FASTA file format error: first line is not a header: {fasta_path}")

            # Skip empty/blank lines until non-empty sequence line is found
            sequence_line = f.readline()
            while sequence_line and not sequence_line.strip():
                next_line = f.readline()
                if not next_line:
                    break
                if next_line.startswith(b'>'):
                    # Header followed by another header -> no sequence
                    raise ValueError(f"FASTA file format error: no sequence data after header: {fasta_path}")
                sequence_line = next_line

            if not sequence_line:
                raise ValueError(f"FASTA file format error: no sequence data: {fasta_path}")

            seq_length = len(sequence_line.rstrip(b'\r\n'))
            if seq_length == 0:
                raise ValueError(f"FASTA file format error: first sequence length is 0: {fasta_path}")

            return seq_length

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"Error reading FASTA file: {fasta_path}, error: {e}")


def format_error_rate_str(item: Dict, max_length: int = 30) -> str:
    """
    Format error rate string for table display (unit: 10^-3 nt^-1)

    Args:
        item: Dictionary containing error rate information
        max_length: Maximum string length (will truncate if exceeded)

    Returns:
        Formatted error rate string (unit: 10^-3 nt^-1)
    """
    if item.get('error_rate_total') is not None:
        error_rate_source = item.get('error_rate_source', 'from file')
        total = item.get('error_rate_total', 0)
        sub = item.get('error_rate_sub', 0)
        ins = item.get('error_rate_ins', 0)
        del_ = item.get('error_rate_del', 0)
        if error_rate_source == "from file":
            error_rate_str = f"{_format_error_rate(total)} ({_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)})"
        elif error_rate_source == "custom total error rate":
            error_rate_str = f"{_format_error_rate(total)}"
        elif error_rate_source == "custom three error rates":
            error_rate_str = f"{_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)}"
        else:
            error_rate_str = f"{_format_error_rate(total)} ({_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)})"
    else:
        error_rate_str = "N/A"

    # Ensure error rate string does not exceed column width
    if len(error_rate_str) > max_length:
        error_rate_str = error_rate_str[:max_length-3] + "..."

    return error_rate_str


def append_test_report(log_file: Path, test_file: str, seq_count: int, elapsed_time: float):
    """Append test report to log file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 85 + "\n")
        f.write(f"Test file: {test_file}\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Sequence count: {seq_count:,}\n")
        f.write(f"Processing time: {elapsed_time:.2f} seconds\n")


def convert_markdown_table_to_plain_text(markdown_lines: List[str]) -> List[str]:
    """Convert markdown table to plain text table (remove | symbols)"""
    plain_lines = []
    for line in markdown_lines:
        # If it's a table row (contains |), remove leading/trailing | and split, then recombine
        if '|' in line:
            # Remove leading/trailing | and spaces, then split
            parts = [part.strip() for part in line.split('|') if part.strip()]
            # If it's a separator row (only contains -), skip or convert to empty line
            if parts and all(part.replace('-', '').replace(' ', '') == '' for part in parts):
                # Convert separator row to empty line
                plain_lines.append("")
            else:
                # Recombine with spaces as separator
                plain_line = '  '.join(parts)
                plain_lines.append(plain_line)
        else:
            # Non-table rows are kept as-is
            plain_lines.append(line)
    return plain_lines


def format_error_rate_for_print(item: Dict) -> str:
    """
    Format error rate string for print output (unit: 10^-3 nt^-1)

    Args:
        item: Dictionary containing error rate information

    Returns:
        Formatted error rate string (unit: 10^-3 nt^-1)
    """
    error_rate_source = item.get('error_rate_source', 'from file')
    total = item.get('error_rate_total', 0)
    sub = item.get('error_rate_sub', 0)
    ins = item.get('error_rate_ins', 0)
    del_ = item.get('error_rate_del', 0)
    if error_rate_source == "from file":
        return f"{_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)}"
    elif error_rate_source == "custom total error rate":
        return f"{_format_error_rate(total)}"
    elif error_rate_source == "custom three error rates":
        return f"{_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)}"
    elif error_rate_source == "custom":
        return f"{_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)}"
    else:
        return f"{_format_error_rate(total)} ({_format_error_rate(sub)}/{_format_error_rate(ins)}/{_format_error_rate(del_)})"


def _preview_single_fasta(preview_lines: List[str], fasta_path: Path, total_reads: int = None, seed: int = 2001) -> None:
    """
    Preview from single FASTA file: first 2 sequences + ... + random 1 sequence + ...
    Used for previewing merged single file.
    """
    if not fasta_path.exists():
        preview_lines.append(f"Cannot preview file: file does not exist: {fasta_path}")
        return
    available_end = min(10000, total_reads) if total_reads is not None else 10000
    available_start = 2
    random_idx = available_start
    if available_end > available_start:
        rng = np.random.default_rng(seed)
        random_idx = int(rng.choice(range(available_start, available_end)))
    want_indices = {0, 1, random_idx}
    current_seq_idx = -1
    collecting = False
    current_seq_lines = []
    try:
        with open(fasta_path, 'rb', buffering=128*1024*1024) as f:
            for line in f:
                if line.startswith(b'>'):
                    if collecting and current_seq_lines and (current_seq_idx in want_indices):
                        preview_lines.append(f"(from file: {fasta_path.name}, sequence {current_seq_idx + 1})")
                        for l in current_seq_lines:
                            decoded = l.decode('utf-8', errors='ignore').rstrip()
                            if decoded.strip():
                                preview_lines.append(decoded)
                        if current_seq_idx == 1:
                            preview_lines.append("...")
                    current_seq_idx += 1
                    collecting = current_seq_idx in want_indices
                    current_seq_lines = [line] if collecting else []
                elif collecting:
                    current_seq_lines.append(line)
        if collecting and current_seq_lines and (current_seq_idx in want_indices):
            preview_lines.append(f"(from file: {fasta_path.name}, sequence {current_seq_idx + 1})")
            for l in current_seq_lines:
                decoded = l.decode('utf-8', errors='ignore').rstrip()
                if decoded.strip():
                    preview_lines.append(decoded)
        preview_lines.append("...")
    except Exception as e:
        preview_lines.append(f"Cannot preview file: {e}")


def _preview_chunk_file(preview_lines: list, chunk_file_0: Path) -> None:
    """Preview first 2 sequences from a single chunk file"""
    if not chunk_file_0.exists():
        preview_lines.append(f"Cannot preview file: chunk file does not exist: {chunk_file_0}")
        return

    with open(chunk_file_0, 'rb', buffering=128*1024*1024) as f:
        current_seq_idx = -1
        collecting = False
        current_seq_lines = []
        collected_count = 0

        for line in f:
            if line.startswith(b'>'):
                if collecting and current_seq_lines:
                    preview_lines.append(f"(from chunk file: {chunk_file_0.name}, sequence {current_seq_idx + 1})")
                    for l in current_seq_lines:
                        decoded_line = l.decode('utf-8', errors='ignore').rstrip()
                        if decoded_line.strip():
                            preview_lines.append(decoded_line)
                    current_seq_lines = []
                    collected_count += 1
                    if collected_count >= 2:
                        break

                current_seq_idx += 1
                if current_seq_idx < 2:
                    collecting = True
                    current_seq_lines.append(line)
                else:
                    collecting = False
            elif collecting:
                current_seq_lines.append(line)

        if collecting and current_seq_lines and collected_count < 2:
            preview_lines.append(f"(from chunk file: {chunk_file_0.name}, sequence {current_seq_idx + 1})")
            for l in current_seq_lines:
                decoded_line = l.decode('utf-8', errors='ignore').rstrip()
                if decoded_line.strip():
                    preview_lines.append(decoded_line)


def preview_fasta_file(fasta_path: Path, seed: int = 2001, total_reads: int = None, max_sequences: int = 100000,
                      chunk_dir: str = None, chunk_size: int = None, read_id_offset: int = 1) -> List[str]:
    """
    Preview part of FASTA file content.
    If output directory contains a merged single file (*_merged_*.fasta), preview that single file;
    If it's chunk files (output_chunk_*.fasta), read preview from chunks.
    Display format: first 2 sequences + ... + random 1 sequence + ...
    Note: seed=2001 is a fixed value, not changed by --random-seed parameter

    Args:
        fasta_path: FASTA file path (only for identification, actual reading from chunk or single file)
        seed: Random seed (for selecting middle position sequence, fixed at 2001)
        total_reads: Total reads count (if provided, used to limit random selection range)
        max_sequences: Maximum sequence count limit (deprecated, kept for compatibility, actually uses fixed range 10000)
        chunk_dir: Chunk file directory or single file directory (required)
        chunk_size: Chunk size (required for chunk mode, used to calculate which chunk file to open)
        read_id_offset: Reads sequence ID starting offset (default 1)


    Returns:
        Preview lines list
    """
    preview_lines = []

    try:
        if chunk_dir is None:
            preview_lines.append(f"Cannot preview file: missing output directory parameter")
            return preview_lines

        chunk_dir_path = Path(chunk_dir)
        if not chunk_dir_path.exists():
            preview_lines.append(f"Cannot preview file: output directory does not exist: {chunk_dir}")
            return preview_lines

        # Determine if it's a single file (merged) or chunk:
        # Priority check merged file first (may have been deleted but merge failed)
        # Use regex to match merged file (supports _merged.fasta and _merged_xxx.fasta)
        import re
        all_files = list(chunk_dir_path.iterdir())

        # Find merged files (match _merged.fasta or _merged_*.fasta)
        merged_files_in_dir = [f for f in all_files
                               if re.search(r'_merged.*\.fasta$', f.name, re.IGNORECASE)]
        # Find chunk files
        chunk_files_in_dir = [f for f in all_files
                              if f.name.startswith('output_chunk_') and f.name.endswith('.fasta')]

        # Prioritize previewing merged file (even if chunk files still exist, merged file should be complete)
        if merged_files_in_dir:
            single_file = merged_files_in_dir[0]
            _preview_single_fasta(preview_lines, single_file, total_reads, seed=seed)
            return preview_lines

        # No merged file, check if there are chunk files
        if chunk_files_in_dir:
            chunk_idx_0 = 0
            chunk_file_0 = chunk_dir_path / f'output_chunk_{chunk_idx_0}.fasta'
            if chunk_file_0.exists():
                _preview_chunk_file(preview_lines, chunk_file_0)
                return preview_lines

        # Neither exists, report error
        preview_lines.append(f"Cannot preview file: no merged file (*_merged.fasta) or chunk file (output_chunk_*.fasta) found, directory: {chunk_dir_path}")
        preview_lines.append(f"  Files in directory: {[f.name for f in all_files]}")
        return preview_lines

    except Exception as e:
        preview_lines.append(f"Cannot preview file: {e}")

    return preview_lines


def build_summary_item(stats: Dict,
                       file_name: str,
                       output_file_path: str,
                       elapsed_time: float,
                       ref_count: int = None,
                       total_cpus_default: int = None,
                       output_file_size: int = None) -> Dict:
    """
    Build unified summary_item dictionary (for unified building logic, avoid code duplication)

    Args:
        stats: Dictionary returned by parallel_simulate_errors
        file_name: Input file name
        output_file_path: Output file path (string)
        elapsed_time: Runtime (seconds)
        ref_count: Reference sequence count (if None, use stats['num_ref_seqs'])
        total_cpus_default: Default value for total_cpus (if not in stats)
        output_file_size: Output file size (bytes, if None, use stats['output_file_size'])

    Returns:
        summary_item dictionary
    """
    output_file_name = Path(stats['output_file']).name
    output_file_size_bytes = output_file_size if output_file_size is not None else stats.get('output_file_size', 0)
    output_file_size_mb = output_file_size_bytes / (1024 ** 2)

    summary_item = {
        'file': file_name,
        'ref_count': ref_count if ref_count is not None else stats['num_ref_seqs'],
        'reads_count': stats['total_reads'],
        'elapsed_time': elapsed_time,
        'num_chunks': stats['num_chunks'],
        'chunk_size': stats['chunk_size'],
        'seq_length': stats['seq_length'],
        'synthesis_method': stats['synthesis_method'],
                'target_read_depth': stats['target_read_depth'],
        'num_workers': stats['num_workers'],
        'total_cpus': stats.get('total_cpus', total_cpus_default),
        'actual_cpus': stats.get('actual_cpus'),
        'num_workers_allocated': stats.get('num_workers_allocated'),  # May be None
        'user_input_chunk_size': stats.get('user_input_chunk_size'),
        'error_rate_source': stats.get('error_rate_source', 'from file'),
        'error_rate_total': stats['error_rate_total'],
        'error_rate_sub': stats['error_rate_sub'],
        'error_rate_ins': stats['error_rate_ins'],
        'error_rate_del': stats['error_rate_del'],
        'output_file_name': output_file_name,
        'output_file_path': output_file_path,
        'output_file_size': output_file_size_bytes,
        'output_file_size_mb': output_file_size_mb,
        'num_chunk_batches': stats.get('num_chunk_batches', 0),
        'precompute_time': stats.get('precompute_time', 0),
        'mutation_time': stats.get('mutation_time', 0),
        'merge_time': stats.get('merge_time', 0),
        'chunk_dir': stats.get('chunk_dir'),  # Chunk directory path
        'chunk_file_sizes': stats.get('chunk_file_sizes', {}),  # Chunk file size dictionary
        'total_chunk_count': stats.get('total_chunk_count', 0),  # Total chunk file count
        'total_chunk_size': stats.get('total_chunk_size', 0),  # Total chunk file size (bytes)
        'chunk_idx_width': stats.get('chunk_idx_width'),  # Chunk file name digit width
        'actual_sub': stats.get('actual_sub'),
        'actual_ins': stats.get('actual_ins'),
        'actual_del': stats.get('actual_del'),
    }

    return summary_item


def build_print_output(summary_item: Dict, output_dir: Path, test_number: int = None,
                       input_file_name: str = None, elapsed_time: float = None,
                       total_bases: int = None, estimated_time_100tb: float = None) -> List[str]:
    """
    Build print-formatted output (test number, input file, input parameters, output info, file preview)

    Args:
        summary_item: Summary data for single test
        output_dir: Output directory path
        test_number: Test number (optional, if provided will display at the beginning of output)
        input_file_name: Input file name (optional, if provided will display at the beginning of output)
        elapsed_time: Runtime (seconds)
        total_bases: Total bases count
        estimated_time_100tb: Estimated runtime for 100TB (seconds)

    Returns:
        Output lines list
    """
    output_lines = []

    output_lines.append("=" * 57)
    output_lines.append("Test")
    if input_file_name is not None:
        output_lines.append(f"Input file: {input_file_name}")
    output_lines.append("Input parameters:")

    method_display = get_synthesis_method_display_name(summary_item.get('synthesis_method', 'inkjet'))
    output_lines.append(f"1. Synthesis method: {method_display}")
    output_lines.append(f"2. ref sequence length (bp): {summary_item.get('seq_length', 0)}")
    output_lines.append(f"3. ref sequence count: {summary_item.get('ref_count', 0):,}")

    target_read_depth = summary_item.get('target_read_depth')
    if target_read_depth is not None:
        output_lines.append(f"4. Target read depth (x): {target_read_depth:.2f}")
    else:
        output_lines.append("4. Target read depth (x): -")

    error_rate_str = format_error_rate_for_print(summary_item)
    output_lines.append(f"5. Error rate (10\u207b\u00b3 nt\u207b\u00b9): {error_rate_str}")

    user_input_chunk_size = summary_item.get('user_input_chunk_size')
    if user_input_chunk_size is not None:
        output_lines.append(f"6. Chunk size (reads): {user_input_chunk_size:,}")
    else:
        actual_chunk_size = summary_item.get('chunk_size')
        if actual_chunk_size is not None:
            output_lines.append(f"6. Chunk size (reads): {actual_chunk_size:,}")
        else:
            output_lines.append("6. Chunk size (reads): -")

    num_workers = summary_item.get('num_workers')
    if num_workers is not None:
        output_lines.append(f"7. Parallel workers: {num_workers}")

    output_lines.append("-" * 57)
    output_lines.append("Output:")

    chunk_dir = summary_item.get('chunk_dir', '')
    if chunk_dir:
        chunk_dir_name = Path(chunk_dir).name
        output_lines.append(f"1. Chunk directory: {chunk_dir_name}")
    else:
        output_lines.append("1. Chunk directory: -")

    chunk_dir = summary_item.get('chunk_dir', '')
    merged_file_size = 0
    total_chunk_size = summary_item.get('total_chunk_size', 0)
    has_merged_file = False

    if total_chunk_size == 0 and chunk_dir:
        chunk_dir_path = Path(chunk_dir)
        if chunk_dir_path.exists():
            merged_files = sorted(chunk_dir_path.glob('*_merged_*.fasta'))
            merged_files = [f for f in merged_files if '_shuffled_merged_' not in f.name]
            if merged_files:
                has_merged_file = True
                try:
                    merged_file_size = merged_files[0].stat().st_size
                except (OSError, ValueError):
                    pass
            else:
                patterns = ['output_chunk_*.fasta', 'chunk_*.fasta', '*.fasta']
                for pattern in patterns:
                    for chunk_file in sorted(chunk_dir_path.glob(pattern)):
                        if 'shuffled' not in chunk_file.name:
                            try:
                                total_chunk_size += chunk_file.stat().st_size
                            except (OSError, ValueError):
                                pass
                    if total_chunk_size > 0:
                        break

        if total_chunk_size == 0:
            total_reads = summary_item.get('total_reads') or summary_item.get('reads_count') or 0
            seq_length  = summary_item.get('seq_length') or 0
            if total_reads > 0 and seq_length > 0:
                lines_per_seq = (seq_length + 59) // 60 + 2
                bytes_per_seq = seq_length + lines_per_seq + 10
                total_chunk_size = total_reads * bytes_per_seq

    # Format and output
    if has_merged_file and merged_file_size > 0:
        total_size_gb = merged_file_size / (1024 ** 3)
        total_size_tb = merged_file_size / (1024 ** 4)
        if total_size_tb >= 1:
            total_size_str = f"{total_size_tb:.2f} TB"
        elif total_size_gb >= 1:
            total_size_str = f"{total_size_gb:.2f} GB"
        else:
            total_size_str = f"{merged_file_size / (1024 ** 2):.2f} MB"
        output_lines.append(f"2. Single fasta file size: {total_size_str}")
    elif total_chunk_size > 0:
        total_size_gb = total_chunk_size / (1024 ** 3)
        total_size_tb = total_chunk_size / (1024 ** 4)
        if total_size_tb >= 1:
            total_size_str = f"{total_size_tb:.2f} TB"
        elif total_size_gb >= 1:
            total_size_str = f"{total_size_gb:.2f} GB"
        else:
            total_size_str = f"{total_chunk_size / (1024 ** 2):.2f} MB"
        output_lines.append(f"2. Total chunk file size: {total_size_str}")
    else:
        if has_merged_file:
            output_lines.append("2. Single fasta file size: 0 MB")
        else:
            output_lines.append("2. Total chunk file size: 0 MB")

    reads_count = summary_item.get('reads_count') or summary_item.get('total_reads') or 0
    if reads_count > 0:
        output_lines.append(f"3. Actual output read count: {reads_count:,}")
    else:
        output_lines.append("3. Actual output read count: -")

    # Error statistics available in ref_count.tsv and read_to_ref_*.tsv
    output_lines.append("4. Error statistics: see ref_count.tsv / read_to_ref_*.tsv")

    # 5. Runtime
    if elapsed_time is not None:
        output_lines.append(f"5. Runtime: {elapsed_time:.2f} s")
    else:
        output_lines.append("5. Runtime: -")

    # 6. Runtime efficiency (bp/s)
    if elapsed_time is not None and elapsed_time > 0 and total_chunk_size > 0:
        efficiency_bps = total_chunk_size / elapsed_time
        output_lines.append(f"6. Runtime efficiency: {efficiency_bps:,.2f} bp/s")
    else:
        output_lines.append("6. Runtime efficiency: -")

    # 7. Estimated runtime for 100TB
    if estimated_time_100tb is not None and estimated_time_100tb > 0:
        estimated_time_100tb_hours = estimated_time_100tb / 3600
        estimated_time_100tb_days = estimated_time_100tb / 86400
        if estimated_time_100tb_days >= 1:
            estimated_time_str = f"{estimated_time_100tb_days:.2f} days"
        elif estimated_time_100tb_hours >= 1:
            estimated_time_str = f"{estimated_time_100tb_hours:.2f} hours"
        elif estimated_time_100tb >= 60:
            estimated_time_str = f"{estimated_time_100tb / 60:.2f} minutes"
        else:
            estimated_time_str = f"{estimated_time_100tb:.2f} seconds"
        output_lines.append(f"7. Estimated runtime for 100TB: {estimated_time_str}")
    else:
        output_lines.append("7. Estimated runtime for 100TB: -")

    output_lines.append("")
    output_lines.append("File preview as follows:")
    # Get reads_count from summary_item, pass to preview_fasta_file
    reads_count = summary_item.get('reads_count', None)
    chunk_dir = summary_item.get('chunk_dir', None)
    chunk_size = summary_item.get('chunk_size', None)
    read_id_offset = 1  # Default value is 1
    chunk_idx_width = summary_item.get('chunk_idx_width')
    if chunk_idx_width is None:
        total_chunk_count = summary_item.get('total_chunk_count', 0)
        if total_chunk_count > 0:
            chunk_idx_width = max(1, len(str(total_chunk_count - 1)))

    preview_lines = preview_fasta_file(
        output_dir,
        seed=2001,
        total_reads=reads_count,
        chunk_dir=chunk_dir,
        chunk_size=chunk_size,
        read_id_offset=read_id_offset,
    )
    output_lines.extend(preview_lines)

    return output_lines


def build_summary_table(summary_data: List[Dict], title: str) -> List[str]:
    """
    Build summary table content (unified function)

    Args:
        summary_data: Summary data list
        title: Table title

    Returns:
        Table lines list
    """
    table_lines = []
    table_lines.append("\n\n" + "=" * 200)
    table_lines.append(title)
    table_lines.append("=" * 200)
    table_lines.append("")

    col_widths = {
        'method': 8,
        'seq_length': 15,
        'error_rate': 40,
        'seq_type': 8,
        'input_file': 30,
        'ref_count': 16,
        'reads_count': 18,
        'target_reads': 22,
        'target_depth': 15,
        'num_chunks': 14,
        'chunk_size': 15,
        'total_cpus': 21,
        'reads_per_sec': 14,
        'precompute_time': 18,
        'mutation_time': 18,
        'merge_time': 18,
        'total_time': 18,
        'output_file': 50,
        'output_size': 18,
        'actual_bases': 16,
        'efficiency': 14
    }

    header = (f"| {'Synthesis Method':<{col_widths['method']}} | "
              f"{'ref seq length(bp)':<{col_widths['seq_length']}} | "
              f"{'Cumulative Error Rate(%)':<{col_widths['error_rate']}} | "
              f"{'Sequencing Method':<{col_widths['seq_type']}} | "
              f"{'Input File':<{col_widths['input_file']}} | "
              f"{'ref seq count':>{col_widths['ref_count']}} | "
              f"{'reads seq count':>{col_widths['reads_count']}} | "
              f"{'target reads count':>{col_widths['target_reads']}} | "
              f"{'target read depth(x)':>{col_widths['target_depth']}} | "
              f"{'chunk count':>{col_widths['num_chunks']}} | "
              f"{'chunk size':>{col_widths['chunk_size']}} | "
              f"{'user available total CPUs':>{col_widths['total_cpus']}} | "
              f"{'reads/sec':>{col_widths['reads_per_sec']}} | "
              f"{'precompute time(s)':>{col_widths['precompute_time']}} | "
              f"{'read & mutate time(s)':>{col_widths['mutation_time']}} | "
              f"{'file merge time(s)':>{col_widths['merge_time']}} | "
              f"{'total time(s)':>{col_widths['total_time']}} | "
              f"{'Output File':<{col_widths['output_file']}} | "
              f"{'Output File Size(MB)':>{col_widths['output_size']}} | "
              f"{'Actual Bases':>{col_widths['actual_bases']}} | "
              f"{'Efficiency(bp/s)':>{col_widths['efficiency']}} |")
    table_lines.append(header)

    separator = (f"| {'-' * col_widths['method']} | "
                 f"{'-' * col_widths['seq_length']} | "
                 f"{'-' * col_widths['error_rate']} | "
                 f"{'-' * col_widths['seq_type']} | "
                 f"{'-' * col_widths['input_file']} | "
                 f"{'-' * col_widths['ref_count']} | "
                 f"{'-' * col_widths['reads_count']} | "
                 f"{'-' * col_widths['target_reads']} | "
                 f"{'-' * col_widths['target_depth']} | "
                 f"{'-' * col_widths['num_chunks']} | "
                 f"{'-' * col_widths['chunk_size']} | "
                 f"{'-' * col_widths['total_cpus']} | "
                 f"{'-' * col_widths['reads_per_sec']} | "
                 f"{'-' * col_widths['precompute_time']} | "
                 f"{'-' * col_widths['mutation_time']} | "
                 f"{'-' * col_widths['merge_time']} | "
                 f"{'-' * col_widths['total_time']} | "
                 f"{'-' * col_widths['output_file']} | "
                 f"{'-' * col_widths['output_size']} | "
                 f"{'-' * col_widths['actual_bases']} | "
                 f"{'-' * col_widths['efficiency']} |")
    table_lines.append(separator)

    for item in summary_data:
        method_display = get_synthesis_method_short_name(item.get('synthesis_method', 'inkjet'))
        error_rate_str = format_error_rate_str(item, max_length=col_widths['error_rate'])
        input_file_name = item.get('file', 'N/A')
        output_file_name = item.get('output_file_name', 'N/A')
        output_file_size_mb = item.get('output_file_size_mb', 0.0)

        total_chunk_size_bytes = item.get('total_chunk_size', 0)
        elapsed_time = item.get('elapsed_time', 0)
        reads_count = item.get('reads_count', 0)
        reads_per_sec = (reads_count / elapsed_time) if elapsed_time > 0 else 0.0
        if total_chunk_size_bytes > 0:
            actual_bases_str = _format_bases(total_chunk_size_bytes)
        else:
            reads_count_est = item.get('reads_count', 0) or item.get('total_reads', 0)
            seq_len_est     = item.get('seq_length', 0)
            if reads_count_est > 0 and seq_len_est > 0:
                bytes_per_seq = seq_len_est + (seq_len_est + 59) // 60 + 12
                total_chunk_size_bytes = reads_count_est * bytes_per_seq
                actual_bases_str = _format_bases(total_chunk_size_bytes)
            else:
                actual_bases_str = "-"
        if elapsed_time > 0 and total_chunk_size_bytes > 0:
            efficiency_bps = total_chunk_size_bytes / elapsed_time
            efficiency_str = f"{efficiency_bps:.2f}"
        else:
            efficiency_str = "-"

        precompute_time = item.get('precompute_time', 0)
        mutation_time = item.get('mutation_time', 0)
        merge_time = item.get('merge_time', 0)
        total_time = item.get('elapsed_time', 0)

        target_read_depth = item.get('target_read_depth')
        target_read_depth_str = f"{target_read_depth:.2f}" if target_read_depth is not None else "-"

        user_input_chunk_size = item.get('user_input_chunk_size')
        actual_num_chunks = item.get('num_chunks')
        actual_chunk_size = item.get('chunk_size')

        num_chunks_str = f"{actual_num_chunks:,}" if actual_num_chunks is not None else "-"
        chunk_size_str = f"{user_input_chunk_size:,}" if user_input_chunk_size is not None else (f"{actual_chunk_size:,}" if actual_chunk_size is not None else "-")

        total_cpus = item.get('total_cpus', item.get('num_workers', CONST.DEFAULT_TOTAL_CPUS))

        if len(input_file_name) > col_widths['input_file']:
            input_file_name = input_file_name[:col_widths['input_file']-3] + "..."
        if len(output_file_name) > col_widths['output_file']:
            output_file_name = output_file_name[:col_widths['output_file']-3] + "..."

        table_line = (f"| {method_display:<{col_widths['method']}} | "
                     f"{item.get('seq_length', 0):>{col_widths['seq_length']}} | "
                     f"{error_rate_str:<{col_widths['error_rate']}} | "
                     f"{'Illumina':<{col_widths['seq_type']}} | "
                     f"{input_file_name:<{col_widths['input_file']}} | "
                     f"{item.get('ref_count', 0):>{col_widths['ref_count']},} | "
                     f"{item.get('reads_count', 0):>{col_widths['reads_count']},} | "
                     f"{target_read_depth_str:>{col_widths['target_depth']}} | "
                     f"{num_chunks_str:>{col_widths['num_chunks']}} | "
                     f"{chunk_size_str:>{col_widths['chunk_size']}} | "
                     f"{total_cpus:>{col_widths['total_cpus']}} | "
                     f"{reads_per_sec:>{col_widths['reads_per_sec']}.2f} | "
                     f"{precompute_time:>{col_widths['precompute_time']}.2f} | "
                     f"{mutation_time:>{col_widths['mutation_time']}.2f} | "
                     f"{merge_time:>{col_widths['merge_time']}.2f} | "
                     f"{total_time:>{col_widths['total_time']}.2f} | "
                     f"{output_file_name:<{col_widths['output_file']}} | "
                     f"{item.get('output_file_size_mb', 0.0):>{col_widths['output_size']}.2f} | "
                     f"{actual_bases_str:>{col_widths['actual_bases']}} | "
                     f"{efficiency_str:>{col_widths['efficiency']}} |")
        table_lines.append(table_line)

    table_lines.append("")
    return table_lines


def get_display_width(text: str) -> int:
    """
    Calculate display width of string (considering CJK characters occupy 2 widths)

    Args:
        text: String to calculate

    Returns:
        Display width (CJK characters count as 2, ASCII characters count as 1)
    """
    width = 0
    for char in text:
        # Check if it's a CJK character (including CJK punctuation)
        if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f' or '\uff00' <= char <= '\uffef':
            width += 2
        else:
            width += 1
    return width


def format_with_display_width(text: str, display_width: int, align: str = '<') -> str:
    """
    Format string according to display width (considering CJK characters occupy 2 widths)

    Args:
        text: String to format
        display_width: Target display width
        align: Alignment, '<' left, '>' right, '^' center

    Returns:
        Formatted string
    """
    current_width = get_display_width(text)
    if current_width >= display_width:
        return text

    # Calculate padding needed (spaces are single-byte characters, occupying 1 display width)
    padding = display_width - current_width
    spaces = ' ' * padding

    if align == '<':
        return text + spaces
    elif align == '>':
        return spaces + text
    elif align == '^':
        left_padding = padding // 2
        right_padding = padding - left_padding
        return ' ' * left_padding + text + ' ' * right_padding
    else:
        return text + spaces


def _format_bases(num_bytes: int) -> str:
    """Format byte count to human-readable string (B / KB / MB / GB / TB)."""
    if num_bytes >= 1024**4:
        return f"{num_bytes / (1024**4):.2f} TB"
    elif num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes} B"


def build_plain_table(summary_data: List[Dict]) -> tuple[int, List[str], int, List[str]]:
    """
    Build borderless aligned plain text table (for batch test output)
    Divided into two tables: first table contains input-related columns, second table contains output-related columns

    Args:
        summary_data: Summary data list

    Returns:
        (first table width, first table lines list, second table width, second table lines list) tuple
    """
    if not summary_data:
        return (0, [], 0, [])

    # Define second table header texts
    header_texts_table2 = {
        'output_file': 'Output File',
        'output_size': 'Output File Size (TB)',
        'elapsed_time': 'Runtime (seconds)',
        'estimated_time': '100TB Estimated (days)',
        'actual_bases': 'Actual Bases',
        'efficiency': 'Efficiency (bp/s)'
    }

    data_max_widths_table1 = {
        'input_file': 0,
        'method': 0,
        'seq_length': 0,
        'ref_count': 0,
        'target_depth': 0,
        'error_rate': 0,
        'chunk_size': 0
    }

    data_max_widths_table2 = {
        'output_file': 0,
        'output_size': 0,
        'elapsed_time': 0,
        'estimated_time': 0,
        'actual_bases': 0,
        'efficiency': 0
    }

    for item in summary_data:
        input_file = item.get('file', 'N/A')
        data_max_widths_table1['input_file'] = max(data_max_widths_table1['input_file'], len(input_file))

        method_display = get_synthesis_method_short_name(item.get('synthesis_method', 'inkjet'))
        data_max_widths_table1['method'] = max(data_max_widths_table1['method'], len(method_display))

        seq_length = item.get('seq_length', 0)
        data_max_widths_table1['seq_length'] = max(data_max_widths_table1['seq_length'], len(str(seq_length)))

        ref_count = item.get('ref_count', 0)
        ref_count_str = f"{ref_count:,}"
        data_max_widths_table1['ref_count'] = max(data_max_widths_table1['ref_count'], len(ref_count_str))

        target_read_depth = item.get('target_read_depth')
        target_depth_str = f"{target_read_depth:.2f}" if target_read_depth is not None else "-"
        data_max_widths_table1['target_depth'] = max(data_max_widths_table1['target_depth'], len(target_depth_str))

        error_rate_str = format_error_rate_for_print(item)
        data_max_widths_table1['error_rate'] = max(data_max_widths_table1['error_rate'], len(error_rate_str))

        user_input_chunk_size = item.get('user_input_chunk_size')
        actual_chunk_size = item.get('chunk_size')
        chunk_size = user_input_chunk_size if user_input_chunk_size is not None else actual_chunk_size
        chunk_size_str = f"{chunk_size:,}" if chunk_size is not None else "-"
        data_max_widths_table1['chunk_size'] = max(data_max_widths_table1['chunk_size'], len(chunk_size_str))

        output_file = item.get('output_file_name', 'N/A')
        data_max_widths_table2['output_file'] = max(data_max_widths_table2['output_file'], len(output_file))

        output_file_size_bytes = item.get('output_file_size', 0)
        if output_file_size_bytes == 0:
            output_file_size_mb = item.get('output_file_size_mb', 0.0)
            output_file_size_bytes = int(output_file_size_mb * (1024 ** 2))
        output_file_size_tb = output_file_size_bytes / (1024 ** 4)
        output_size_str = f"{output_file_size_tb:.4f}" if output_file_size_tb > 0 else "0.0000"
        data_max_widths_table2['output_size'] = max(data_max_widths_table2['output_size'], len(output_size_str))

        elapsed_time = item.get('elapsed_time', 0)
        elapsed_time_str = f"{elapsed_time:.2f}" if elapsed_time > 0 else "-"
        data_max_widths_table2['elapsed_time'] = max(data_max_widths_table2['elapsed_time'], len(elapsed_time_str))

        reads_count = item.get('reads_count', 0)
        seq_length_val = item.get('seq_length', 0)
        total_bases = reads_count * seq_length_val if reads_count > 0 and seq_length_val > 0 else 0
        hundred_tb_bases = 100 * (10 ** 12)
        if total_bases > 0:
            estimated_time_100tb = elapsed_time * (hundred_tb_bases / total_bases)
            estimated_time_100tb_days = estimated_time_100tb / 86400
            estimated_time_str = f"{estimated_time_100tb_days:.2f}"
        else:
            estimated_time_str = "-"
        data_max_widths_table2['estimated_time'] = max(data_max_widths_table2['estimated_time'], len(estimated_time_str))

        total_chunk_size_bytes = item.get('total_chunk_size', 0)
        if total_chunk_size_bytes == 0:
            total_chunk_size_bytes = output_file_size_bytes
        if total_chunk_size_bytes > 0:
            actual_bases_str = _format_bases(total_chunk_size_bytes)
        else:
            reads_count_est = item.get('reads_count', 0) or item.get('total_reads', 0)
            seq_len_est     = item.get('seq_length', 0)
            if reads_count_est > 0 and seq_len_est > 0:
                bytes_per_seq = seq_len_est + (seq_len_est + 59) // 60 + 12
                total_chunk_size_bytes = reads_count_est * bytes_per_seq
                actual_bases_str = _format_bases(total_chunk_size_bytes)
            else:
                actual_bases_str = "-"
        data_max_widths_table2['actual_bases'] = max(data_max_widths_table2['actual_bases'], len(actual_bases_str))

        if elapsed_time > 0 and total_chunk_size_bytes > 0:
            efficiency_bps = total_chunk_size_bytes / elapsed_time
            efficiency_str = f"{efficiency_bps:.2f}"
        else:
            efficiency_str = "-"
        data_max_widths_table2['efficiency'] = max(data_max_widths_table2['efficiency'], len(efficiency_str))

    col_widths_table1 = {
        key: max(get_display_width(header_texts_table1[key]), data_max_widths_table1[key])
        for key in header_texts_table1.keys()
    }

    col_widths_table2 = {
        key: max(get_display_width(header_texts_table2[key]), data_max_widths_table2[key])
        for key in header_texts_table2.keys()
    }

    header_table1 = (
        format_with_display_width(header_texts_table1['input_file'], col_widths_table1['input_file'], '<') + ' '
        + format_with_display_width(header_texts_table1['method'], col_widths_table1['method'], '<') + ' '
        + format_with_display_width(header_texts_table1['seq_length'], col_widths_table1['seq_length'], '<') + ' '
        + format_with_display_width(header_texts_table1['ref_count'], col_widths_table1['ref_count'], '<') + ' '
        + format_with_display_width(header_texts_table1['target_depth'], col_widths_table1['target_depth'], '<') + ' '
        + format_with_display_width(header_texts_table1['error_rate'], col_widths_table1['error_rate'], '<') + ' '
        + format_with_display_width(header_texts_table1['chunk_size'], col_widths_table1['chunk_size'], '<')
    )

    header_table2 = (
        format_with_display_width(header_texts_table2['output_file'], col_widths_table2['output_file'], '<') + ' '
        + format_with_display_width(header_texts_table2['output_size'], col_widths_table2['output_size'], '<') + ' '
        + format_with_display_width(header_texts_table2['elapsed_time'], col_widths_table2['elapsed_time'], '<') + ' '
        + format_with_display_width(header_texts_table2['estimated_time'], col_widths_table2['estimated_time'], '<') + ' '
        + format_with_display_width(header_texts_table2['actual_bases'], col_widths_table2['actual_bases'], '<') + ' '
        + format_with_display_width(header_texts_table2['efficiency'], col_widths_table2['efficiency'], '<')
    )

    table1_width = get_display_width(header_table1) + 4
    table2_width = get_display_width(header_table2) + 4

    table1_lines = []
    table1_lines.append(header_table1)

    table2_lines = []
    table2_lines.append(header_table2)

    for item in summary_data:
        input_file = item.get('file', 'N/A')
        if get_display_width(input_file) > col_widths_table1['input_file']:
            while get_display_width(input_file) > col_widths_table1['input_file'] - 3:
                input_file = input_file[:-1]
            input_file = input_file + '...'

        method_display = get_synthesis_method_short_name(item.get('synthesis_method', 'inkjet'))

        seq_length = item.get('seq_length', 0)
        seq_length_str = str(seq_length)

        ref_count = item.get('ref_count', 0)
        ref_count_str = f"{ref_count:,}"

        target_read_depth = item.get('target_read_depth')
        target_depth_str = f"{target_read_depth:.2f}" if target_read_depth is not None else "-"

        error_rate_str = format_error_rate_for_print(item)
        if get_display_width(error_rate_str) > col_widths_table1['error_rate']:
            while get_display_width(error_rate_str) > col_widths_table1['error_rate'] - 3:
                error_rate_str = error_rate_str[:-1]
            error_rate_str = error_rate_str + '...'

        user_input_chunk_size = item.get('user_input_chunk_size')
        actual_chunk_size = item.get('chunk_size')
        chunk_size = user_input_chunk_size if user_input_chunk_size is not None else actual_chunk_size
        chunk_size_str = f"{chunk_size:,}" if chunk_size is not None else "-"

        row_table1 = (
            format_with_display_width(input_file, col_widths_table1['input_file'], '<') + ' '
            + format_with_display_width(method_display, col_widths_table1['method'], '<') + ' '
            + format_with_display_width(seq_length_str, col_widths_table1['seq_length'], '<') + ' '
            + format_with_display_width(ref_count_str, col_widths_table1['ref_count'], '<') + ' '
            + format_with_display_width(target_depth_str, col_widths_table1['target_depth'], '<') + ' '
            + format_with_display_width(error_rate_str, col_widths_table1['error_rate'], '<') + ' '
            + format_with_display_width(chunk_size_str, col_widths_table1['chunk_size'], '<')
        )
        table1_lines.append(row_table1)

        output_file = item.get('output_file_name', 'N/A')
        if get_display_width(output_file) > col_widths_table2['output_file']:
            while get_display_width(output_file) > col_widths_table2['output_file'] - 3:
                output_file = output_file[:-1]
            output_file = output_file + '...'

        output_file_size_bytes = item.get('output_file_size', 0)
        if output_file_size_bytes == 0:
            output_file_size_mb = item.get('output_file_size_mb', 0.0)
            output_file_size_bytes = int(output_file_size_mb * (1024 ** 2))
        output_file_size_tb = output_file_size_bytes / (1024 ** 4)
        output_size_str = f"{output_file_size_tb:.4f}" if output_file_size_tb > 0 else "0.0000"

        elapsed_time = item.get('elapsed_time', 0)
        elapsed_time_str = f"{elapsed_time:.2f}" if elapsed_time > 0 else "-"

        reads_count = item.get('reads_count', 0)
        seq_length_val = item.get('seq_length', 0)
        total_bases = reads_count * seq_length_val if reads_count > 0 and seq_length_val > 0 else 0
        hundred_tb_bases = 100 * (10 ** 12)
        if total_bases > 0:
            estimated_time_100tb = elapsed_time * (hundred_tb_bases / total_bases)
            estimated_time_100tb_days = estimated_time_100tb / 86400
            estimated_time_str = f"{estimated_time_100tb_days:.2f}"
        else:
            estimated_time_str = "-"

        total_chunk_size_bytes = item.get('total_chunk_size', 0)
        if total_chunk_size_bytes == 0:
            total_chunk_size_bytes = output_file_size_bytes
        if total_chunk_size_bytes > 0:
            actual_bases_str = _format_bases(total_chunk_size_bytes)
        else:
            # Estimate from statistics
            reads_count_est = item.get('reads_count', 0) or item.get('total_reads', 0)
            seq_len_est     = item.get('seq_length', 0)
            if reads_count_est > 0 and seq_len_est > 0:
                bytes_per_seq = seq_len_est + (seq_len_est + 59) // 60 + 12
                total_chunk_size_bytes = reads_count_est * bytes_per_seq
                actual_bases_str = _format_bases(total_chunk_size_bytes)
            else:
                actual_bases_str = "-"

        # Runtime efficiency (bp/s)
        if elapsed_time > 0 and total_chunk_size_bytes > 0:
            efficiency_bps = total_chunk_size_bytes / elapsed_time
            efficiency_str = f"{efficiency_bps:.2f}"
        else:
            efficiency_str = "-"

        # Build second table data row
        row_table2 = (
            format_with_display_width(output_file, col_widths_table2['output_file'], '<') + ' '
            + format_with_display_width(output_size_str, col_widths_table2['output_size'], '<') + ' '
            + format_with_display_width(elapsed_time_str, col_widths_table2['elapsed_time'], '<') + ' '
            + format_with_display_width(estimated_time_str, col_widths_table2['estimated_time'], '<') + ' '
            + format_with_display_width(actual_bases_str, col_widths_table2['actual_bases'], '<') + ' '
            + format_with_display_width(efficiency_str, col_widths_table2['efficiency'], '<')
        )
        table2_lines.append(row_table2)

    # Return widths and line lists for both tables
    return (table1_width, table1_lines, table2_width, table2_lines)


def append_summary_report(log_file: Path, summary_data: List[Dict]):
    """Output summary report to log file (not to console)"""
    # Use unified function to build table content
    table_lines = build_summary_table(summary_data, "Batch Test Complete - Complete Summary Report")

    # Add ending information
    table_lines.append("=" * 200)
    table_lines.append(f"Complete report saved to: {log_file}")
    table_lines.append("")

    # Save original formatters and create no-timestamp formatter
    original_formatters = []
    no_timestamp_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Temporarily replace all handlers' formatter with no-timestamp version
    for handler in logger.handlers:
        original_formatters.append(handler.formatter)
        handler.setFormatter(no_timestamp_formatter)

    # Write to file (markdown format)
    with open(log_file, 'a', encoding='utf-8') as f:
        for line in table_lines:
            f.write(line + "\n")


def run_batch_tests(synthesis_method: str = "electro",
                   target_read_depth: float = None,
                   target_num_chunks: int = None,
                   dist_name: str = None,
                   cv: float = None,
                   beta_min: float = None,
                   beta_max: float = None,
                   num_workers: int = 52,
                   timestamp_suffix: bool = False,
                   random_seed: int = 42):
    """
    Batch test multiple FASTA files

    Args:
        synthesis_method: DNA synthesis method (default "electro")
            - "inkjet": Inkjet-based DNA synthesis
            - "electro": Electrochemical DNA synthesis
            - "photo": Photochemical DNA synthesis
        target_read_depth: Target read depth (average reads per ref, mutually exclusive with target_num_reads)
        target_num_chunks: Target total chunk count (total chunks for all processes to handle, default None, auto-calculated. Default chunk_size is DEFAULT_CHUNK_SIZE(100000), if reads count < DEFAULT_CHUNK_SIZE then use actual reads count)
        num_workers: Number of parallel worker processes (default 64)

    Chunk explanation:
        - Total chunk count = total chunks for all processes to handle
        - Chunks per process ≈ total chunks / num_workers
    """

    header_texts_table1 = {
        'input_file': 'Input File',
        'method': 'Method',
        'seq_length': 'Seq Len',
        'ref_count': 'ref Count',
        'target_depth': 'Depth',
        'error_rate': 'Error Rate',
        'chunk_size': 'Chunk Size'
    }

    script_dir = Path(__file__).parent
    input_dir = script_dir / 'input_dir'
    logs_dir = script_dir / 'logs'
    output_dir = script_dir / 'output_dir'

    input_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directly specify file path list (relative to current file's directory)
    test_files = [
        'input_dir/seq_n5_l150.fasta',  # 5
        'input_dir/seq_n50_l150.fasta',  # 50
        'input_dir/seq_n500_l150.fasta',  # 500
        'input_dir/seq_n5000_l150.fasta',  # 5000
        'input_dir/seq_n50000_l150.fasta',  # 50k
        'input_dir/seq_n500000_l150.fasta',  # 500k
        'input_dir/seq_n5000000_l150.fasta',  # 5M
        # 'input_dir/seq_n50000000_l150.fasta',  # 50M
        # 'input_dir/seq_n500000000_l150.fasta',  # 500M
        # 'input_dir/seq_n1000000000_l117.fasta',  # 1B 20T scenario
       #  'input_dir/seq_n5000000000_l150.fasta',  # 5B
       # 'input_dir/ILSVRC2010_devkit-1.0.tar.gz.fasta',

    ]


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"ScaleDS_{timestamp}.log"

    method_display_name = get_synthesis_method_display_name(synthesis_method)

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write("DNATerra: a computational prelude to large-scale DNA storage\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Synthesis method: {method_display_name}\n")

    # Add file handler to logger, so all logs (including errors) are written to log file
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Record all levels of logs
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Ensure all logs output immediately (avoid delayed output during multiprocess creation)
    for handler in logger.handlers:
        handler.flush()

    # Summary data list
    summary_data = []
    # Flag whether table header has been initialized
    table_header_written = False
    # Test number counter
    test_number = 0
    original_formatters_prev = []

    for test_file_rel in test_files:
        test_number += 1
        # Build full path (relative to script directory)
        input_fasta = script_dir / test_file_rel
        test_file_name = Path(test_file_rel).name

        # Check if file exists
        if not input_fasta.exists():
            error_msg = f"Error: file does not exist: {input_fasta}\nPlease ensure the file exists at the specified path."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Detect sequence length
        try:
            detected_seq_length = detect_fasta_sequence_length(input_fasta)
        except Exception as e:
            error_msg = f"Error: failed to process file: {input_fasta}\n{str(e)}"
            logger.error(error_msg)
            raise

        if summary_data:
            for prev_idx, prev_summary_item in enumerate(summary_data, start=1):
                try:
                    prev_total_reads = prev_summary_item.get('reads_count', 0)
                    prev_seq_length = prev_summary_item.get('seq_length', 0)
                    prev_total_bases = prev_total_reads * prev_seq_length
                    prev_elapsed_time = prev_summary_item.get('elapsed_time', 0)

                    hundred_tb_bases = 100 * (10 ** 12)
                    if prev_total_bases > 0:
                        prev_estimated_time_100tb = prev_elapsed_time * (hundred_tb_bases / prev_total_bases)
                    else:
                        prev_estimated_time_100tb = 0

                    prev_output_path = prev_summary_item.get('output_file_path')
                    if prev_output_path:
                        prev_output_dir = Path(prev_output_path)
                    else:
                        prev_chunk_dir = prev_summary_item.get('chunk_dir', '')
                        if prev_chunk_dir:
                            prev_output_dir = Path(prev_chunk_dir)
                        else:
                            prev_output_dir = None

                    if prev_output_dir is None or not prev_output_dir.exists():
                        logger.warning(f"Test {prev_idx} output directory does not exist: {prev_output_dir}, skipping preview")

                    if prev_output_dir is not None:
                        prev_print_output_lines = build_print_output(
                            prev_summary_item, prev_output_dir,
                            test_number=prev_idx,
                            input_file_name=prev_summary_item.get('file', ''),
                            elapsed_time=prev_elapsed_time,
                            total_bases=prev_total_bases,
                            estimated_time_100tb=prev_estimated_time_100tb
                        )
                    else:
                        prev_print_output_lines = []
                    for line in prev_print_output_lines:
                        print(line)

                    for handler, original_formatter in zip(logger.handlers, original_formatters_prev):
                        handler.setFormatter(original_formatter)
                except Exception as e:
                    logger.warning(f"Error outputting test {prev_idx} results: {e}")
                    error_output_lines = [
                        f"Test {prev_idx}",
                        f"Input file: {prev_summary_item.get('file', 'unknown')}",
                        "(output file preview failed)",
                        "-" * 80
                    ]
                    for line in error_output_lines:
                        print(line)

        try:
            # Record start time
            start_time = time.time()

            from .normal_mode import parallel_simulate_errors

            stats = parallel_simulate_errors(
                input_fasta=str(input_fasta),
                output_dir=str(output_dir),
                synthesis_method=synthesis_method,
                seq_length=None,
                chunk_size=None,
                random_seed=random_seed,
                target_read_depth=target_read_depth,
                target_num_chunks=None,
                dist_name=dist_name,
                cv=cv,
                beta_min=beta_min,
                beta_max=beta_max,
                num_workers_global=num_workers,
                custom_position_rates=None,
                error_rate_input_type=None,
                command_line=" ".join(sys.argv),
                timestamp_suffix=timestamp_suffix,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            actual_seq_count = stats['num_ref_seqs']

            append_test_report(log_file, test_file_name, actual_seq_count, elapsed_time)

            summary_item = build_summary_item(
                stats=stats,
                file_name=test_file_name,
                output_file_path=str(output_dir),
                elapsed_time=elapsed_time,
                ref_count=actual_seq_count,
                total_cpus_default=num_workers
            )
            summary_data.append(summary_item)

            if summary_data:
                table1_width, table1_lines, table2_width, table2_lines = build_plain_table(summary_data)

                max_width = max(table1_width, table2_width)

                original_formatters_plain = []
                no_timestamp_formatter_plain = logging.Formatter('%(levelname)s - %(message)s')

                for handler in logger.handlers:
                    original_formatters_plain.append(handler.formatter)
                    handler.setFormatter(no_timestamp_formatter_plain)

                print("\n" + "=" * max_width)
                for line in table1_lines:
                    print(line)
                print("-" * max_width)
                for line in table2_lines:
                    print(line)
                print("=" * max_width)

                for handler, original_formatter in zip(logger.handlers, original_formatters_plain):
                    handler.setFormatter(original_formatter)

            if not table_header_written:
                table_header_written = True

            table_lines = build_summary_table(summary_data, "Batch Test Progress - Summary Report")

            for handler, original_formatter in zip(logger.handlers, original_formatters):
                handler.setFormatter(original_formatter)

            total_reads = stats['total_reads']
            seq_length = stats['seq_length']
            total_bases = total_reads * seq_length
            hundred_tb_bases = 100 * (10 ** 12)

            if total_bases > 0:
                estimated_time_100tb = elapsed_time * (hundred_tb_bases / total_bases)
            else:
                estimated_time_100tb = 0

            summary_item['output_file_size'] = stats['output_file_size']

            print_output_lines = build_print_output(
                summary_item, output_dir,
                test_number=test_number,
                input_file_name=test_file_name,
                elapsed_time=elapsed_time,
                total_bases=total_bases,
                estimated_time_100tb=estimated_time_100tb
            )
            for line in print_output_lines:
                print(line)

            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
                    break

            try:
                from .errors import cleanup_orphaned_shared_memory
                cleanup_orphaned_shared_memory(
                    prefixes=['buffer_batch', 'buffer_worker', 'psm_', 'bucket_sampling_'],
                    max_retries=2,
                    force_delete=True
                )
            except Exception as e:
                logger.warning(f"Error cleaning up orphaned shared memory (can be ignored): {e}")

            try:
                script_dir = Path(__file__).parent
                temp_bucket_dir = script_dir / '.temp_bucket_data'
                if temp_bucket_dir.exists():
                    import shutil
                    shutil.rmtree(temp_bucket_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up bucket_sampling_counts temp files (can be ignored): {e}")

            try:
                from .errors import cleanup_orphaned_temp_dirs
                cleanup_orphaned_temp_dirs(pattern='seq_sampling_split_*', max_retries=2)
            except Exception as e:
                logger.warning(f"Error cleaning up orphaned temp directories (can be ignored): {e}")

            time.sleep(0.5)

        except Exception as e:
            import traceback
            error_msg = f"Error processing file: {test_file_name}\n{str(e)}"
            print(f"\nx {error_msg}")
            logger.error(error_msg)
            tb_str = traceback.format_exc()
            traceback.print_exc()
            logger.error(f"Detailed error info:\n{tb_str}")
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nx Error: error processing file: {test_file_name}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write(f"Detailed stack trace:\n{tb_str}\n")
                    f.write("=" * 80 + "\n")
                    f.flush()
            except Exception as log_error:
                print(f"[WARNING] Error writing to log file: {log_error}")
            continue

    if summary_data:
        # If table has been initialized, append end marker
        if table_header_written:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 85 + "\n")
                f.write("Batch test complete\n")
                f.write("=" * 85 + "\n")

        # Output borderless aligned table to console (divided into two tables)
        table1_width, table1_lines, table2_width, table2_lines = build_plain_table(summary_data)

        max_width = max(table1_width, table2_width)

        original_formatters_plain = []
        no_timestamp_formatter_plain = logging.Formatter('%(levelname)s - %(message)s')

        for handler in logger.handlers:
            original_formatters_plain.append(handler.formatter)
            handler.setFormatter(no_timestamp_formatter_plain)

        print("\n" + "=" * max_width)
        for line in table1_lines:
            print(line)
        print("-" * max_width)
        for line in table2_lines:
            print(line)
        print("=" * max_width)

        for handler, original_formatter in zip(logger.handlers, original_formatters_plain):
            handler.setFormatter(original_formatter)

        append_summary_report(log_file, summary_data)
    else:
        print("[WARNING] No files were successfully processed")

    try:
        log_file_str = str(log_file)
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler_path = str(handler.baseFilename)
                if os.path.normpath(handler_path) == os.path.normpath(log_file_str):
                    logger.removeHandler(handler)
                    handler.close()
    except Exception as e:
        print(f"[WARNING] Error cleaning up file handler: {e}")


def process_error_rate_input(error_rate_input: list,
                            input_fasta_path: str,
                            config_position_rates: Dict,
                            target_seq_length: int) -> Tuple[Dict, str]:
    """
    Process user-input error rate parameters

    Args:
        error_rate_input: User-input error rate parameter list
        input_fasta_path: Input FASTA file path
        config_position_rates: Position error rate dictionary from config file
        target_seq_length: Target sequence length

    Returns:
        Processed position error rate dictionary, containing:
        - total_error_rate: Total error rate array
        - substitution_rate: Substitution error rate array
        - insertion_rate: Insertion error rate array
        - deletion_rate: Deletion error rate array
    """
    if not error_rate_input:
        return config_position_rates, "from file"

    if len(error_rate_input) == 1:
        input_str = error_rate_input[0]

        try:
            total_error_rate_perk = float(input_str)
            if total_error_rate_perk < 0:
                raise ValueError(f"Error rate cannot be negative, current value: {total_error_rate_perk}")
            if total_error_rate_perk > 1000:
                raise ValueError(
                    f"Total error rate must be in [0, 1000] 10" + "⁻³" + f" nt" + "⁻¹" + " range (i.e. 0%~100%), "
                    f"current value: {total_error_rate_perk}"
                )

            total_error_rate_value = total_error_rate_perk / 1000.0

            if total_error_rate_perk == 0:
                pass

            config_total = np.mean(config_position_rates['total_error_rate'])
            config_sub_mean = np.mean(config_position_rates['substitution_rate'])
            config_ins_mean = np.mean(config_position_rates['insertion_rate'])
            config_del_mean = np.mean(config_position_rates['deletion_rate'])

            if config_total == 0:
                if total_error_rate_value == 0:
                    seq_length = len(config_position_rates['total_error_rate'])
                    return {
                        'total_error_rate': np.zeros(seq_length, dtype=np.float64),
                        'substitution_rate': np.zeros(seq_length, dtype=np.float64),
                        'insertion_rate': np.zeros(seq_length, dtype=np.float64),
                        'deletion_rate': np.zeros(seq_length, dtype=np.float64)
                    }, "custom total error rate"
                else:
                    raise ValueError("Config file total error rate is 0, cannot calculate multiplier")

            multiplier = total_error_rate_value / config_total

            substitution_rate = config_position_rates['substitution_rate'] * multiplier
            insertion_rate = config_position_rates['insertion_rate'] * multiplier
            deletion_rate = config_position_rates['deletion_rate'] * multiplier

            total_error_rate = substitution_rate + insertion_rate + deletion_rate

            return {
                'total_error_rate': total_error_rate,
                'substitution_rate': substitution_rate,
                'insertion_rate': insertion_rate,
                'deletion_rate': deletion_rate,
                'user_input_total_error_rate': total_error_rate_value
            }, "custom total error rate"
        except ValueError as e:
                if "could not convert" in str(e).lower():
                    raise ValueError(
                        f"Cannot parse error rate parameter: {input_str}\n"
                        f"Please provide:\n"
                        f"  1. Total error rate (single number, unit: 10⁻³ nt⁻¹, e.g.: 1.0 means 0.1%)\n"
                        f"  2. Substitution, insertion, deletion error rates (three numbers, e.g.: 0.5 0.3 0.2)"
                    )
                raise

    elif len(error_rate_input) == 3:
        try:
            sub_perk = float(error_rate_input[0])
            ins_perk = float(error_rate_input[1])
            del_perk = float(error_rate_input[2])

            if any(r < 0 for r in [sub_perk, ins_perk, del_perk]):
                raise ValueError("Error rate cannot be negative")
            if any(r > 1000 for r in [sub_perk, ins_perk, del_perk]):
                raise ValueError(
                    "Single error rate must be in [0, 1000] 10⁻³ nt⁻¹ range (i.e. 0%~100%)"
                )

            sub_rate = sub_perk / 1000.0
            ins_rate = ins_perk / 1000.0
            del_rate = del_perk / 1000.0

            total_input_rate = sub_rate + ins_rate + del_rate
            if total_input_rate > 1:
                raise ValueError(
                    f"Sum of three error rates ({_format_error_rate(total_input_rate)} 10⁻³ nt⁻¹) exceeds 1000 10⁻³ nt⁻¹ (i.e. 100%)\n"
                    f"Substitution: {sub_perk:.4f}, Insertion: {ins_perk:.4f}, Deletion: {del_perk:.4f}\n"
                    f"Three are mutually exclusive (a base can only have one error), sum cannot exceed 1000 10⁻³ nt⁻¹"
                )

            if sub_perk == 0 and ins_perk == 0 and del_perk == 0:
                pass

            sub_perk_display = sub_rate * 1000.0
            ins_perk_display = ins_rate * 1000.0
            del_perk_display = del_rate * 1000.0
            total_perk_display = total_input_rate * 1000.0

            config_sub_mean = np.mean(config_position_rates['substitution_rate'])
            config_ins_mean = np.mean(config_position_rates['insertion_rate'])
            config_del_mean = np.mean(config_position_rates['deletion_rate'])

            sub_multiplier = sub_rate / config_sub_mean if config_sub_mean > 0 else 0
            ins_multiplier = ins_rate / config_ins_mean if config_ins_mean > 0 else 0
            del_multiplier = del_rate / config_del_mean if config_del_mean > 0 else 0

            substitution_rate = config_position_rates['substitution_rate'] * sub_multiplier
            insertion_rate = config_position_rates['insertion_rate'] * ins_multiplier
            deletion_rate = config_position_rates['deletion_rate'] * del_multiplier

            total_error_rate = substitution_rate + insertion_rate + deletion_rate

            return {
                'total_error_rate': total_error_rate,
                'substitution_rate': substitution_rate,
                'insertion_rate': insertion_rate,
                'deletion_rate': deletion_rate,
                'user_input_sub_error_rate': sub_rate,  # Internal decimal storage
                'user_input_ins_error_rate': ins_rate,
                'user_input_del_error_rate': del_rate
            }, "custom three error rates"
        except ValueError as e:
            if "could not convert" in str(e).lower():
                raise ValueError(
                    f"Cannot parse error rate parameter: {error_rate_input}\n"
                    f"Please provide three numbers (substitution, insertion, deletion error rates, unit: 10⁻³ nt⁻¹), e.g.: 0.5 0.3 0.2"
                )
            raise

    else:
        # Other cases report error
        raise ValueError(
            f"Incorrect number of error rate parameters: provided {len(error_rate_input)} parameters\n"
            f"Please provide one of the following formats:\n"
            f"  1. Total error rate (single number, unit: 10⁻³ nt⁻¹, e.g.: 1.0 means 0.1%)\n"
            f"  2. Substitution, insertion, deletion error rates (three numbers, e.g.: 0.5 0.3 0.2)"
        )


def parse_ref_copy_file(ref_copy_path: str, num_ref_seqs: int) -> Dict[int, int]:
    """
    Parse ref_copy.txt file, return mapping from each ref_seq_global_idx (0-based) to copy_count.

    Format (per line): seq_index copy_count
    - seq_index: 1-based (user perspective, which reference sequence)
    - copy_count: Number of copies to generate for this sequence (>=1 is valid)

    Returns:
        ref_copy_map: Dict[ref_seq_global_idx (0-based), copy_count]
        seq_index not present defaults to copy_count = 0.

    Error handling:
        - seq_index is not a positive integer -> error
        - copy_count < 1 -> ignore this line
        - Same seq_index appears multiple times -> error and exit
    """
    ref_copy_map: Dict[int, int] = {}
    seen_seq_indices = set()

    if ref_copy_path is None:
        return ref_copy_map

    ref_copy_file = Path(ref_copy_path)
    if not ref_copy_file.exists():
        raise FileNotFoundError(f"ref_copy.txt file does not exist: {ref_copy_path}")

    with open(ref_copy_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"ref_copy.txt line {line_no} format error: insufficient fields, expected 2, actual {len(parts)}")

            try:
                seq_index = int(parts[0])
            except ValueError:
                raise ValueError(f"ref_copy.txt line {line_no}: seq_index must be integer, current value '{parts[0]}'")
            if seq_index < 1:
                raise ValueError(f"ref_copy.txt line {line_no}: seq_index must be positive integer >= 1, current value {seq_index}")

            try:
                copy_count = int(parts[1])
            except ValueError:
                raise ValueError(f"ref_copy.txt line {line_no}: copy_count must be integer, current value '{parts[1]}'")

            if seq_index in seen_seq_indices:
                raise ValueError(f"ref_copy.txt line {line_no}: seq_index={seq_index} appears multiple times, each seq_index can only appear once")
            seen_seq_indices.add(seq_index)

            if copy_count < 1:
                continue

            ref_seq_global_idx = seq_index - 1
            if ref_seq_global_idx >= num_ref_seqs:
                raise ValueError(
                    f"ref_copy.txt line {line_no}: seq_index={seq_index} exceeds reference sequence count "
                    f"(input FASTA has {num_ref_seqs} sequences, seq_index range 1~{num_ref_seqs})"
                )

            ref_copy_map[ref_seq_global_idx] = copy_count

    return ref_copy_map


def parse_read_error_file(read_error_path: str, total_expanded_reads: int) -> Dict[int, List[Dict]]:
    """
    Parse read_error.txt file, return error plan list for each expanded_read_global_idx (0-based).
    """
    VALID_BASES = {'A', 'T', 'G', 'C'}
    DELETION_PLACEHOLDER = '255'

    if read_error_path is None:
        return {}

    read_error_file = Path(read_error_path)
    if not read_error_file.exists():
        raise FileNotFoundError(f"read_error.txt file does not exist: {read_error_path}")

    raw_errors: List[Dict] = []
    with open(read_error_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"read_error.txt line {line_no} format error: insufficient fields, expected 4, actual {len(parts)}")

            try:
                seq_index = int(parts[0])
            except ValueError:
                raise ValueError(f"read_error.txt line {line_no}: seq_index must be integer, current value '{parts[0]}'")
            if seq_index < 1:
                raise ValueError(f"read_error.txt line {line_no}: seq_index must be positive integer >= 1, current value {seq_index}")

            try:
                pos = int(parts[1])
            except ValueError:
                raise ValueError(f"read_error.txt line {line_no}: pos must be integer, current value '{parts[1]}'")
            if pos < 1:
                raise ValueError(f"read_error.txt line {line_no}: pos must be positive integer >= 1, current value {pos}")

            err_type = parts[2].upper()
            if err_type not in ('S', 'I', 'D'):
                raise ValueError(f"read_error.txt line {line_no}: type must be one of S/I/D, current value '{parts[2]}'")

            base = parts[3].upper()
            if err_type in ('S', 'I'):
                if base not in VALID_BASES:
                    raise ValueError(f"read_error.txt line {line_no}: when type={err_type}, base must be one of A/T/G/C, current value '{base}'")
            elif err_type == 'D':
                if base != DELETION_PLACEHOLDER:
                    raise ValueError(f"read_error.txt line {line_no}: when type=D, base must be fixed at 255, current value '{base}'")
                base = None

            expanded_read_global_idx = seq_index - 1
            pos_0based = pos - 1

            if expanded_read_global_idx >= total_expanded_reads:
                logger.warning(f"read_error.txt line {line_no}: seq_index={seq_index} exceeds total expanded reads count (total {total_expanded_reads}), this error will be ignored")
                continue

            raw_errors.append({
                'expanded_idx': expanded_read_global_idx,
                'pos': pos_0based,
                'type': err_type,
                'base': base,
                'line_no': line_no,
            })

    errors_by_expanded_idx: Dict[int, List[Dict]] = {}
    for rec in raw_errors:
        idx = rec['expanded_idx']
        if idx not in errors_by_expanded_idx:
            errors_by_expanded_idx[idx] = []
        errors_by_expanded_idx[idx].append(rec)

    for idx in errors_by_expanded_idx:
        err_list = errors_by_expanded_idx[idx]
        s_list = [e for e in err_list if e['type'] == 'S']
        indels = [e for e in err_list if e['type'] in ('I', 'D')]
        s_list.sort(key=lambda x: x['line_no'])
        indels.sort(key=lambda x: (-x['pos'], x['line_no']))
        errors_by_expanded_idx[idx] = s_list + indels

    total_err_count = sum(len(v) for v in errors_by_expanded_idx.values())

    return errors_by_expanded_idx


def apply_explicit_errors(original_seq: str, errors: List[Dict]) -> str:
    """Apply explicit error plan to a single clean sequence."""
    if not errors:
        return original_seq

    seq_bytes = bytearray(original_seq.encode('ascii'))

    # S: numpy batch replacement (does not change sequence length, not affected by I/D offset)
    s_errors = [e for e in errors if e['type'] == 'S']
    if s_errors:
        positions = np.array([e['pos'] for e in s_errors], dtype=np.intp)
        subs_arr = np.frombuffer(b''.join(e['base'].encode() for e in s_errors), dtype=np.uint8)
        seq_bytes_np = np.frombuffer(seq_bytes, dtype=np.uint8).copy()
        seq_bytes_np[positions] = subs_arr
        seq_bytes = bytearray(seq_bytes_np)

    # I/D: process bytearray from end to start (avoid earlier I/D affecting later I/D indices)
    indel_errors = [e for e in errors if e['type'] in ('I', 'D')]
    indel_errors.sort(key=lambda x: (-x['pos'], x['line_no']))

    for err in indel_errors:
        if err['type'] == 'I':
            seq_bytes.insert(err['pos'], ord(err['base']))
        elif err['type'] == 'D':
            if 0 <= err['pos'] < len(seq_bytes):
                del seq_bytes[err['pos']]

    return seq_bytes.decode()


def validate_fasta_format_and_length(fasta_file: str) -> int:
    """
    Validate FASTA format and sequence length (merged check, only open file once)

    Check content:
    1. Format validation: strict two-line format (Header line + sequence line)
       - Header line must start with '>'
       - Sequence line must not start with '>'
       - Strictly alternating
       - No multi-line sequences allowed

    2. Sequence content validation (first few sequences only):
       - No spaces or tabs allowed in sequence lines
       - Only ATCG (uppercase) base characters (A/T/C/G) allowed
       - N or other IUPAC degenerate bases not allowed

    3. Sequence length validation: all sequences must have the same length
       - Check first 4 sequences (8 lines)
       - First sequence length as reference
       - Subsequent sequence lengths must match the first

    Args:
        fasta_file: FASTA file path

    Returns:
        Sequence length (bp)

    Raises:
        ValueError: Format or length does not meet requirements
    """
    VALID_BASES = set(b'ATCG')

    try:
        with open(fasta_file, 'rb') as f:
            seq_count = 0
            reference_length = None
            max_seqs_to_check = 4
            pending_header = None  # Temporarily store when new header found in inner while

            while seq_count < max_seqs_to_check:
                # Prioritize processing stored header
                if pending_header is not None:
                    header = pending_header
                    pending_header = None
                else:
                    header = f.readline()
                    if not header:
                        if seq_count == 0:
                            raise ValueError(
                                f"\n{'='*70}\n"
                                f"FASTA format error: no valid sequence found\n"
                                f"{'='*70}\n"
                                f"File may be empty or not in FASTA format\n"
                                f"{'='*70}"
                            )
                        break

                if not header.strip():
                    continue

                if not header.startswith(b'>'):
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"FASTA format error: expected header line (starting with '>')\n"
                        f"{'='*70}\n"
                        f"Actual: {header[:80].decode('utf-8', errors='replace')}\n"
                        f"{'='*70}"
                    )

                # Read corresponding sequence line (skip empty lines in between)
                seq = f.readline()
                while seq and not seq.strip():
                    next_line = f.readline()
                    if not next_line:
                        break
                    if next_line.startswith(b'>'):
                        pending_header = next_line
                        seq = None
                        break
                    seq = next_line

                if pending_header is not None:
                    continue

                if seq is None or not seq:
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"FASTA format error: no sequence line after header\n"
                        f"{'='*70}\n"
                        f"Header: {header[:80].decode('utf-8', errors='replace')}\n"
                        f"{'='*70}"
                    )

                seq = seq.rstrip(b'\r\n')
                if not seq:
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"FASTA format error: sequence line is empty\n"
                        f"{'='*70}\n"
                        f"Header: {header[:80].decode('utf-8', errors='replace')}\n"
                        f"{'='*70}"
                    )

                if seq.startswith(b'>'):
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"FASTA format error: expected sequence line (not starting with '>')\n"
                        f"{'='*70}\n"
                        f"Actual: {seq[:80].decode('utf-8', errors='replace')}\n"
                        f"Previous Header: {header[:80].decode('utf-8', errors='replace')}\n"
                        f"{'='*70}"
                    )

                if b' ' in seq or b'\t' in seq:
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"Sequence contains spaces or tabs\n"
                        f"{'='*70}\n"
                        f"Header: {header[:80].decode('utf-8', errors='replace')}\n"
                        f"Sequence (truncated): {seq[:80].decode('utf-8', errors='replace')}\n"
                        f"{'='*70}"
                    )

                illegal_bases = set(seq) - VALID_BASES
                if illegal_bases:
                    illegal_str = ''.join(chr(b) for b in sorted(illegal_bases))
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"Sequence contains illegal base characters\n"
                        f"{'='*70}\n"
                        f"Header: {header[:80].decode('utf-8', errors='replace')}\n"
                        f"Illegal characters: {illegal_str}\n"
                        f"Only ATCG (uppercase) allowed, N or other degenerate bases not allowed\n"
                        f"{'='*70}"
                    )

                seq_len = len(seq)
                if reference_length is None:
                    reference_length = seq_len
                else:
                    if seq_len != reference_length:
                        raise ValueError(
                            f"\n{'='*70}\n"
                            f"Sequence length inconsistent (sequence {seq_count+1})\n"
                            f"{'='*70}\n"
                            f"Reference length: {reference_length} bp\n"
                            f"Current length: {seq_len} bp\n"
                            f"Header: {header[:80].decode('utf-8', errors='replace')}\n"
                            f"{'='*70}"
                        )

                seq_count += 1

        if seq_count == 0:
            raise ValueError(
                f"\n{'='*70}\n"
                f"FASTA format error: no valid sequence found\n"
                f"{'='*70}\n"
                f"File may be empty or not in FASTA format\n"
                f"{'='*70}"
            )

        return reference_length

    except Exception as e:
        logger.error(f"FASTA validation failed")
        raise


def merge_chunks_parallel(
    output_file: Path,
    chunk_list: List[Dict],
    chunk_offsets: Dict[int, int],
    chunk_file_prefix: str,
    output_dir: Path,
    total_file_size: float,
    total_cpus: int,
    progress_interval: int = None,
) -> Tuple[bool, Path, float]:
    """
    Parallel merge multiple chunk files into single output file (zero-copy, general orchestration logic).

    Args:
        output_file: Merged output file path
        chunk_list: Chunk metadata list (each element contains chunk_idx)
        chunk_offsets: chunk_idx -> file offset dictionary
        chunk_file_prefix: Chunk file name prefix (e.g. "output_chunk_" or "output_chunk_shuffled_")
        chunk_idx_width: Chunk index width (e.g. 6 -> chunk_000000.fasta)
        output_dir: Directory where chunk files are located
        total_file_size: Total merged file size (bytes, for logging)
        total_cpus: Available CPU count (for deciding merge worker count)
        progress_interval: Progress output interval (default = len(chunk_list))

    Returns:
        (success, output file path, elapsed seconds)
    """
    if not chunk_list:
        return True, None, 0.0

    pi = progress_interval if progress_interval else len(chunk_list)

    # Build merge task list
    merge_tasks = [
        {'chunk_idx': m['chunk_idx'], 'file_offset': chunk_offsets[m['chunk_idx']]}
        for m in sorted(chunk_list, key=lambda x: x['chunk_idx'])
    ]

    num_workers = max(1, min(total_cpus - 2, len(merge_tasks)))
    # Sequential allocation: each worker's tasks are consecutive in merged file, to avoid concurrent copy_file_range overwrite issues
    tasks_per_worker = (len(merge_tasks) + num_workers - 1) // num_workers
    worker_tasks = [[] for _ in range(num_workers)]
    for i, task in enumerate(merge_tasks):
        wid = i // tasks_per_worker
        wid = min(wid, num_workers - 1)
        worker_tasks[wid].append(task)

    Path(output_file).touch()

    counter = multiprocessing.Value('i', 0)
    max_extend_target = multiprocessing.Value('L', 0)
    procs = []
    for wid in range(num_workers):
        if not worker_tasks[wid]:
            continue
        p = multiprocessing.Process(
            target=_merge_worker,
            args=(
                wid, worker_tasks[wid], output_file, output_dir,
                counter,
                len(merge_tasks), pi,
                max_extend_target,
                chunk_file_prefix,
            )
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    return True, output_file, 0.0


def cleanup_trailing_newlines(file_path: Path):
    """Clean up extra trailing newlines in file"""
    try:
        fd = os.open(str(file_path), os.O_RDWR)
        try:
            file_size = os.fstat(fd).st_size
            if file_size > 0:
                os.lseek(fd, max(0, file_size - 64), os.SEEK_SET)
                tail = os.read(fd, 64)
                if tail:
                    stripped = tail.rstrip(b'\n\r')
                    new_size = file_size - (len(tail) - len(stripped))
                    if new_size < file_size:
                        os.ftruncate(fd, new_size)
        finally:
            os.close(fd)
    except Exception:
        pass


def build_cigar_md_from_errors(ref_seq: str, errors: List[dict],
                              skip_self_match_check: bool = False,
                              ref_length: int = None) -> Tuple[str, str]:
    r"""
    Construct CIGAR and MD from error records, processing in order by position.

    Rules (consistent with BWA/SAM):
      S (substitution): CIGAR=M (don't break M segment), MD records ref base
      D (deletion): CIGAR=D (break M segment), MD records ^refbase
      I (insertion): CIGAR=I (break M segment), MD does not appear

    Args:
        ref_seq: Reference sequence (simple mode) or empty string (normal mode)
        errors: Error record list, format:
            - SUB: {'type': 'S', 'pos': int, 'orig_base': str}  # orig_base: original base
            - INS: {'type': 'I', 'pos': int}
            - DEL: {'type': 'D', 'pos': int, 'orig_base': str}  # orig_base: deleted base
        skip_self_match_check:
            False (simple mode): filter out "self-to-self" S events (compare with ref_seq[pos])
            True  (normal mode): don't check "self-to-self", directly use orig_base in errors
        ref_length: Reference sequence length (used when ref_seq is empty). Normal mode must provide this parameter.

    MD specification (regex: [0-9]+(([A-Z]|\^[A-Z]+)[0-9]+)*'):
      - Number = match length, [A-Z] = mismatch, ^bases = deletion
      - Must start with number, trailing 0 can be omitted
      - Each [A-Z] or ^bases must be followed by a number
      - If no matching bases between adjacent events, must write 0
    """
    if not errors:
        return f"{len(ref_seq) if ref_seq else ref_length}M", str(len(ref_seq) if ref_seq else ref_length)

    ref_len = len(ref_seq) if ref_seq else ref_length

    # Build position-to-event mapping, handle "self-to-self" filtering
    events_by_pos = {}
    orig_base_by_pos = {}  # Quick lookup for orig_base
    for e in errors:
        pos = e['pos']
        if e['type'] == 'S' and not skip_self_match_check:
            # Simple mode: check if "self-to-self"
            if ref_seq[pos] == e.get('base', ref_seq[pos]):
                continue
        events_by_pos[pos] = e['type']  # No multiple types at same position
        if 'orig_base' in e:
            orig_base_by_pos[pos] = e['orig_base']

    cigar_ops = []
    cumulative_ref = 0

    for pos in sorted(events_by_pos.keys()):
        op_type = events_by_pos[pos]

        if cumulative_ref < pos:
            m_len = pos - cumulative_ref
            cigar_ops.append((m_len, 'M'))
            cumulative_ref = pos

        if op_type == 'I':
            cigar_ops.append((1, 'I'))
        elif op_type == 'D':
            cigar_ops.append((1, 'D'))
            cumulative_ref += 1
        elif op_type == 'S':
            cigar_ops.append((1, 'M'))
            cumulative_ref += 1

    if cumulative_ref < ref_len:
        cigar_ops.append((ref_len - cumulative_ref, 'M'))

    merged_cigar = []
    for cnt, op in cigar_ops:
        if merged_cigar and merged_cigar[-1][1] == op:
            prev_cnt, _ = merged_cigar.pop()
            merged_cigar.append((prev_cnt + cnt, op))
        else:
            merged_cigar.append((cnt, op))
    cigar_str = ''.join(f'{c}{op}' for c, op in merged_cigar)

    md_parts = []
    cumulative_ref = 0
    last_was_event = False
    last_event_type = None
    processed = set()

    for pos in sorted(events_by_pos.keys()):
        if pos in processed:
            continue

        op_type = events_by_pos[pos]
        last_event_type = op_type

        if op_type == 'D':
            # Output match length before D
            match_len = pos - cumulative_ref
            if match_len > 0:
                md_parts.append(str(match_len))
                last_was_event = False
            elif match_len == 0 and (last_was_event or not md_parts):
                md_parts.append('0')
                last_was_event = False

            # Collect consecutive D bases - use orig_base_by_pos for quick lookup
            del_base = orig_base_by_pos.get(pos)
            del_bases = del_base if del_base is not None else '?'
            processed.add(pos)
            cumulative_ref = pos + 1

            next_pos = pos + 1
            while next_pos in events_by_pos and events_by_pos[next_pos] == 'D' and next_pos not in processed:
                next_base = orig_base_by_pos.get(next_pos)
                del_bases += next_base if next_base is not None else '?'
                cumulative_ref = next_pos + 1
                processed.add(next_pos)
                next_pos += 1

            md_parts.append('^' + del_bases)
            last_was_event = True

        elif op_type == 'S':
            match_len = pos - cumulative_ref
            if match_len > 0:
                md_parts.append(str(match_len))
                last_was_event = False
            elif match_len == 0 and (last_was_event or not md_parts):
                md_parts.append('0')
                last_was_event = False

            orig_base = orig_base_by_pos.get(pos)
            if orig_base is not None:
                md_parts.append(orig_base)
            else:
                md_parts.append('?')
            cumulative_ref = pos + 1
            last_was_event = True

        elif op_type == 'I':
            pass

    final_match = ref_len - cumulative_ref
    if final_match > 0:
        md_parts.append(str(final_match))
    elif final_match == 0 and last_event_type in ('D', 'S'):
        md_parts.append('0')

    md_str = ''.join(md_parts)

    return cigar_str, md_str


def merge_tsv_chunks(chunk_files: List[Path], output_file: Path,
                     delete_chunks: bool = True) -> Tuple[int, bool]:
    """
    Merge multiple TSV chunk files into a single file.

    Prefer sendfile(2) zero-copy merge; automatically degrade to normal merge if sendfile fails.

    Args:
        chunk_files: Chunk file path list (should be sorted by chunk_idx)
        output_file: Output merged file path
        delete_chunks: Whether to delete chunk files after successful merge

    Returns:
        (total_lines, was_zero_copy): Total lines after merge, whether zero-copy was used
    """
    import logging as _logger
    _logger = _logger.getLogger(__name__)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Filter out non-existent files
    chunk_files = [p for p in chunk_files if p.exists()]
    if not chunk_files:
        _logger.warning(f"merge_tsv_chunks: no valid chunk files, skipping")
        return 0, False

    # ── Prefer zero-copy merge (sendfile) ──────────────────────────────
    was_zero_copy = False
    try:
        output_file.unlink(missing_ok=True)
        fout = open(output_file, 'wb', buffering=0)
        try:
            for chunk_path in chunk_files:
                chunk_size = chunk_path.stat().st_size
                if chunk_size == 0:
                    _logger.debug(f"  [zero-copy] Skip empty chunk: {chunk_path.name}")
                    if delete_chunks:
                        try:
                            chunk_path.unlink()
                        except Exception:
                            pass
                    continue

                fin = os.open(str(chunk_path), os.O_RDONLY)
                try:
                    offset = 0
                    while offset < chunk_size:
                        # sendfile(2): dst_fd, src_fd, offset, count
                        copied = os.sendfile(fout.fileno(), fin, offset, chunk_size - offset)
                        if copied <= 0:
                            raise IOError(f"sendfile returned {copied}, kernel may not support it")
                        offset += copied
                finally:
                    os.close(fin)

                _logger.debug(f"  [zero-copy] Merge chunk: {chunk_path.name} ({chunk_size:,} bytes)")
                if delete_chunks:
                    try:
                        chunk_path.unlink()
                    except Exception:
                        pass
        finally:
            fout.close()

        was_zero_copy = True
        return 0, was_zero_copy

    except (AttributeError, OSError, IOError) as e:
        _logger.warning(f"  [zero-copy failed] sendfile not available: {e}, automatically degrading to normal merge")

    # ── Degrade: normal line-by-line merge ──────────────────────────────────────
    total_lines = 0
    buffer_size = 100000
    output_file.unlink(missing_ok=True)

    with open(output_file, 'w', encoding='utf-8', buffering=1024*1024) as fout:
        buffer = []

        for chunk_path in chunk_files:
            if not chunk_path.exists():
                continue

            with open(chunk_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    buffer.append(line)
                    if len(buffer) >= buffer_size:
                        fout.write(''.join(buffer))
                        total_lines += len(buffer)
                        buffer.clear()

            if delete_chunks:
                try:
                    chunk_path.unlink()
                except Exception:
                    pass
            _logger.debug(f"  [normal merge] Merge chunk: {chunk_path.name}")

        if buffer:
            fout.write(''.join(buffer))
            total_lines += len(buffer)

    return total_lines, was_zero_copy


def _calculate_tsv_chunk_offsets(chunk_files: List[Path]) -> Tuple[dict, int]:
    """
    Calculate offsets for TSV chunk files, used for zero-copy merge.
    Since TSV file is text format, uses sequential merge.

    Returns:
        (chunk_offsets dict, total_size)
    """
    chunk_offsets = {}
    current_offset = 0

    for chunk_idx in range(len(chunk_files)):
        chunk_path = chunk_files[chunk_idx]
        if not chunk_path.exists():
            chunk_offsets[chunk_idx] = current_offset
            continue

        size = chunk_path.stat().st_size
        chunk_offsets[chunk_idx] = current_offset
        current_offset += size

    return chunk_offsets, current_offset


# IDS table symbol definitions (shared by both modes)
# 0=match, E/F/H/J=substitution, K=insertion, L/O/P/Q=deletion


_IDS_SUBSTITUTE_REVERSE = {ord('E'): b'A', ord('F'): b'T', ord('H'): b'C', ord('J'): b'G'}
_IDS_DELETE_REVERSE = {ord('L'): b'A', ord('O'): b'T', ord('P'): b'C', ord('Q'): b'G'}
_IDS_DELETE_VALUES = set(_IDS_DELETE_REVERSE.keys())

# Forward tables: original base byte -> IDS table code
_IDS_SUBSTITUTE = {b'A': ord('E'), b'T': ord('F'), b'C': ord('H'), b'G': ord('J')}
_IDS_DELETE = {b'A': ord('L'), b'T': ord('O'), b'C': ord('P'), b'G': ord('Q')}


def _ids_table_to_cigar(ids_row: np.ndarray) -> str:
    """
    Generate CIGAR string from IDS table row.

    CIGAR rules (consistent with BWA/SAM):
      EFHJ (substitution) -> M (don't break M segment)
      LOPQ (deletion) -> D (break M segment)
      K (insertion)    -> I (break M segment)

    Algorithm (single-pass scan, consistent with build_cigar_md_from_errors):
      1. Process in order by position, encounter M -> count continuous segment;
      2. Encounter D/I -> output individually;
      3. Merge adjacent same-type operations.
    """
    n = len(ids_row)
    if n == 0:
        return ''

    cigar_ops = []
    cumulative_ref = 0

    for pos in range(n):
        v = ids_row[pos]

        if cumulative_ref < pos:
            cigar_ops.append((pos - cumulative_ref, 'M'))
            cumulative_ref = pos

        if v == ord('K'):
            cigar_ops.append((1, 'I'))
        elif v in _IDS_DELETE_VALUES:
            cigar_ops.append((1, 'D'))
            cumulative_ref += 1
        else:
            pass

    if cumulative_ref < n:
        cigar_ops.append((n - cumulative_ref, 'M'))

    merged = []
    for cnt, op in cigar_ops:
        if merged and merged[-1][1] == op:
            merged[-1] = (merged[-1][0] + cnt, op)
        else:
            merged.append((cnt, op))
    return ''.join(f'{c}{op}' for c, op in merged)


def _ids_table_to_md(ids_row: np.ndarray) -> str:
    """
    Generate MD string from IDS table row (consistent with build_cigar_md_from_errors algorithm).

    MD rules:
      EFHJ (substitution) -> write original base (A/T/C/G) —— mismatch
      LOPQ (deletion) -> write ^original base (A/T/C/G) —— deletion
      K (insertion)    -> does not appear in MD

    MD specification (regex: [0-9]+(([A-Z]|\\^[A-Z]+)[0-9]+)*'):
      - Number = match length, [A-Z] = mismatch, ^bases = deletion
      - Must start with number, trailing 0 can be omitted
      - Each [A-Z] or ^bases must be followed by a number
      - If no matching bases between adjacent events, must write 0
      - Consecutive deletions merged into single ^bases
      - Trailing D without trailing match then add 0; trailing S without trailing match then don't add 0
    """
    n = len(ids_row)
    if n == 0:
        return ''

    # Build position->event type and original base mapping
    events_by_pos = {}
    orig_base_by_pos = {}
    for pos in range(n):
        v = ids_row[pos]
        if v in _IDS_DELETE_VALUES:
            events_by_pos[pos] = 'D'
            orig_base_by_pos[pos] = _IDS_DELETE_REVERSE.get(v, b'N').decode()
        elif v in _IDS_SUBSTITUTE_REVERSE:
            events_by_pos[pos] = 'S'
            orig_base_by_pos[pos] = _IDS_SUBSTITUTE_REVERSE.get(v, b'N').decode()

    if not events_by_pos:
        return str(n)

    md_parts = []
    cumulative_ref = 0
    last_was_event = False
    last_event_type = None
    processed = set()

    for pos in sorted(events_by_pos.keys()):
        if pos in processed:
            continue

        op_type = events_by_pos[pos]
        last_event_type = op_type

        if op_type == 'D':
            match_len = pos - cumulative_ref
            if match_len > 0:
                md_parts.append(str(match_len))
                last_was_event = False
            elif match_len == 0 and (last_was_event or not md_parts):
                md_parts.append('0')
                last_was_event = False

            # Collect consecutive D
            del_bases = orig_base_by_pos.get(pos, 'N')
            cumulative_ref = pos + 1
            processed.add(pos)
            next_pos = pos + 1
            while next_pos in events_by_pos and events_by_pos[next_pos] == 'D':
                del_bases += orig_base_by_pos.get(next_pos, 'N')
                cumulative_ref = next_pos + 1
                processed.add(next_pos)
                next_pos += 1

            md_parts.append('^' + del_bases)
            last_was_event = True

        elif op_type == 'S':
            match_len = pos - cumulative_ref
            if match_len > 0:
                md_parts.append(str(match_len))
                last_was_event = False
            elif match_len == 0 and (last_was_event or not md_parts):
                md_parts.append('0')
                last_was_event = False

            orig_base = orig_base_by_pos.get(pos, 'N')
            md_parts.append(orig_base)
            cumulative_ref = pos + 1
            last_was_event = True

        # I (ord('K')) does not appear in MD, skip

    # Output trailing match
    final_match = n - cumulative_ref
    if final_match > 0:
        md_parts.append(str(final_match))
    elif final_match == 0 and last_event_type in ('D', 'S'):
        md_parts.append('0')

    return ''.join(md_parts)
