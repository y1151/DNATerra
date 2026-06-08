"""Simple mode module"""
import atexit as _atexit
import logging
import multiprocessing
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import CONST
from .coverage import (
    count_sequences_with_seqkit,
    generate_shuffle_mapping,
    write_shuffle_chunks_parallel,
    write_ref_count_and_read_to_ref_tsv,
)
from .utils import (
    get_timestamp_string,
    get_merge_filename,
    validate_fasta_format_and_length,
    _ids_table_to_cigar,
    _ids_table_to_md,
    _IDS_DELETE,
    _IDS_SUBSTITUTE,
)

logger = logging.getLogger(__name__)


def simple_mode_stage0_validate(input_fasta: str, seq_length: int = None) -> Tuple[str, int]:
    """Validate FASTA format and sequence length"""
    try:
        detected_seq_length = validate_fasta_format_and_length(input_fasta)
    except Exception as e:
        logger.error(f"FASTA validation failed: {e}")
        raise

    if seq_length is None:
        seq_length = detected_seq_length
    elif seq_length != detected_seq_length:
        logger.warning(f"User-specified sequence length ({seq_length}bp) does not match detected length ({detected_seq_length}bp), using detected length")
        seq_length = detected_seq_length

    return input_fasta, seq_length


def simple_mode_stage1_count_refs(input_fasta: str, num_workers: int = 10) -> int:
    """Count reference sequences"""
    try:
        num_ref_seqs = count_sequences_with_seqkit(input_fasta, num_threads=max(1, num_workers - 2))
        return num_ref_seqs
    except FileNotFoundError:
        logger.warning("seqkit is not available, counting sequences using Python method")
        count = 0
        with open(input_fasta, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count


def simple_mode_stage2_parse_ref_copy(ref_copy_path: str, num_ref_seqs: int) -> Dict[int, int]:
    """Parse ref_copy.txt, return ref_copy_map"""
    return parse_ref_copy_file(ref_copy_path, num_ref_seqs)


def simple_mode_stage3_build_expanded_index(ref_copy_map: Dict[int, int], num_ref_seqs: int) -> Tuple[int, np.ndarray]:
    """Build expanded index table based on ref_copy_map"""
    return build_expanded_read_index(ref_copy_map, num_ref_seqs)


def simple_mode_stage4_parse_read_error(read_error_path: str, expanded_index: np.ndarray, total_expanded_reads: int) -> Dict[int, List[Dict]]:
    """Parse read_error.txt, distribute errors by expanded_read_global_idx"""
    return parse_read_error_file(read_error_path, expanded_index, total_expanded_reads)


def simple_mode_stage5_build_chunks(total_expanded_reads: int, chunk_size: int = None,
                                    target_num_chunks: int = None) -> Tuple[int, int, List[Dict]]:
    """Calculate chunk parameters and build chunk metadata list"""

    if chunk_size is not None:
        target_chunk_size = chunk_size
        num_chunks = (total_expanded_reads + target_chunk_size - 1) // target_chunk_size
    elif target_num_chunks is not None:
        num_chunks = target_num_chunks
        target_chunk_size = (total_expanded_reads + num_chunks - 1) // num_chunks
    else:
        target_chunk_size = 100000
        num_chunks = (total_expanded_reads + target_chunk_size - 1) // target_chunk_size


    chunk_metadata_list = []
    for chunk_idx in range(num_chunks):
        chunk_global_start = chunk_idx * target_chunk_size
        chunk_global_end = min((chunk_idx + 1) * target_chunk_size, total_expanded_reads)
        chunk_reads_count = chunk_global_end - chunk_global_start
        if chunk_reads_count <= 0:
            continue
        chunk_metadata_list.append({
            'chunk_idx': chunk_idx,
            'chunk_global_start': chunk_global_start,
            'chunk_global_end': chunk_global_end,
            'total_reads_in_chunk': chunk_reads_count,
        })

    return target_chunk_size, num_chunks, chunk_metadata_list


def simple_mode_merge_ordered_chunks(
    output_dir: Path,
    chunk_metadata_list: List[Dict],
    existing_chunk_files: set,
    total_cpus: int,
    merge_enabled: bool,
    timestamp_suffix: bool,
    input_fasta: str,
    progress_interval: int = None,
) -> Tuple[bool, Path, float]:
    """
    Merge ordered chunk files

    Returns:
        (success, merged file path, elapsed time)
    """
    if not merge_enabled or not CONST.COPY_FILE_RANGE_AVAILABLE:
        return True, None, 0.0

    from .mutations import _calculate_chunk_offsets_from_actual_sizes
    from .utils import (
        merge_chunks_parallel,
        cleanup_trailing_newlines,
        get_timestamp_string,
        get_merge_filename,
    )

    merge_start_time = time.time()
    merge_timestamp = get_timestamp_string() if timestamp_suffix else None
    merge_basename = Path(input_fasta).stem
    merged_filename = get_merge_filename(merge_basename, merge_timestamp, shuffled=False)
    output_merged = output_dir / merged_filename

    successful_chunks = sorted(
        [m for m in chunk_metadata_list if m['chunk_idx'] in existing_chunk_files],
        key=lambda x: x['chunk_idx']
    )
    if not successful_chunks:
        logger.warning("[WARNING] No successful chunks to merge, skipping")
        return True, None, 0.0

    chunk_offsets, total_file_size = _calculate_chunk_offsets_from_actual_sizes(
        successful_chunks, output_dir
    )

    merge_chunks_parallel(
        output_file=output_merged,
        chunk_list=successful_chunks,
        chunk_offsets=chunk_offsets,
        chunk_file_prefix="output_chunk_",
        output_dir=output_dir,
        total_file_size=total_file_size,
        total_cpus=total_cpus,
        progress_interval=progress_interval,
    )

    cleanup_trailing_newlines(output_merged)
    merge_elapsed = time.time() - merge_start_time
    print(f"[OK] Sequential file merge completed: {output_merged}")
    print(f"  Elapsed: {merge_elapsed:.2f}s, Size: {total_file_size / (1024**3):.2f} GB")
    return True, output_merged, merge_elapsed


def simple_mode_merge_shuffled_chunks(
    output_dir: Path,
    chunk_metadata_list: List[Dict],
    total_cpus: int,
    merge_enabled: bool,
    timestamp_suffix: bool,
    input_fasta: str,
    progress_interval: int = None,
    shuffle_split_dir=None,
) -> Tuple[bool, Path, float]:
    """
    Merge shuffled chunk files

    Returns:
        (success, merged file path, elapsed time)
    """
    if not merge_enabled or not CONST.COPY_FILE_RANGE_AVAILABLE:
        return True, None, 0.0

    if shuffle_split_dir is None or len(chunk_metadata_list) == 0:
        return True, None, 0.0

    from .mutations import _calculate_chunk_offsets_from_actual_sizes
    from .utils import (
        merge_chunks_parallel,
        cleanup_trailing_newlines,
        get_timestamp_string,
        get_merge_filename,
    )

    shuffled_successful = []
    shuffled_offsets = {}
    current_offset = 0
    for s in range(len(chunk_metadata_list)):
        chunk_file = output_dir / f"output_chunk_shuffled_{s}.fasta"
        if not chunk_file.exists():
            continue
        size = chunk_file.stat().st_size
        if size == 0:
            continue
        shuffled_offsets[s] = current_offset
        shuffled_successful.append({"chunk_idx": s})
        current_offset += size
    shuffled_total_size = current_offset
    if not shuffled_successful:
        return True, None, 0.0

    merge_start_time = time.time()
    merge_timestamp = get_timestamp_string() if timestamp_suffix else None
    merge_basename = Path(input_fasta).stem
    merged_shuffled_filename = get_merge_filename(merge_basename, merge_timestamp, shuffled=True)
    output_merged_shuffled = output_dir / merged_shuffled_filename

    merge_chunks_parallel(
        output_file=output_merged_shuffled,
        chunk_list=shuffled_successful,
        chunk_offsets=shuffled_offsets,
        chunk_file_prefix="output_chunk_shuffled_",
        output_dir=output_dir,
        total_file_size=shuffled_total_size,
        total_cpus=total_cpus,
        progress_interval=progress_interval,
    )

    cleanup_trailing_newlines(output_merged_shuffled)
    merge_elapsed = time.time() - merge_start_time
    print(f"[OK] Shuffled file merge completed: {output_merged_shuffled}")
    print(f"  Elapsed: {merge_elapsed:.2f}s, Size: {shuffled_total_size / (1024**3):.2f} GB")
    return True, output_merged_shuffled, merge_elapsed


def simple_mode_worker(
    worker_id: int,
    worker_tasks: List[Dict],
    input_fasta: str,
    expanded_index: np.ndarray,
    errors_by_expanded_idx: Dict[int, List[Dict]],
    seq_length: int,
    chunk_size: int,
    read_id_offset: int,
    output_dir: Path,
    split_dir: Path,
    rng_seed: int,
    progress_counter_worker,
    total_chunks: int,
    progress_interval: int,
    output_stats: bool = False,
):
    """Simple Mode worker function"""
    processed_count = 0
    fasta_handle = open(input_fasta, 'r', encoding='utf-8')
    ref_seq_cache: Dict[int, str] = {}
    rng = np.random.default_rng(rng_seed)

    try:
        for task in worker_tasks:
            chunk_idx = task['chunk_idx']
            chunk_global_start = task['chunk_global_start']
            chunk_global_end = task['chunk_global_end']
            chunk_reads_count = chunk_global_end - chunk_global_start
            if chunk_reads_count <= 0:
                continue

            needed_ref_indices = set()
            for global_idx in range(chunk_global_start, chunk_global_end):
                ref_idx = int(expanded_index[global_idx, 0])
                needed_ref_indices.add(ref_idx)

            for ref_idx in needed_ref_indices:
                if ref_idx not in ref_seq_cache:
                    fasta_handle.seek(0)
                    current_ref_idx = -1
                    seq_lines = []
                    found_target = False
                    for line in fasta_handle:
                        line = line.strip()
                        if line.startswith('>'):
                            if found_target:
                                break
                            current_ref_idx += 1
                            if current_ref_idx == ref_idx:
                                found_target = True
                        else:
                            if found_target:
                                seq_lines.append(line)
                    if not seq_lines:
                        raise ValueError(f"SimpleModeWorker {worker_id} Chunk {chunk_idx}: Cannot read ref_idx={ref_idx}")
                    ref_seq_cache[ref_idx] = ''.join(seq_lines)

            ids_table = np.zeros((chunk_reads_count, seq_length), dtype=np.uint8)
            read_ids = []

            from collections import defaultdict
            all_errors_by_read = []

            for local_idx in range(chunk_reads_count):
                global_idx = chunk_global_start + local_idx
                ref_idx = int(expanded_index[global_idx, 0])
                local_copy_idx = int(expanded_index[global_idx, 1])
                reads_seq_id = read_id_offset + global_idx
                read_id_str = f"seq_{reads_seq_id}"

                original_seq = ref_seq_cache[ref_idx]
                actual_len = min(len(original_seq), seq_length)
                errors = errors_by_expanded_idx.get(global_idx, [])

                s_by_pos = defaultdict(list)
                for e in errors:
                    if e['type'] == 'S':
                        s_by_pos[e['pos']].append(e)
                for pos, es in s_by_pos.items():
                    if len(es) > 1:
                        kept = es[-1]
                        ctx_start = max(0, pos - 5)
                        ctx_end = min(len(original_seq), pos + 6)
                        ref_ctx = original_seq[ctx_start:ctx_end]
                        logger.warning(
                            f"[{read_id_str}] Same position S at ref_pos={pos}: "
                            f"{[e['base'] for e in es]} -> keep '{kept['base']}' "
                            f"(ref_ctx: {ctx_start}:{ctx_end} = ...{ref_ctx}...)"
                        )
                        errors = [e for e in errors if not (e['type'] == 'S' and e['pos'] == pos and e is not kept)]

                read_ids.append(read_id_str)
                all_errors_by_read.append((read_id_str, errors))

                for e in errors:
                    pos = e['pos']
                    if pos >= seq_length:
                        continue
                    if e['type'] == 'I':
                        ids_table[local_idx, pos] = ord('K')
                    elif e['type'] == 'D':
                        orig_base = original_seq[pos]
                        ids_table[local_idx, pos] = _IDS_DELETE[orig_base.encode()]
                    elif e['type'] == 'S':
                        orig_base = original_seq[pos]
                        ids_table[local_idx, pos] = _IDS_SUBSTITUTE[orig_base.encode()]

            output_buffer = []
            for local_idx, (read_id_str, errors) in enumerate(all_errors_by_read):
                global_idx = chunk_global_start + local_idx
                ref_idx = int(expanded_index[global_idx, 0])
                original_seq = ref_seq_cache[ref_idx]
                noisy_seq = apply_explicit_errors(original_seq, errors)
                output_buffer.append(f">{read_id_str}\n")
                output_buffer.append(noisy_seq + "\n")

            chunk_filename = f"output_chunk_{chunk_idx}.fasta"
            chunk_path = output_dir / chunk_filename

            with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
                chunk_f.writelines(output_buffer)

            if output_stats:
                chunk_tsv_path = output_dir / f"read_to_ref_ordered_{chunk_idx}.tsv"
                tsv_buffer = []
                tsv_buffer_size = 100000
                _ref_counts = {}
                for _gi in range(chunk_global_start, chunk_global_end):
                    _ri = int(expanded_index[_gi, 0])
                    _ref_counts[_ri] = _ref_counts.get(_ri, 0) + 1
                _nonzero = [_r for _r, _c in _ref_counts.items() if _c > 0]
                _ref_idx_arr = np.array(_nonzero, dtype=np.uint64)
                _count_arr = np.array([_ref_counts[_r] for _r in _nonzero], dtype=np.uint64)
                _lookup = np.empty(chunk_reads_count, dtype=np.uint64)
                _pos = 0
                for _ki in range(len(_count_arr)):
                    _c = int(_count_arr[_ki])
                    _lookup[_pos:_pos + _c] = _ref_idx_arr[_ki]
                    _pos += _c

                for local_idx in range(chunk_reads_count):
                    reads_seq_global_id = read_id_offset + chunk_global_start + local_idx
                    read_id_str = f"seq_{reads_seq_global_id}"
                    ref_idx_0based = int(_lookup[local_idx])
                    ref_id_str = f"ref_{ref_idx_0based + 1}"
                    cigar = _ids_table_to_cigar(ids_table[local_idx])
                    md = _ids_table_to_md(ids_table[local_idx])
                    tsv_buffer.append(f"{read_id_str}\t{ref_id_str}\t{cigar}\t{md}\n")
                    if len(tsv_buffer) >= tsv_buffer_size:
                        with open(chunk_tsv_path, 'a', encoding='utf-8') as f_tsv:
                            f_tsv.write(''.join(tsv_buffer))
                        tsv_buffer.clear()
                if tsv_buffer:
                    with open(chunk_tsv_path, 'a', encoding='utf-8') as f_tsv:
                        f_tsv.write(''.join(tsv_buffer))
                del tsv_buffer, _lookup

            ref_counts_in_chunk = {}
            for global_idx in range(chunk_global_start, chunk_global_end):
                ref_idx = int(expanded_index[global_idx, 0])
                ref_counts_in_chunk[ref_idx] = ref_counts_in_chunk.get(ref_idx, 0) + 1
            nonzero_refs = [r for r, c in ref_counts_in_chunk.items() if c > 0]
            if nonzero_refs:
                ref_indices_arr = np.array(nonzero_refs, dtype=np.uint64)
                counts_arr = np.array([ref_counts_in_chunk[r] for r in nonzero_refs], dtype=np.uint64)
                split_data = np.vstack([ref_indices_arr, counts_arr])
                split_file = split_dir / f"chunk_{chunk_idx}_split.npy"
                np.save(split_file, split_data, allow_pickle=False)

                try:
                    with open(split_file, 'r+b') as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
                try:
                    split_dir_fd = os.open(str(split_dir), os.O_RDONLY | os.O_DIRECTORY)
                    os.fsync(split_dir_fd)
                    os.close(split_dir_fd)
                except Exception:
                    pass

            processed_count += chunk_reads_count
            with progress_counter_worker.get_lock():
                progress_counter_worker.value += 1
    finally:
        fasta_handle.close()
        ref_seq_cache.clear()


def parse_ref_copy_file(ref_copy_path: str, num_ref_seqs: int) -> Dict[int, int]:
    """
    Parse ref_copy.txt, return {ref_seq_global_idx (0-based): copy_count}.

    Format: ref_index (1-based) copy_count
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
                raise ValueError(f"ref_copy.txt line {line_no}: format error - expected 2 fields, got {len(parts)}")

            try:
                seq_index = int(parts[0])
            except ValueError:
                raise ValueError(f"ref_copy.txt line {line_no}: seq_index must be an integer, got '{parts[0]}'")
            if seq_index < 1:
                raise ValueError(f"ref_copy.txt line {line_no}: seq_index must be a positive integer >= 1")

            try:
                copy_count = int(parts[1])
            except ValueError:
                raise ValueError(f"ref_copy.txt line {line_no}: copy_count must be an integer, got '{parts[1]}'")

            if seq_index in seen_seq_indices:
                raise ValueError(f"ref_copy.txt line {line_no}: seq_index={seq_index} appears more than once")
            seen_seq_indices.add(seq_index)


            ref_seq_global_idx = seq_index - 1
            if ref_seq_global_idx >= num_ref_seqs:
                raise ValueError(
                    f"ref_copy.txt line {line_no}: seq_index={seq_index} exceeds number of reference sequences ({num_ref_seqs})"
                )
            ref_copy_map[ref_seq_global_idx] = copy_count

    return ref_copy_map


def parse_read_error_file(read_error_path: str, expanded_index: np.ndarray, total_expanded_reads: int) -> Dict[int, List[Dict]]:
    """
    Parse read_error.txt, distribute errors to corresponding reads by expanded_read_global_idx.

    Format: read_index (1-based) pos (1-based) type base
    - type: S=substitution, I=insertion, D=deletion
    - base: A/T/G/C for S/I, fixed 255 for D

    Returns:
        {expanded_read_global_idx (0-based): [errors]}
    """
    VALID_BASES = {'A', 'T', 'G', 'C'}
    DELETION_PLACEHOLDER = '255'

    if read_error_path is None:
        return {}

    read_error_file = Path(read_error_path)
    if not read_error_file.exists():
        raise FileNotFoundError(f"read_error.txt file does not exist: {read_error_path}")

    total_expanded_reads = expanded_index.shape[0] if expanded_index is not None and expanded_index.size > 0 else 0

    raw_errors: List[Dict] = []
    with open(read_error_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"read_error.txt line {line_no}: format error - expected 4 fields, got {len(parts)}")

            try:
                seq_index = int(parts[0])
            except ValueError:
                raise ValueError(f"read_error.txt line {line_no}: read_index must be an integer")
            if seq_index < 1:
                raise ValueError(f"read_error.txt line {line_no}: read_index must be a positive integer >= 1")

            try:
                pos = int(parts[1])
            except ValueError:
                raise ValueError(f"read_error.txt line {line_no}: pos must be an integer")
            if pos < 1:
                raise ValueError(f"read_error.txt line {line_no}: pos must be a positive integer >= 1")

            err_type = parts[2].upper()
            if err_type not in ('S', 'I', 'D'):
                raise ValueError(f"read_error.txt line {line_no}: type must be one of S/I/D")

            base = parts[3].upper()
            if err_type in ('S', 'I'):
                if base not in VALID_BASES:
                    raise ValueError(f"read_error.txt line {line_no}: for type={err_type}, base must be A/T/G/C")
            elif err_type == 'D':
                if base != DELETION_PLACEHOLDER:
                    raise ValueError(f"read_error.txt line {line_no}: for type=D, base must be fixed at 255")
                base = None

            expanded_read_global_idx = seq_index - 1
            pos_0based = pos - 1

            if expanded_read_global_idx >= total_expanded_reads:
                logger.warning(
                    f"read_error.txt line {line_no}: read_index={seq_index} exceeds total expanded reads"
                    f" ({total_expanded_reads}), this error will be ignored"
                )
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

    return errors_by_expanded_idx




def build_expanded_read_index(ref_copy_map: Dict[int, int], num_ref_seqs: int) -> Tuple[int, np.ndarray]:
    """Build expanded noise-free sequence index table based on ref_copy_map"""
    total_expanded_reads = sum(ref_copy_map.values())

    if total_expanded_reads == 0:
        logger.warning("All copy_count values in ref_copy_map are 0, total reads is 0")
        return 0, np.zeros((0, 2), dtype=np.uint64)

    expanded_index = np.zeros((total_expanded_reads, 2), dtype=np.uint64)
    pos = 0
    for ref_idx in range(num_ref_seqs):
        copy_count = ref_copy_map.get(ref_idx, 0)
        if copy_count > 0:
            for local_copy_idx in range(copy_count):
                expanded_index[pos, 0] = ref_idx
                expanded_index[pos, 1] = local_copy_idx
                pos += 1

    if pos != total_expanded_reads:
        raise RuntimeError(f"build_expanded_read_index internal error: pos={pos} != total_expanded_reads={total_expanded_reads}")

    return total_expanded_reads, expanded_index




def apply_explicit_errors(original_seq: str, errors: List[Dict]) -> str:
    """Apply explicit error plan to a single noise-free sequence"""
    if not errors:
        return original_seq

    seq_bytes = bytearray(original_seq.encode('ascii'))

    s_errors = [e for e in errors if e['type'] == 'S']
    if s_errors:
        positions = np.array([e['pos'] for e in s_errors], dtype=np.intp)
        subs_arr = np.frombuffer(b''.join(e['base'].encode() for e in s_errors), dtype=np.uint8)
        seq_bytes_np = np.frombuffer(seq_bytes, dtype=np.uint8).copy()
        seq_bytes_np[positions] = subs_arr
        seq_bytes = bytearray(seq_bytes_np)

    indel_errors = [e for e in errors if e['type'] in ('I', 'D')]
    indel_errors.sort(key=lambda x: (-x['pos'], x['line_no']))

    for err in indel_errors:
        if err['type'] == 'I':
            seq_bytes.insert(err['pos'], ord(err['base']))
        elif err['type'] == 'D':
            if 0 <= err['pos'] < len(seq_bytes):
                del seq_bytes[err['pos']]

    return seq_bytes.decode()




def parallel_simulate_errors_simple_mode(
    input_fasta: str,
    output_dir: str,
    seq_length: int,
    chunk_size: int = None,
    random_seed: int = 42,
    read_id_offset: int = 1,
    target_num_chunks: int = None,
    num_workers_global: int = 10,
    merge_files_enabled: bool = False,
    command_line: str = None,
    ref_copy_path: str = None,
    read_error_path: str = None,
    timestamp_suffix: bool = False,
    output_stats: bool = False,
):
    """Simple Mode (user-defined concise mode) main entry function"""
    if platform.system() == 'Windows':
        logger.error("This program only supports Linux/macOS, not Windows. Program terminated.")
        raise RuntimeError("Windows environment is not supported. Please use Linux or macOS.")

    total_start_time = time.time()

    total_cpus = num_workers_global
    num_workers = total_cpus - 2
    if num_workers < 1:
        num_workers = 1

    _, seq_length = simple_mode_stage0_validate(input_fasta, seq_length)
    num_ref_seqs = simple_mode_stage1_count_refs(input_fasta, num_workers)
    ref_copy_map = simple_mode_stage2_parse_ref_copy(ref_copy_path, num_ref_seqs)
    total_expanded_reads, expanded_index = simple_mode_stage3_build_expanded_index(ref_copy_map, num_ref_seqs)

    if total_expanded_reads == 0:
        logger.warning("Total expanded reads is 0, no output files generated")
        return {
            'num_ref_seqs': num_ref_seqs, 'total_reads': 0,
            'num_chunks': 0, 'chunk_size': 0,
            'seq_length': seq_length, 'simple_mode': True,
        }

    errors_by_expanded_idx = parse_read_error_file(read_error_path, expanded_index, total_expanded_reads)
    target_chunk_size, num_chunks, chunk_metadata_list = simple_mode_stage5_build_chunks(
        total_expanded_reads, chunk_size, target_num_chunks
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_chunks(output_dir)

    split_dir = output_dir / 'split'
    split_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_split_dir():
        if split_dir.exists():
            try:
                shutil.rmtree(split_dir)
            except Exception:
                pass
    _atexit.register(_cleanup_split_dir)

    if command_line:
        try:
            (output_dir / "command_line.txt").write_text(command_line.strip(), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write command line record: {e}")

    if len(chunk_metadata_list) > 0 and num_workers > len(chunk_metadata_list):
        num_workers = len(chunk_metadata_list)
    worker_tasks_list = [[] for _ in range(num_workers)]
    for chunk_meta in chunk_metadata_list:
        worker_id = chunk_meta['chunk_idx'] % num_workers
        worker_tasks_list[worker_id].append(chunk_meta)
    for worker_id in range(num_workers):
        worker_tasks_list[worker_id].sort(key=lambda x: x['chunk_idx'])

    progress_counter = multiprocessing.Value('i', 0)
    progress_interval = max(1, len(chunk_metadata_list) // 100)
    ss = np.random.SeedSequence(random_seed)
    child_seeds = ss.spawn(num_workers)

    workers = []
    for worker_id in range(num_workers):
        if not worker_tasks_list[worker_id]:
            continue
        p = multiprocessing.Process(
            target=simple_mode_worker,
            args=(
                worker_id, worker_tasks_list[worker_id], input_fasta, expanded_index,
                errors_by_expanded_idx, seq_length, target_chunk_size, read_id_offset,
                output_dir, split_dir, child_seeds[worker_id], progress_counter,
                len(chunk_metadata_list), progress_interval,
                output_stats,
            )
        )
        p.start()
        workers.append(p)
        if worker_id < num_workers - 1:
            time.sleep(0.1)

    for w in workers:
        w.join()

    expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
    existing_chunk_files = set()
    for chunk_file in output_dir.glob('output_chunk_*.fasta'):
        try:
            chunk_idx_str = chunk_file.stem[len('output_chunk_'):]
            chunk_idx = int(chunk_idx_str)
            if chunk_file.stat().st_size > 0:
                existing_chunk_files.add(chunk_idx)
        except (ValueError, OSError):
            continue

    missing_chunks = expected_chunk_indices - existing_chunk_files
    if missing_chunks:
        logger.warning(f"[WARNING] Missing {len(missing_chunks)} chunk files")

    shuffle_split_dir = None
    if not missing_chunks:
        shuffle_split_dir = generate_shuffle_mapping(
            chunk_metadata_list=chunk_metadata_list,
            split_dir=split_dir,
            total_reads=total_expanded_reads,
            chunk_size=target_chunk_size,
            output_dir=output_dir,
            master_seed=random_seed,
            read_id_offset=read_id_offset,
        )

        write_shuffle_chunks_parallel(
            shuffle_split_dir=shuffle_split_dir,
            output_dir=output_dir,
            chunk_metadata_list=chunk_metadata_list,
            chunk_size=target_chunk_size,
            read_id_offset=read_id_offset,
            num_workers=max(1, min(len(chunk_metadata_list), total_cpus - 2)),
        )

    if merge_files_enabled and not missing_chunks and output_dir.exists():
        simple_mode_merge_ordered_chunks(
            output_dir, chunk_metadata_list, existing_chunk_files,
            total_cpus, merge_files_enabled, timestamp_suffix, input_fasta, progress_interval
        )
        simple_mode_merge_shuffled_chunks(
            output_dir, chunk_metadata_list,
            total_cpus, merge_files_enabled, timestamp_suffix, input_fasta, progress_interval,
            shuffle_split_dir=shuffle_split_dir
        )

    if output_stats and not missing_chunks:
        write_ref_count_and_read_to_ref_tsv(
            chunk_metadata_list=chunk_metadata_list,
            split_dir=split_dir,
            num_ref_seqs=num_ref_seqs,
            total_reads=total_expanded_reads,
            output_dir=output_dir,
            read_id_offset=read_id_offset,
            shuffle_split_dir=shuffle_split_dir,
            chunk_size=target_chunk_size,
        )
        print("[OK] TSV file generation completed")

    if platform.system() != 'Linux':
        if split_dir and split_dir.exists():
            try:
                shutil.rmtree(split_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up seq_sampling_split directory: {e}")
        if shuffle_split_dir and shuffle_split_dir.exists():
            try:
                shutil.rmtree(shuffle_split_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up shuffle mapping directory: {e}")

    total_time = time.time() - total_start_time
    return {
        'num_ref_seqs': num_ref_seqs,
        'total_reads': total_expanded_reads,
        'num_chunks': len(chunk_metadata_list),
        'chunk_size': target_chunk_size,
        'seq_length': seq_length,
        'simple_mode': True,
        'output_file': str(output_dir),
        'num_workers': num_workers,
        'total_time': total_time,
    }


def _cleanup_old_chunks(output_dir: Path):
    """Clean up old chunk files in output directory"""
    old_ordered = list(output_dir.glob('output_chunk_[0-9]*.fasta'))
    old_shuffled = list(output_dir.glob('output_chunk_shuffled_*.fasta'))
    old_idx = list(output_dir.glob('output_chunk_*.fasta.idx.npy'))
    old_shuffled_idx = list(output_dir.glob('output_chunk_shuffled_*.fasta.idx.npy'))
    all_old = old_ordered + old_shuffled + old_idx + old_shuffled_idx

    if all_old:
        for f in all_old:
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"  Failed to clean up old file {f}: {e}")
