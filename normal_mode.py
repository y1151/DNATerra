"""Normal mode main pipeline module"""
import gc
import logging
import multiprocessing
import os
import platform
import random
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import CONST
from .coverage import (
    _build_fasta_chunk_record_index,
    _save_chunk_index,
)
from .errors import sample_errors_from_bucket
from .mutations import (
    _calculate_chunk_offsets_for_shuffled,
    _calculate_chunk_offsets_from_actual_sizes,
    _merge_chunk_zero_copy,
    _merge_worker,
    apply_indels_and_write,
    apply_substitutions,
    apply_substitutions_kmer,
    prepare_indels,
    prepare_indels_kmer,
    sample_insertions_kmer,
    sample_substitutions_kmer,
)
from .utils import (
    _format_error_rate,
    _ids_table_to_cigar,
    _ids_table_to_md,
    _IDS_SUBSTITUTE,
    _IDS_DELETE,
    _load_split_file,
    _open_fasta_file_with_retry,
    _readline_with_retry,
    _seek_with_retry,
)

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)

INDEL_SAFETY_FACTOR = 1.3


def error_simulator_worker(worker_id: int,
                          worker_tasks: List[Dict],
                          fasta_file: str,
                          split_dir: Path,
                          error_rates: Dict,
                          bucket_metadata: Dict,
                          bucket_sampling_counts: np.ndarray,
                          bucket_idx_to_array_idx: Dict[int, int],
                          sub_choices: np.ndarray,
                          sub_cum_probs: np.ndarray,
                          insertion_probs: np.ndarray,
                          seq_length: int,
                          chunk_size: int,
                          read_id_offset: int,
                          output_dir: Path,
                          rng_seed: int,
                          progress_counter_worker = None,
                          num_ref_seqs: int = 0,
                          failed_chunks = None,
                          total_chunks: int = 0,
                          progress_interval: int = 100,
                          use_kmer: bool = False,
                          kmer_sub: Optional[np.ndarray] = None,
                          kmer_ins: Optional[np.ndarray] = None,
                          output_stats: bool = False):
    """Error simulation Worker process: pre-read + sampling + mutation + write integration"""
    fasta_file_handle = None
    current_seq_idx = 0
    
    rng = np.random.default_rng(rng_seed)
    processed_count = 0
    
    worker_exit_code = 0
    
    chunk_io_retry_count = {}
    
    try:
        try:
            fasta_file_handle = _open_fasta_file_with_retry(fasta_file, worker_id, max_retries=10)
            
            split_data_cache = {}
            if chunk_size >= 1000000:
                cache_max_size = 2
                load_batch_size = 1
                preload_batch_size = 2
            else:
                preload_batch_size = CONST.SPLIT_FILE_PRELOAD_BATCH_SIZE
                load_batch_size = CONST.SPLIT_FILE_LOAD_BATCH_SIZE
                cache_max_size = CONST.SPLIT_FILE_CACHE_MAX_SIZE
            
            preload_count = min(preload_batch_size, len(worker_tasks))
            
            for task in worker_tasks[:preload_count]:
                chunk_idx = task['chunk_idx']
                try:
                    ref_indices, counts = _load_split_file(worker_id, chunk_idx, split_dir)
                    split_data_cache[chunk_idx] = (ref_indices, counts)
                except Exception as e:
                    raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Failed to preload split file: {e}")
            
            
            def _ensure_split_in_cache(chunk_idx: int):
                if chunk_idx in split_data_cache:
                    return
                
                if len(split_data_cache) >= cache_max_size:
                    current_task_idx = None
                    for idx, task in enumerate(worker_tasks):
                        if task['chunk_idx'] == chunk_idx:
                            current_task_idx = idx
                            break
                    
                    if current_task_idx is not None:
                        chunks_to_delete = []
                        for idx in range(max(0, current_task_idx - load_batch_size), current_task_idx):
                            task_chunk_idx = worker_tasks[idx]['chunk_idx']
                            if task_chunk_idx in split_data_cache:
                                chunks_to_delete.append(task_chunk_idx)
                        
                        if len(chunks_to_delete) < load_batch_size:
                            for idx in range(max(0, current_task_idx - cache_max_size), max(0, current_task_idx - load_batch_size)):
                                task_chunk_idx = worker_tasks[idx]['chunk_idx']
                                if task_chunk_idx in split_data_cache and task_chunk_idx not in chunks_to_delete:
                                    chunks_to_delete.append(task_chunk_idx)
                                    if len(chunks_to_delete) >= load_batch_size:
                                        break
                        
                        for chunk_idx_to_del in chunks_to_delete[:load_batch_size]:
                            if chunk_idx_to_del in split_data_cache:
                                del split_data_cache[chunk_idx_to_del]

                    else:
                        cached_count = len(split_data_cache)
                        split_data_cache.clear()
                
                current_task_idx = None
                for idx, task in enumerate(worker_tasks):
                    if task['chunk_idx'] == chunk_idx:
                        current_task_idx = idx
                        break
                
                if current_task_idx is None:
                    try:
                        ref_indices, counts = _load_split_file(worker_id, chunk_idx, split_dir)
                        split_data_cache[chunk_idx] = (ref_indices, counts)
                    except Exception as e:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Failed to load split file on demand: {e}")
                    return
                
                loaded_count = 0
                for idx in range(current_task_idx, min(current_task_idx + load_batch_size, len(worker_tasks))):
                    task_chunk_idx = worker_tasks[idx]['chunk_idx']
                    if task_chunk_idx not in split_data_cache:
                        try:
                            ref_indices, counts = _load_split_file(worker_id, task_chunk_idx, split_dir)
                            split_data_cache[task_chunk_idx] = (ref_indices, counts)
                            loaded_count += 1
                        except Exception as e:
                            raise ValueError(f"Worker{worker_id} Chunk {task_chunk_idx}: Failed to batch load split files: {e}")
            
            for task_idx, task in enumerate(worker_tasks):
                chunk_idx = task['chunk_idx']
                chunk_meta = task['chunk_meta']
                bucket_idx = chunk_idx
                
                chunk_buffer = None
                
                try:
                    reads_seq_global_id_start = chunk_idx * chunk_size + read_id_offset
                    
                    _ensure_split_in_cache(chunk_idx)
                    ref_indices, counts = split_data_cache[chunk_idx]
                    
                    seq_sampling_split = dict(zip(ref_indices, counts))
                    split_keys = list(seq_sampling_split.keys())
                    
                    del ref_indices, counts
                    
                    if not split_keys:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Empty chunk (split_keys is empty), which indicates data inconsistency or file corruption.")
                    
                    required_seq_indices = set(split_keys)
                    original_sequences = {}
                    
                    sorted_required_indices = sorted(required_seq_indices)
                    if not sorted_required_indices:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: required_seq_indices is empty")
                    
                    first_required_idx = sorted_required_indices[0]
                    
                    if current_seq_idx > first_required_idx:
                        _seek_with_retry(fasta_file_handle, 0, worker_id, chunk_idx, max_retries=10)
                        current_seq_idx = 0
                    
                    max_required_idx = max(sorted_required_indices)
                    min_required_idx = min(sorted_required_indices)
                    if max_required_idx >= num_ref_seqs:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: ref_idx recorded in split file exceeds range (max ref_idx={max_required_idx} >= num_ref_seqs={num_ref_seqs})")
                    if min_required_idx < 0:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: ref_idx recorded in split file is negative (min ref_idx={min_required_idx}), which is invalid")
                    
                    while current_seq_idx < first_required_idx:
                        
                        header_line = _readline_with_retry(fasta_file_handle, worker_id, chunk_idx, max_retries=10)
                        if not header_line:
                            raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: File ended but first required sequence not reached yet (current_seq_idx={current_seq_idx} < first_required_idx={first_required_idx}, skipped {skip_count} sequences)")
                        if not header_line.startswith(b'>'):
                                    continue
                        sequence_line = _readline_with_retry(fasta_file_handle, worker_id, chunk_idx, max_retries=10)
                        if not sequence_line:
                            raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: File ended but first required sequence not reached yet (current_seq_idx={current_seq_idx} < first_required_idx={first_required_idx}, skipped {skip_count} sequences)")
                        current_seq_idx += 1
                    
                    if current_seq_idx != first_required_idx:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: File pointer positioning error (current_seq_idx={current_seq_idx} != first_required_idx={first_required_idx}, skipped {skip_count} sequences)")
                    
                    next_required_idx_ptr = 0
                    
                    while next_required_idx_ptr < len(sorted_required_indices):
                        
                        if current_seq_idx > sorted_required_indices[next_required_idx_ptr]:
                            missing_idx = sorted_required_indices[next_required_idx_ptr]
                            logger.error(f"Worker{worker_id} Chunk {chunk_idx}: Sequence index mismatch")
                            logger.error(f"  Required sequence indices: {sorted_required_indices}")
                            logger.error(f"  Already read sequence indices: {sorted(original_sequences.keys())}")
                            logger.error(f"  Current file position: current_seq_idx={current_seq_idx}")
                            logger.error(f"  Missing sequence index: {missing_idx}")
                            raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Sequence {missing_idx} is missing (file position has passed this sequence index, current_seq_idx={current_seq_idx} > required={missing_idx}). This may be because ref_idx in chunk is non-contiguous but the file pointer missed this sequence during positioning.")
                        
                        header_line = _readline_with_retry(fasta_file_handle, worker_id, chunk_idx, max_retries=10)
                        if not header_line:
                            missing_indices = sorted_required_indices[next_required_idx_ptr:]
                            logger.error(f"Worker{worker_id} Chunk {chunk_idx}: File ended but there are still sequences not read")
                            logger.error(f"  Required sequence indices: {sorted_required_indices}")
                            logger.error(f"  Already read sequence indices: {sorted(original_sequences.keys())}")
                            logger.error(f"  Missing sequence indices: {missing_indices}")
                            raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Sequences {missing_indices} are missing (file ended, current position current_seq_idx={current_seq_idx})")
                        
                        if not header_line.startswith(b'>'):
                                    continue
                        
                        seq_idx = current_seq_idx
                        
                        sequence_line = _readline_with_retry(fasta_file_handle, worker_id, chunk_idx, max_retries=10)
                        if not sequence_line:
                            missing_indices = sorted_required_indices[next_required_idx_ptr:]
                            logger.error(f"Worker{worker_id} Chunk {chunk_idx}: File ended but there are still sequences not read")
                            logger.error(f"  Required sequence indices: {sorted_required_indices}")
                            logger.error(f"  Already read sequence indices: {sorted(original_sequences.keys())}")
                            logger.error(f"  Missing sequence indices: {missing_indices}")
                            raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Sequences {missing_indices} are missing (file ended, current position current_seq_idx={current_seq_idx})")
                        
                        sequence = sequence_line.rstrip(b'\r\n')
                        
                        if seq_idx in required_seq_indices:
                            original_sequences[seq_idx] = sequence
                            next_required_idx_ptr += 1
                        
                        current_seq_idx += 1
                    
                    still_missing = required_seq_indices - set(original_sequences.keys())
                    if still_missing:
                        logger.error(f"Worker{worker_id} Chunk {chunk_idx}: Sequence missing details")
                        logger.error(f"  Required sequences: {sorted(required_seq_indices)}")
                        logger.error(f"  Already read sequences: {sorted(original_sequences.keys())}")
                        logger.error(f"  Missing sequences: {sorted(still_missing)}")
                        logger.error(f"  File pointer current position: current_seq_idx={current_seq_idx}")
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: Sequences {still_missing} are missing (total {len(still_missing)} sequences missing, required {len(required_seq_indices)}, already read {len(original_sequences)})")
                    
                    chunk_buffer = np.empty((chunk_size, seq_length), dtype=np.uint8)
                    
                    buffer_offset = 0
                    for ref_idx in sorted(seq_sampling_split.keys()):
                        count = seq_sampling_split[ref_idx]
                        if count == 0:
                                    continue
                        seq_bytes = original_sequences[ref_idx]
                        seq_array = np.frombuffer(seq_bytes, dtype=np.uint8)
                        seq_len = len(seq_array)
                        chunk_buffer[buffer_offset:buffer_offset+count, :seq_len] = seq_array
                        buffer_offset += count
                        del seq_array
                    
                    chunk_reads_actual_count = buffer_offset
                    
                    if chunk_buffer is None or chunk_buffer.shape != (chunk_size, seq_length):
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: chunk_buffer creation failed or shape incorrect")
                    if chunk_reads_actual_count == 0:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: chunk_reads_actual_count is 0, no sequences were filled into buffer")
                    
                    del seq_sampling_split, split_keys, required_seq_indices, sorted_required_indices, seq_bytes
                    
                    error_records = sample_errors_from_bucket(
                        bucket_idx=bucket_idx,
                        bucket_metadata=bucket_metadata,
                        bucket_sampling_counts=bucket_sampling_counts,
                        bucket_idx_to_array_idx=bucket_idx_to_array_idx,
                        error_rates=error_rates,
                        rng=rng
                    )
                    
                    if len(error_records) > 0:
                        sort_order = np.lexsort((-error_records['error_pos'], error_records['chunk_local_reads_idx']))
                        error_records = error_records[sort_order]
                        del sort_order
                    
                    buffer = chunk_buffer[:chunk_reads_actual_count, :seq_length]

                    if buffer is None or buffer.shape[0] != chunk_reads_actual_count:
                        raise ValueError(f"Worker{worker_id} Chunk {chunk_idx}: buffer creation failed or shape incorrect")

                    ids_table = np.zeros((chunk_reads_actual_count, seq_length), dtype=np.uint8)

                    sub_records = None
                    sub_local_idx = None
                    indel_records = None
                    indel_local_idx = None
                    sorted_read_idx, sorted_pos, sorted_types, sorted_bases = None, None, None, None
                    n_indels = 0
                    sub_new_bases = None

                    if len(error_records) > 0:
                        chunk_local_idx = error_records['chunk_local_reads_idx']

                        sub_mask = (error_records['error_type'] == CONST.SUB)
                        indel_mask = (error_records['error_type'] != CONST.SUB)

                        if np.sum(sub_mask) > 0:
                            sub_records = error_records[sub_mask]
                            sub_local_idx = chunk_local_idx[sub_mask]
                        else:
                            sub_records = None
                            sub_local_idx = None

                        if sub_records is not None:
                            for i in range(len(sub_records)):
                                local_idx = sub_local_idx[i]
                                pos = sub_records['error_pos'][i]
                                orig_base_byte = buffer[local_idx, pos]
                                ids_table[local_idx, pos] = _IDS_SUBSTITUTE[bytes([orig_base_byte])]

                        if np.sum(indel_mask) > 0:
                            indel_records = error_records[indel_mask]
                            indel_local_idx = chunk_local_idx[indel_mask]
                            sort_order = np.lexsort((-indel_records['error_pos'], indel_local_idx))
                            indel_records = indel_records[sort_order]
                            indel_local_idx = indel_local_idx[sort_order]
                            del sort_order
                        else:
                            indel_records = None
                            indel_local_idx = None

                        if indel_records is not None:
                            for i in range(len(indel_records)):
                                local_idx = indel_local_idx[i]
                                pos = indel_records['error_pos'][i]
                                etype = indel_records['error_type'][i]
                                if etype == CONST.INS:
                                    ids_table[local_idx, pos] = ord('K')
                                elif etype == CONST.DEL:
                                    orig_base_byte = buffer[local_idx, pos]
                                    ids_table[local_idx, pos] = _IDS_DELETE[bytes([orig_base_byte])]

                        if indel_records is not None:
                            if use_kmer:
                                ins_mask_kmer = (indel_records['error_type'] == CONST.INS)
                                if np.any(ins_mask_kmer):
                                    ins_bases_pre = sample_insertions_kmer(
                                        buffer,
                                        indel_local_idx[ins_mask_kmer],
                                        indel_records['error_pos'][ins_mask_kmer],
                                        kmer_ins, insertion_probs, seq_length, rng
                                    )
                                else:
                                    ins_bases_pre = np.empty(0, dtype=np.uint8)
                                sorted_read_idx, sorted_pos, sorted_types, sorted_bases, n_ins, n_dels, n_indels = prepare_indels_kmer(
                                    indel_local_idx, indel_records['error_pos'],
                                    indel_records['error_type'], ins_bases_pre
                                )
                            else:
                                sorted_read_idx, sorted_pos, sorted_types, sorted_bases, n_ins, n_dels, n_indels = prepare_indels(
                                    indel_local_idx, indel_records['error_pos'],
                                    indel_records['error_type'], rng, insertion_probs
                                )
                        else:
                            sorted_read_idx = None
                            sorted_pos = None
                            sorted_types = None
                            sorted_bases = None
                            n_indels = 0

                        if sub_records is not None:
                            if use_kmer:
                                sub_new_bases = sample_substitutions_kmer(
                                    buffer, sub_local_idx, sub_records['error_pos'],
                                    kmer_sub, sub_choices, sub_cum_probs, seq_length, rng
                                )
                                apply_substitutions_kmer(
                                    buffer, sub_local_idx, sub_records['error_pos'], sub_new_bases
                                )
                            else:
                                apply_substitutions(
                                    buffer, sub_local_idx, sub_records['error_pos'],
                                    sub_choices, sub_cum_probs, rng
                                )

                    output_file = output_dir / f'output_chunk_{chunk_idx}.fasta'
                    with open(output_file, 'wb', buffering=8*1024*1024) as f_out:
                        actual_file_size = apply_indels_and_write(
                            buffer, chunk_reads_actual_count,
                            sorted_read_idx, sorted_pos, sorted_types, sorted_bases, n_indels,
                            f_out, b'>seq_', b'\n', reads_seq_global_id_start,
                            None
                        )
                        try:
                            f_out.flush()
                            os.fsync(f_out.fileno())
                        except Exception as e:
                            logger.warning(f"Worker{worker_id} Chunk {chunk_idx}: fsync failed: {e}, data may not be fully written to disk")

                    if output_stats:
                        _split_file = split_dir / f"chunk_{chunk_idx}_split.npy"
                        _split_data = np.load(_split_file, allow_pickle=False)
                        _ref_indices = _split_data[0]
                        _counts = _split_data[1]
                        _lookup = np.empty(chunk_reads_actual_count, dtype=np.uint64)
                        _pos = 0
                        for _ki in range(len(_counts)):
                            _c = int(_counts[_ki])
                            _lookup[_pos:_pos + _c] = _ref_indices[_ki]
                            _pos += _c
                        del _split_data

                        chunk_tsv_path = output_dir / f'read_to_ref_ordered_{chunk_idx}.tsv'
                        tsv_buffer = []
                        tsv_buffer_size = 100000
                        for local_idx in range(chunk_reads_actual_count):
                            reads_seq_global_id = reads_seq_global_id_start + local_idx
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

                    try:
                        chunk_index = _build_fasta_chunk_record_index(output_file)
                        _save_chunk_index(output_file, chunk_index)
                    except Exception as e:
                        logger.warning(f"Worker{worker_id} Chunk {chunk_idx}: Index generation/save failed: {e}, will build index on-the-fly during shuffle phase")

                    processed_count += 1
                    if progress_counter_worker is not None:
                        with progress_counter_worker.get_lock():
                            progress_counter_worker.value += 1
                    
                    if chunk_idx in split_data_cache:
                        del split_data_cache[chunk_idx]
                    
                    del error_records, buffer, chunk_buffer, original_sequences

                    try:
                        del sorted_read_idx, sorted_pos, sorted_types, sorted_bases
                    except NameError:
                        pass

                    try:
                        del sub_records, indel_records, chunk_local_idx
                    except NameError:
                        pass

                    try:
                        del sub_local_idx, indel_local_idx
                    except NameError:
                        pass

                    try:
                        del sub_mask, indel_mask
                    except NameError:
                        pass

                    try:
                        del sub_new_bases
                    except NameError:
                        pass

                    try:
                        del ins_bases_pre, ins_mask_kmer
                    except NameError:
                        pass

                    try:
                        del ids_table
                    except NameError:
                        pass

                    chunk_buffer = None

                    if (task_idx + 1) % 10 == 0:
                        gc.collect()
                
                except (IOError, OSError, PermissionError) as io_err:
                    chunk_retry_count = chunk_io_retry_count.get(chunk_idx, 0)
                    max_chunk_retries = 3
                    
                    if chunk_retry_count < max_chunk_retries:
                        base_delay = 0.5 * (2 ** chunk_retry_count)
                        retry_delay = base_delay + random.uniform(0, base_delay * 0.2)
                        
                        logger.warning(f"Worker{worker_id} Chunk {chunk_idx}: I/O exception (retry {chunk_retry_count+1}/{max_chunk_retries}): {io_err}, will retry current chunk after {retry_delay:.2f}s...")
                        
                        chunk_io_retry_count[chunk_idx] = chunk_retry_count + 1
                        
                        time.sleep(retry_delay)
                        
                        try:
                            if 'chunk_buffer' in locals() and chunk_buffer is not None:
                                del chunk_buffer
                            chunk_buffer = None
                        except:
                            pass
                        
                        task_idx -= 1
                        continue
                    else:
                        logger.error(f"Worker{worker_id} Chunk {chunk_idx}: I/O exception retry attempts exhausted (tried {max_chunk_retries} times): {io_err}")
                        if chunk_idx in chunk_io_retry_count:
                            del chunk_io_retry_count[chunk_idx]
                        e = io_err
                        
                except Exception as e:
                    import traceback as _tb
                    logger.error(
                        f"Worker{worker_id} Chunk {chunk_idx}: Processing exception, skipping this chunk!\n"
                        f"  use_kmer={use_kmer}\n"
                        f"  Exception type: {type(e).__name__}: {e}\n"
                        f"  Stack trace:\n{_tb.format_exc()}"
                    )
                    try:
                        if 'chunk_buffer' in locals() and chunk_buffer is not None:
                            del chunk_buffer
                        chunk_buffer = None
                    except:
                        pass

                    if chunk_idx in split_data_cache:
                        try:
                            del split_data_cache[chunk_idx]
                        except:
                            pass

                    continue

                if chunk_idx in chunk_io_retry_count:
                    del chunk_io_retry_count[chunk_idx]
            
        except Exception as init_err:
            worker_exit_code = 1
            logger.error(f"Worker{worker_id} Uncaught exception occurred (initialization or main loop): {init_err}")
            import traceback
            logger.error(f"Worker{worker_id} Exception stack trace:\n{traceback.format_exc()}")
            raise
        
        finally:
            try:
                if fasta_file_handle is not None:
                    fasta_file_handle.close()
            except Exception as cleanup_err:
                logger.warning(f"Worker{worker_id} Error cleaning up FASTA file handle: {cleanup_err}")
            
            try:
                if 'split_data_cache' in locals():
                    split_data_cache.clear()
            except Exception as cleanup_err:
                logger.warning(f"Worker{worker_id} Error cleaning up split cache: {cleanup_err}")
            
            if worker_exit_code == 0:
                pass
            else:
                expected_count = len(worker_tasks) if 'worker_tasks' in locals() else 0
                logger.error(f"Mutation process Worker{worker_id} exited abnormally: processed {processed_count}/{expected_count} chunks")
    
    except Exception as outer_err:
        worker_exit_code = 1
        logger.critical(f"Worker{worker_id} Critical exception occurred, process will exit: {outer_err}")
        import traceback
        logger.critical(f"Worker{worker_id} Full exception stack trace:\n{traceback.format_exc()}")
        
        raise


def parallel_simulate_errors(input_fasta: str,
                            output_dir: str,
                            synthesis_method: str = "electro",
                            seq_length: int = None,
                            chunk_size: int = None,
                            random_seed: int = 42,
                            read_id_offset: int = 1,
                            target_read_depth: float = None,
                            target_num_chunks: int = None,
                            dist_name: str = None,
                            cv: float = None,
                            beta_min: float = None,
                            beta_max: float = None,
                            drop_rate: float = None,
                            num_workers_global: int = 10,
                            custom_position_rates: Dict = None,
                            error_rate_input_type: str = None,
                            num_ref_seqs: int = None,
                            merge_files_enabled: bool = False,
                            command_line: str = None,
                            use_kmer: bool = False,
                            timestamp_suffix: bool = False,
                            shuffle_enabled: bool = False,
                            output_stats: bool = False,
                            precomputed_data: Dict = None):
    """Parallel DNA sequence error simulator - main function"""
    from . import CONST
    from .utils import (
        validate_fasta_format_and_length,
        get_timestamp_string,
        get_merge_filename,
    )
    from .config import load_synthesis_config, build_substitution_lookup_from_matrix
    from .errors import compute_bucket_metadata
    from .coverage import (
        count_sequences_with_seqkit,
        calculate_depth_distribution_params,
        split_refs_into_chunks_streaming,
        write_ref_count_and_read_to_ref_tsv,
        generate_shuffle_mapping,
        write_shuffle_chunks_parallel,
    )

    if precomputed_data is not None:
        synthesis_config = precomputed_data['synthesis_config']
        seq_length = precomputed_data['seq_length']
        num_ref_seqs = precomputed_data['num_ref_seqs']
        chunk_metadata_list = precomputed_data['chunk_metadata_list']
        total_reads = precomputed_data['total_reads']
        dist_name = precomputed_data['dist_name']
        dist_params = precomputed_data['dist_params']
        target_read_depth = precomputed_data['target_read_depth']
        error_rates = precomputed_data['error_rates']
        sub_choices = precomputed_data['sub_choices']
        sub_cum_probs = precomputed_data['sub_cum_probs']
        insertion_probs = precomputed_data['insertion_probs']
        kmer_sub = precomputed_data['kmer_sub']
        kmer_ins = precomputed_data['kmer_ins']
        _use_kmer = precomputed_data['_use_kmer']
        error_rate_source = precomputed_data['error_rate_source']
        user_input_total_error_rate = precomputed_data.get('user_input_total_error_rate')
        user_input_sub_error_rate = precomputed_data.get('user_input_sub_error_rate')
        user_input_ins_error_rate = precomputed_data.get('user_input_ins_error_rate')
        user_input_del_error_rate = precomputed_data.get('user_input_del_error_rate')
        total_cpus = precomputed_data['total_cpus']
        reads_seq_id_width = precomputed_data['reads_seq_id_width']
        ref_id_width = len(str(num_ref_seqs))
        split_dir = Path(precomputed_data['split_dir'])
        chunk_size = precomputed_data.get('chunk_size')
        user_input_chunk_size = chunk_size
        output_dir = Path(precomputed_data['output_dir'])
        merge_basename = Path(input_fasta).stem
        merge_timestamp = get_timestamp_string() if timestamp_suffix else None
        merge_ts_flag = timestamp_suffix
        precompute_start_time = None
    else:
        script_dir = Path(__file__).parent
        
        print("\nStage 1: Parameter validation and processing")
        
        try:
            detected_seq_length = validate_fasta_format_and_length(input_fasta)
            print(f"  Detected sequence length: {detected_seq_length} bp")
        except Exception as e:
            logger.error(f"FASTA validation failed, program terminating")
            raise
        
        if seq_length is not None:
            if seq_length != detected_seq_length:
                error_msg = (
                    f"Error: Specified sequence length ({seq_length}bp) does not match "
                    f"the sequence length in the input FASTA file ({detected_seq_length}bp)!"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        print(f"  Sequence length: {seq_length} bp")
        
        if seq_length > 65535:
            error_msg = (
                f"Error: Sequence length ({seq_length}bp) exceeds uint16 maximum value (65,535bp).\n"
                f"Current code uses uint16 to store error positions, does not support sequences longer than 65535bp."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        print(f"  Data type safety check: PASSED (sequence length < 65535)")
        
        synthesis_config = load_synthesis_config(
            synthesis_method,
            script_dir,
            target_seq_length=seq_length,
            use_kmer=use_kmer,
            drop_rate=drop_rate,
        )
        if drop_rate is None:
            effective_drop_rate = synthesis_config.get('dropout_rate', 0.0)
        else:
            effective_drop_rate = drop_rate
        config_seq_length = synthesis_config['seq_length']
        if config_seq_length != seq_length:
            print(f"  Scaled sequence length: {config_seq_length} bp (scaled from {seq_length} bp)")
        seq_length = config_seq_length
        print(f"  Using config: {synthesis_config['method_name']}")
        
        num_workers_global = num_workers_global if num_workers_global > 0 else DEFAULT_TOTAL_CPUS
        total_cpus = num_workers_global
        
        print("\nStage 2: Generate sequence-level sequencing coverage depth")
        from . import CONST
        seqkit_threads = min(CONST.SEQKIT_MAX_THREADS, total_cpus - 2)
        actual_num_ref_seqs = count_sequences_with_seqkit(input_fasta, num_threads=seqkit_threads)
        
        if num_ref_seqs is not None:
            if num_ref_seqs != actual_num_ref_seqs:
                warning_msg = (
                    f"\nWarning: User-provided ref sequence count ({num_ref_seqs:,}) does not match "
                    f"the actual sequence count in the input file ({actual_num_ref_seqs:,})!\n"
                    f"We will not split the file, but proceed with the actual sequence count ({actual_num_ref_seqs:,}) for subsequent processing."
                )
                print(warning_msg)
                logger.warning(warning_msg.strip())
                num_ref_seqs = actual_num_ref_seqs
            else:
                num_ref_seqs = actual_num_ref_seqs
        else:
            num_ref_seqs = actual_num_ref_seqs
        
        print(f"  Ref sequence count: {num_ref_seqs:,}")
        
        print("Sampling distribution parameters")
        
        if dist_name is not None:
            print(f"  Distribution type: {dist_name}")
            print(f"  Target sequencing depth: {target_read_depth}x")
            print(f"  Coefficient of variation (CV): {cv}")
            depth_dist_info = calculate_depth_distribution_params(dist_name, target_read_depth, cv,
                                                                    beta_min=beta_min, beta_max=beta_max)
            print(f"  Distribution parameters: {depth_dist_info['params']}")
        else:
            default_config = DEFAULT_DEPTH_DISTRIBUTION.get(synthesis_method)
            if default_config is None:
                avg_coverage = synthesis_config['depth_params']['avg_coverage']
                cv = synthesis_config['depth_params']['cv']
                depth_dist_info = calculate_depth_distribution_params('lognormal', avg_coverage, cv=cv)
            else:
                dist_name = default_config['dist']
                cv = default_config['cv']
                print(f"  Distribution type: {dist_name}")
                print(f"  Coefficient of variation (CV): {cv}")
                avg_coverage = synthesis_config['depth_params']['avg_coverage']
                if target_read_depth is None:
                    target_read_depth = avg_coverage
                depth_dist_info = calculate_depth_distribution_params(
                    dist_name, target_read_depth, cv,
                    beta_min=beta_min if beta_min is not None else default_config.get('beta_min'),
                    beta_max=beta_max if beta_max is not None else default_config.get('beta_max'),
                )
            
            print(f"  Distribution parameters: {depth_dist_info['params']}")
        
        dist_name = depth_dist_info['dist_name']
        dist_params = depth_dist_info['params']
        
        if target_read_depth is not None:
            target_num_read_seqs = int(num_ref_seqs * target_read_depth)
        else:
            target_read_depth = 133.4
            target_num_read_seqs = int(num_ref_seqs * target_read_depth)
        
        estimated_total_reads = target_num_read_seqs
        
        if chunk_size is not None:
            pass
        else:
            if estimated_total_reads < DEFAULT_CHUNK_SIZE:
                chunk_size = estimated_total_reads
            else:
                chunk_size = DEFAULT_CHUNK_SIZE
        
        if chunk_size >= 2**32:
            error_msg = (
                f"Error: chunk_size ({chunk_size:,}) exceeds uint32 maximum value (4,294,967,295).\n"
                f"Current code uses uint32 to store chunk internal sequence indices, does not support chunk_size exceeding 4.2 billion."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        chunk_metadata_list, split_dir, total_reads, num_nonzero = split_refs_into_chunks_streaming(
            num_ref_seqs=num_ref_seqs,
            target_read_depth=target_read_depth,
            dist_name=dist_name,
            dist_params=dist_params,
            random_seed=random_seed,
            chunk_size=chunk_size,
            split_dir=output_dir / 'split',
            num_parallel_workers=total_cpus - 1,
            target_num_chunks=None,
            total_cpus=total_cpus,
            drop_rate=effective_drop_rate,
            seq_length=seq_length,
        )
        split_dir = Path(split_dir)

        max_reads_seq_global_id = read_id_offset + total_reads - 1
        reads_seq_id_width = len(str(int(max_reads_seq_global_id)))
        ref_id_width = len(str(num_ref_seqs))
        
        print(f"  Total chunk count={len(chunk_metadata_list):,}, total reads={total_reads:,}")
        
        user_input_total_error_rate = None
        user_input_sub_error_rate = None
        user_input_ins_error_rate = None
        user_input_del_error_rate = None

        from .errors import load_error_rates_from_config
        if custom_position_rates is not None:
            user_input_total_error_rate = custom_position_rates.get('user_input_total_error_rate')
            user_input_sub_error_rate = custom_position_rates.get('user_input_sub_error_rate')
            user_input_ins_error_rate = custom_position_rates.get('user_input_ins_error_rate')
            user_input_del_error_rate = custom_position_rates.get('user_input_del_error_rate')
            
            error_rates = load_error_rates_from_config(custom_position_rates, seq_length)
            if error_rate_input_type:
                error_rate_source = error_rate_input_type
            elif user_input_total_error_rate is not None:
                error_rate_source = "Custom total error rate"
            elif user_input_sub_error_rate is not None:
                error_rate_source = "Custom three error rates"
            else:
                error_rate_source = "Custom"
        else:
            error_rates = load_error_rates_from_config(
                synthesis_config['position_rates'], 
                seq_length
            )
            error_rate_source = "From file"
        
        print(f"  Error rate source: {error_rate_source}")
        
        sub_choices, sub_cum_probs = build_substitution_lookup_from_matrix(
            synthesis_config['error_matrix']['substitution']
        )
        
        insertion_vector = synthesis_config['error_matrix']['insertion']
        insertion_total = np.sum(insertion_vector)
        if insertion_total > 0:
            insertion_probs = insertion_vector.astype(np.float64) / insertion_total
        else:
            insertion_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        
        _kmer_data = synthesis_config.get('kmer')
        _use_kmer = synthesis_config.get('use_kmer', False)
        kmer_sub = _kmer_data['substitution'] if (_use_kmer and _kmer_data is not None) else None
        kmer_ins = _kmer_data['insertion']    if (_use_kmer and _kmer_data is not None) else None

    precompute_start_time = time.time()
    bucket_metadata, bucket_sampling_counts = compute_bucket_metadata(
        error_rates=error_rates,
        seq_length=seq_length,
        chunk_metadata_list=chunk_metadata_list,
        chunk_size=chunk_size,
        rng=np.random.default_rng(random_seed),
        num_parallel_workers=None,
        total_cpus=total_cpus
    )
    precompute_end_time = time.time()
    precompute_time = precompute_end_time - precompute_start_time
    
    error_rate_total = None
    error_rate_sub = None
    error_rate_ins = None
    error_rate_del = None
    if user_input_total_error_rate is not None:
        error_rate_total = user_input_total_error_rate
        error_rate_sub = np.mean(error_rates['substitution']) if error_rates else None
        error_rate_ins = np.mean(error_rates['insertion']) if error_rates else None
        error_rate_del = np.mean(error_rates['deletion']) if error_rates else None
    elif user_input_sub_error_rate is not None:
        error_rate_sub = user_input_sub_error_rate
        error_rate_ins = user_input_ins_error_rate
        error_rate_del = user_input_del_error_rate
        error_rate_total = error_rate_sub + error_rate_ins + error_rate_del
    else:
        error_rate_total = np.mean(error_rates['total']) if error_rates else None
        error_rate_sub = np.mean(error_rates['substitution']) if error_rates else None
        error_rate_ins = np.mean(error_rates['insertion']) if error_rates else None
        error_rate_del = np.mean(error_rates['deletion']) if error_rates else None
    actual_sub = actual_ins = actual_del = 0
    expected_total_errors = 0
    _bucket_counts_sum = int(np.sum(bucket_sampling_counts))
    expected_sub = total_reads * seq_length * error_rate_sub if error_rate_sub is not None else None
    expected_ins = total_reads * seq_length * error_rate_ins if error_rate_ins is not None else None
    expected_del = total_reads * seq_length * error_rate_del if error_rate_del is not None else None
    if expected_total_errors_tmp := (expected_sub + expected_ins + expected_del):
        actual_sub = int(_bucket_counts_sum * (expected_sub / expected_total_errors_tmp))
        actual_ins = int(_bucket_counts_sum * (expected_ins / expected_total_errors_tmp))
        actual_del = _bucket_counts_sum - actual_sub - actual_ins
        expected_total_errors = expected_total_errors_tmp
    
    mutation_start_time = time.time()
    print("\nGenerating ordered chunks...")
    
    num_chunks = len(chunk_metadata_list)

    num_workers = total_cpus - 2
    if num_workers < 1:
        num_workers = 1
    if num_chunks > 0 and num_workers > num_chunks:
        num_workers = num_chunks

    workers = []
    manager = None

    def cleanup_temp_dirs():
        if platform.system() != 'Linux':
            if shuffle_split_dir is not None and shuffle_split_dir.exists():
                try:
                    shutil.rmtree(shuffle_split_dir)
                except Exception as e:
                    logger.warning(f"Error cleaning up shuffle mapping directory: {e}")
    
    def cleanup_chunk_index_files():
        if output_dir is None:
            return
        output_dir_path = Path(output_dir)
        for pattern in ['output_chunk_*.idx.npy', 'output_chunk_shuffled_*.idx.npy']:
            for idx_file in output_dir_path.glob(pattern):
                try:
                    idx_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up index file {idx_file}: {e}")
        split_dir = output_dir_path / 'split'
        if split_dir.exists():
            try:
                shutil.rmtree(split_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up split directory {split_dir}: {e}")
    
    try:
        progress_counters = {
            'worker_chunks': multiprocessing.Value('i', 0),
            'merge_chunks': multiprocessing.Value('i', 0),
        }
        total_chunks_count = len(chunk_metadata_list)
        
        worker_progress_interval = max(1, total_chunks_count // 100)
        merge_progress_interval = max(1, total_chunks_count // 100)
        
        manager = multiprocessing.Manager()
        failed_chunks = manager.dict()
        
        worker_tasks_list = [[] for _ in range(num_workers)]
        for chunk_meta in chunk_metadata_list:
            chunk_idx = chunk_meta['chunk_idx']
            worker_id = chunk_idx % num_workers
            worker_tasks_list[worker_id].append({
                'chunk_idx': chunk_idx,
                'chunk_meta': chunk_meta
            })

        for worker_id in range(num_workers):
            worker_tasks_list[worker_id].sort(key=lambda x: x['chunk_idx'])
        
        total_assigned = sum(len(tasks) for tasks in worker_tasks_list)
        if total_assigned != len(chunk_metadata_list):
            logger.error(f"Task allocation error: total chunks={len(chunk_metadata_list)}, assigned={total_assigned}")
        
        worker_bucket_data = []
        for worker_id in range(num_workers):
            worker_bucket_indices = set()
            for task in worker_tasks_list[worker_id]:
                chunk_idx = task['chunk_idx']
                bucket_idx = chunk_idx
                worker_bucket_indices.add(bucket_idx)
            
            worker_bucket_indices_sorted = sorted(worker_bucket_indices)
            bucket_idx_to_array_idx = {bucket_idx: idx for idx, bucket_idx in enumerate(worker_bucket_indices_sorted)}
            
            bucket_indices_global = sorted(bucket_metadata.keys())
            worker_bucket_array_indices = [bucket_indices_global.index(bucket_idx) for bucket_idx in worker_bucket_indices_sorted]
            
            worker_bucket_sampling_counts = bucket_sampling_counts[worker_bucket_array_indices, :].copy()
            
            worker_bucket_data.append({
                'bucket_sampling_counts': worker_bucket_sampling_counts,
                'bucket_idx_to_array_idx': bucket_idx_to_array_idx
            })
        
        ss = np.random.SeedSequence(random_seed)
        child_seeds = ss.spawn(num_workers)
        
        workers = []
        startup_stagger_delay = 0.1
        
        for worker_id in range(num_workers):
            if not worker_tasks_list[worker_id]:
                continue

            p = multiprocessing.Process(
                target=error_simulator_worker,
                args=(
                    worker_id,
                    worker_tasks_list[worker_id],
                    input_fasta,
                    split_dir,
                    error_rates,
                    bucket_metadata,
                    worker_bucket_data[worker_id]['bucket_sampling_counts'],
                    worker_bucket_data[worker_id]['bucket_idx_to_array_idx'],
                    sub_choices,
                    sub_cum_probs,
                    insertion_probs,
                    seq_length,
                    chunk_size,
                    read_id_offset,
                    output_dir,
                    child_seeds[worker_id],
                    progress_counters['worker_chunks'],
                    num_ref_seqs,
                    failed_chunks,
                    total_chunks_count,
                    worker_progress_interval,
                    _use_kmer,
                    kmer_sub,
                    kmer_ins,
                    output_stats,
                )
            )
            p.start()
            workers.append(p)
            
            if worker_id < num_workers - 1:
                time.sleep(startup_stagger_delay)
        
        pbar = None
        if _TQDM_AVAILABLE and total_chunks_count > 0:
            pbar = tqdm(
                total=total_chunks_count,
                desc="Generating ordered chunks",
                unit="chunk",
                ncols=100,
            )
        
        try:
            while any(w.is_alive() for w in workers):
                for w in workers:
                    w.join(timeout=0.5)
                if pbar is not None:
                    pbar.n = progress_counters['worker_chunks'].value
                    pbar.refresh()
                time.sleep(0.2)
        finally:
            if pbar is not None:
                pbar.n = total_chunks_count
                pbar.refresh()
                pbar.close()
                print()
        
        workers.clear()
        
        if progress_counters['worker_chunks'] is not None:
            final_count = progress_counters['worker_chunks'].value
            if final_count < total_chunks_count:
                missing_count = total_chunks_count - final_count
                logger.warning(f"Warning: {missing_count} chunks were not processed completely!")
                
                if output_dir is not None and output_dir.exists():
                    chunk_files = list(output_dir.glob('output_chunk_*.fasta'))
        
        missing_chunk_files = set()
        if output_dir is not None and output_dir.exists():
            expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
            
            existing_chunk_files = set()
            for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                try:
                    filename = chunk_file.stem
                    if filename.startswith('output_chunk_'):
                        chunk_idx_str = filename[len('output_chunk_'):]
                        chunk_idx = int(chunk_idx_str)
                        if chunk_file.stat().st_size > 0:
                            existing_chunk_files.add(chunk_idx)
                except (ValueError, IndexError):
                    continue
            
            missing_chunk_files = expected_chunk_indices - existing_chunk_files

        # Compute chunk file size statistics
        chunk_file_sizes_dict = {}
        total_chunk_size = 0
        if output_dir is not None and output_dir.exists():
            for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                try:
                    chunk_file_sizes_dict[chunk_file.name] = chunk_file.stat().st_size
                    total_chunk_size += chunk_file.stat().st_size
                except (OSError, PermissionError):
                    pass
        total_chunk_count = len(chunk_file_sizes_dict)
        avg_chunk_size = total_chunk_size / total_chunk_count if total_chunk_count > 0 else 0
        process_round = 0
        max_process_rounds = 50
        consecutive_no_progress_rounds = 0
        max_consecutive_no_progress = 3
        min_progress_threshold = 0.001
        all_failed_chunk_ids = set()
        
        if missing_chunk_files:
            all_failed_chunk_ids.update(missing_chunk_files)
        
        while all_failed_chunk_ids and process_round < max_process_rounds:
            process_round += 1
            
            if output_dir is not None and output_dir.exists():
                expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
                
                existing_chunk_files = set()
                for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                    try:
                        filename = chunk_file.stem
                        if filename.startswith('output_chunk_'):
                            chunk_idx_str = filename[len('output_chunk_'):]
                            chunk_idx = int(chunk_idx_str)
                            if chunk_file.stat().st_size > 0:
                                existing_chunk_files.add(chunk_idx)
                    except (ValueError, IndexError, OSError, PermissionError):
                                continue
                
                missing_chunk_files = expected_chunk_indices - existing_chunk_files
                all_failed_chunk_ids = missing_chunk_files.copy()
            
            if not all_failed_chunk_ids:
                break
            
            failed_chunk_ids = sorted(all_failed_chunk_ids)
            
            print(f"\nStarting round {process_round} retry (still {len(failed_chunk_ids)} chunks unfinished)...")
            
            process_tasks = []
            for chunk_idx in failed_chunk_ids:
                chunk_meta = None
                for meta in chunk_metadata_list:
                    if meta.get('chunk_idx') == chunk_idx:
                        chunk_meta = meta
                        break
                
                if chunk_meta is None:
                    logger.warning(f"Chunk {chunk_idx:06d}: Cannot find chunk_meta, skipping processing")
                    all_failed_chunk_ids.discard(chunk_idx)
                    continue
                
                process_tasks.append({
                    'chunk_idx': chunk_idx,
                    'chunk_meta': chunk_meta
                })
            
            if not process_tasks:
                logger.warning(f"[WARNING] No chunks available for processing (all pending chunks are missing chunk_meta)")
                break
            
            retry_worker_tasks_list = [[] for _ in range(num_workers)]
            for task in process_tasks:
                chunk_idx = task['chunk_idx']
                original_worker_id = chunk_idx % num_workers
                retry_worker_tasks_list[original_worker_id].append(task)

            for worker_id in range(num_workers):
                if retry_worker_tasks_list[worker_id]:
                    retry_worker_tasks_list[worker_id].sort(key=lambda x: x['chunk_idx'])
            
            retry_worker_bucket_data = []
            for worker_id in range(num_workers):
                if not retry_worker_tasks_list[worker_id]:
                    retry_worker_bucket_data.append({
                        'bucket_sampling_counts': np.array([]),
                        'bucket_idx_to_array_idx': {}
                    })
                    continue
                
                worker_bucket_indices = set()
                for task in retry_worker_tasks_list[worker_id]:
                    chunk_idx = task['chunk_idx']
                    bucket_idx = chunk_idx
                    worker_bucket_indices.add(bucket_idx)
                
                worker_bucket_indices_sorted = sorted(worker_bucket_indices)
                bucket_idx_to_array_idx = {bucket_idx: idx for idx, bucket_idx in enumerate(worker_bucket_indices_sorted)}
                
                bucket_indices_global = sorted(bucket_metadata.keys())
                worker_bucket_array_indices = [bucket_indices_global.index(bucket_idx) for bucket_idx in worker_bucket_indices_sorted]
                
                worker_bucket_sampling_counts = bucket_sampling_counts[worker_bucket_array_indices, :].copy()
                
                retry_worker_bucket_data.append({
                    'bucket_sampling_counts': worker_bucket_sampling_counts,
                    'bucket_idx_to_array_idx': bucket_idx_to_array_idx
                })
            
            retry_workers = []
            startup_stagger_delay = 0.1
            
            active_worker_ids = [w_id for w_id in range(num_workers) if retry_worker_tasks_list[w_id]]
            
            for idx, original_worker_id in enumerate(active_worker_ids):
                p = multiprocessing.Process(
                    target=error_simulator_worker,
                    args=(
                        original_worker_id,
                        retry_worker_tasks_list[original_worker_id],
                        input_fasta,
                        split_dir,
                        error_rates,
                        bucket_metadata,
                        retry_worker_bucket_data[original_worker_id]['bucket_sampling_counts'],
                        retry_worker_bucket_data[original_worker_id]['bucket_idx_to_array_idx'],
                        sub_choices,
                        sub_cum_probs,
                        insertion_probs,
                        seq_length,
                        chunk_size,
                        read_id_offset,
                        output_dir,
                        child_seeds[original_worker_id],
                        progress_counters['worker_chunks'],
                        num_ref_seqs,
                        failed_chunks,
                        total_chunks_count,
                        worker_progress_interval,
                        _use_kmer,
                        kmer_sub,
                        kmer_ins,
                        output_stats,
                    )
                )
                p.start()
                retry_workers.append((original_worker_id, p))
                
                if idx < len(active_worker_ids) - 1:
                    time.sleep(startup_stagger_delay)
            
            retry_workers_count = len(retry_workers)
            
            for worker_id, w in retry_workers:
                w.join()
            
            retry_workers.clear()
            
            try:
                if 'retry_worker_bucket_data' in locals() and retry_worker_bucket_data is not None:
                    for worker_data in retry_worker_bucket_data:
                        if isinstance(worker_data, dict):
                            if 'bucket_sampling_counts' in worker_data:
                                del worker_data['bucket_sampling_counts']
                    retry_worker_bucket_data.clear()
                    del retry_worker_bucket_data
            except Exception as e:
                logger.warning(f"Error cleaning up retry_worker_bucket_data: {e}")
            
            process_success_count = 0
            process_failed_count = 0
            
            expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
            
            existing_chunk_files = set()
            if output_dir is not None and output_dir.exists():
                for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                    try:
                        filename = chunk_file.stem
                        if filename.startswith('output_chunk_'):
                            chunk_idx_str = filename[len('output_chunk_'):]
                            chunk_idx = int(chunk_idx_str)
                            if chunk_file.stat().st_size > 0:
                                existing_chunk_files.add(chunk_idx)
                    except (ValueError, IndexError, OSError, PermissionError):
                                continue
                
                missing_chunk_files = expected_chunk_indices - existing_chunk_files
                
                for chunk_idx in failed_chunk_ids:
                    chunk_file = output_dir / f'output_chunk_{chunk_idx}.fasta'
                    try:
                        if chunk_file.exists() and chunk_file.stat().st_size > 0:
                            process_success_count += 1
                            all_failed_chunk_ids.discard(chunk_idx)
                        else:
                            process_failed_count += 1
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Error checking Chunk {chunk_idx:06d} file status: {e}, will keep in pending list")
                        process_failed_count += 1
                
                newly_found_missing = missing_chunk_files - all_failed_chunk_ids
                if newly_found_missing:
                    logger.warning(f"[WARNING] After round {process_round} processing check: found {len(newly_found_missing)} newly untracked missing chunks, adding to pending list")
                    for chunk_idx in newly_found_missing:
                        all_failed_chunk_ids.add(chunk_idx)
            
            if output_dir is not None and output_dir.exists():
                expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
                
                existing_chunk_files = set()
                for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                    try:
                        filename = chunk_file.stem
                        if filename.startswith('output_chunk_'):
                            chunk_idx_str = filename[len('output_chunk_'):]
                            chunk_idx = int(chunk_idx_str)
                            if chunk_file.stat().st_size > 0:
                                existing_chunk_files.add(chunk_idx)
                    except (ValueError, IndexError, OSError, PermissionError):
                                continue
                
                missing_chunk_files = expected_chunk_indices - existing_chunk_files
                
                successfully_fixed = all_failed_chunk_ids - missing_chunk_files
                if successfully_fixed:
                    for chunk_idx in successfully_fixed:
                        all_failed_chunk_ids.discard(chunk_idx)
                
                newly_found_missing = missing_chunk_files - all_failed_chunk_ids
                if newly_found_missing:
                    logger.warning(f"[WARNING] Round {process_round} fallback check: found {len(newly_found_missing)} new missing chunks")
                    for chunk_idx in newly_found_missing:
                        all_failed_chunk_ids.add(chunk_idx)
                
            round_success_rate = process_success_count / len(failed_chunk_ids) if len(failed_chunk_ids) > 0 else 0
            
            if round_success_rate < min_progress_threshold:
                consecutive_no_progress_rounds += 1
                logger.warning(f"[WARNING] Round {process_round} processing: success rate {round_success_rate*100:.1f}% is below threshold {min_progress_threshold*100:.1f}%")
                logger.warning(f"   Consecutive rounds without significant progress: {consecutive_no_progress_rounds}/{max_consecutive_no_progress}")
                
                if consecutive_no_progress_rounds >= max_consecutive_no_progress:
                    logger.warning(f"[WARNING] Consecutive {consecutive_no_progress_rounds} rounds without significant progress, exiting retry loop early")
                    logger.warning(f"   The remaining {len(all_failed_chunk_ids)} failed chunks will be reported in final check")
                    break
                
                wait_time = min(60 * (2 ** (consecutive_no_progress_rounds - 1)), 600)
                time.sleep(wait_time)
            else:
                consecutive_no_progress_rounds = 0
        
        if output_dir is not None and output_dir.exists():
            expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
            
            existing_chunk_files = set()
            for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                try:
                    filename = chunk_file.stem
                    if filename.startswith('output_chunk_'):
                        chunk_idx_str = filename[len('output_chunk_'):]
                        chunk_idx = int(chunk_idx_str)
                        if chunk_file.stat().st_size > 0:
                            existing_chunk_files.add(chunk_idx)
                except (ValueError, IndexError):
                            continue
            
            fs_missing_chunks = expected_chunk_indices - existing_chunk_files
            
            successfully_fixed = all_failed_chunk_ids - fs_missing_chunks
            newly_found_missing = fs_missing_chunks - all_failed_chunk_ids
            
            for chunk_idx in successfully_fixed:
                all_failed_chunk_ids.discard(chunk_idx)
            
            if newly_found_missing:
                logger.warning(f"[WARNING] Retry loop ended: found {len(newly_found_missing)} new missing chunks (filesystem scan)")
                all_failed_chunk_ids.update(newly_found_missing)
        
        if process_round >= max_process_rounds and all_failed_chunk_ids:
            logger.error(f"[ERROR] Maximum processing rounds {max_process_rounds} reached, stopping processing")
            logger.error(f"   {len(all_failed_chunk_ids)} chunks still failed to process:")
            for chunk_idx in sorted(all_failed_chunk_ids)[:20]:
                logger.error(f"     Chunk {chunk_idx}: file missing or empty")
            if len(all_failed_chunk_ids) > 20:
                logger.error(f"     ... and {len(all_failed_chunk_ids) - 20} more failed chunks not shown")
        elif all_failed_chunk_ids:
            logger.warning(f"[WARNING] After {process_round} rounds of processing, still {len(all_failed_chunk_ids)} chunks cannot be processed")
        
        final_missing_chunk_files = set()
        final_existing_chunk_files = set()
        
        if output_dir is not None and output_dir.exists():
            expected_chunk_indices = {meta['chunk_idx'] for meta in chunk_metadata_list}
            
            for chunk_file in output_dir.glob('output_chunk_*.fasta'):
                try:
                    filename = chunk_file.stem
                    if filename.startswith('output_chunk_'):
                        chunk_idx_str = filename[len('output_chunk_'):]
                        chunk_idx = int(chunk_idx_str)
                        if chunk_file.stat().st_size > 0:
                            final_existing_chunk_files.add(chunk_idx)
                except (ValueError, IndexError, OSError, PermissionError):
                            continue
            
            final_missing_chunk_files = expected_chunk_indices - final_existing_chunk_files
        
        if final_missing_chunk_files:
            logger.warning(f"[WARNING] Final check (filesystem scan): detected {len(final_missing_chunk_files)} missing or empty chunk files")
            logger.warning(f"   Total chunks: {len(chunk_metadata_list)}")
            logger.warning(f"   Actually exist: {len(final_existing_chunk_files)}")
            logger.warning(f"   Actually missing: {len(final_missing_chunk_files)}")

        shuffle_split_dir = None
        if shuffle_enabled and not final_missing_chunk_files:
            print("\nShuffling: building mapping...")
            shuffle_split_dir = generate_shuffle_mapping(
                chunk_metadata_list=chunk_metadata_list,
                split_dir=split_dir,
                total_reads=total_reads,
                chunk_size=chunk_size,
                output_dir=output_dir,
                master_seed=random_seed,
                read_id_offset=read_id_offset,
            )

            write_shuffle_chunks_parallel(
                shuffle_split_dir=shuffle_split_dir,
                output_dir=output_dir,
                chunk_metadata_list=chunk_metadata_list,
                chunk_size=chunk_size,
                read_id_offset=read_id_offset,
                num_workers=max(1, min(len(chunk_metadata_list), total_cpus - 2)),
            )
        elif shuffle_enabled:
            pass
        else:
            pass
            
        if merge_files_enabled and not final_missing_chunk_files and output_dir is not None and output_dir.exists():
            if not CONST.COPY_FILE_RANGE_AVAILABLE:
                logger.warning("[WARNING] Merge enabled but copy_file_range not available, skipping file merge")
            else:
                merge_start_time = time.time()
                if merge_timestamp is None:
                    merge_timestamp = get_timestamp_string() if merge_ts_flag else None
                merged_filename = get_merge_filename(merge_basename, merge_timestamp, shuffled=False)
                output_merged = output_dir / merged_filename
                successful_chunks = sorted(
                    [m for m in chunk_metadata_list if m['chunk_idx'] in final_existing_chunk_files],
                    key=lambda x: x['chunk_idx']
                )
                if not successful_chunks:
                    logger.warning("[WARNING] No successful chunks to merge, skipping")
                else:
                    chunk_offsets, total_file_size = _calculate_chunk_offsets_from_actual_sizes(
                        successful_chunks, output_dir
                    )
                    merge_tasks_list = [
                        {'chunk_idx': m['chunk_idx'], 'file_offset': chunk_offsets.get(m['chunk_idx'])}
                        for m in successful_chunks
                    ]
                    merge_tasks_list = [t for t in merge_tasks_list if t['file_offset'] is not None]
                    if not merge_tasks_list:
                        logger.error(f"  All chunk files are empty (0 bytes), cannot merge, skipping ordered merge phase")
                        output_merged = None
                    else:
                        Path(output_merged).touch()
                        print(f"\nMerging ordered chunks...")

                        num_merge_workers = max(1, min(total_cpus - 2, len(merge_tasks_list)))
                        tasks_per_merge_worker = (len(merge_tasks_list) + num_merge_workers - 1) // num_merge_workers
                        merge_worker_tasks = [[] for _ in range(num_merge_workers)]
                        for i, task in enumerate(merge_tasks_list):
                            wid = i // tasks_per_merge_worker
                            wid = min(wid, num_merge_workers - 1)
                            merge_worker_tasks[wid].append(task)
                        with progress_counters['merge_chunks'].get_lock():
                            progress_counters['merge_chunks'].value = 0
                        merge_procs = []
                        max_extend_target = multiprocessing.Value('L', 0)
                        for wid in range(num_merge_workers):
                            if not merge_worker_tasks[wid]:
                                continue
                            p = multiprocessing.Process(
                                target=_merge_worker,
                                args=(
                                    wid,
                                    merge_worker_tasks[wid],
                                    output_merged,
                                    output_dir,
                                    progress_counters['merge_chunks'],
                                    len(merge_tasks_list),
                                    merge_progress_interval,
                                    max_extend_target,
                                )
                            )
                            p.start()
                            merge_procs.append(p)

                        merge_pbar = None
                        if _TQDM_AVAILABLE and len(merge_tasks_list) > 0:
                            merge_pbar = tqdm(
                                total=len(merge_tasks_list),
                                desc="Merging ordered chunks",
                                unit="chunk",
                                ncols=100,
                            )
                        try:
                            while any(p.is_alive() for p in merge_procs):
                                for p in merge_procs:
                                    p.join(timeout=0.5)
                                if merge_pbar is not None:
                                    merge_pbar.n = progress_counters['merge_chunks'].value
                                    merge_pbar.refresh()
                                time.sleep(0.2)
                        finally:
                            if merge_pbar is not None:
                                merge_pbar.n = len(merge_tasks_list)
                                merge_pbar.refresh()
                                merge_pbar.close()
                                print()
                        merge_elapsed = time.time() - merge_start_time
                        print(f"[OK] {output_merged.name}")
                        
                        try:
                            fd = os.open(str(output_merged), os.O_RDWR)
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
                        except Exception as e:
                            logger.warning(f"  Failed to clean up trailing newline in merged file: {e}")
                        
                if shuffle_enabled and shuffle_split_dir is not None and len(chunk_metadata_list) > 0:
                    shuffled_offsets, shuffled_total_size, shuffled_successful = _calculate_chunk_offsets_for_shuffled(
                        output_dir, len(chunk_metadata_list), chunk_file_prefix="output_chunk_shuffled_"
                    )
                    if shuffled_successful:
                        merge_shuffled_start = time.time()
                        merged_shuffled_filename = get_merge_filename(merge_basename, merge_timestamp, shuffled=True)
                        output_merged_shuffled = output_dir / merged_shuffled_filename
                        merge_shuffled_tasks = [
                            {"chunk_idx": m["chunk_idx"], "file_offset": shuffled_offsets[m["chunk_idx"]]}
                            for m in shuffled_successful
                        ]
                        Path(output_merged_shuffled).touch()
                        print(f"\nMerging shuffled chunks...")

                        num_merge_workers_s = max(1, min(total_cpus - 2, len(merge_shuffled_tasks)))
                        tasks_per_shuffled_worker = (len(merge_shuffled_tasks) + num_merge_workers_s - 1) // num_merge_workers_s
                        merge_shuffled_worker_tasks = [[] for _ in range(num_merge_workers_s)]
                        for i, task in enumerate(merge_shuffled_tasks):
                            wid = i // tasks_per_shuffled_worker
                            wid = min(wid, num_merge_workers_s - 1)
                            merge_shuffled_worker_tasks[wid].append(task)
                        with progress_counters["merge_chunks"].get_lock():
                            progress_counters["merge_chunks"].value = 0
                        merge_shuffled_procs = []
                        max_extend_target_shuffled = multiprocessing.Value('L', 0)
                        for wid in range(num_merge_workers_s):
                            if not merge_shuffled_worker_tasks[wid]:
                                continue
                            p = multiprocessing.Process(
                                target=_merge_worker,
                                args=(
                                    wid,
                                    merge_shuffled_worker_tasks[wid],
                                    output_merged_shuffled,
                                    output_dir,
                                    progress_counters["merge_chunks"],
                                    len(merge_shuffled_tasks),
                                    merge_progress_interval,
                                    max_extend_target_shuffled,
                                ),
                                kwargs={"chunk_file_prefix": "output_chunk_shuffled_"},
                            )
                            p.start()
                            merge_shuffled_procs.append(p)

                        merge_shuffled_pbar = None
                        if _TQDM_AVAILABLE and len(merge_shuffled_tasks) > 0:
                            merge_shuffled_pbar = tqdm(
                                total=len(merge_shuffled_tasks),
                                desc="Merging shuffled chunks",
                                unit="chunk",
                                ncols=100,
                            )
                        try:
                            while any(p.is_alive() for p in merge_shuffled_procs):
                                for p in merge_shuffled_procs:
                                    p.join(timeout=0.5)
                                if merge_shuffled_pbar is not None:
                                    merge_shuffled_pbar.n = progress_counters["merge_chunks"].value
                                    merge_shuffled_pbar.refresh()
                                time.sleep(0.2)
                        finally:
                            if merge_shuffled_pbar is not None:
                                merge_shuffled_pbar.n = len(merge_shuffled_tasks)
                                merge_shuffled_pbar.refresh()
                                merge_shuffled_pbar.close()
                                print()
                        merge_shuffled_elapsed = time.time() - merge_shuffled_start
                        print(f"[OK] {output_merged_shuffled.name}")
                        
                        try:
                            fd = os.open(str(output_merged_shuffled), os.O_RDWR)
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
                        except Exception as e:
                            logger.warning(f"  Failed to clean up trailing newline in shuffled merged file: {e}")
                        
        if output_stats and not final_missing_chunk_files:
            write_ref_count_and_read_to_ref_tsv(
                chunk_metadata_list=chunk_metadata_list,
                split_dir=split_dir,
                num_ref_seqs=num_ref_seqs,
                total_reads=total_reads,
                output_dir=output_dir,
                read_id_offset=read_id_offset,
                shuffle_split_dir=shuffle_split_dir,
                chunk_size=chunk_size,
                output_stats=output_stats,
            )

    except Exception as e:
        logger.error(f"Parallel processing stage error: {e}")
        raise
    finally:
        try:
            if 'worker_bucket_data' in locals() and worker_bucket_data is not None:
                for worker_data in worker_bucket_data:
                    if isinstance(worker_data, dict) and 'bucket_sampling_counts' in worker_data:
                        del worker_data['bucket_sampling_counts']
                worker_bucket_data.clear()
                del worker_bucket_data
        except Exception as e:
            logger.warning(f"Error cleaning up worker_bucket_data: {e}")
        
        try:
            if 'retry_worker_bucket_data' in locals() and retry_worker_bucket_data is not None:
                for worker_data in retry_worker_bucket_data:
                    if isinstance(worker_data, dict) and 'bucket_sampling_counts' in worker_data:
                        del worker_data['bucket_sampling_counts']
                retry_worker_bucket_data.clear()
                del retry_worker_bucket_data
        except Exception as e:
            logger.warning(f"Error cleaning up retry_worker_bucket_data: {e}")
        
        try:
            if manager is not None:
                manager.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down Manager object: {e}")
        
        try:
            if 'bucket_sampling_counts' in locals():
                del bucket_sampling_counts
        except Exception as e:
            logger.warning(f"Error cleaning up bucket_sampling_counts: {e}")
        
        cleanup_temp_dirs()
        cleanup_chunk_index_files()
        
        try:
            gc.collect()
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")
    
    mutation_end_time = time.time()
    total_end_time = time.time()
    print(f"\n{'='*80}")
    print(f"DNATerra finished ({datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*80}")

    precompute_time_calculated = precompute_end_time - precompute_start_time if precompute_start_time is not None else 0.0
    mutation_time_calculated = mutation_end_time - mutation_start_time
    
    actual_total_cpus = total_cpus
    num_workers_global_total = num_workers
    
    return {
        'num_ref_seqs': num_ref_seqs,
        'total_reads': total_reads,
        'num_chunks': len(chunk_metadata_list),
        'chunk_size': chunk_size,
        'seq_length': seq_length,
        'synthesis_method': synthesis_method,
        'target_read_depth': target_read_depth,
        'num_workers': num_workers_global_total,
        'total_cpus': total_cpus,
        'actual_cpus': actual_total_cpus,
        'user_input_chunk_size': user_input_chunk_size,
        'error_rate_source': error_rate_source,
        'error_rate_total': error_rate_total,
        'error_rate_sub': error_rate_sub,
        'error_rate_ins': error_rate_ins,
        'error_rate_del': error_rate_del,
        'output_file': str(output_dir),
        'chunk_file_sizes': chunk_file_sizes_dict,
        'chunk_dir': str(output_dir),
        'total_chunk_count': total_chunk_count,
        'total_chunk_size': total_chunk_size,
        'avg_chunk_size': avg_chunk_size,
        'output_file_size': total_chunk_size,
        'precompute_time': precompute_time_calculated,
        'mutation_time': mutation_time_calculated,
        'actual_sub': actual_sub,
        'actual_ins': actual_ins,
        'actual_del': actual_del
    }
