"""Error sampling and precomputation module."""
import logging
import multiprocessing
import signal
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import CONST

logger = logging.getLogger(__name__)


def load_error_rates_from_config(position_rates: Dict, seq_length: int) -> Dict:
    error_rates = {
        'total': np.array(position_rates['total_error_rate'], dtype=np.float64),
        'substitution': np.array(position_rates['substitution_rate'], dtype=np.float64),
        'insertion': np.array(position_rates['insertion_rate'], dtype=np.float64),
        'deletion': np.array(position_rates['deletion_rate'], dtype=np.float64)
    }
    
    for key, rates in error_rates.items():
        if len(rates) != seq_length:
            logger.warning(f"{key}_rate has length {len(rates)}, expected {seq_length}")
    
    total_sum = (error_rates['substitution'] + 
                error_rates['insertion'] + 
                error_rates['deletion'])
    diff = np.abs(error_rates['total'] - total_sum)
    max_diff = np.max(diff)
    tolerance = 1e-6
    if max_diff > tolerance:
        logger.warning(f"Error rate data has minor inconsistency (max difference: {max_diff:.2e}), possibly due to floating point precision")
    
    from .utils import _format_error_rate
    return error_rates


def _reset_signal_handlers():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _compute_position_chunk_shm(args):
    (position_range, error_rates_total, num_read_seqs, bucket_indices, 
     bucket_metadata, base_rng_seed, shm_name, n_buckets, L) = args
    
    shm = shared_memory.SharedMemory(name=shm_name)
    bucket_sampling_counts = np.ndarray((n_buckets, L), dtype=np.uint32, buffer=shm.buf)
    
    chunk_total_samples = 0
    
    try:
        for pos in position_range:
            error_rate_at_pos = error_rates_total[pos]
            
            if error_rate_at_pos == 0:
                bucket_sampling_counts[:, pos] = 0
                continue
            
            worker_rng = np.random.default_rng(base_rng_seed + pos)
            
            total_errors_in_population = int(np.round(num_read_seqs * error_rate_at_pos))
            total_non_errors = num_read_seqs - total_errors_in_population

            exceeds_hypergeometric_limit = (total_errors_in_population >= CONST.HYPERGEOMETRIC_LIMIT or
                                           total_non_errors >= CONST.HYPERGEOMETRIC_LIMIT)
            use_binomial = (num_read_seqs >= CONST.BINOMIAL_VS_HYPERGEOMETRIC_THRESHOLD) or exceeds_hypergeometric_limit
            
            if (not use_binomial 
                and error_rate_at_pos > 0 
                and num_read_seqs > 0 
                and total_errors_in_population == 0):
                use_binomial = True
            
            pos_total = 0
            
            if use_binomial:
                for i, bucket_idx in enumerate(bucket_indices):
                    actual_bucket_size = bucket_metadata[bucket_idx]['bucket_actual_size']
                    if actual_bucket_size == 0:
                        count = 0
                    else:
                        count = worker_rng.binomial(n=actual_bucket_size, p=error_rate_at_pos)
                    bucket_sampling_counts[i, pos] = count
                    pos_total += count
            else:
                total_errors_in_population = max(0, min(total_errors_in_population, num_read_seqs))
                total_non_errors = max(0, num_read_seqs - total_errors_in_population)
                remaining_errors = total_errors_in_population
                remaining_non_errors = total_non_errors
                
                for i, bucket_idx in enumerate(bucket_indices):
                    actual_bucket_size = bucket_metadata[bucket_idx]['bucket_actual_size']
                    
                    if actual_bucket_size == 0 or remaining_errors == 0:
                        count = 0
                    else:
                        remaining_total = remaining_errors + remaining_non_errors
                        nsample = min(actual_bucket_size, remaining_total)
                        ngood = min(remaining_errors, remaining_total)
                        nbad = max(0, remaining_total - ngood)
                        
                        if ngood >= CONST.HYPERGEOMETRIC_LIMIT or nbad >= CONST.HYPERGEOMETRIC_LIMIT:
                            if remaining_total > 0:
                                error_prob = ngood / remaining_total
                                count = worker_rng.binomial(n=nsample, p=error_prob)
                            else:
                                count = 0
                        elif nsample > 0 and remaining_total > 0 and ngood > 0:
                            count = worker_rng.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample)
                        else:
                            count = 0
                        
                        if count > 0 or nsample > 0:
                            remaining_errors -= count
                            remaining_non_errors -= (nsample - count)
                            remaining_non_errors = max(0, remaining_non_errors)
                            remaining_errors = max(0, remaining_errors)
                    
                    bucket_sampling_counts[i, pos] = count
                    pos_total += count
            
            chunk_total_samples += pos_total
        
        return chunk_total_samples
        
    finally:
        shm.close()


def compute_error_counts_per_position(error_rates: Dict,
                                     num_read_seqs: int,
                                     bucket_metadata: Dict,
                                     rng,
                                     num_parallel_workers: int = None,
                                     total_cpus: int = None) -> np.ndarray:
    L = len(error_rates['total'])
    bucket_indices = sorted(bucket_metadata.keys())
    n_buckets = len(bucket_indices)
    
    expected_total = np.sum(error_rates['total'] * num_read_seqs)
    
    if total_cpus is None:
        total_cpus = multiprocessing.cpu_count()
    max_workers = total_cpus - 1
    
    if num_parallel_workers is None:
        if n_buckets < max_workers:
            num_parallel_workers = n_buckets
        else:
            num_parallel_workers = max_workers
    elif num_parallel_workers < 1:
        num_parallel_workers = 1
    
    base_rng_seed = rng.integers(0, 2**31)
    
    old_size_gb = n_buckets * L * 8 / (1024**3)
    new_size_gb = n_buckets * L * 4 / (1024**3)
    
    shm_size = n_buckets * L * 4
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    bucket_sampling_counts = np.ndarray((n_buckets, L), dtype=np.uint32, buffer=shm.buf)
    bucket_sampling_counts[:] = 0
    
    try:
        chunk_size = max(1, L // num_parallel_workers)
        position_chunks = []
        for i in range(num_parallel_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, L) if i < num_parallel_workers - 1 else L
            if start < L:
                position_chunks.append(range(start, end))
        
        chunk_args = [
            (
                pos_range,
                error_rates['total'],
                num_read_seqs,
                bucket_indices,
                bucket_metadata,
                base_rng_seed,
                shm.name,
                n_buckets,
                L
            )
            for pos_range in position_chunks
        ]
        
        pool = None
        try:
            pool = multiprocessing.Pool(
                processes=len(position_chunks),
                initializer=_reset_signal_handlers
            )
            
            total_chunks = len(chunk_args)
            chunk_totals = []
            completed = 0
            
            for result in pool.imap(_compute_position_chunk_shm, chunk_args):
                chunk_totals.append(result)
                completed += 1
            
            total_samples = sum(chunk_totals)
            
        except Exception as e:
            if pool is not None:
                pool.terminate()
            raise e
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        
        result = bucket_sampling_counts.copy()
        del bucket_sampling_counts
        return result
        
    finally:
        shm.close()
        shm.unlink()


def sample_errors_from_bucket(bucket_idx: int,
                             bucket_metadata: Dict,
                             bucket_sampling_counts: np.ndarray,
                             bucket_idx_to_array_idx: Dict[int, int],
                             error_rates: Dict,
                             rng) -> np.ndarray:
    bucket_range_size = int(bucket_metadata[bucket_idx]['bucket_actual_size'])
    L = len(error_rates['total'])
    
    if bucket_range_size == 0:
        dtype = [('chunk_local_reads_idx', np.uint32), ('error_pos', np.uint16), ('error_type', np.uint8)]
        return np.empty(0, dtype=dtype)
    
    bucket_array_idx = bucket_idx_to_array_idx[bucket_idx]
    exact_error_count = sum(int(x) for x in bucket_sampling_counts[bucket_array_idx, :])
    
    dtype = [('chunk_local_reads_idx', np.uint32), ('error_pos', np.uint16), ('error_type', np.uint8)]
    error_records = np.empty(exact_error_count, dtype=dtype)
    
    current_idx = 0
    base_rng_seed = rng.integers(0, 2**31)
    
    for pos in range(L - 1, -1, -1):
        count = int(bucket_sampling_counts[bucket_array_idx, pos])
        
        if count == 0:
            continue
        
        pos_rng = np.random.default_rng(base_rng_seed + pos)
        
        if count <= bucket_range_size:
            sampled_chunk_local_reads_idx = pos_rng.choice(
                bucket_range_size,
                size=count,
                replace=False
            ).astype(np.uint32)
        else:
            sampled_chunk_local_reads_idx = pos_rng.choice(
                bucket_range_size,
                size=count,
                replace=True
            ).astype(np.uint32)
        
        total_rate = error_rates['total'][pos]
        if total_rate > 0:
            sub_ratio = error_rates['substitution'][pos] / total_rate
            ins_ratio = error_rates['insertion'][pos] / total_rate
            sub_threshold = sub_ratio
            ins_threshold = sub_ratio + ins_ratio
        else:
            continue
        
        random_vals = pos_rng.random(count, dtype=np.float32)
        error_types = np.empty(count, dtype=np.uint8)
        error_types[random_vals < sub_threshold] = CONST.SUB
        error_types[(random_vals >= sub_threshold) & (random_vals < ins_threshold)] = CONST.INS
        error_types[random_vals >= ins_threshold] = CONST.DEL
        
        error_records['chunk_local_reads_idx'][current_idx:current_idx+count] = sampled_chunk_local_reads_idx
        error_records['error_pos'][current_idx:current_idx+count] = pos
        error_records['error_type'][current_idx:current_idx+count] = error_types
        
        current_idx += count
        del sampled_chunk_local_reads_idx, random_vals, error_types, pos_rng
    
    return error_records


def compute_bucket_metadata(error_rates: Dict,
                           seq_length: int,
                           chunk_metadata_list: List[Dict],
                           chunk_size: int,
                           rng,
                           num_parallel_workers: int = None,
                           total_cpus: int = None) -> Tuple[Dict, np.ndarray]:
    L = seq_length
    n_buckets = len(chunk_metadata_list)
    
    total_reads = sum(chunk_info['total_reads_in_chunk'] for chunk_info in chunk_metadata_list)
    
    bucket_metadata = {}
    for chunk_info in chunk_metadata_list:
        bucket_idx = chunk_info['chunk_idx']
        bucket_metadata[bucket_idx] = {
            'bucket_actual_size': int(chunk_info['total_reads_in_chunk'])
        }
    
    base_rng_seed = rng.integers(0, 2**31)
    precompute_rng = np.random.default_rng(base_rng_seed)
    
    bucket_sampling_counts = compute_error_counts_per_position(
        error_rates, total_reads, bucket_metadata, precompute_rng,
        num_parallel_workers=num_parallel_workers,
        total_cpus=total_cpus
    )
    
    counts_sum = sum(int(x) for x in bucket_sampling_counts.flat)
    non_zero_count = np.count_nonzero(bucket_sampling_counts)
    max_value = bucket_sampling_counts.max()
    
    if counts_sum == 0:
        pass
    
    return bucket_metadata, bucket_sampling_counts


def cleanup_orphaned_shared_memory(prefixes: List[str] = None, max_retries: int = 3, force_delete: bool = False):
    if prefixes is None:
        prefixes = ['psm_', 'bucket_sampling_', 'buffer_batch', 'buffer_worker']
    
    shm_dir = Path('/dev/shm')
    if not shm_dir.exists():
        return
    
    cleaned_count = 0
    total_size = 0
    skipped_count = 0
    force_deleted_count = 0
    
    for prefix in prefixes:
        for shm_file in shm_dir.glob(f'{prefix}*'):
            if not shm_file.is_file():
                continue
            
            try:
                file_size = shm_file.stat().st_size
            except OSError:
                continue
            
            cleaned = False
            for retry in range(max_retries):
                try:
                    shm = shared_memory.SharedMemory(name=shm_file.name)
                    shm.close()
                    shm.unlink()
                    cleaned = True
                    cleaned_count += 1
                    total_size += file_size
                    break
                except FileNotFoundError:
                    cleaned = True
                    break
                except (PermissionError, OSError) as e:
                    if force_delete and retry == max_retries - 1:
                        try:
                            shm_file.unlink()
                            cleaned = True
                            force_deleted_count += 1
                            total_size += file_size
                            break
                        except (PermissionError, OSError):
                            skipped_count += 1
                            break
                    else:
                        if retry == max_retries - 1:
                            skipped_count += 1
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        skipped_count += 1
                        break
    
    if cleaned_count > 0 or force_deleted_count > 0:
        pass
    if skipped_count > 0:
        pass


def cleanup_orphaned_temp_dirs(pattern: str = 'seq_sampling_split_*', max_retries: int = 3):
    temp_base = Path('/tmp')
    if not temp_base.exists():
        return
    
    cleaned_count = 0
    total_size = 0
    
    for temp_dir in temp_base.glob(pattern):
        if not temp_dir.is_dir():
            continue
        
        try:
            dir_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
        except (OSError, PermissionError):
            dir_size = 0
        
        cleaned = False
        for retry in range(max_retries):
            try:
                shutil.rmtree(temp_dir)
                cleaned = True
                cleaned_count += 1
                total_size += dir_size
                break
            except PermissionError:
                logger.warning(f"[WARNING] Cannot clean up temporary directory {temp_dir.name} (insufficient permissions)")
                break
            except OSError as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                else:
                    logger.warning(f"[WARNING] Failed to clean up temporary directory {temp_dir.name} (retried {max_retries} times): {e}")
    
    if cleaned_count > 0:
        pass
