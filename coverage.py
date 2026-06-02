"""Coverage sampling and chunking module"""
import atexit
import logging
import multiprocessing
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import CONST

FLUSH_THRESHOLD_MB = 1

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)

def count_sequences_with_seqkit(fasta_path: str, num_threads: int = 60) -> int:
    """Use seqkit stats to count sequences in a FASTA file.

    Args:
        fasta_path: Path to the FASTA file
        num_threads: Number of threads

    Returns:
        Total number of sequences

    Raises:
        FileNotFoundError: seqkit is not installed
        ValueError: Unable to parse seqkit output
    """
    result = subprocess.run(
        ['seqkit', 'stats', '-j', str(num_threads), fasta_path],
        capture_output=True,
        text=True,
        check=True
    )

    lines = result.stdout.strip().split('\n')
    header_line = lines[0]
    headers = header_line.split()
    try:
        num_seqs_idx = headers.index('num_seqs')
    except ValueError:
        if 'num_seqs' in header_line.lower():
            for i, h in enumerate(headers):
                if 'num' in h.lower() and 'seq' in h.lower():
                    num_seqs_idx = i
                    break
            else:
                _ = headers[999]
        else:
            _ = headers[999]

    data_line = lines[1]
    data_fields = data_line.split()
    num_seqs_str = data_fields[num_seqs_idx]
    seq_count = int(num_seqs_str.replace(",", ""))

    return seq_count


def sample_lognormal(rng, size, mu, sigma):
    """Sample from lognormal distribution"""
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def sample_gamma(rng, size, k, theta):
    """Sample from gamma distribution"""
    return rng.gamma(shape=k, scale=theta, size=size)

def sample_normal(rng, size, mu, sigma):
    """Sample from normal distribution"""
    return rng.normal(loc=mu, scale=sigma, size=size)

def sample_weibull(rng, size, k, lambda_val):
    """Sample from Weibull distribution"""
    return rng.weibull(k, size=size) * lambda_val

def sample_exponential(rng, size, scale):
    """Sample from exponential distribution"""
    return rng.exponential(scale=scale, size=size)

def sample_poisson(rng, size, lam):
    """Sample from Poisson distribution"""
    return rng.poisson(lam=lam, size=size)

def sample_uniform(rng, size, low, high):
    """Sample from uniform distribution"""
    return rng.uniform(low=low, high=high, size=size)

def sample_nbinom(rng, size, r, p):
    """Sample from negative binomial distribution"""
    return rng.negative_binomial(n=r, p=p, size=size)

def sample_beta(rng, size, alpha, beta, low, scale):
    """Sample from beta distribution (scaled to [low, high] interval)"""
    return rng.beta(alpha, beta, size=size) * scale + low

def sample_distribution(rng, size, dist_name, **params):
    """Top-level sampling function (module-level, pickle-compatible).

    Args:
        rng: numpy.random.Generator instance
        size: Number of samples
        dist_name: Distribution type (lognormal, gamma, normal, weibull, exponential, poisson, uniform, nbinom, beta)
        **params: Distribution parameters

    Returns:
        np.ndarray: Sampling results
    """
    if dist_name == 'lognormal':
        return rng.lognormal(mean=params['mu'], sigma=params['sigma'], size=size)
    elif dist_name == 'gamma':
        return rng.gamma(shape=params['k'], scale=params['theta'], size=size)
    elif dist_name == 'normal':
        return rng.normal(loc=params['mu'], scale=params['sigma'], size=size)
    elif dist_name == 'weibull':
        return rng.weibull(a=params['k'], size=size) * params['lambda']
    elif dist_name == 'exponential':
        return rng.exponential(scale=params['scale'], size=size)
    elif dist_name == 'poisson':
        return rng.poisson(lam=params['lambda'], size=size)
    elif dist_name == 'uniform':
        return rng.uniform(low=params['a'], high=params['b'], size=size)
    elif dist_name == 'nbinom':
        return rng.negative_binomial(n=params['r'], p=params['p'], size=size)
    elif dist_name == 'beta':
        return rng.beta(params['alpha'], params['beta'], size=size) * params['scale'] + params['low']
    else:
        raise ValueError(f"Unsupported distribution type: {dist_name}")

def calculate_depth_distribution_params(dist_name: str, mean_depth: float, cv: float = None,
                                        beta_min: float = None, beta_max: float = None) -> dict:
    """Calculate sampling parameters for sequencing coverage depth based on distribution type.

    Args:
        dist_name: Distribution type
        mean_depth: Target mean sequencing coverage depth
        cv: Coefficient of variation

    Returns:
        dict: {'dist_name': str, 'params': dict}
    """
    from scipy.special import gamma as gamma_func

    if dist_name == 'lognormal':
        sigma_sq = np.log(1 + cv**2)
        sigma = np.sqrt(sigma_sq)
        mu = np.log(mean_depth) - sigma_sq / 2
        return {
            'dist_name': 'lognormal',
            'params': {'mu': mu, 'sigma': sigma}
        }

    elif dist_name == 'gamma':
        k = 1.0 / (cv ** 2)
        theta = mean_depth * (cv ** 2)
        return {
            'dist_name': 'gamma',
            'params': {'k': k, 'theta': theta}
        }

    elif dist_name == 'normal':
        sigma = mean_depth * cv
        return {
            'dist_name': 'normal',
            'params': {'mu': mean_depth, 'sigma': sigma}
        }

    elif dist_name == 'weibull':
        def solve_weibull_k(target_cv_sq, max_iter=100):
            k_low, k_high = 0.1, 20.0
            for _ in range(max_iter):
                k_mid = (k_low + k_high) / 2
                gamma_1_plus_k = gamma_func(1 + 1/k_mid)
                gamma_1_plus_2k = gamma_func(1 + 2/k_mid)
                cv_sq_mid = (gamma_1_plus_2k - gamma_1_plus_k**2) / gamma_1_plus_k**2
                if abs(cv_sq_mid - target_cv_sq) < 1e-8:
                    return k_mid
                elif cv_sq_mid < target_cv_sq:
                    k_low = k_mid
                else:
                    k_high = k_mid
            return (k_low + k_high) / 2

        k = solve_weibull_k(cv**2)
        lambda_val = mean_depth / gamma_func(1 + 1/k)
        return {
            'dist_name': 'weibull',
            'params': {'k': k, 'lambda': lambda_val}
        }

    elif dist_name == 'exponential':
        scale = mean_depth
        return {
            'dist_name': 'exponential',
            'params': {'scale': scale}
        }

    elif dist_name == 'poisson':
        return {
            'dist_name': 'poisson',
            'params': {'lambda': mean_depth}
        }

    elif dist_name == 'uniform':
        range_val = np.sqrt(12) * mean_depth * cv
        a = mean_depth - range_val / 2
        b = mean_depth + range_val / 2
        return {
            'dist_name': 'uniform',
            'params': {'a': a, 'b': b}
        }

    elif dist_name == 'nbinom':
        min_cv_sq = 1.0 / mean_depth
        if cv**2 < min_cv_sq:
            raise ValueError(f"Negative Binomial distribution CV^2 must be >= 1/mean = {min_cv_sq:.6f}")
        r = 1.0 / (cv**2 - 1.0/mean_depth)
        p = r / (r + mean_depth)
        return {
            'dist_name': 'nbinom',
            'params': {'r': r, 'p': p}
        }

    elif dist_name == 'beta':
        if beta_min is not None and beta_max is not None:
            low = float(beta_min)
            high = float(beta_max)
        else:
            low = 0.0
            high = mean_depth * 4.0

        scale = high - low
        mean_norm = (mean_depth - low) / scale
        if mean_norm <= 0 or mean_norm >= 1:
            raise ValueError(
                f"Beta distribution parameters invalid: mean_depth={mean_depth} is not in (beta_min={low}, beta_max={high}) range"
            )
        var_norm = (cv * mean_depth / scale) ** 2
        func = mean_norm * (1 - mean_norm) / var_norm - 1
        if func <= 0:
            raise ValueError(
                f"Beta distribution parameters invalid: func={func:.4f} <= 0. "
                f"Please reduce cv or expand [beta_min, beta_max] range. "
                f"(current scale={scale:.1f}, need scale > {cv * mean_depth / np.sqrt(mean_norm * (1 - mean_norm)):.1f})"
            )
        alpha = mean_norm * func
        beta_param = (1 - mean_norm) * func
        return {
            'dist_name': 'beta',
            'params': {'alpha': alpha, 'beta': beta_param, 'low': low, 'scale': scale}
        }

    else:
        raise ValueError(f"Unsupported distribution type: {dist_name}")

def _process_batch_direct(batch_args):
    """Directly generate counts data (for parallel processing, no scaling needed)"""
    batch_idx, batch_start, batch_end, dist_name, dist_params, batch_seed = batch_args

    rng = np.random.default_rng(batch_seed)
    batch_size_actual = batch_end - batch_start

    counts_batch = sample_distribution(rng, batch_size_actual, dist_name, **dist_params)

    clipped_count = int(np.sum(counts_batch < 0))
    counts_batch = np.clip(counts_batch, 0, None)
    counts_batch = np.round(counts_batch).astype(np.uint64)

    return {
        'batch_idx': batch_idx,
        'batch_start': batch_start,
        'batch_end': batch_end,
        'counts': counts_batch,
        'clipped_count': clipped_count,
    }

def _write_chunk_file_worker(chunk_info):
    """Write a single chunk file (for parallel processing, module-level function for pickle)"""
    import os

    chunk_idx = chunk_info['chunk_idx']
    split_data = chunk_info['split_data']
    split_dir = Path(chunk_info['split_dir'])
    split_file = split_dir / f'chunk_{chunk_idx}_split.npy'

    if split_data.ndim != 2:
        raise ValueError(f"Chunk {chunk_idx}: split_data dimension error, expected 2D array, got {split_data.ndim}D, shape={split_data.shape}")
    if split_data.shape[0] != 2:
        raise ValueError(f"Chunk {chunk_idx}: split_data first dimension error, expected 2, got shape={split_data.shape}")
    if split_data.shape[1] == 0:
        raise ValueError(f"Chunk {chunk_idx}: split_data is empty (shape={split_data.shape})")

    if 'expected_min_size' in chunk_info and 'expected_max_size' in chunk_info:
        expected_min_size = chunk_info['expected_min_size']
        expected_max_size = chunk_info['expected_max_size']
    else:
        dtype_size = split_data.dtype.itemsize
        data_size = split_data.shape[0] * split_data.shape[1] * dtype_size
        expected_min_size = 128 + data_size
        expected_max_size = 512 + data_size

    try:
        np.save(split_file, split_data, allow_pickle=False)
    except Exception as write_err:
        raise IOError(f"Chunk {chunk_idx}: File write failed: {write_err}")

    try:
        with open(split_file, 'r+b') as f:
            os.fsync(f.fileno())
    except Exception as sync_err:
        logger.warning(f"Chunk {chunk_idx}: Filesystem sync failed: {sync_err}, file may have been written but not fully flushed")

    import time

    max_wait_retries = 20
    wait_retry_idx = 0
    file_exists = False

    while not file_exists and wait_retry_idx < max_wait_retries:
        if wait_retry_idx == 0:
            wait_time = 1.0
        else:
            wait_time = min(2 ** wait_retry_idx, 3600.0)

        time.sleep(wait_time)

        if split_file.exists():
            file_exists = True
            if wait_retry_idx > 0:
                total_wait_time = sum(1.0 if i == 0 else min(2**i, 3600.0) for i in range(wait_retry_idx + 1))
        else:
            if wait_retry_idx < 3:
                pass
            else:
                logger.warning(f"Chunk {chunk_idx}: File does not exist after write (waited {wait_retry_idx+1} times, waited {wait_time:.1f}s), continuing to wait...")
            wait_retry_idx += 1

    if not file_exists:
        raise IOError(f"Chunk {chunk_idx}: File does not exist after write (waited {max_wait_retries} times, waited up to 60 minutes and still failed): {split_file}")

    try:
        actual_file_size = split_file.stat().st_size
        if actual_file_size < expected_min_size:
            raise IOError(
                f"Chunk {chunk_idx}: File size abnormal (actual {actual_file_size} bytes, expected minimum {expected_min_size} bytes), "
                f"file may not have been fully written or is corrupted: {split_file}"
            )
        elif actual_file_size > expected_max_size:
            logger.warning(
                f"Chunk {chunk_idx}: File size exceeds expected (actual {actual_file_size} bytes, expected maximum {expected_max_size} bytes), "
                f"file header may be large or format unusual, but continuing: {split_file}"
            )
        else:
            logger.debug(f"Chunk {chunk_idx}: File size validation passed ({actual_file_size} bytes, expected range [{expected_min_size}, {expected_max_size}])")
    except Exception as size_check_err:
        raise IOError(f"Chunk {chunk_idx}: File size validation failed: {size_check_err}")

    del split_data

    return chunk_idx

def _background_write_worker(write_queue, stop_event, failed_chunks=None, failed_chunks_lock=None,
                            written_counter=None, written_condition=None):
    """Background write thread: takes chunks from queue and writes to files.

    Args:
        write_queue: Write queue
        stop_event: Stop event
        failed_chunks: Failed chunks dict
        failed_chunks_lock: Lock protecting failed_chunks
        written_counter: Shared counter
        written_condition: Shared Condition
    """
    written_count = 0
    thread_name = threading.current_thread().name

    while True:
        try:
            try:
                chunk_info = write_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if chunk_info is None:
                break

            chunk_idx = chunk_info.get('chunk_idx', 'unknown')
            logger.debug(f"[{thread_name}] Starting to write chunk {chunk_idx}...")

            try:
                _write_chunk_file_worker(chunk_info)
                written_count += 1

                if written_counter is not None and written_condition is not None:
                    with written_condition:
                        written_counter.value += 1
                        written_condition.notify()

            except Exception as e:
                logger.error(f"[{thread_name}] Failed to write chunk {chunk_idx}: {e}")
                if failed_chunks is not None:
                    if failed_chunks_lock is not None:
                        with failed_chunks_lock:
                            failed_chunks[chunk_idx] = str(e)
                    else:
                        failed_chunks[chunk_idx] = str(e)

            if 'split_data' in chunk_info:
                del chunk_info['split_data']

            write_queue.task_done()

        except Exception as e:
            logger.error(f"[{thread_name}] Background write thread error: {e}")

    return written_count

def _write_pending_chunks_batch_nonblocking(pending_chunks, split_dir, write_queue):
    """Non-blocking batch writing of pending_chunks to queue.

    Args:
        pending_chunks: List of chunks to write
        split_dir: Split directory
        write_queue: Write queue

    Returns:
        Number of chunks submitted to queue
    """
    if len(pending_chunks) == 0:
        return 0

    total_chunks = 0
    for chunk_info in pending_chunks:
        chunk_info_with_dir = chunk_info.copy()
        chunk_info_with_dir['split_dir'] = str(split_dir)
        try:
            write_queue.put_nowait(chunk_info_with_dir)
            total_chunks += 1
        except queue.Full:
            write_queue.put(chunk_info_with_dir, timeout=1.0)
            total_chunks += 1

    pending_chunks.clear()

    return total_chunks

def _chunked_vectorized_allocation(batch_results, chunk_size, num_ref_seqs, split_dir, block_size=10_000_000,
                                   write_batch_size=10000, num_parallel_workers=None, total_cpus=None):
    """Chunked vectorized processing for chunk allocation (memory-friendly, supports batch writing)"""
    import time
    start_time = time.time()

    total_refs = sum(len(br['counts']) for br in batch_results)
    total_reads_preview = int(sum(br['counts'].sum() for br in batch_results))
    num_blocks = (total_refs + block_size - 1) // block_size

    est_num_chunks = max(1, total_reads_preview // chunk_size)

    global_chunk_idx = 0
    global_chunk_reads_count = 0
    global_chunk_ref_start = 0
    global_chunk_ref_indices = []
    global_chunk_counts = []

    total_reads = 0
    num_nonzero = 0
    min_count = float('inf')
    max_count = 0
    all_counts_for_stats = []
    first_3_ref_counts = {}
    last_3_ref_counts = {}

    chunk_metadata_list = []
    pending_chunks = []

    total_written_chunks = 0

    write_queue = queue.Queue(maxsize=100000)
    stop_event = threading.Event()
    failed_chunks = {}
    failed_chunks_lock = threading.Lock()

    written_counter = multiprocessing.Value('L', 0)
    written_condition = threading.Condition()

    if total_cpus is not None and total_cpus > 2:
        base_threads = min(128, total_cpus - 2)
    else:
        base_threads = 32
    num_write_threads = min(base_threads, est_num_chunks)

    write_threads = []
    for i in range(num_write_threads):
        write_thread = threading.Thread(
            target=_background_write_worker,
            args=(write_queue, stop_event, failed_chunks, failed_chunks_lock,
                  written_counter, written_condition),
            daemon=True,
            name=f"BackgroundWriteThread-{i+1}"
        )
        write_thread.start()
        write_threads.append(write_thread)

    all_counts_list = []
    for batch_result in batch_results:
        all_counts_list.append(batch_result['counts'])

    processed_refs = 0
    last_logged_pct = -1

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, total_refs)
        block_actual_size = block_end - block_start

        if block_actual_size == 0:
            break

        block_start_time = time.time()

        block_counts_list = []
        block_ref_start_global = None

        current_global_pos = 0
        for batch_result in batch_results:
            batch_counts = batch_result['counts']
            batch_start = batch_result['batch_start']
            batch_size_actual = len(batch_counts)

            batch_global_start = current_global_pos
            batch_global_end = current_global_pos + batch_size_actual

            if batch_global_end <= block_start:
                current_global_pos = batch_global_end
                continue
            elif batch_global_start >= block_end:
                break
            else:
                local_start = max(0, block_start - batch_global_start)
                local_end = min(batch_size_actual, block_end - batch_global_start)

                if local_start < local_end:
                    overlap_counts = batch_counts[local_start:local_end]
                    block_counts_list.append(overlap_counts)

                    if block_ref_start_global is None:
                        block_ref_start_global = batch_start + local_start

                current_global_pos = batch_global_end

        if len(block_counts_list) == 0:
            continue

        block_counts = np.concatenate(block_counts_list).astype(np.uint64)
        block_ref_indices = np.arange(block_ref_start_global, block_ref_start_global + len(block_counts), dtype=np.uint64)

        block_num_nonzero = int((block_counts > 0).sum())
        block_total_reads = int(block_counts.sum())
        block_min = int(block_counts[block_counts > 0].min()) if block_num_nonzero > 0 else None
        block_max = int(block_counts.max())

        total_reads += block_total_reads
        num_nonzero += block_num_nonzero
        if block_min is not None:
            min_count = min(min_count, block_min)
        max_count = max(max_count, block_max)

        if processed_refs < 3:
            for i in range(min(3 - processed_refs, len(block_counts))):
                ref_idx = block_ref_start_global + i
                first_3_ref_counts[ref_idx] = int(block_counts[i])

        if processed_refs + len(block_counts) > total_refs - 3:
            start_idx = max(0, total_refs - 3 - processed_refs)
            for i in range(start_idx, len(block_counts)):
                ref_idx = block_ref_start_global + i
                last_3_ref_counts[ref_idx] = int(block_counts[i])

        if len(all_counts_for_stats) < 1000:
            remaining = 1000 - len(all_counts_for_stats)
            all_counts_for_stats.extend(block_counts[:remaining].astype(np.float64).tolist())

        for i in range(len(block_counts)):
            ref_idx = block_ref_indices[i]
            count = block_counts[i]

            if count == 0:
                continue

            if global_chunk_reads_count + count <= chunk_size:
                global_chunk_ref_indices.append(ref_idx)
                global_chunk_counts.append(count)
                global_chunk_reads_count += count

                if global_chunk_reads_count == chunk_size:
                    _save_chunk_from_vectors(
                        global_chunk_idx, global_chunk_ref_indices, global_chunk_counts,
                        global_chunk_ref_start, ref_idx, global_chunk_reads_count,
                        chunk_metadata_list, pending_chunks
                    )

                    if len(pending_chunks) > 0:
                        written = _write_pending_chunks_batch_nonblocking(pending_chunks, split_dir, write_queue)
                        total_written_chunks += written
                        est_num_chunks = max(1, est_num_chunks)
                        if int(total_written_chunks * 100 / est_num_chunks) > int((total_written_chunks - written) * 100 / est_num_chunks):
                            pct = int(100 * total_written_chunks / est_num_chunks)

                    global_chunk_idx += 1
                    global_chunk_reads_count = 0
                    global_chunk_ref_indices = []
                    global_chunk_counts = []
                    global_chunk_ref_start = ref_idx + 1

            else:
                remaining = count

                while remaining > 0:
                    space_left = chunk_size - global_chunk_reads_count

                    if space_left > 0:
                        take = min(remaining, space_left)
                        global_chunk_ref_indices.append(ref_idx)
                        global_chunk_counts.append(take)
                        global_chunk_reads_count += take
                        remaining -= take

                    if global_chunk_reads_count == chunk_size:
                        _save_chunk_from_vectors(
                            global_chunk_idx, global_chunk_ref_indices, global_chunk_counts,
                            global_chunk_ref_start, ref_idx, global_chunk_reads_count,
                            chunk_metadata_list, pending_chunks
                        )

                        if len(pending_chunks) > 0:
                            written = _write_pending_chunks_batch_nonblocking(pending_chunks, split_dir, write_queue)
                            total_written_chunks += written
                            est_num_chunks = max(1, est_num_chunks)
                            if int(total_written_chunks * 100 / est_num_chunks) > int((total_written_chunks - written) * 100 / est_num_chunks):
                                pct = int(100 * total_written_chunks / est_num_chunks)

                        global_chunk_idx += 1
                        global_chunk_reads_count = 0
                        global_chunk_ref_indices = []
                        global_chunk_counts = []
                        global_chunk_ref_start = ref_idx

                if remaining > 0:
                    logger.warning(f"  Warning: ref {ref_idx} still has {remaining} reads unallocated")

        processed_refs += len(block_counts)
        block_time = time.time() - block_start_time
        progress_pct = 100 * processed_refs / total_refs

        current_pct_int = int(progress_pct)
        if current_pct_int > last_logged_pct:
            last_logged_pct = current_pct_int

    if global_chunk_reads_count > 0:
        if len(global_chunk_ref_indices) > 0:
            last_ref_idx = global_chunk_ref_indices[-1]
            _save_chunk_from_vectors(
                global_chunk_idx, global_chunk_ref_indices, global_chunk_counts,
                global_chunk_ref_start, num_ref_seqs, global_chunk_reads_count,
                chunk_metadata_list, pending_chunks
            )

    if len(pending_chunks) > 0:
        written = _write_pending_chunks_batch_nonblocking(pending_chunks, split_dir, write_queue)
        total_written_chunks += written
    else:
        pass

    last_logged_pct = -1
    while True:
        done = written_counter.value
        total = total_written_chunks
        pct = int(100 * done / total) if total > 0 else 100
        if pct > last_logged_pct and pct < 100:
            last_logged_pct = pct
        if done >= total:
            break
        with written_condition:
            written_condition.wait(timeout=0.5)

    for _ in range(num_write_threads):
        write_queue.put(None)
    stop_event.set()

    for i, write_thread in enumerate(write_threads):
        write_thread.join()

    if len(failed_chunks) > 0:
        logger.warning(f"  [WARNING] {len(failed_chunks)} chunks failed to write: {list(failed_chunks.keys())[:10]}...")
    else:
        pass

    import os
    try:
        split_dir_fd = os.open(split_dir, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(split_dir_fd)
        os.close(split_dir_fd)
    except Exception as sync_dir_err:
        logger.warning(f"  split_dir fsync failed (does not affect main flow): {sync_dir_err}")
    time.sleep(2.0)

    num_zero = num_ref_seqs - num_nonzero
    median_count = float(np.median(all_counts_for_stats)) if len(all_counts_for_stats) > 0 else 0.0

    total_time = time.time() - start_time

    return (chunk_metadata_list, pending_chunks, total_reads,
            num_nonzero, min_count, max_count, median_count,
            first_3_ref_counts, last_3_ref_counts)

def _save_chunk_from_vectors(chunk_idx, ref_indices_list, counts_list,
                             ref_start, ref_end, total_reads_in_chunk,
                             chunk_metadata_list, pending_chunks):
    """Save chunk from vector data (helper function)"""
    if len(ref_indices_list) == 0:
        return

    ref_indices = np.array(ref_indices_list, dtype=np.uint64)
    counts = np.array(counts_list, dtype=np.uint64)

    nonzero_mask = counts > 0
    if nonzero_mask.sum() > 0:
        ref_indices_nonzero = ref_indices[nonzero_mask]
        counts_nonzero = counts[nonzero_mask]

        split_data = np.vstack([ref_indices_nonzero, counts_nonzero])

        dtype_size = split_data.dtype.itemsize
        data_size = split_data.shape[0] * split_data.shape[1] * dtype_size
        expected_min_size = 128 + data_size
        expected_max_size = 512 + data_size

        actual_ref_start = int(ref_indices_nonzero[0])
        actual_ref_end = int(ref_indices_nonzero[-1])

        pending_chunks.append({
            'chunk_idx': chunk_idx,
            'split_data': split_data,
            'ref_start': actual_ref_start,
            'ref_end': actual_ref_end,
            'total_reads_in_chunk': total_reads_in_chunk,
            'expected_min_size': expected_min_size,
            'expected_max_size': expected_max_size
        })

        chunk_metadata_list.append({
            'chunk_idx': chunk_idx,
            'ref_start': actual_ref_start,
            'ref_end': actual_ref_end,
            'total_reads_in_chunk': total_reads_in_chunk,
            'expected_min_size': expected_min_size,
            'expected_max_size': expected_max_size
        })

def _vectorized_chunk_allocation(batch_results, chunk_size, num_ref_seqs, split_dir):
    """Vectorized chunk allocation (alternative to the original loop)"""
    import time
    start_time = time.time()

    all_counts_list = []
    for batch_result in batch_results:
        all_counts_list.append(batch_result['counts'])
    all_counts = np.concatenate(all_counts_list).astype(np.uint64)

    all_ref_indices = np.arange(0, len(all_counts), dtype=np.uint64)

    num_nonzero = int((all_counts > 0).sum())
    total_reads = int(all_counts.sum())
    min_count = int(all_counts[all_counts > 0].min()) if num_nonzero > 0 else 0
    max_count = int(all_counts.max())

    first_3_ref_counts = {}
    last_3_ref_counts = {}
    if len(all_counts) >= 3:
        for i in range(min(3, len(all_counts))):
            first_3_ref_counts[i + 1] = int(all_counts[i])
    if len(all_counts) > 3:
        for i in range(max(0, len(all_counts) - 3), len(all_counts)):
            last_3_ref_counts[i + 1] = int(all_counts[i])

    all_counts_for_stats = all_counts[:min(1000, len(all_counts))].astype(np.float64)
    median_count = float(np.median(all_counts_for_stats)) if len(all_counts_for_stats) > 0 else 0.0

    merge_time = time.time() - start_time

    cumsum_start = time.time()
    cumsum = np.cumsum(all_counts, dtype=np.uint64)
    cumsum_time = time.time() - cumsum_start

    boundary_start = time.time()
    chunk_boundaries = []
    current_chunk_end = chunk_size

    while current_chunk_end <= cumsum[-1]:
        boundary_idx = np.searchsorted(cumsum, current_chunk_end, side='left')
        chunk_boundaries.append(boundary_idx)
        current_chunk_end += chunk_size

    if len(chunk_boundaries) == 0 or chunk_boundaries[-1] < len(all_counts):
        chunk_boundaries.append(len(all_counts))

    num_chunks = len(chunk_boundaries)
    boundary_time = time.time() - boundary_start

    assign_start = time.time()
    chunk_boundaries_array = np.array(chunk_boundaries, dtype=np.int64)
    chunk_assignments = np.searchsorted(chunk_boundaries_array,
                                       np.arange(len(all_counts)),
                                       side='right')
    assign_time = time.time() - assign_start

    gen_start = time.time()
    chunk_metadata_list = []
    pending_chunks = []

    for chunk_idx in range(num_chunks):
        mask = chunk_assignments == chunk_idx
        chunk_ref_indices = all_ref_indices[mask]
        chunk_counts = all_counts[mask]

        nonzero_mask = chunk_counts > 0
        if nonzero_mask.sum() > 0:
            ref_indices_nonzero = chunk_ref_indices[nonzero_mask]
            counts_nonzero = chunk_counts[nonzero_mask]

            chunk_ref_start = int(ref_indices_nonzero[0])
            chunk_ref_end = int(ref_indices_nonzero[-1])
            chunk_actual_reads_count = int(chunk_counts.sum())

            split_data = np.vstack([ref_indices_nonzero, counts_nonzero])

            dtype_size = split_data.dtype.itemsize
            data_size = split_data.shape[0] * split_data.shape[1] * dtype_size
            expected_min_size = 128 + data_size
            expected_max_size = 512 + data_size

            pending_chunks.append({
                'chunk_idx': chunk_idx,
                'split_data': split_data,
                'ref_start': chunk_ref_start,
                'ref_end': chunk_ref_end,
                'total_reads_in_chunk': chunk_actual_reads_count,
                'expected_min_size': expected_min_size,
                'expected_max_size': expected_max_size
            })

            chunk_metadata_list.append({
                'chunk_idx': chunk_idx,
                'ref_start': chunk_ref_start,
                'ref_end': chunk_ref_end,
                'total_reads_in_chunk': chunk_actual_reads_count,
                'expected_min_size': expected_min_size,
                'expected_max_size': expected_max_size
            })
        else:
            continue

    gen_time = time.time() - gen_start
    total_time = time.time() - start_time

    del all_counts, all_ref_indices, cumsum, chunk_boundaries, chunk_boundaries_array, chunk_assignments

    return (chunk_metadata_list, pending_chunks, total_reads,
            num_nonzero, min_count, max_count, median_count,
            first_3_ref_counts, last_3_ref_counts)

def split_refs_into_chunks_streaming(num_ref_seqs: int,
                                     target_read_depth: float,
                                     dist_name: str,
                                     dist_params: dict,
                                     random_seed: int,
                                     chunk_size: int,
                                     split_dir: Path = None,
                                     num_parallel_workers: int = None,
                                     target_num_chunks: int = None,
                                     total_cpus: int = None,
                                     drop_rate: float = 0.0,
                                     seq_length: int = 150) -> Tuple[List[Dict], Path, int, int]:
    """Directly sample and generate coverage depth counts and split into chunks (memory-friendly, supports parallel).

    Args:
        num_ref_seqs: Total number of ref sequences
        target_read_depth: Target mean sequencing depth
        dist_name: Distribution type name
        dist_params: Distribution parameters dict
        random_seed: Random seed
        chunk_size: Chunk size
        split_dir: Split save directory
        num_parallel_workers: Number of parallel processes
        target_num_chunks: Target number of chunks
        total_cpus: Total number of CPUs
        drop_rate: Drop rate
        seq_length: Sequence length

    Returns:
        (chunk_metadata_list, split_dir, total_reads, num_nonzero)
    """
    if num_parallel_workers is None:
        if total_cpus is not None:
            num_parallel_workers = total_cpus - 1
        else:
            num_parallel_workers = multiprocessing.cpu_count() - 1

    if num_ref_seqs > 100_000_000:
        batch_size = chunk_size * 10
    else:
        batch_size = chunk_size * CONST.COVERAGE_BATCH_MULTIPLIER
    if batch_size > num_ref_seqs:
        batch_size = num_ref_seqs
    num_batches = (num_ref_seqs + batch_size - 1) // batch_size

    ss = np.random.SeedSequence(random_seed)
    batch_seeds = ss.spawn(num_batches)

    batch_args_list = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_ref_seqs)
        if batch_start >= num_ref_seqs:
            break
        batch_args_list.append((
            batch_idx, batch_start, batch_end,
            dist_name, dist_params, batch_seeds[batch_idx]
        ))

    if num_parallel_workers > 1 and num_batches > 1:
        pool = None
        try:
            pool = multiprocessing.Pool(processes=num_parallel_workers)
            batch_results = pool.map(_process_batch_direct, batch_args_list)
        except Exception:
            if pool is not None:
                pool.terminate()
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    else:
        batch_results = []
        for batch_args in batch_args_list:
            batch_results.append(_process_batch_direct(batch_args))

    batch_results.sort(key=lambda x: x['batch_idx'])

    total_clipped = sum(br.get('clipped_count', 0) for br in batch_results)

    all_counts = np.concatenate([br['counts'] for br in batch_results])

    current_zero_count = int(np.sum(all_counts == 0))
    target_zero_count = int(num_ref_seqs * drop_rate)

    drop_seed = ss.spawn(1)[0]
    drop_rng = np.random.default_rng(drop_seed)

    if current_zero_count > target_zero_count:
        surplus = current_zero_count - target_zero_count
        surplus_indices = np.where(all_counts == 0)[0]
        recover_indices = drop_rng.choice(surplus_indices, size=surplus, replace=False)
        all_counts[recover_indices] = 1
    elif current_zero_count < target_zero_count:
        deficit = target_zero_count - current_zero_count
        nonzero_indices = np.where(all_counts > 0)[0]
        drop_indices = drop_rng.choice(nonzero_indices, size=deficit, replace=False)
        all_counts[drop_indices] = 0

    offset = 0
    for br in batch_results:
        batch_size = len(br['counts'])
        br['counts'] = all_counts[offset:offset + batch_size].copy()
        offset += batch_size

    if split_dir is None:
        split_dir = Path(tempfile.mkdtemp(prefix='seq_sampling_split_'))
    else:
        split_dir = Path(split_dir)

    split_dir.mkdir(parents=True, exist_ok=True)

    total_reads_from_batches = int(sum(br['counts'].sum() for br in batch_results))
    total_refs_to_process = sum(len(br['counts']) for br in batch_results)
    use_chunked_vectorized = total_reads_from_batches > 1_000_000_000

    if use_chunked_vectorized:
        try:
            if total_refs_to_process > 100000000:
                dynamic_block_size = chunk_size * 10
            else:
                dynamic_block_size = chunk_size * 100

            (chunk_metadata_list, pending_chunks, total_reads,
             num_nonzero, min_count, max_count, median_count,
             first_3_ref_counts, last_3_ref_counts) = _chunked_vectorized_allocation(
                batch_results, chunk_size, num_ref_seqs, split_dir,
                block_size=dynamic_block_size,
                write_batch_size=10000, num_parallel_workers=num_parallel_workers, total_cpus=total_cpus,
            )

            if len(pending_chunks) > 0:
                logger.warning(f"  Warning: {len(pending_chunks)} chunks still not written, serializing...")
                for chunk_info in pending_chunks:
                    chunk_info_with_dir = chunk_info.copy()
                    chunk_info_with_dir['split_dir'] = str(split_dir)
                    _write_chunk_file_worker(chunk_info_with_dir)
                pending_chunks.clear()

            num_zero = num_ref_seqs - num_nonzero
            all_counts_for_stats = []

            skip_original_loop = True
        except Exception as e:
            logger.warning(f"  Chunked vectorized processing failed, falling back to original method: {e}")
            import traceback
            skip_original_loop = False
            chunk_metadata_list = []
    else:
        skip_original_loop = False
        chunk_metadata_list = []

    est_chunks = len(chunk_metadata_list) if chunk_metadata_list else max(1, (total_refs_to_process + chunk_size - 1) // chunk_size)

    if not skip_original_loop:
        chunk_idx = 0
        chunk_actual_reads_count = 0
        chunk_seq_sampling_split = {}
        chunk_ref_start_idx = 0

        total_reads = 0
        num_nonzero = 0
        min_count = float('inf')
        max_count = 0
        all_counts_for_stats = []

        first_3_ref_counts = {}
        last_3_ref_counts = {}

        pending_chunks = []

        def save_chunk_data(chunk_idx, chunk_seq_sampling_split, chunk_ref_start_idx, chunk_ref_end_idx, chunk_actual_reads_count):
            """Save chunk data to pending write list (lazy write optimization)"""
            if not chunk_seq_sampling_split:
                return

            ref_indices_list = []
            counts_list = []
            for ref_idx, count in sorted(chunk_seq_sampling_split.items()):
                ref_indices_list.append(ref_idx)
                counts_list.append(count)

            ref_indices = np.array(ref_indices_list, dtype=np.uint64)
            counts = np.array(counts_list, dtype=np.uint64)

            actual_ref_start = int(ref_indices[0])
            actual_ref_end = int(ref_indices[-1])

            split_data = np.vstack([ref_indices, counts])

            dtype_size = split_data.dtype.itemsize
            data_size = split_data.shape[0] * split_data.shape[1] * dtype_size
            expected_min_size = 128 + data_size
            expected_max_size = 512 + data_size

            pending_chunks.append({
                'chunk_idx': chunk_idx,
                'split_data': split_data,
                'ref_start': actual_ref_start,
                'ref_end': actual_ref_end,
                'total_reads_in_chunk': chunk_actual_reads_count,
                'expected_min_size': expected_min_size,
                'expected_max_size': expected_max_size,
            })

        for batch_result in batch_results:
            counts_batch = batch_result['counts']
            batch_start = batch_result['batch_start']
            batch_size_actual = len(counts_batch)

            for local_idx in range(batch_size_actual):
                ref_idx = batch_start + local_idx
                count = counts_batch[local_idx]

                if ref_idx < 3:
                    first_3_ref_counts[ref_idx] = count
                elif ref_idx >= num_ref_seqs - 3:
                    last_3_ref_counts[ref_idx] = count

                if len(all_counts_for_stats) < 1000:
                    all_counts_for_stats.append(count)

                if count > 0:
                    num_nonzero += 1
                    min_count = min(min_count, count)
                    max_count = max(max_count, count)
                    total_reads += count

                    if chunk_actual_reads_count + count <= chunk_size:
                        chunk_seq_sampling_split[ref_idx] = count
                        chunk_actual_reads_count += count

                        if chunk_actual_reads_count == chunk_size:
                            save_chunk_data(chunk_idx, chunk_seq_sampling_split, chunk_ref_start_idx, ref_idx, chunk_actual_reads_count)

                            actual_ref_start = min(chunk_seq_sampling_split.keys())
                            actual_ref_end = max(chunk_seq_sampling_split.keys())
                            chunk_metadata_list.append({
                                'chunk_idx': chunk_idx,
                                'ref_start': actual_ref_start,
                                'ref_end': actual_ref_end,
                                'total_reads_in_chunk': chunk_actual_reads_count
                            })

                            chunk_idx += 1
                            chunk_seq_sampling_split = {}
                            chunk_actual_reads_count = 0
                            chunk_ref_start_idx = ref_idx + 1

                    else:
                        remaining = count

                        while remaining > 0:
                            space_left = chunk_size - chunk_actual_reads_count

                            if space_left > 0:
                                take = min(remaining, space_left)
                                chunk_seq_sampling_split[ref_idx] = take
                                chunk_actual_reads_count += take
                                remaining -= take

                            if chunk_actual_reads_count == chunk_size:
                                save_chunk_data(chunk_idx, chunk_seq_sampling_split, chunk_ref_start_idx, ref_idx, chunk_actual_reads_count)

                                actual_ref_start = min(chunk_seq_sampling_split.keys())
                                actual_ref_end = max(chunk_seq_sampling_split.keys())
                                chunk_metadata_list.append({
                                    'chunk_idx': chunk_idx,
                                    'ref_start': actual_ref_start,
                                    'ref_end': actual_ref_end,
                                    'total_reads_in_chunk': chunk_actual_reads_count
                                })

                                chunk_idx += 1
                                chunk_seq_sampling_split = {}
                                chunk_actual_reads_count = 0
                                chunk_ref_start_idx = ref_idx

        if chunk_seq_sampling_split:
            if not chunk_seq_sampling_split:
                raise ValueError(f"Last Chunk {chunk_idx}: Attempted to save empty chunk, chunk_seq_sampling_split is empty")

            save_chunk_data(chunk_idx, chunk_seq_sampling_split, chunk_ref_start_idx, num_ref_seqs - 1, chunk_actual_reads_count)

            actual_ref_start = min(chunk_seq_sampling_split.keys())
            actual_ref_end = max(chunk_seq_sampling_split.keys())
            chunk_metadata_list.append({
                'chunk_idx': chunk_idx,
                'ref_start': actual_ref_start,
                'ref_end': actual_ref_end,
                'total_reads_in_chunk': chunk_actual_reads_count
            })

    total_chunks = len(pending_chunks)
    if total_chunks > 0:

        write_queue = queue.Queue(maxsize=100000)
        stop_event = threading.Event()
        failed_chunks = {}
        failed_chunks_lock = threading.Lock()

        written_counter = multiprocessing.Value('L', 0)
        written_condition = threading.Condition()

        if total_cpus is not None and total_cpus > 2:
            base_threads = min(128, total_cpus - 2)
        else:
            base_threads = 32
        num_write_workers = min(base_threads, total_chunks)

        write_threads = []
        for i in range(num_write_workers):
            write_thread = threading.Thread(
                target=_background_write_worker,
                args=(write_queue, stop_event, failed_chunks, failed_chunks_lock,
                      written_counter, written_condition),
                daemon=True,
                name=f"BackgroundWriteThread-{i+1}"
            )
            write_thread.start()
            write_threads.append(write_thread)

        if total_chunks < 1000:
            batch_size = total_chunks
        elif total_chunks < 10000:
            batch_size = 1000
        else:
            batch_size = 10000

        total_submitted = 0
        num_batches = (total_chunks + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = pending_chunks[batch_start:batch_end]

            for chunk_info in batch_chunks:
                chunk_info_with_dir = chunk_info.copy()
                chunk_info_with_dir['split_dir'] = str(split_dir)

                try:
                    write_queue.put_nowait(chunk_info_with_dir)
                    total_submitted += 1
                except queue.Full:
                    write_queue.put(chunk_info_with_dir, timeout=1.0)
                    total_submitted += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                progress_pct = 100 * (batch_idx + 1) / num_batches

        write_queue.put(None)
        stop_event.set()

        _wait_start = time.time()
        _last_pct = -1
        while written_counter.value < total_chunks:
            with written_condition:
                written_condition.wait(timeout=0.5)
            written = written_counter.value
            pct = int(100 * written / total_chunks)
            if pct != _last_pct:
                elapsed = time.time() - _wait_start
                rate = written / elapsed if elapsed > 0 else 0
                _last_pct = pct

        for wt in write_threads:
            wt.join()

        written_final = written_counter.value
        elapsed_total = time.time() - _wait_start
        rate_total = written_final / elapsed_total if elapsed_total > 0 else 0
        logger.info(
            f"Time elapsed: {elapsed_total:.1f}s, rate: {rate_total:.0f} files/s")

        if failed_chunks:
            logger.error(f"  [ERROR] {len(failed_chunks)} chunks failed to write:")
            for chunk_idx, error_msg in list(failed_chunks.items())[:10]:
                logger.error(f"     - Chunk {chunk_idx}: {error_msg}")
            if len(failed_chunks) > 10:
                logger.error(f"     - ... and {len(failed_chunks)-10} more chunks failed to write")
            raise RuntimeError(f"{len(failed_chunks)} chunks failed to write, cannot continue")

        import random

        max_global_wait_retries = 40
        global_wait_retry_idx = 0
        all_files_exist = False

        sample_indices = list(range(total_chunks))
        sample_size = len(sample_indices)

        while not all_files_exist and global_wait_retry_idx < max_global_wait_retries:
                if global_wait_retry_idx == 0:
                    wait_time = 0.1
                else:
                    wait_time = min(0.5 * (2 ** (global_wait_retry_idx - 1)), 60.0)

                time.sleep(wait_time)

                missing_files = []
                incomplete_files = []
                checked_count = 0
                for idx in sample_indices:
                    chunk_info = pending_chunks[idx]
                    chunk_idx = chunk_info['chunk_idx']
                    split_file = split_dir / f'chunk_{chunk_idx}_split.npy'
                    expected_min_size = chunk_info['expected_min_size']
                    expected_max_size = chunk_info['expected_max_size']

                    if not split_file.exists():
                        missing_files.append((chunk_idx, "File does not exist"))
                    else:
                        try:
                            file_size = split_file.stat().st_size
                            if file_size < expected_min_size:
                                reason = f"File too small ({file_size} bytes, expected >={expected_min_size} bytes)"
                                incomplete_files.append((chunk_idx, reason))
                                missing_files.append((chunk_idx, reason))
                            elif file_size > expected_max_size:
                                reason = f"File too large ({file_size} bytes, expected <={expected_max_size} bytes)"
                                incomplete_files.append((chunk_idx, reason))
                                missing_files.append((chunk_idx, reason))
                        except Exception as e:
                            reason = f"Cannot get file size: {e}"
                            incomplete_files.append((chunk_idx, reason))
                            missing_files.append((chunk_idx, reason))

                    checked_count += 1

                    if checked_count % max(1, total_chunks // 5) == 0:
                        progress_pct = 100 * checked_count / total_chunks

                if not missing_files:
                    all_files_exist = True
                    if global_wait_retry_idx > 0:
                        pass
                    else:
                        pass
                else:
                    problem_count = len(missing_files)
                    true_missing = [item for item in missing_files if "File does not exist" in item[1]]

                    if global_wait_retry_idx < 2:
                        pass
                    else:
                        logger.warning(f"  [WARNING] Detected {problem_count} files with issues (waited {global_wait_retry_idx+1} times, waited {wait_time:.1f}s):")
                        if true_missing:
                            logger.warning(f"     File does not exist: {len(true_missing)} files")
                            for chunk_idx, reason in true_missing[:3]:
                                logger.warning(f"       - Chunk {chunk_idx}")
                        if incomplete_files:
                            logger.warning(f"     File size abnormal: {len(incomplete_files)} files")
                            for chunk_idx, reason in incomplete_files[:3]:
                                logger.warning(f"       - Chunk {chunk_idx}: {reason}")
                        if problem_count > 6:
                            logger.warning(f"     - ... and {problem_count-6} more files with issues")
                        logger.warning(f"  Continuing to wait for file writes to complete...")
                    global_wait_retry_idx += 1

        if not all_files_exist:
            logger.error(f"  [ERROR] After waiting {max_global_wait_retries} times, still {len(missing_files)} files with issues")
            logger.error(f"  Problem file list:")
            for chunk_idx, reason in missing_files[:20]:
                split_file = split_dir / f'chunk_{chunk_idx}_split.npy'
                logger.error(f"     - Chunk {chunk_idx}: {reason} ({split_file})")
            if len(missing_files) > 20:
                logger.error(f"     - ... and {len(missing_files)-20} more files with issues")

            raise RuntimeError(f"{len(missing_files)} split files have issues (do not exist or size abnormal), cannot continue. Please check filesystem and disk space.")

        for chunk_info in pending_chunks:
            if 'split_data' in chunk_info:
                del chunk_info['split_data']

        try:
            import os as _os
            split_dir_fd = _os.open(split_dir, _os.O_RDONLY | _os.O_DIRECTORY)
            _os.fsync(split_dir_fd)
            _os.close(split_dir_fd)
        except Exception as sync_dir_err:
            logger.warning(f"  split_dir fsync failed (does not affect main flow): {sync_dir_err}")
        time.sleep(2.0)

    if not skip_original_loop:
        num_zero = num_ref_seqs - num_nonzero
        median_count = np.median(all_counts_for_stats) if all_counts_for_stats else 0

    print(f"Read-depth: {num_nonzero:,}/{num_ref_seqs:,} refs, {total_reads:,} reads, "
          f"coverage {total_reads / num_ref_seqs:.2f}x (range {min_count if min_count != float('inf') else 0}-{max_count}, median {median_count:.0f})")

    print()

    total_reads = int(total_reads)
    return chunk_metadata_list, split_dir, total_reads, num_nonzero

def _ref_idx_from_split_and_local(split_data: "np.ndarray", local_read_idx: int) -> int:
    """Given split data and local_read_idx within chunk, return the corresponding ref_idx."""
    if split_data.ndim != 2 or split_data.shape[0] != 2:
        raise ValueError("split_data format error, expected shape (2, n)")
    ref_indices = split_data[0]
    counts = split_data[1]
    pos = 0
    for k in range(len(counts)):
        c = int(counts[k])
        if pos <= local_read_idx < pos + c:
            return int(ref_indices[k])
        pos += c
    raise IndexError(f"local_read_idx={local_read_idx} exceeds chunk reads (pos={pos})")

def _build_lookup_table_from_split(split_data: "np.ndarray", total_reads_in_chunk: int) -> "np.ndarray":
    """Build lookup table for split_data: local_read_idx → ref_idx.

    Args:
        split_data: Array with shape=(2, n), first row is ref_indices, second row is counts
        total_reads_in_chunk: Total reads within chunk

    Returns:
        lookup_table: Array with shape=(total_reads_in_chunk,), lookup_table[local_idx] = ref_idx
    """
    if split_data.ndim != 2 or split_data.shape[0] != 2:
        raise ValueError("split_data format error, expected shape (2, n)")
    ref_indices = split_data[0]
    counts = split_data[1]

    lookup_table = np.zeros(int(total_reads_in_chunk), dtype=np.uint64)
    pos = 0
    for k in range(len(counts)):
        c = int(counts[k])
        ref_idx = int(ref_indices[k])
        lookup_table[pos:pos+c] = ref_idx
        pos += c

    return lookup_table

def write_ref_count_and_read_to_ref_tsv(
    chunk_metadata_list: list,
    split_dir: "Path",
    num_ref_seqs: int,
    total_reads: int,
    output_dir: "Path" = None,
    read_id_offset: int = 1,
    shuffle_split_dir: "Path" = None,
    chunk_size: int = None,
    output_stats: bool = False,
) -> None:
    """Write TSV based on split.

    Args:
        chunk_metadata_list: Chunk metadata list
        split_dir: Split file directory
        num_ref_seqs: Total number of ref sequences
        total_reads: Total number of reads
        output_dir: Output directory
        read_id_offset: Read sequence ID starting offset
        shuffle_split_dir: Shuffle mapping directory
        chunk_size: Chunk size
        output_stats: Whether to output statistics
    """
    import os
    from .utils import merge_tsv_chunks

    split_dir = Path(split_dir)

    def fmt_seq(sid):
        return f"seq_{sid}"

    def fmt_reads(rid):
        return f"read_{rid}"

    def fmt_ref(ridx):
        return f"ref_{int(ridx) + 1}"

    out_dir = Path(output_dir) if output_dir is not None else split_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_meta = [{**m, "total_reads_in_chunk": int(m["total_reads_in_chunk"])}
                   for m in sorted(chunk_metadata_list, key=lambda m: m["chunk_idx"])]

    if output_stats:
        ref_count_path = out_dir / "ref_count.tsv"
        ref_counts = np.zeros(num_ref_seqs, dtype=np.uint64)
        for meta in sorted_meta:
            chunk_idx = meta["chunk_idx"]
            split_file = split_dir / f"chunk_{chunk_idx}_split.npy"
            if not split_file.exists():
                logger.warning(f"write_ref_count_and_read_to_ref_tsv: Skipping non-existent {split_file}")
                continue
            split_data = np.load(split_file, allow_pickle=False)
            if split_data.ndim != 2 or split_data.shape[0] != 2:
                logger.warning(f"write_ref_count_and_read_to_ref_tsv: chunk_{chunk_idx} format abnormal, skipping")
                continue
            ref_indices = split_data[0]
            counts = split_data[1]
            for i in range(len(counts)):
                ref_idx = int(ref_indices[i])
                c = int(counts[i])
                if 0 <= ref_idx < num_ref_seqs:
                    ref_counts[ref_idx] += c
        with open(ref_count_path, "w", encoding="utf-8") as f:
            for ref_idx in range(num_ref_seqs):
                f.write(f"{fmt_ref(ref_idx)}\t{ref_counts[ref_idx]}\n")

    if output_stats:
        ordered_tsv_chunks = []
        for meta in sorted_meta:
            chunk_idx = meta["chunk_idx"]
            chunk_tsv_path = out_dir / f"read_to_ref_ordered_{chunk_idx}.tsv"
            if chunk_tsv_path.exists():
                ordered_tsv_chunks.append(chunk_tsv_path)

        read_to_ref_ordered_path = out_dir / "read_to_ref_ordered.tsv"
        if ordered_tsv_chunks:
            merge_tsv_chunks(ordered_tsv_chunks, read_to_ref_ordered_path, delete_chunks=True)
        else:
            logger.warning("No valid ordered TSV chunks, skipping merge")

    if output_stats and shuffle_split_dir is not None and chunk_size is not None:
        shuffle_split_dir = Path(shuffle_split_dir)

        ordered_tsv_path = out_dir / "read_to_ref_ordered.tsv"
        if not ordered_tsv_path.exists():
            logger.warning("read_to_ref_ordered.tsv does not exist, shuffled TSV CIGAR/MD will be empty")
        else:
            chunk_offset: Dict[int, int] = {}
            offset = 0
            for meta in sorted_meta:
                chunk_idx = meta["chunk_idx"]
                chunk_offset[chunk_idx] = offset
                offset += meta["total_reads_in_chunk"]

            def _load_ordered_tsv() -> List[str]:
                with open(ordered_tsv_path, 'r', encoding='utf-8') as f:
                    return f.readlines()

            all_ordered_lines = _load_ordered_tsv()
            total_ordered_lines = len(all_ordered_lines)

            shuffled_tsv_chunks = []
            num_chunks = len(chunk_metadata_list)

            for s in range(num_chunks):
                map_path = shuffle_split_dir / f"shuffle_chunk_{s}_map.raw"
                if not map_path.exists():
                    logger.warning(f"write_ref_count_and_read_to_ref_tsv: Skipping non-existent {map_path}")
                    continue

                mapping = np.fromfile(map_path, dtype=np.uint64).reshape(-1, 4)
                if len(mapping) == 0:
                    continue

                shuffled_read_ids = mapping[:, 0]
                ref_indices_map = mapping[:, 1]
                ordered_chunk_indices = mapping[:, 2]
                local_read_indices = mapping[:, 3]

                chunk_tsv_path = out_dir / f"read_to_ref_shuffled_{s}.tsv"
                shuffled_tsv_chunks.append(chunk_tsv_path)

                buffer = []

                for i in range(len(mapping)):
                    shuffled_read_id = int(shuffled_read_ids[i])
                    ref_idx = int(ref_indices_map[i])
                    oc_idx = int(ordered_chunk_indices[i])
                    lr_idx = int(local_read_indices[i])

                    cigar, md = '', ''
                    global_line = chunk_offset.get(oc_idx, 0) + lr_idx
                    if 0 <= global_line < total_ordered_lines:
                        parts = all_ordered_lines[global_line].rstrip('\n\r').split('\t')
                        if len(parts) >= 4:
                            cigar, md = parts[2], parts[3]

                    ref_id_str = fmt_ref(ref_idx)
                    read_id_str = fmt_reads(shuffled_read_id)
                    buffer.append(f"{read_id_str}\t{ref_id_str}\t{cigar}\t{md}\n")

                    if len(buffer) >= 100000 and i < len(mapping) - 1:
                        with open(chunk_tsv_path, 'a', encoding='utf-8') as f_out:
                            f_out.write(''.join(buffer))
                        buffer.clear()

                if buffer:
                    with open(chunk_tsv_path, 'a', encoding='utf-8') as f_out:
                        f_out.write(''.join(buffer))

            read_to_ref_shuffled_path = out_dir / "read_to_ref_shuffled.tsv"
            if shuffled_tsv_chunks:
                merge_tsv_chunks(shuffled_tsv_chunks, read_to_ref_shuffled_path, delete_chunks=True)
            else:
                logger.warning("No valid shuffled TSV chunks, skipping merge")


def largest_remainder_method(ideal_counts, total):
    """Use Largest Remainder Method to convert floating-point allocation to integer allocation.

    Args:
        ideal_counts: List of ideal allocations
        total: Target sum

    Returns:
        List of integer allocations, sum == total
    """
    n = len(ideal_counts)
    if n == 0:
        return []

    ideal_counts = np.array(ideal_counts, dtype=np.float64)

    floor_counts = ideal_counts.astype(np.int64)

    remainders = ideal_counts - floor_counts

    current_sum = int(floor_counts.sum())
    remaining = int(total - current_sum)

    if remaining < 0:
        raise ValueError(
            f"Largest remainder method error: integer part sum {current_sum} > target sum {total}"
        )
    if remaining > n:
        raise ValueError(
            f"Largest remainder method error: remaining count {remaining} > array length {n}"
        )

    if remaining > 0:
        sorted_indices = np.argsort(remainders)[::-1]
        for i in sorted_indices[:remaining]:
            floor_counts[i] += 1

    final_sum = int(floor_counts.sum())
    if final_sum != total:
        raise ValueError(
            f"Largest remainder method validation failed: final sum {final_sum} != target sum {total}"
        )

    return floor_counts.tolist()

def _shuffle_window_indices(i, N, w):
    """Band shuffle: set of shuffled chunk indices that ordered chunk i is allowed to write to (circular)."""
    if w <= 0:
        return [i % N]
    seen = set()
    out = []
    for k in range(-w, w + 1):
        j = (i + k) % N
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out

def generate_shuffle_mapping(
    chunk_metadata_list: list,
    split_dir: "Path",
    total_reads: int,
    chunk_size: int,
    output_dir: "Path" = None,
    master_seed: int = 42,
    read_id_offset: int = 1,
) -> "Path":
    """Generate shuffle mapping and write to shuffle_split_dir.

    Args:
        chunk_metadata_list: Chunk metadata list
        split_dir: Split file directory
        total_reads: Total number of reads
        chunk_size: Chunk size
        output_dir: Output directory
        master_seed: Master random seed
        read_id_offset: Read sequence ID starting offset

    Returns:
        shuffle_split_dir path
    """
    split_dir = Path(split_dir)
    num_chunks = len(chunk_metadata_list)
    if num_chunks == 0:
        raise ValueError("chunk_metadata_list is empty, cannot generate shuffle mapping")
    if total_reads <= 0:
        raise ValueError("total_reads must be > 0")

    import tempfile as _tempfile
    import uuid as _uuid
    _temp_base = Path(_tempfile.gettempdir())
    shuffle_split_dir = _temp_base / f"dnaterra_shuffle_{_uuid.uuid4().hex[:8]}"
    shuffle_split_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_shuffle_split_dir():
        if shuffle_split_dir.exists():
            import shutil as _shutil
            try:
                _shutil.rmtree(shuffle_split_dir)
            except Exception:
                pass
    atexit.register(_cleanup_shuffle_split_dir)

    sorted_meta = [{**m, "total_reads_in_chunk": int(m["total_reads_in_chunk"])}
                   for m in sorted(chunk_metadata_list, key=lambda m: m["chunk_idx"])]
    reads_count_per_chunk = np.array([meta["total_reads_in_chunk"] for meta in sorted_meta], dtype=np.uint64)
    actual_total = sum(reads_count_per_chunk)
    if actual_total != total_reads:
        logger.warning(
            f"generate_shuffle_mapping: Sum of chunk reads {actual_total} != total_reads {total_reads}, "
            f"using actual total {actual_total} for calculation"
        )
        total_reads = actual_total

    cap_last = int(total_reads - (num_chunks - 1) * chunk_size)
    if cap_last < 0:
        raise ValueError(
            f"total_reads ({total_reads}) < (num_chunks-1)*chunk_size ({(num_chunks-1)*chunk_size}), cannot fit perfectly"
        )
    cap_target = np.array([chunk_size] * (num_chunks - 1) + [cap_last], dtype=np.uint64)

    logger.info(
        f"Shuffled chunk target capacity: first {num_chunks-1} chunks with {chunk_size} each, "
        f"last 1 chunk with {cap_last}, total {total_reads} reads"
    )

    import struct
    shuffled_chunk_counts = np.zeros(num_chunks, dtype=np.uint64)

    if num_chunks == 1:
        num_reads_in_chunk = reads_count_per_chunk[0]
        if num_reads_in_chunk > 0:
            split_file = split_dir / f"chunk_{sorted_meta[0]['chunk_idx']}_split.npy"
            if split_file.exists():
                split_data = np.load(split_file, allow_pickle=False)
                ref_indices = split_data[0].astype(np.int64)
                counts = split_data[1].astype(np.int64)
                ref_expanded = np.repeat(ref_indices, counts)
            else:
                ref_expanded = np.zeros(num_reads_in_chunk, dtype=np.int64)

            seed_c = master_seed + (2**20)
            rng = np.random.default_rng(seed_c)
            perm = rng.permutation(num_reads_in_chunk)
            mapping = np.empty((num_reads_in_chunk, 4), dtype=np.uint64)
            mapping[:, 0] = np.arange(num_reads_in_chunk, dtype=np.uint64) + read_id_offset
            mapping[:, 1] = ref_expanded[perm.astype(np.int64)].astype(np.uint64)
            mapping[:, 2] = 0
            mapping[:, 3] = perm.astype(np.uint64)
            map_path = shuffle_split_dir / f"shuffle_chunk_0_map.raw"
            with open(map_path, "wb") as fh:
                mapping.tofile(fh, "")
        shuffled_chunk_counts[0] = num_reads_in_chunk
        logger.info(
            "Shuffle mapping allocation validation passed (N=1 in-chunk shuffle): total sequences=%d",
            int(shuffled_chunk_counts.sum()),
        )
    else:
        def _compute_cap_in_window(w_val):
            """Given window half-width w, calculate the upper limit of total reads each shuffled chunk j can receive from within the window"""
            cap_in_window = np.zeros(num_chunks, dtype=np.uint64)
            for i in range(num_chunks):
                reads_i = reads_count_per_chunk[i]
                if reads_i == 0:
                    continue
                window_js = _window_js(i, w_val, num_chunks)
                cap_in_win = sum(int(cap_target[j]) for j in window_js)
                if cap_in_win <= 0:
                    continue
                for j in window_js:
                    cap_in_window[j] += round(reads_i * int(cap_target[j]) / cap_in_win)
            return cap_in_window

        def _window_js(i, w_val, n):
            """Shuffled chunk indices within ordered chunk i's window (circular)"""
            out = []
            seen = set()
            for k in range(-w_val, w_val + 1):
                j = (i + k) % n
                if j not in seen:
                    seen.add(j)
                    out.append(j)
            return out

        w = 0
        cap_in_window = None
        for w_candidate in range(num_chunks):
            cap_in_window = _compute_cap_in_window(w_candidate)
            feasible = all(int(cap_in_window[j]) >= int(cap_target[j]) for j in range(num_chunks))
            if feasible:
                w = w_candidate
                break
        if w == 0:
            w = num_chunks - 1

        logger.info(
            f"Adaptive window half-width w={w} (num_chunks={num_chunks}, "
            f"total_reads={total_reads})")
        if cap_in_window is not None:
            col_sums = cap_in_window
            max_diff_j = max(range(num_chunks), key=lambda j: abs(int(col_sums[j]) - int(cap_target[j])))
            logger.debug(
                f"cap_in_window[0..2]=[{', '.join(str(int(col_sums[j])) for j in range(min(3, num_chunks)))}]")
            logger.debug(
                f"cap_target[0..2]=[{', '.join(str(int(cap_target[j])) for j in range(min(3, num_chunks)))}]")
            if num_chunks > 6:
                logger.debug(
                    f"cap_in_window[-3..]=[{', '.join(str(int(col_sums[j])) for j in range(max(3, num_chunks-3), num_chunks))}]")
                logger.debug(
                    f"cap_target[-3..]=[{', '.join(str(int(cap_target[j])) for j in range(max(3, num_chunks-3), num_chunks))}]")
            logger.debug(
                f"Maximum error column: j={max_diff_j}, "
                f"cap_in_window={int(col_sums[max_diff_j])}, "
                f"cap_target={int(cap_target[max_diff_j])}, diff={int(col_sums[max_diff_j]) - int(cap_target[max_diff_j])}")

        try:
            def _in_band(o, j_target, w_val, n):
                """Whether shuffled chunk j_target is within ordered chunk o's window (circular distance <= w)"""
                d = abs(o - j_target)
                return min(d, n - d) <= w_val

            allocation_matrix = np.zeros((num_chunks, num_chunks), dtype=np.uint64)

            for ordered_chunk_idx in range(num_chunks):
                num_reads_in_chunk = reads_count_per_chunk[ordered_chunk_idx]
                if num_reads_in_chunk == 0:
                    continue

                window_js = _window_js(ordered_chunk_idx, w, num_chunks)
                cap_in_window = sum(cap_target[j] for j in window_js)
                if cap_in_window <= 0:
                    raise ValueError(
                        f"Ordered chunk {ordered_chunk_idx} window cap_target sum is 0"
                    )
                ideal_counts = [
                    num_reads_in_chunk * cap_target[j] / cap_in_window
                    for j in window_js
                ]
                reads_per_win = largest_remainder_method(
                    ideal_counts, num_reads_in_chunk
                )
                for idx, j in enumerate(window_js):
                    allocation_matrix[ordered_chunk_idx, j] = reads_per_win[idx]

            def _circular_dist(a, b):
                return min(abs(a - b), num_chunks - abs(a - b))

            max_iterations = num_chunks * 2
            for iteration in range(max_iterations):
                changed = False
                for ordered_chunk_idx in range(num_chunks):
                    num_reads_in_chunk = reads_count_per_chunk[ordered_chunk_idx]
                    if num_reads_in_chunk == 0:
                        continue
                    row_sum = int(allocation_matrix[ordered_chunk_idx].sum())
                    if row_sum == num_reads_in_chunk:
                        continue
                    surplus = row_sum - num_reads_in_chunk
                    if surplus <= 0:
                        continue

                    candidates = []
                    for j_target in range(num_chunks):
                        deficit_j = int(cap_target[j_target] - shuffled_chunk_counts[j_target])
                        if deficit_j <= 0:
                            continue
                        if _in_band(ordered_chunk_idx, j_target, w, num_chunks):
                            dist = _circular_dist(ordered_chunk_idx, j_target)
                            candidates.append((dist, j_target, deficit_j))

                    candidates.sort(key=lambda x: x[0])

                    remaining_surplus = surplus
                    for _, j_target, deficit_j in candidates:
                        if remaining_surplus <= 0:
                            break
                        transfer_amount = min(remaining_surplus, deficit_j, int(allocation_matrix[ordered_chunk_idx, j_target]))
                        if transfer_amount > 0:
                            allocation_matrix[ordered_chunk_idx, j_target] -= transfer_amount
                            shuffled_chunk_counts[j_target] += transfer_amount
                            remaining_surplus -= transfer_amount
                            if remaining_surplus != surplus:
                                changed = True

                    if remaining_surplus > 0:
                        logger.warning(f"Iteration {iteration}: Ordered chunk {ordered_chunk_idx} has {remaining_surplus} reads that cannot be redistributed within the window")

                if not changed:
                    break

            for ordered_chunk_idx in range(num_chunks):
                num_reads_in_chunk = reads_count_per_chunk[ordered_chunk_idx]
                row_sum = int(allocation_matrix[ordered_chunk_idx].sum())
                if row_sum != num_reads_in_chunk:
                    raise RuntimeError(
                        f"Allocation validation failed: Ordered chunk {ordered_chunk_idx} allocation sum {row_sum} != original count {num_reads_in_chunk}"
                    )

            total_reads = int(reads_count_per_chunk.sum())
            total_cap = int(cap_target.sum())
            if total_reads == total_cap:
                deficit_cols = []
                surplus_cols = []
                col_sums = allocation_matrix.sum(axis=0).astype(np.uint64)
                for j in range(num_chunks):
                    diff = int(cap_target[j]) - int(col_sums[j])
                    if diff > 0:
                        deficit_cols.append((j, diff))
                    elif diff < 0:
                        surplus_cols.append((j, -diff))

                if not deficit_cols:
                    pass
                elif not surplus_cols:
                    logger.warning("No extra chunks available for fixing, chunk error cannot be eliminated")
                else:
                    total_deficit = sum(d for _, d in deficit_cols)
                    total_surplus = sum(s for _, s in surplus_cols)

                    surplus_idx = 0
                    fixed = []
                    for deficit_col, deficit_amt in deficit_cols:
                        while deficit_amt > 0 and surplus_idx < len(surplus_cols):
                            surplus_col, surplus_amt = surplus_cols[surplus_idx]
                            if surplus_amt <= 0:
                                surplus_idx += 1
                                continue
                            moved = 0
                            for i_row in range(num_chunks):
                                src_val = int(allocation_matrix[i_row, surplus_col])
                                if src_val == 0:
                                    continue
                                take = min(src_val, surplus_amt - moved, deficit_amt - moved)
                                if take > 0:
                                    allocation_matrix[i_row, surplus_col] -= take
                                    allocation_matrix[i_row, deficit_col] += take
                                    moved += take
                                    if moved >= deficit_amt or moved >= surplus_amt:
                                        break
                            if moved > 0:
                                deficit_amt -= moved
                                surplus_cols[surplus_idx] = (surplus_col, surplus_amt - moved)
                                if surplus_cols[surplus_idx][1] <= 0:
                                    surplus_idx += 1
                            else:
                                surplus_idx += 1
                        if deficit_amt == 0:
                            fixed.append(deficit_col)
                        else:
                            logger.warning(f"Cannot fix shuffled chunk {deficit_col} (still short {deficit_amt} reads)")

                    if fixed:
                        pass

                    for surplus_col, surplus_amt in surplus_cols:
                        if surplus_amt <= 0:
                            continue
                        col_sum = int(allocation_matrix[:, surplus_col].sum())
                        cap_j = int(cap_target[surplus_col])
                        excess = col_sum - cap_j
                        if excess <= 0:
                            continue
                        for i_row in range(num_chunks):
                            src_val = int(allocation_matrix[i_row, surplus_col])
                            if src_val == 0:
                                continue
                            take = min(src_val, excess)
                            if take > 0:
                                allocation_matrix[i_row, surplus_col] -= take
                                excess -= take
                                if excess <= 0:
                                    break

                remaining_issues = []
                final_col_sums = allocation_matrix.sum(axis=0).astype(np.uint64)
                for j in range(num_chunks):
                    if int(final_col_sums[j]) != int(cap_target[j]):
                        remaining_issues.append(j)

                if remaining_issues:
                    total_err = sum(abs(int(final_col_sums[j]) - int(cap_target[j])) for j in remaining_issues)
                    logger.warning(f"{len(remaining_issues)} columns still have errors, total error {total_err} reads")
                else:
                    logger.info("Column errors have been fully fixed")

            remaining_issues = []
            for j_target in range(num_chunks):
                col_sum = int(allocation_matrix[:, j_target].sum())
                cap_j = int(cap_target[j_target])
                if col_sum != cap_j:
                    remaining_issues.append(j_target)
                    logger.warning(f"Allocation error (column {j_target}: received {col_sum} vs target {cap_j}), but total is consistent, will continue")

            if remaining_issues:
                pass
            else:
                logger.info("Allocation validation passed, column sums match target capacity perfectly")

            FLUSH_THRESHOLD_BYTES = int(FLUSH_THRESHOLD_MB * 1024 * 1024)
            ROW_BYTES = 4 * 8
            FLUSH_THRESHOLD_ROWS = FLUSH_THRESHOLD_BYTES // ROW_BYTES

            last_col = num_chunks - 1
            buffers = [np.empty((int(cap_target[s]), 4), dtype=np.uint64) for s in range(num_chunks)]
            pos_in_shuffled = np.zeros(num_chunks, dtype=np.uint64)
            written_counts = np.zeros(num_chunks, dtype=np.uint64)

            _flush_log_every = max(1, num_chunks // 100)
            _ordered_chunk_logged = 0

            def _flush_buffer(s):
                """Append filled data in buffer[s] to .raw file, then clear buffer and reset pos."""
                nonlocal _ordered_chunk_logged
                buf_len = int(pos_in_shuffled[s])
                if buf_len == 0:
                    return
                raw_path = shuffle_split_dir / f"shuffle_chunk_{s}_map.raw"
                with open(raw_path, "ab") as fh:
                    buffers[s][:buf_len].tofile(fh, "")
                written_counts[s] += buf_len
                pos_in_shuffled[s] = 0
                _ordered_chunk_logged += 1
                if _ordered_chunk_logged == num_chunks or _ordered_chunk_logged % _flush_log_every == 0:
                    logger.debug(f"[FLUSH] chunk={s:06d}, rows={buf_len}, total={written_counts[s]}")

            for ordered_chunk_idx in range(num_chunks):
                num_reads_in_chunk = reads_count_per_chunk[ordered_chunk_idx]
                if num_reads_in_chunk == 0:
                    continue

                _chunk_idx = sorted_meta[ordered_chunk_idx]["chunk_idx"]
                _split_file = split_dir / f"chunk_{_chunk_idx}_split.npy"
                if not _split_file.exists():
                    logger.warning(f"generate_shuffle_mapping: Skipping missing ordered_chunk {ordered_chunk_idx}")
                    continue
                split_data = np.load(_split_file, allow_pickle=False)
                if split_data.ndim != 2 or split_data.shape[0] != 2:
                    logger.warning(f"generate_shuffle_mapping: chunk_{ordered_chunk_idx} format abnormal, skipping")
                    continue

                ref_indices = split_data[0]
                counts = split_data[1]
                n_refs = len(counts)

                ref_expanded = np.repeat(ref_indices.astype(np.int64), counts.astype(np.int64))

                cumsum = np.empty(n_refs + 1, dtype=np.uint64)
                cumsum[0] = 0
                cumsum[1:] = np.cumsum(counts.astype(np.uint64))

                inner_pos = np.concatenate([
                    np.arange(int(counts[k]), dtype=np.uint64) + cumsum[k]
                    for k in range(n_refs)
                ])

                seed_c = master_seed + ordered_chunk_idx
                rng = np.random.default_rng(seed_c)
                perm = rng.permutation(num_reads_in_chunk)
                shuffled_pos  = inner_pos[perm.astype(np.uint64)]
                shuffled_refs = ref_expanded[perm.astype(np.int64)]

                offset = 0
                for shuffled_chunk_idx in range(num_chunks):
                    alloc_count = int(allocation_matrix[ordered_chunk_idx, shuffled_chunk_idx])
                    if alloc_count <= 0:
                        continue

                    buf_pos = int(pos_in_shuffled[shuffled_chunk_idx])
                    buf_rows = buffers[shuffled_chunk_idx]
                    rows_slice = slice(buf_pos, buf_pos + alloc_count)

                    buf_rows[rows_slice, 0] = (
                        read_id_offset + int(shuffled_chunk_counts[:shuffled_chunk_idx].sum()) +
                        np.arange(alloc_count, dtype=np.uint64)
                    )
                    buf_rows[rows_slice, 1] = shuffled_refs[offset:offset + alloc_count]
                    buf_rows[rows_slice, 2] = ordered_chunk_idx
                    buf_rows[rows_slice, 3] = shuffled_pos[offset:offset + alloc_count]

                    pos_in_shuffled[shuffled_chunk_idx] += alloc_count
                    if pos_in_shuffled[shuffled_chunk_idx] >= FLUSH_THRESHOLD_ROWS:
                        _flush_buffer(shuffled_chunk_idx)
                    shuffled_chunk_counts[shuffled_chunk_idx] += alloc_count
                    offset += alloc_count

            for s in range(num_chunks):
                _flush_buffer(s)

            total_written = 0
            for j in range(num_chunks):
                total_written += int(written_counts[j])
                if int(written_counts[j]) != int(cap_target[j]):
                    logger.warning(f"Shuffled chunk {j:06d}: Actually wrote {int(written_counts[j])} reads, target {int(cap_target[j])} reads")

        except Exception:
            raise

    return shuffle_split_dir

def _build_fasta_chunk_record_index(fasta_path: "Path") -> list:
    """Build record index for ordered chunk FASTA file: each record is (sequence line start offset, sequence line byte length)."""
    index = []
    with open(fasta_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b">"):
                seq_start = f.tell()
                seq_line = f.readline()
                if not seq_line:
                    break
                seq_byte_len = len(seq_line)
                index.append((seq_start, seq_byte_len))
    return index

def _save_chunk_index(chunk_file: "Path", index: list) -> None:
    """Save ordered chunk index to .idx.npy file (optimization: avoid rebuilding index during shuffle phase)."""
    index_file = chunk_file.parent / (chunk_file.name + '.idx.npy')
    try:
        index_array = np.array(index, dtype=[('start', np.int64), ('length', np.int64)])
        np.save(index_file, index_array, allow_pickle=False)
    except Exception as e:
        logger.warning(f"Failed to save chunk index {chunk_file}: {e}, will build index during shuffle phase")

def _load_chunk_index(chunk_file: "Path") -> list:
    """Load pre-saved ordered chunk index, or build index on the fly if not exists."""
    index_file = chunk_file.parent / (chunk_file.name + '.idx.npy')
    if index_file.exists():
        try:
            index_array = np.load(index_file, allow_pickle=False)
            index = [(int(row['start']), int(row['length'])) for row in index_array]
            return index
        except Exception as e:
            pass

    return _build_fasta_chunk_record_index(chunk_file)

def _read_sequences_by_indices(
    fasta_path: "Path",
    index: list,
    local_indices: "np.ndarray",
) -> list:
    """Read sequences from indexed FASTA by local_indices (returns bytes list without newlines)."""
    result = []
    with open(fasta_path, "rb") as f:
        for local_idx in local_indices:
            start, length = index[int(local_idx)]
            f.seek(start)
            raw = f.read(length)
            result.append(raw.rstrip(b"\n"))
    return result

def _validate_shuffle_mapping_caps(shuffle_split_dir, cap_target_arr):
    """Validate that the number of lines in each shuffled chunk's mapping file equals the target capacity cap_target_arr[s]"""
    num_chunks = len(cap_target_arr)
    total_mapped = 0
    mismatches = []
    for s in range(num_chunks):
        raw_path = shuffle_split_dir / f"shuffle_chunk_{s}_map.raw"
        if not raw_path.exists():
            mismatches.append(f"chunk {s}: Mapping file does not exist {raw_path}")
            continue
        total_bytes = raw_path.stat().st_size
        cap_s = total_bytes // (4 * 8)
        total_mapped += cap_s
        if cap_s != cap_target_arr[s]:
            mismatches.append(
                f"chunk {s}: Mapping line count {cap_s} != target capacity {cap_target_arr[s]}"
            )
        logger.debug(
            f"[Shuffle Map Validate] Total mapped reads={total_mapped}, expected={int(cap_target_arr.sum())}, "
            f"num_chunks={num_chunks}"
        )
    if mismatches:
        for m in mismatches:
            logger.error(f"[Shuffle Map Validate] {m}")
        raise ValueError(
            f"Shuffle mapping validation failed: {len(mismatches)} chunks have mismatched capacity, "
            f"total_mapped={total_mapped}, expected={int(cap_target_arr.sum())}"
        )
    if total_mapped != int(cap_target_arr.sum()):
        raise ValueError(
            f"Shuffle mapping total {total_mapped} != expected {int(cap_target_arr.sum())}"
        )

def _write_one_shuffle_chunk_worker(task: dict, total_reads: int = None, num_chunks: int = None) -> int:
    """Worker for writing a single shuffled chunk: reads mapping, fetches sequences from ordered chunks, writes shuffled chunk file.

    Mapping file format: (shuffled_read_id, ref_idx, ordered_chunk_idx, local_read_idx)

    Returns: shuffled_chunk_idx_s
    """
    s = task["shuffled_chunk_idx_s"]
    shuffle_split_dir = Path(task["shuffle_split_dir"])
    output_dir = Path(task["output_dir"])
    ordered_chunk_dir = Path(task["ordered_chunk_dir"])
    read_id_offset = task["read_id_offset"]
    chunk_size = task["chunk_size"]

    mapping_path = shuffle_split_dir / f"shuffle_chunk_{s}_map.raw"
    mapping = np.fromfile(mapping_path, dtype=np.uint64).reshape(-1, 4)
    cap_s = len(mapping)

    cap_target = np.array([chunk_size] * (num_chunks - 1) + [int(total_reads - (num_chunks - 1) * chunk_size)],
                          dtype=np.uint64)
    reads_seq_global_id_start = read_id_offset + int(cap_target[:s].sum())

    if cap_s != cap_target[s]:
        logger.error(
            f"Shuffled chunk {s:06d}: Actual reads count {cap_s} != target capacity {cap_target[s]}, "
            f"read ID will be calculated incorrectly!"
        )
    elif s == 0:
        logger.debug(
            f"First shuffled chunk {s:06d}: read_id_start={reads_seq_global_id_start}, total_reads={total_reads}, num_chunks={num_chunks}")
    elif s == num_chunks - 1:
        logger.debug(
            f"Last shuffled chunk {s:06d}: "
            f"last read_id_start={reads_seq_global_id_start}, "
            f"last read_id_end={reads_seq_global_id_start + cap_s - 1}")

    newline = b"\n"

    output_buffer = np.empty(cap_s, dtype=object)
    orig_chunk_indices = mapping[:, 2]
    local_indices = mapping[:, 3]
    unique_c = np.unique(orig_chunk_indices)

    for c in unique_c:
        mask = orig_chunk_indices == c
        output_positions = np.where(mask)[0]
        local_idx_list = local_indices[mask]
        chunk_file = ordered_chunk_dir / f"output_chunk_{c}.fasta"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Ordered chunk file does not exist: {chunk_file}")

        index = _load_chunk_index(chunk_file)
        seqs = _read_sequences_by_indices(chunk_file, index, local_idx_list)
        for pos, seq in zip(output_positions, seqs):
            output_buffer[pos] = seq

    out_path = output_dir / f"output_chunk_shuffled_{s}.fasta"
    temp_path = output_dir / f"output_chunk_shuffled_{s}.fasta.tmp"

    try:
        BATCH_SIZE = 10000
        with open(temp_path, "wb", buffering=8 * 1024 * 1024) as f_out:
            for batch_start in range(0, cap_s, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, cap_s)
                batch_data = []
                for i in range(batch_start, batch_end):
                    seq_id = reads_seq_global_id_start + i
                    batch_data.append(b">read_")
                    batch_data.append(str(seq_id).encode("ascii"))
                    batch_data.append(newline)
                    batch_data.append(output_buffer[i])
                    batch_data.append(newline)
                f_out.write(b"".join(batch_data))
            try:
                os.fsync(f_out.fileno())
            except Exception as e:
                logger.warning(f"Shuffled chunk {s:06d}: fsync failed: {e}")

        if out_path.exists():
            try:
                out_path.unlink()
            except Exception as e:
                logger.warning(f"Shuffled chunk {s:06d}: Failed to delete existing file: {e}, trying force rename")

        try:
            temp_path.rename(out_path)
        except OSError as rename_err:
            logger.warning(f"Shuffled chunk {s:06d}: rename failed ({rename_err}), using copy+delete instead")
            import shutil
            shutil.copy2(temp_path, out_path)
            try:
                temp_path.unlink()
            except Exception:
                pass

    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise IOError(f"Shuffled chunk {s:06d}: Write failed: {e}")

    return s

def write_shuffle_chunks_parallel(
    shuffle_split_dir: "Path",
    output_dir: "Path",
    chunk_metadata_list: list,
    chunk_size: int,
    read_id_offset: int,
    num_workers: int = None,
) -> None:
    """Parallel write all shuffled chunks: each worker processes several shuffled chunks.

    Args:
        shuffle_split_dir: Shuffle mapping directory
        output_dir: Output directory
        chunk_metadata_list: Chunk metadata list
        chunk_size: Chunk size
        read_id_offset: Read sequence ID starting offset
        num_workers: Number of parallel workers
    """
    shuffle_split_dir = Path(shuffle_split_dir)
    output_dir = Path(output_dir)
    num_chunks = len(chunk_metadata_list)
    if num_workers is None:
        num_workers = max(1, min(num_chunks, (os.cpu_count() or 4) - 1))

    total_reads = sum(int(m["total_reads_in_chunk"]) for m in chunk_metadata_list)

    cap_last = int(total_reads - (num_chunks - 1) * chunk_size)
    if cap_last < 0:
        raise ValueError(f"total_reads ({total_reads}) < (num_chunks-1)*chunk_size ({(num_chunks-1)*chunk_size})")
    cap_target_arr = np.array([chunk_size] * (num_chunks - 1) + [cap_last], dtype=np.uint64)

    _validate_shuffle_mapping_caps(shuffle_split_dir, cap_target_arr)

    logger.info(
        f"[Shuffle Write] total_reads={total_reads}, num_chunks={num_chunks}, "
        f"chunk_size={chunk_size}, cap_last={cap_last}"
    )

    tasks = [
        {
            "shuffled_chunk_idx_s": s,
            "shuffle_split_dir": str(shuffle_split_dir),
            "output_dir": str(output_dir),
            "ordered_chunk_dir": str(output_dir),
            "read_id_offset": read_id_offset,
            "chunk_size": chunk_size,
        }
        for s in range(num_chunks)
    ]
    worker_tasks = [[] for _ in range(num_workers)]
    for i, t in enumerate(tasks):
        worker_tasks[i % num_workers].append(t)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for wid in range(num_workers):
            for t in worker_tasks[wid]:
                futures.append(executor.submit(_write_one_shuffle_chunk_worker, t, total_reads, num_chunks))
        pbar = None
        if _TQDM_AVAILABLE and num_chunks > 0:
            pbar = tqdm(total=num_chunks, desc="Generating shuffled chunks", unit="chunk", ncols=100)
        try:
            for fut in as_completed(futures):
                try:
                    fut.result()
                    if pbar is not None:
                        pbar.update(1)
                except Exception as e:
                    logger.exception(f"Shuffled chunk write failed: {e}")
                    raise
        finally:
            if pbar is not None:
                pbar.n = num_chunks
                pbar.refresh()
                pbar.close()
                print()
