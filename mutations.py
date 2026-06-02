"""Sequence mutation application module"""
import ctypes
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from . import CONST

logger = logging.getLogger(__name__)

def apply_substitutions(seq_array: np.ndarray, 
                       sub_seq_idx: np.ndarray,
                       sub_pos: np.ndarray,
                       sub_choices: np.ndarray,
                       sub_cum_probs: np.ndarray,
                       rng) -> int:
    """
    Apply substitution errors to numpy array (in-place modification, optimized with lookup tables).
    
    Performance optimizations:
    - Pre-built lookup tables (LUT)
    - Vectorized random sampling (generate all random numbers at once)
    - CDF + conditional mask method for weighted sampling (optimized for 3 elements, 20-30% faster than searchsorted)
    
    Args:
        seq_array: Sequence array, shape (chunk_size, seq_length)
        sub_seq_idx: Chunk-local index array for substitution errors (0 to chunk_size-1)
        sub_pos: Position array for substitution errors (0 to seq_length-1)
        sub_choices: Substitution lookup table
        sub_cum_probs: Cumulative probability array
        rng: Random number generator
    
    Returns:
        n_subs: Number of substitution errors actually applied
    """
    n_subs = len(sub_seq_idx)
    
    if n_subs == 0:
        return 0
    
    original_bases = seq_array[sub_seq_idx, sub_pos]
    choices = sub_choices[original_bases]  
    cum_probs = sub_cum_probs[original_bases]
    random_vals = rng.random(n_subs, dtype=np.float32)
    selected_idx = np.zeros(n_subs, dtype=np.uint8)
    selected_idx[random_vals >= cum_probs[:, 0]] = 1
    selected_idx[random_vals >= cum_probs[:, 1]] = 2
    new_bases = np.choose(selected_idx, choices.T)
    seq_array[sub_seq_idx, sub_pos] = new_bases
    
    return n_subs


def apply_substitutions_return_bases(seq_array: np.ndarray,
                                     sub_seq_idx: np.ndarray,
                                     sub_pos: np.ndarray,
                                     sub_choices: np.ndarray,
                                     sub_cum_probs: np.ndarray,
                                     rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Variant of apply_substitutions: modifies sequence in-place while returning original and new base arrays.
    Used for generating CIGAR/MD.

    Returns:
        (original_bases, new_bases): Two uint8 arrays, shape (n_subs,), ASCII code values.
    """
    n_subs = len(sub_seq_idx)
    if n_subs == 0:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)

    original_bases = seq_array[sub_seq_idx, sub_pos].copy()
    choices = sub_choices[original_bases]
    cum_probs = sub_cum_probs[original_bases]
    random_vals = rng.random(n_subs, dtype=np.float32)
    selected_idx = np.zeros(n_subs, dtype=np.uint8)
    selected_idx[random_vals >= cum_probs[:, 0]] = 1
    selected_idx[random_vals >= cum_probs[:, 1]] = 2
    new_bases = np.choose(selected_idx, choices.T)
    seq_array[sub_seq_idx, sub_pos] = new_bases
    return original_bases, new_bases


def sample_substitutions_kmer(seq_array: np.ndarray,
                              sub_seq_idx: np.ndarray,
                              sub_pos: np.ndarray,
                              kmer_sub: np.ndarray,
                              sub_choices: np.ndarray,
                              sub_cum_probs: np.ndarray,
                              seq_length: int,
                              rng) -> np.ndarray:
    """
    Kmer substitution pre-sampling: for each SUB record, sample target substitution bases
    based on 5mer context in the noise-free buffer, return sub_new_bases array without writing to buffer.

    Boundary rules (consistent with document 7.4):
        kmer region: KMER_N_HEAD <= pos < seq_length - KMER_N_TAIL
        other positions: fall back to single-base preference (sub_choices / sub_cum_probs)

    Kmer substitution definition (document 7.1):
        5mer = buffer[seq_idx, pos-3 : pos+2]
        ref_3mer = 5mer[:3]  →  i_3mer (0..63)
        ref4     = 5mer[3]   →  ref4_idx (0..3)
        probs    = kmer_sub[i_3mer, ref4_idx, :]  (length-4 probability vector)
        sample read4_idx  →  write IDX_TO_ASCII[read4_idx] to get target base

    Args:
        seq_array:      Noise-free sequence array, shape (chunk_size, seq_length), read-only
        sub_seq_idx:    SUB chunk-local sequence indices (0..chunk_size-1)
        sub_pos:        SUB position array (0..seq_length-1)
        kmer_sub:       Kmer substitution probability matrix, shape (64, 4, 4), normalized
        sub_choices:    Single-base substitution lookup table, shape (256, 3)
        sub_cum_probs:  Single-base substitution cumulative probabilities, shape (256, 3)
        seq_length:     Sequence length
        rng:            Numpy random number generator

    Returns:
        sub_new_bases:  Target base ASCII array, shape (n_subs,), dtype=uint8
    """
    n_subs = len(sub_seq_idx)
    if n_subs == 0:
        return np.empty(0, dtype=np.uint8)

    kmer_mask = (sub_pos >= CONST.KMER_N_HEAD) & (sub_pos < seq_length - CONST.KMER_N_TAIL)
    sub_new_bases = np.empty(n_subs, dtype=np.uint8)

    if np.any(~kmer_mask):
        fb_idx = np.where(~kmer_mask)[0]
        fb_seq = sub_seq_idx[fb_idx]
        fb_pos = sub_pos[fb_idx]

        orig_bases = seq_array[fb_seq, fb_pos]
        choices    = sub_choices[orig_bases]
        cum_probs  = sub_cum_probs[orig_bases]
        rand_vals  = rng.random(len(fb_idx), dtype=np.float32)

        sel = np.zeros(len(fb_idx), dtype=np.uint8)
        sel[rand_vals >= cum_probs[:, 0]] = 1
        sel[rand_vals >= cum_probs[:, 1]] = 2
        sub_new_bases[fb_idx] = np.choose(sel, choices.T)

    if np.any(kmer_mask):
        km_idx = np.where(kmer_mask)[0]
        km_seq = sub_seq_idx[km_idx]
        km_pos = sub_pos[km_idx]

        b0 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos - 3]].astype(np.uint32)
        b1 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos - 2]].astype(np.uint32)
        b2 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos - 1]].astype(np.uint32)
        i_3mer = b0 * 16 + b1 * 4 + b2

        ref4 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos]].astype(np.uint32)

        probs = kmer_sub[i_3mer, ref4, :]
        cum = np.cumsum(probs, axis=1)
        rand_vals = rng.random(len(km_idx), dtype=np.float32)[:, np.newaxis]
        read4_idx = np.sum(rand_vals >= cum, axis=1).astype(np.uint8)
        read4_idx = np.clip(read4_idx, 0, 3)
        sub_new_bases[km_idx] = CONST._IDX_TO_ASCII[read4_idx]

    return sub_new_bases


def apply_substitutions_kmer(seq_array: np.ndarray,
                              sub_seq_idx: np.ndarray,
                              sub_pos: np.ndarray,
                              sub_new_bases: np.ndarray) -> int:
    """
    Kmer substitution application: batch write pre-sampled sub_new_bases back to seq_array, no sampling.

    Must be called after sample_substitutions_kmer completes (sampled when buffer is still noise-free)
    and when entering the application phase.

    Args:
        seq_array:      Sequence array, shape (chunk_size, seq_length), in-place modification
        sub_seq_idx:    SUB chunk-local sequence indices
        sub_pos:        SUB position array
        sub_new_bases:  Target base ASCII array returned by sample_substitutions_kmer

    Returns:
        n_subs: Number of substitutions written
    """
    n_subs = len(sub_seq_idx)
    if n_subs == 0:
        return 0
    seq_array[sub_seq_idx, sub_pos] = sub_new_bases
    return n_subs


def apply_substitutions_kmer_return_bases(seq_array: np.ndarray,
                                          sub_seq_idx: np.ndarray,
                                          sub_pos: np.ndarray,
                                          sub_new_bases: np.ndarray) -> np.ndarray:
    """
    Variant of apply_substitutions_kmer: modifies sequence in-place, returns original base array.
    sub_new_bases must be pre-sampled by sample_substitutions_kmer when buffer is noise-free.

    Returns:
        original_bases: uint8 array, shape (n_subs,), ASCII code values.
    """
    n_subs = len(sub_seq_idx)
    if n_subs == 0:
        return np.empty(0, dtype=np.uint8)
    original_bases = seq_array[sub_seq_idx, sub_pos].copy()
    seq_array[sub_seq_idx, sub_pos] = sub_new_bases
    return original_bases


def sample_insertions_kmer(seq_array: np.ndarray,
                            ins_seq_idx: np.ndarray,
                            ins_pos: np.ndarray,
                            kmer_ins: np.ndarray,
                            insertion_probs: np.ndarray,
                            seq_length: int,
                            rng) -> np.ndarray:
    """
    Kmer insertion pre-sampling: for each INS record, sample insertion bases
    based on 4mer context in the noise-free buffer, return ins_bases array without writing to buffer.

    Boundary rules (consistent with document 7.4, shared with substitutions for N_HEAD/N_TAIL):
        kmer region: KMER_N_HEAD <= pos < seq_length - KMER_N_TAIL
        other positions: fall back to single-base insertion preference (insertion_probs)

    Kmer insertion definition (document 7.2):
        Insertion position definition: insert before ref's pos-th base (pos can be 0)
        4mer takes buffer[seq_idx, pos-3 : pos+1] (4 bp total, 4th position is ref[pos])
        ref_3mer = 4mer[:3]  →  i_3mer (0..63)
        ref4     = 4mer[3]   →  ref4_idx (0..3, i.e., ref[pos])
        probs    = kmer_ins[i_3mer, ref4_idx, :]  (length-4 probability vector)
        sample ins_idx  →  _IDX_TO_ASCII[ins_idx] to get insertion base

    Args:
        seq_array:       Noise-free sequence array, shape (chunk_size, seq_length), read-only
        ins_seq_idx:     INS chunk-local sequence indices
        ins_pos:         INS position array (insert before ref[pos])
        kmer_ins:        Kmer insertion probability matrix, shape (64, 4, 4), normalized
        insertion_probs: Single-base insertion preference probabilities, shape (4,) (A/T/C/G)
        seq_length:      Sequence length
        rng:             Numpy random number generator

    Returns:
        ins_bases: Insertion base ASCII array, shape (n_ins,), dtype=uint8
    """
    n_ins = len(ins_seq_idx)
    if n_ins == 0:
        return np.empty(0, dtype=np.uint8)

    ins_bases = np.empty(n_ins, dtype=np.uint8)
    kmer_mask = (ins_pos >= CONST.KMER_N_HEAD) & (ins_pos < seq_length - CONST.KMER_N_TAIL)

    if np.any(~kmer_mask):
        fb_idx = np.where(~kmer_mask)[0]
        n_fb = len(fb_idx)
        if insertion_probs is not None:
            rand_idx = rng.choice(4, size=n_fb, p=insertion_probs).astype(np.uint8)
        else:
            rand_idx = rng.integers(0, 4, size=n_fb, dtype=np.uint8)
        ins_bases[fb_idx] = CONST._IDX_TO_ASCII[rand_idx]

    if np.any(kmer_mask):
        km_idx = np.where(kmer_mask)[0]
        km_seq = ins_seq_idx[km_idx]
        km_pos = ins_pos[km_idx]

        b0 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos - 3]].astype(np.uint32)
        b1 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos - 2]].astype(np.uint32)
        b2 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos - 1]].astype(np.uint32)
        i_3mer = b0 * 16 + b1 * 4 + b2

        ref4 = CONST._ASCII_TO_IDX[seq_array[km_seq, km_pos]].astype(np.uint32)

        probs = kmer_ins[i_3mer, ref4, :]
        cum = np.cumsum(probs, axis=1)
        rand_vals = rng.random(len(km_idx), dtype=np.float32)[:, np.newaxis]
        ins_idx = np.sum(rand_vals >= cum, axis=1).astype(np.uint8)
        ins_idx = np.clip(ins_idx, 0, 3)
        ins_bases[km_idx] = CONST._IDX_TO_ASCII[ins_idx]

    return ins_bases


def prepare_indels(indel_seq_idx: np.ndarray, indel_pos: np.ndarray, 
                   indel_types: np.ndarray, rng, insertion_probs: np.ndarray = None) -> Tuple:
    """
    Prepare insertion/deletion data (generate insertion bases).
    
    Args:
        indel_seq_idx: indel error chunk-local index array (0 to chunk_size-1, sorted)
        indel_pos: indel error position array (0 to seq_length-1, unsorted)
        indel_types: indel error type array (INS=1/DEL=2)
        rng: Random number generator
        insertion_probs: Insertion base preference probability array, shape=(4,), probabilities for ATCG
                         If None, use uniform distribution
    
    Returns:
        (indel_seq_idx, indel_pos, indel_types, indel_bases, n_ins, n_dels, n_indels)
    """
    n_indels = len(indel_seq_idx)
    
    if n_indels == 0:
        return None, None, None, None, 0, 0, 0
    
    ins_mask = (indel_types == CONST.INS)
    n_ins = np.sum(ins_mask)
    n_dels = n_indels - n_ins
    
    indel_bases = np.zeros(n_indels, dtype=np.uint8)
    n_ins_actual = np.sum(ins_mask)
    
    if n_ins_actual > 0:
        if insertion_probs is not None:
            random_bases = rng.choice(4, size=n_ins_actual, p=insertion_probs).astype(np.uint8)
        else:
            random_bases = rng.integers(0, 4, size=n_ins_actual, dtype=np.uint8)
        indel_bases[ins_mask] = random_bases
    
    return indel_seq_idx, indel_pos, indel_types, indel_bases, n_ins, n_dels, n_indels


def prepare_indels_kmer(indel_seq_idx: np.ndarray,
                         indel_pos: np.ndarray,
                         indel_types: np.ndarray,
                         ins_bases_presample: np.ndarray) -> Tuple:
    """
    Kmer INDEL packing function: receives ins_bases pre-sampled by sample_insertions_kmer,
    merges with DEL records, sorts and packs, outputs tuple for apply_indels_and_write.

    No sampling inside this function -- INS bases are determined by caller during noise-free buffer phase;
    DEL does not need bases (apply_indels_and_write only uses pos and type for DEL).

    Args:
        indel_seq_idx:       INDEL chunk-local sequence indices (INS + DEL mixed, sorted by seq↑/pos↓)
        indel_pos:           INDEL position array
        indel_types:         INDEL type array (INS=1, DEL=2)
        ins_bases_presample: Pre-sampled base ASCII array for INS records only, shape (n_ins,)

    Returns:
        (indel_seq_idx, indel_pos, indel_types, indel_bases, n_ins, n_dels, n_indels)
        Format identical to prepare_indels, downstream can directly pass to apply_indels_and_write.
    """
    n_indels = len(indel_seq_idx)
    if n_indels == 0:
        return None, None, None, None, 0, 0, 0

    ins_mask = (indel_types == CONST.INS)
    n_ins = int(np.sum(ins_mask))
    n_dels = n_indels - n_ins

    indel_bases = np.full(n_indels, 255, dtype=np.uint8)
    if n_ins > 0:
        indel_bases[ins_mask] = CONST._ASCII_TO_IDX[ins_bases_presample]

    return indel_seq_idx, indel_pos, indel_types, indel_bases, n_ins, n_dels, n_indels


def apply_indels_and_write(seq_array: np.ndarray, N: int,
                          sorted_seq_idx, sorted_pos, sorted_types, sorted_bases, n_indels: int,
                          f_out, header_prefix: bytes, newline: bytes, reads_seq_global_id_start: int,
                          del_orig_bases_collector: dict = None) -> int:
    """
    Apply insertion/deletion errors and directly write to output file (performance-optimized version).

    Performance optimizations:
    1. Pre-compute header strings to reduce repeated encoding
    2. Pre-allocate lists to reduce memory allocation
    3. Batch process indels to reduce dictionary lookups
    4. Avoid creating temporary objects in loops

    Args:
        seq_array: Pre-allocated sequence array, shape (chunk_size, seq_length)
        N: Actual number of sequences to process
        sorted_seq_idx: indel error chunk-local reads index array (0 to N-1, sorted, np.uint64)
        sorted_pos: indel error position array (0 to seq_length-1, 0-based)
        sorted_types: indel error type array (INS=1/DEL=2)
        sorted_bases: indel error insertion base array
        n_indels: Actual number of indels
        f_out: Output file object
        header_prefix: Header prefix (e.g., b'>seq_')
        newline: Newline character (e.g., b'\n')
        reads_seq_global_id_start: Global starting reads ID (Python int, arbitrary precision, for FASTA header, e.g., 347001)
        del_orig_bases_collector: Optional dictionary to collect DEL original bases.
                                  Format: {local_read_idx: [(pos, base_byte), ...]}
                                  base_byte is ASCII code (e.g., ord('A'))

    Returns:
        written_bytes: Total bytes written
    """
    written_bytes = 0

    INS_VAL = CONST.INS
    DEL_VAL = CONST.DEL

    def format_seq_id(seq_id):
        return str(seq_id).encode('ascii')
    
    if sorted_seq_idx is None or n_indels == 0:
        header_prefix_str = header_prefix
        newline_bytes = newline
        write_buffer = bytearray()
        for chunk_local_reads_idx in range(N):
            reads_seq_global_id = reads_seq_global_id_start + chunk_local_reads_idx
            seq_view = memoryview(seq_array[chunk_local_reads_idx])
            header = header_prefix_str + format_seq_id(reads_seq_global_id) + newline_bytes
            write_buffer.extend(header)
            write_buffer.extend(seq_view)
            write_buffer.extend(newline_bytes)
            written_bytes += len(header) + len(seq_view) + len(newline_bytes)
        while write_buffer and write_buffer[-1] == ord(newline_bytes):
            write_buffer.pop()
        write_buffer.append(ord(newline_bytes))
        written_bytes = len(write_buffer)
        f_out.write(write_buffer)
        return written_bytes
    
    seq_idx_view = sorted_seq_idx[:n_indels]
    unique_seq_idx, start_indices = np.unique(seq_idx_view, return_index=True)
    end_indices = np.append(start_indices[1:], n_indels)
    
    max_seq_idx = unique_seq_idx.max() if len(unique_seq_idx) > 0 else -1
    if max_seq_idx >= 0:
        indel_lookup = np.full(max_seq_idx + 1, -1, dtype=np.int32)
        for i, seq_idx in enumerate(unique_seq_idx):
            indel_lookup[seq_idx] = i
    
    header_prefix_str = header_prefix
    newline_bytes = newline
    
    estimated_seq_len = seq_array.shape[1] if len(seq_array.shape) > 1 else 0
    estimated_buffer_size = N * (len(header_prefix_str) + 20 + len(newline_bytes) +
                                 int(estimated_seq_len * 1.3) + len(newline_bytes))
    write_buffer = bytearray()
    
    for chunk_local_reads_idx in range(N):
        reads_seq_global_id = reads_seq_global_id_start + chunk_local_reads_idx
        
        seq_view = memoryview(seq_array[chunk_local_reads_idx])
        seq_bytes = bytearray(seq_view)
        
        if chunk_local_reads_idx <= max_seq_idx and indel_lookup[chunk_local_reads_idx] >= 0:
            lookup_idx = indel_lookup[chunk_local_reads_idx]
            start = start_indices[lookup_idx]
            end = end_indices[lookup_idx]
            group_size = end - start
            
            if group_size > 1:
                group_pos = sorted_pos[start:end]
                group_types = sorted_types[start:end]
                group_bases = sorted_bases[start:end]
                
                sort_indices = np.argsort(-group_pos)
                
                for i in sort_indices:
                    pos = group_pos[i]
                    err_type = group_types[i]

                    if err_type == INS_VAL:
                        base_idx = group_bases[i]
                        seq_bytes.insert(pos, CONST.BASE_ASCII[base_idx])
                    else:
                        if del_orig_bases_collector is not None:
                            orig_base = seq_bytes[pos]
                            if chunk_local_reads_idx not in del_orig_bases_collector:
                                del_orig_bases_collector[chunk_local_reads_idx] = []
                            del_orig_bases_collector[chunk_local_reads_idx].append((pos, orig_base))
                        del seq_bytes[pos]
                
                del group_pos, group_types, group_bases, sort_indices
            else:
                pos = sorted_pos[start]
                err_type = sorted_types[start]

                if err_type == INS_VAL:
                    base_idx = sorted_bases[start]
                    seq_bytes.insert(pos, CONST.BASE_ASCII[base_idx])
                else:
                    if del_orig_bases_collector is not None:
                        orig_base = seq_bytes[pos]
                        if chunk_local_reads_idx not in del_orig_bases_collector:
                            del_orig_bases_collector[chunk_local_reads_idx] = []
                        del_orig_bases_collector[chunk_local_reads_idx].append((pos, orig_base))
                    del seq_bytes[pos]
        
        header = header_prefix_str + format_seq_id(reads_seq_global_id) + newline_bytes
        write_buffer.extend(header)
        write_buffer.extend(seq_bytes)
        write_buffer.extend(newline_bytes)
        written_bytes += len(header) + len(seq_bytes) + len(newline_bytes)
        
        del seq_bytes
    
    while write_buffer and write_buffer[-1] == ord(newline_bytes):
        write_buffer.pop()
    write_buffer.append(ord(newline_bytes))
    written_bytes = len(write_buffer)
    f_out.write(write_buffer)
    
    try:
        del seq_idx_view, unique_seq_idx, start_indices, end_indices
    except NameError:
        pass
    
    try:
        del indel_lookup
    except NameError:
        pass
    
    try:
        del write_buffer
    except NameError:
        pass
    
    return written_bytes


def preallocate_output_file(output_file: Path, total_size: int):
    """
    Pre-allocate output file space (using truncate).
    
    Note: For extremely large files (exceeding filesystem limits), pre-allocation may fail.
    If it fails, a warning will be logged but execution continues, and the file will grow dynamically.
    
    Args:
        output_file: Output file path
        total_size: Total file size (bytes)
    """
    size_gb = total_size / (1024**3)
    
    try:
        with open(output_file, 'wb') as f:
            f.truncate(total_size)
    except OSError as e:
        if e.errno == 27:  # ENOSPC / File too large
            logger.warning(f"[WARNING] File space pre-allocation failed: file too large ({size_gb:.2f} GB), exceeds filesystem limit")
            logger.warning(f"   Skipping pre-allocation, file will grow dynamically (performance may degrade slightly)")
            logger.warning(f"   Error message: {e}")
            try:
                with open(output_file, 'wb') as f:
                    pass
            except Exception as e2:
                logger.error(f"x Failed to create output file: {e2}")
                raise
        else:
            logger.error(f"x File space pre-allocation failed: {e}")
            raise
    except Exception as e:
        logger.error(f"x File space pre-allocation failed: {e}")
        raise


def _calculate_chunk_offsets_from_actual_sizes(chunk_metadata_list: List[Dict],
                                               output_dir: Path) -> Tuple[Dict[int, int], int]:
    """
    Calculate chunk offsets based on actual file sizes (for zero-copy merging).
    Obtains actual file size of each chunk by scanning the filesystem, ensuring consistency with actual files.
    Called at the beginning of file merging to uniformly calculate each chunk's offset in the final file.
    """
    chunk_offsets = {}
    current_offset = 0
    sorted_chunks = sorted(chunk_metadata_list, key=lambda x: x['chunk_idx'])
    missing_files = []
    empty_chunks = []
    for chunk_meta in sorted_chunks:
        chunk_idx = chunk_meta['chunk_idx']
        chunk_file = output_dir / f'output_chunk_{chunk_idx}.fasta'
        if not chunk_file.exists():
            missing_files.append(chunk_idx)
            continue
        size = chunk_file.stat().st_size
        if size == 0:
            empty_chunks.append(chunk_idx)
            continue
        chunk_offsets[chunk_idx] = current_offset
        current_offset += size
    if missing_files:
        raise ValueError(f"{len(missing_files)} chunk files are missing: {missing_files[:20]}{'...' if len(missing_files) > 20 else ''}")
    if empty_chunks:
        raise ValueError(f"Found empty chunk files: {empty_chunks[:20]}{'...' if len(empty_chunks) > 20 else ''}")
    return chunk_offsets, current_offset


def _fallback_copy(src_fd: int, dst_fd: int, file_offset: int,
                    already_copied: int, chunk_size: int):
    """Fallback: use Python read/write to continue copying remaining data.

    Called when copy_file_range fails or returns 0.
    Parameter already_copied is the number of bytes successfully copied; the function continues from here.
    """
    buf_size = 64 * 1024 * 1024  # 64MB buffer
    total = already_copied
    while total < chunk_size:
        remaining = chunk_size - total
        to_read = min(buf_size, remaining)
        data = os.read(src_fd, to_read)
        if not data:
            break
        os.lseek(dst_fd, file_offset + total, os.SEEK_SET)
        os.write(dst_fd, data)
        total += len(data)


def _merge_chunk_zero_copy(chunk_file: Path, output_file: Path, file_offset: int,
                             dst_fd: int = None, max_extend_target=None):
    """Zero-copy merge single chunk file using copy_file_range.
    
    Deletes chunk file after successful merge.
    File is extended on demand: when workers compete to extend, use a shared counter to record the maximum extension target,
    atomically compare and only extend to the maximum value never reached before, avoiding overwriting already-written data regions.
    """
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _copy_file_range_func = _libc.copy_file_range
        _copy_file_range_func.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_uint
        ]
        _copy_file_range_func.restype = ctypes.c_ssize_t
    except Exception:
        raise RuntimeError("copy_file_range not available, cannot perform zero-copy merge")
    
    src_fd = os.open(str(chunk_file), os.O_RDONLY)
    chunk_size = os.fstat(src_fd).st_size
    should_close_dst = False
    if dst_fd is None:
        try:
            dst_fd = os.open(str(output_file), os.O_RDWR)
        except OSError:
            dst_fd = os.open(str(output_file), os.O_WRONLY | os.O_CREAT, 0o644)
        should_close_dst = True
    try:
        src_offset = ctypes.c_int64(0)
        dst_offset = ctypes.c_int64(file_offset)
        total_copied = 0
        max_copy_size = 1024 * 1024 * 1024  # 1GB per call

        required_end = file_offset + chunk_size

        if max_extend_target is not None:
            with max_extend_target.get_lock():
                if required_end > max_extend_target.value:
                    max_extend_target.value = required_end
                    os.ftruncate(dst_fd, required_end)

        while total_copied < chunk_size:
            remaining = chunk_size - total_copied
            to_copy = min(remaining, max_copy_size)

            copied = _copy_file_range_func(
                src_fd, ctypes.byref(src_offset),
                dst_fd, ctypes.byref(dst_offset),
                to_copy, 0
            )
            if copied < 0:
                errno_val = ctypes.get_errno()
                if errno_val not in (27, 28):  # EINVAL, ENOSYS
                    _fallback_copy(src_fd, dst_fd, file_offset, total_copied, chunk_size)
                    total_copied = chunk_size
                    break
                required_end = file_offset + chunk_size
                if max_extend_target is not None:
                    with max_extend_target.get_lock():
                        if required_end > max_extend_target.value:
                            max_extend_target.value = required_end
                            os.ftruncate(dst_fd, required_end)
                copied = _copy_file_range_func(
                    src_fd, ctypes.byref(src_offset),
                    dst_fd, ctypes.byref(dst_offset),
                    to_copy, 0
                )
                if copied < 0:
                    _fallback_copy(src_fd, dst_fd, file_offset, total_copied, chunk_size)
                    total_copied = chunk_size
                    break
                if copied == 0:
                    _fallback_copy(src_fd, dst_fd, file_offset, total_copied, chunk_size)
                    total_copied = chunk_size
                    break
            if copied == 0:
                _fallback_copy(src_fd, dst_fd, file_offset, total_copied, chunk_size)
                total_copied = chunk_size
                break
            total_copied += copied
            src_offset.value += copied
            dst_offset.value += copied
        if total_copied != chunk_size:
            raise IOError(f"Incomplete copy: expected {chunk_size} bytes, actually copied {total_copied} bytes")
    finally:
        os.close(src_fd)
        if should_close_dst:
            os.close(dst_fd)
            dst_fd = None
    return dst_fd


def _calculate_chunk_offsets_for_shuffled(
    output_dir: Path,
    num_chunks: int,
    chunk_file_prefix: str = "output_chunk_shuffled_",
) -> Tuple[Dict[int, int], int, List[Dict]]:
    """
    Calculate offsets based on actual sizes of shuffled chunk files (for zero-copy merging of shuffled FASTA).
    Scans {chunk_file_prefix}{idx}.fasta under output_dir, sorts by chunk_idx, then calculates offsets.
    """
    output_dir = Path(output_dir)
    chunk_offsets = {}
    current_offset = 0
    successful_chunks = []
    for s in range(num_chunks):
        chunk_file = output_dir / f"{chunk_file_prefix}{s}.fasta"
        if not chunk_file.exists() or chunk_file.stat().st_size == 0:
            logger.warning(f"Shuffled chunk {s} file is missing or empty: {chunk_file}")
            continue
        size = chunk_file.stat().st_size
        chunk_offsets[s] = current_offset
        successful_chunks.append({"chunk_idx": s})
        current_offset += size
    if not successful_chunks:
        return {}, 0, []
    return chunk_offsets, current_offset, successful_chunks


def _merge_worker(worker_id: int,
                  merge_tasks: List[Dict],
                  output_file: Path,
                  output_dir: Path,
                  progress_counter,
                  total_chunks: int,
                  progress_interval: int,
                  max_extend_target,
                  chunk_file_prefix: str = "output_chunk_"):
    """Parallel merge worker: performs zero-copy merge for each chunk in the task list.

    chunk_file_prefix distinguishes between sequential/shuffled.
    max_extend_target is a shared multiprocessing.Value('L', 0), recording the triggered maximum extension target,
    workers compete to extend via atomic comparison only extending to the maximum value never reached before,
    avoiding overwriting already-written data regions.
    """
    dst_fd = os.open(str(output_file), os.O_RDWR)
    try:
        merged_count = 0
        local_progress = 0
        last_logged = -1
        for task in merge_tasks:
            chunk_idx = task['chunk_idx']
            file_offset = task['file_offset']
            chunk_file = output_dir / f"{chunk_file_prefix}{chunk_idx}.fasta"
            try:
                _merge_chunk_zero_copy(chunk_file, output_file, file_offset, dst_fd, max_extend_target)
                os.fsync(dst_fd)
                merged_count += 1
                local_progress += 1
                if local_progress >= progress_interval or merged_count == len(merge_tasks):
                    with progress_counter.get_lock():
                        progress_counter.value += local_progress
                        current_total = progress_counter.value
                        should_log = (current_total % progress_interval == 0 or current_total == total_chunks) and current_total != last_logged
                        if should_log:
                            last_logged = current_total
                    if should_log:
                        pct = (current_total / total_chunks) * 100 if total_chunks else 0
                    local_progress = 0
            except Exception as e:
                logger.error(f"MergeWorker{worker_id}: Failed to merge Chunk {chunk_idx}: {e}")
                raise
        if local_progress > 0:
            with progress_counter.get_lock():
                progress_counter.value += local_progress
    finally:
        os.fsync(dst_fd)
        os.close(dst_fd)
