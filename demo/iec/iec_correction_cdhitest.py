"""
IEC Correction - CD-HIT Clustering Version (Indel Error Detection + Majority Voting) - Parallel Version

Prerequisite:
  - The input fasta file MUST be sorted by read_id in ascending order.
    Pre-sort using: seqkit sort -s input.fasta -o input_sorted.fasta
                    sort -k1,1 -V input.fasta > input_sorted.fasta

Usage:
  python iec_voting_cdhit_parallel.py -f reads.fa -c cluster_result.clstr -l 120 -o output.txt
  python iec_voting_cdhit_parallel.py -f reads.fa -c cluster_result.clstr -l 120 -o output.txt -t 4 --batch-clusters 50

Optional arguments:
  -s shift          - Maximum window offset (default: 1)
  -ws windowsize    - Window size (default: 5)
  -cl commonlen     - Common sequence length (default: 5)
  -ct commonth      - Common sequence threshold (default: 3)
  -t threads        - Number of parallel processes (default: os.cpu_count())
  --batch-clusters N - Number of clusters per batch (default: 50)
"""

import argparse
import mmap
import os
import struct
import sys
import shutil
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

# Hamming distance threshold for triggering indel detection.
# The global constant has been removed; use the --hamming-threshold CLI argument instead.

# Per-process private index (loaded once via MmapIndex in worker_init).
_worker_index = None
_worker_fasta_file = None


# ============================================================
# Utility Functions
# ============================================================

map_list = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
rev_map_list = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}


def Num2Base(Num):
    return [rev_map_list[n] for n in Num]


def hammingDist(seq1, seq2):
    return sum(s1 != s2 for s1, s2 in zip(seq1, seq2))


def getCommonSubseq(seq1, seq2, corr_len):
    lseq1 = len(seq1)
    lseq2 = len(seq2)
    record = [[0 for _ in range(lseq2 + 1)] for _ in range(lseq1 + 1)]
    sub_seq_len = 0
    ind = 0
    for i in range(lseq1):
        for j in range(lseq2):
            if seq1[i] == seq2[j]:
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > sub_seq_len:
                    sub_seq_len = record[i + 1][j + 1]
                    ind = i + 1
                    if sub_seq_len >= corr_len:
                        return seq1[ind - sub_seq_len:ind]
    return seq1[ind - sub_seq_len:ind]


# ============================================================
# Stage 1: Build .bidx Binary Index (fasta must be sorted by read_id)
# ============================================================

def _natural_key(s):
    """Split string into numeric/non-numeric segments for natural sort comparison.
    Example: "read_10" -> ("read_", 10), "read_2" -> ("read_", 2).
    This ensures 10 is correctly recognized as greater than 2.
    """
    import re
    parts = re.split(r'(\d+)', s)
    return tuple((int(p) if p.isdigit() else p) for p in parts)


def _check_fasta_sorted(fasta_file, sample_size=1000):
    """
    Check whether the fasta file is sorted by read_id in ascending order (natural sort).

    Returns:
        True  - fasta is sorted by read_id in ascending order
        False - fasta is not sorted by read_id (or read failed)
    """
    print(f'  Checking fasta sorted by read_id (sampling first {sample_size} reads, natural sort)...')
    prev_id = None
    prev_key = None
    with open(fasta_file, 'rb') as f:
        for i, line in enumerate(f):
            if line.startswith(b'>'):
                read_id = line[1:].strip().split()[0].decode('ascii')
                current_key = _natural_key(read_id)
                if prev_key is not None and current_key < prev_key:
                    print(f'  ERROR: read_id "{prev_id}" is followed by "{read_id}" (should be ascending by natural sort)')
                    return False
                prev_id = read_id
                prev_key = current_key
                if i >= sample_size * 2:
                    break
    print(f'  fasta is sorted by read_id in ascending order (natural sort) OK')
    return True


def _build_binary_index(fasta_file, index_file, batch_size=1000000):
    """
    Build .bidx binary index for MmapIndex.

    Binary format (fixed-length, 56 bytes per entry):
      [0:40]  read_id (ASCII, right-space-padded)
      [40:48] seq_start (uint64 LE)
      [48:56] seq_end   (uint64 LE)

    Prerequisite: fasta must be sorted by read_id in ascending order.
    Strategy: single-pass traversal, write in batches of batch_size (no sorting, append-only).
    Peak memory: batch_size x ~72 bytes (Python tuple) ≈ ~72 MB, released immediately after write.
    """
    import struct

    # Check sorting first
    if not _check_fasta_sorted(fasta_file):
        print('=' * 70)
        print('ERROR: fasta file is not sorted by read_id in ascending order (natural sort)!')
        print()
        print('Index building requires fasta sorted by read_id (numeric part by value). Sort with:')
        print()
        print('  Method 1: seqkit sort -s input.fasta -o input_sorted.fasta')
        print('  Method 2: sort -k1,1 -V input.fasta > input_sorted.fasta')
        print()
        print('Note: Natural sort ensures read_2 < read_10 (lexicographic would give read_10 < read_2).')
        print('=' * 70)
        raise SystemExit(1)

    print(f'Building .bidx index (batched write, {batch_size} reads per batch): {index_file}')
    count = 0

    with open(index_file, 'wb') as fout:
        with open(fasta_file, 'rb') as f:
            pos = 0
            current_id = None
            seq_start = None
            batch = []

            for line in f:
                line_len = len(line)
                if line.startswith(b'>'):
                    if current_id is not None:
                        batch.append((current_id, seq_start, pos))
                        count += 1
                        if len(batch) >= batch_size:
                            _write_binary_batch_to_file(fout, batch)
                            if count % (batch_size * 10) == 0:
                                print(f'  Written {count} entries...', flush=True)
                            batch = []
                    current_id = line[1:].strip().split()[0].decode('ascii')
                    seq_start = pos + line_len
                pos += line_len

            if current_id is not None:
                batch.append((current_id, seq_start, pos))
                count += 1

            if batch:
                _write_binary_batch_to_file(fout, batch)

    fsize = os.path.getsize(index_file)
    print(f'Index built: {count} reads, file {fsize/1e6:.1f} MB')


def _write_binary_batch_to_file(fout, batch):
    """Write a sorted batch to an open file object; batch is released after write."""
    import struct
    for read_id, seq_start, seq_end in batch:
        read_id_bytes = read_id.encode('ascii')[:40].ljust(40)
        fout.write(read_id_bytes + struct.pack('<QQ', seq_start, seq_end))


class MmapIndex:
    """
    Read-only mmap index, replacing dict[str, (int, int)].
    Each worker process creates its own mmap object (fork COW shares the same
    physical pages, no longer N copies of Python dict).

    Note: read_id is stored as a 40-byte right-space-padded ASCII string.
    Binary search uses natural sort comparison (numeric parts compared by value, e.g. read_2 < read_10).
    """

    ENTRY_SIZE = 56   # 40 (id) + 8 (start) + 8 (end)
    ID_SIZE = 40

    def __init__(self, path):
        fd = os.open(path, os.O_RDONLY)
        self._fd = fd
        self._file_size = os.path.getsize(path)
        self._num_entries = self._file_size // self.ENTRY_SIZE
        self._mmap_obj = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

    def _read_id_at(self, idx):
        """Read read_id at the specified index (bytes, right-space-padded 40 bytes).
        Explicitly convert to bytes to avoid mmap slice vs bytes comparison issues.
        """
        id_off = idx * self.ENTRY_SIZE
        return bytes(self._mmap_obj[id_off:id_off + self.ID_SIZE])

    def _cmp_natural(self, key_bytes, mem_bytes):
        """Natural sort comparison: strip both sides, then compare by numeric parts.
        key_bytes: bytes (right-space-padded to 40 bytes)
        mem_bytes: bytes (right-space-padded to 40 bytes)
        Returns: -1 / 0 / 1
        """
        mem_str = mem_bytes.rstrip(b' ').decode('ascii')
        key_str = key_bytes.decode('ascii').rstrip()
        nk_key = _natural_key(key_str)
        nk_mem = _natural_key(mem_str)
        if nk_key < nk_mem:
            return -1
        elif nk_key > nk_mem:
            return 1
        else:
            return 0

    def get(self, read_id):
        """Binary search, returns (start, end) or None. O(log N), using natural sort comparison."""
        key_bytes = read_id.encode('ascii')[:self.ID_SIZE].ljust(self.ID_SIZE)
        lo, hi = 0, self._num_entries - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mem_id = self._read_id_at(mid)
            cmp = self._cmp_natural(key_bytes, mem_id)
            if cmp < 0:
                hi = mid - 1
            elif cmp > 0:
                lo = mid + 1
            else:
                # Match found: use os.pread to read positions (avoid mmap slice issues).
                pos_off = mid * self.ENTRY_SIZE + self.ID_SIZE
                pos_bytes = bytes(self._mmap_obj[pos_off:pos_off + 16])
                if len(pos_bytes) < 16:
                    return None
                start, end = struct.unpack('<QQ', pos_bytes)
                return start, end
        return None

    def close(self):
        self._mmap_obj.close()
        os.close(self._fd)


def fetch_reads_by_ids(fasta_file, index, read_ids):
    """
    Batch-fetch sequences by read_ids (on-demand loading, not full load).
    index is an MmapIndex instance (mmap read-only, no dict).
    """
    reads = {}

    with open(fasta_file, 'rb') as f:
        for read_id in read_ids:
            pos = index.get(read_id)
            if pos is None:
                continue
            seq_start, seq_end = pos
            if seq_end <= seq_start:
                continue

            f.seek(seq_start)
            raw = f.read(seq_end - seq_start)
            reads[read_id] = raw.rstrip(b'\r\n').decode('ascii')

    return reads


# ============================================================
# Stage 2 (Parallel): clstr Parsing Utilities
# ============================================================

def count_clusters(clstr_file):
    """Quick scan of clstr to count total number of clusters."""
    count = 0
    with open(clstr_file, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                count += 1
    return count


def parse_clstr_range(clstr_file, start_idx, end_idx):
    """
    Parse clusters in clstr file within the range [start_idx, end_idx).

    Returns: list[(cluster_id, [read_id, ...])].
    Only returns clusters within range; out-of-range clusters are skipped immediately (no memory accumulation).

    Note: end_idx points to the "first cluster not to process" (left-closed, right-open interval).
    For example, parse_clstr_range(clstr, 5, 8) returns data for clusters 5, 6, 7.
    """
    clusters = []
    current_cluster_id = None
    current_cluster_reads = []
    cluster_counter = -1

    with open(clstr_file, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith('>Cluster'):
                if current_cluster_id is not None and current_cluster_reads:
                    if start_idx <= cluster_counter < end_idx:
                        clusters.append((current_cluster_id, current_cluster_reads))

                parts = line_stripped.split()
                if len(parts) >= 2:
                    current_cluster_id = parts[1]
                else:
                    current_cluster_id = str(cluster_counter + 1)
                current_cluster_reads = []
                cluster_counter += 1

                if cluster_counter >= end_idx:
                    break

            elif line_stripped:
                parts = line_stripped.split('\t')
                if len(parts) >= 2:
                    seq_info = parts[1]
                    gt_pos = seq_info.find('>')
                    if gt_pos != -1:
                        read_id_full = seq_info[gt_pos + 1:]
                        read_id = read_id_full.split()[0].split('...')[0]
                        current_cluster_reads.append(read_id)

        if current_cluster_id is not None and current_cluster_reads:
            if start_idx <= cluster_counter < end_idx:
                clusters.append((current_cluster_id, current_cluster_reads))

    return clusters


def clstr_range_generator(clstr_file, cluster_batch_size):
    """
    Single-pass traversal of clstr, yield (start_idx, end_idx) for each batch.
    Does not contain any read_ids; main process only records range info.

    Each yield item: only two integers (start_idx, end_idx), ~50 bytes.
    90M reads / 55 reads/cluster / 50 clusters/batch ≈ 33K batches × 50 B ≈ 1.6 MB.
    """
    batch_start = None
    cluster_counter = -1

    with open(clstr_file, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith('>Cluster'):
                if batch_start is None:
                    batch_start = cluster_counter + 1

                cluster_counter += 1

                if cluster_counter - batch_start + 1 >= cluster_batch_size:
                    yield (batch_start, cluster_counter + 1)
                    batch_start = cluster_counter + 1

            elif line_stripped:
                pass

    if batch_start is not None and batch_start <= cluster_counter:
        yield (batch_start, cluster_counter + 1)


# ============================================================
# Indel Correction Core Algorithm
# ============================================================

def slicingHammingDistBasedIndelDetection(seq_indel, seq_repre, l_seq, shift, window_size, common_len, common_threshold, z_threshold):
    """Sliding-window-based Indel error detection."""
    while common_len >= common_threshold:
        res = getCommonSubseq(seq_indel, seq_repre, common_len)
        a_ind = seq_indel.find(res)
        b_ind = seq_repre.find(res)
        if seq_indel.count(res) == 1 and seq_repre.count(res) == 1 and abs(a_ind - b_ind) <= shift:
            break
        else:
            common_len -= 1

    if len(res) < common_threshold or common_len >= common_threshold * 3:
        return 'Z' * l_seq

    a = list(seq_indel[a_ind:] + seq_indel[:a_ind])
    b = list(seq_repre[b_ind:] + seq_repre[:b_ind])
    k = common_len - 1

    while k < min(len(a), l_seq):
        if a[k] == b[k]:
            k += 1
            continue

        hammingDist_list_repre = []
        hammingDist_list_indel = []

        for d in range(shift + 1):
            start_ind = k
            end_ind = min(window_size + k, len(a), l_seq)

            if end_ind + d <= l_seq:
                seq_indel_int = str(a[start_ind:end_ind])
                seq_repre_int = str(b[(start_ind + d):(end_ind + d)])
                hammingDist_list_repre.append(hammingDist(seq_indel_int, seq_repre_int))

            if end_ind + d <= len(a):
                seq_indel_int = str(a[(start_ind + d):(end_ind + d)])
                seq_repre_int = str(b[start_ind:end_ind])
                hammingDist_list_indel.append(hammingDist(seq_indel_int, seq_repre_int))

        if hammingDist_list_repre and hammingDist_list_indel:
            if (hammingDist_list_repre.index(min(hammingDist_list_repre)) ==
                hammingDist_list_indel.index(min(hammingDist_list_indel)) == 0):
                k += 1
                continue

            if min(hammingDist_list_repre) < min(hammingDist_list_indel):
                a.insert(k, 'Z')
            elif min(hammingDist_list_repre) > min(hammingDist_list_indel):
                del a[k]
            else:
                k += 1
        else:
            k += 1

    if len(a) < l_seq:
        a.extend(['Z'] * (l_seq - len(a)))
    if len(a) > l_seq:
        a = a[:l_seq]

    if len(a) != l_seq or a.count('Z') >= z_threshold:
        a = ['Z'] * l_seq

    if b_ind != 0:
        seq_noindel = a[l_seq - b_ind:] + a[:l_seq - b_ind]
    else:
        seq_noindel = a

    return ''.join(seq_noindel)


def majorityVotingForIndelDetection(seq_list, l_seq, z_threshold):
    """Majority voting on correct-length sequences to generate a representative sequence."""
    voting_counter = [[0] * 4 for _ in range(l_seq)]

    for seq_str in seq_list:
        if len(seq_str) != l_seq:
            continue
        for k in range(l_seq):
            if seq_str[k] == 'A':
                voting_counter[k][0] += 1
            elif seq_str[k] == 'T':
                voting_counter[k][1] += 1
            elif seq_str[k] == 'G':
                voting_counter[k][2] += 1
            elif seq_str[k] == 'C':
                voting_counter[k][3] += 1

    voting_max_result = np.argmax(voting_counter, 1)
    voting_str_list = Num2Base(voting_max_result)

    for k in range(l_seq):
        temp = sorted(voting_counter[k])
        if temp[-1] < z_threshold and temp[-1] == temp[-2] == temp[-3] == temp[-4]:
            voting_str_list[k] = 'Z'

    voting_str = ''.join(voting_str_list)
    return voting_str


def clusterIndelDetectionAndVoting(cluster_seqs, l_seq, shift, window_size, common_len, common_threshold,
                                   hamming_threshold, z_threshold):
    """
    Perform Indel error detection and majority voting correction on a cluster.

    Returns:
        corrected_seqs: list[str], the list of corrected sequences
        cluster_size: number of sequences in the cluster
    """
    right_len_seqs = [s for s in cluster_seqs if len(s) == l_seq]
    wrong_len_seqs = [s for s in cluster_seqs if len(s) != l_seq]

    if not right_len_seqs:
        return cluster_seqs, len(cluster_seqs)

    seq_repre = majorityVotingForIndelDetection(right_len_seqs, l_seq, z_threshold)

    corrected_right_seqs = []
    for seq in right_len_seqs:
        if hammingDist(seq, seq_repre) >= hamming_threshold:
            seq_noindel = slicingHammingDistBasedIndelDetection(
                seq, seq_repre, l_seq, shift, window_size, common_len, common_threshold, z_threshold)
            if hammingDist(seq_noindel, seq_repre) < z_threshold:
                corrected_right_seqs.append(seq_noindel)
            else:
                corrected_right_seqs.append(seq)
        else:
            corrected_right_seqs.append(seq)

    corrected_wrong_seqs = []
    for seq in wrong_len_seqs:
        seq_noindel = slicingHammingDistBasedIndelDetection(
            seq, seq_repre, l_seq, shift, window_size, common_len, common_threshold, z_threshold)
        if hammingDist(seq_noindel, seq_repre) < z_threshold:
            corrected_wrong_seqs.append(seq_noindel)

    all_corrected = corrected_right_seqs + corrected_wrong_seqs

    if not all_corrected:
        return cluster_seqs, len(cluster_seqs)

    voting_counter = [[0] * 4 for _ in range(l_seq)]
    for seq_str in all_corrected:
        for k in range(min(len(seq_str), l_seq)):
            if seq_str[k] == 'A':
                voting_counter[k][0] += 1
            elif seq_str[k] == 'T':
                voting_counter[k][1] += 1
            elif seq_str[k] == 'G':
                voting_counter[k][2] += 1
            elif seq_str[k] == 'C':
                voting_counter[k][3] += 1

    voting_max_result = np.argmax(voting_counter, 1)
    voting_str = ''.join(Num2Base(voting_max_result))

    return [voting_str], len(cluster_seqs)


# ============================================================
# Parallel Worker Functions (Memory-Safe Version)
# ============================================================

def worker_init(fasta_file, index_file):
    """
    Called once when each worker process starts: map .bidx mmap index into process address space.
    MmapIndex only creates mmap objects (fork COW shares physical pages), no dict copy.
    """
    global _worker_index, _worker_fasta_file
    _worker_fasta_file = fasta_file
    _worker_index = MmapIndex(index_file)


def worker_lifecycle(args):
    """
    Worker main loop: process a single task batch (a range of clusters).

    args = (batch_idx, start_idx, end_idx, clstr_file, fasta_file, index_file,
            temp_file, l_seq, shift, window_size, common_len, common_threshold,
            hamming_threshold, z_threshold)

    - Use parse_clstr_range to parse clusters in the given range (read_ids only, no sequences).
    - For each cluster: read reads -> Indel detection + voting -> write to separate temp file.
    - Release all temporary objects immediately after processing.
    Returns (batch_idx, cluster_count, temp_file).
    """
    (batch_idx, start_idx, end_idx, clstr_file, fasta_file, index_file,
     temp_file, l_seq, shift, window_size, common_len, common_threshold,
     hamming_threshold, z_threshold) = args

    # worker_init already created mmap via MmapIndex(index_file) (fork COW shares physical pages).
    index = _worker_index

    clusters = parse_clstr_range(clstr_file, start_idx, end_idx)
    if not clusters:
        return (batch_idx, 0, temp_file)

    result_lines = []

    for cluster_id, read_ids in clusters:
        reads = fetch_reads_by_ids(fasta_file, index, read_ids)
        if not reads:
            del reads
            continue

        cluster_seqs = []
        for read_id in read_ids:
            if read_id in reads:
                cluster_seqs.append(reads[read_id])
            else:
                for key in reads.keys():
                    if key.startswith(read_id) or read_id.startswith(key.split()[0]):
                        cluster_seqs.append(reads[key])
                        break

        if not cluster_seqs:
            del reads, cluster_seqs
            continue

        corrected_seqs, _ = clusterIndelDetectionAndVoting(
            cluster_seqs, l_seq, shift, window_size, common_len, common_threshold,
            hamming_threshold, z_threshold)

        for seq in corrected_seqs:
            if len(seq) == l_seq:
                result_lines.append(seq + '\n')

        del reads, cluster_seqs, corrected_seqs

    if result_lines:
        with open(temp_file, 'w') as out_f:
            out_f.writelines(result_lines)

    return (batch_idx, len(clusters), temp_file)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='IEC Correction - CD-HIT Clustering Version (Indel Detection + Majority Voting) - Parallel')
    parser.add_argument('--fasta', '-f', required=True, help='Input fasta file')
    parser.add_argument('--cluster', '-c', required=True, help='CD-HIT clustering result .clstr file')
    parser.add_argument('--length', '-l', required=True, type=int, help='DNA sequence target length')
    parser.add_argument('--output', '-o', required=True, help='Output txt file')

    parser.add_argument('--shift', '-s', default=1, type=int, help='Maximum window offset (default: 1)')
    parser.add_argument('--windowsize', '-ws', default=5, type=int, help='Window size (default: 5)')
    parser.add_argument('--commonlen', '-cl', default=5, type=int, help='Common sequence length (default: 5)')
    parser.add_argument('--commonth', '-ct', default=3, type=int, help='Common sequence threshold (default: 3)')
    parser.add_argument('--hamming-threshold', default=5, type=int,
                        help='Hamming distance threshold to trigger indel detection (default: 5)')
    parser.add_argument('--z-threshold', default=5, type=int,
                        help='Z tolerance threshold: indel detection failure / tie / post-correction verification (default: 5)')

    parser.add_argument('--threads', '-t', default=None, type=int,
                        help='Number of parallel processes (default: os.cpu_count())')
    parser.add_argument('--batch-clusters', default=50, type=int,
                        help='Number of clusters per batch (default: 50)')

    args = parser.parse_args()

    l_seq = args.length
    shift = args.shift
    window_size = args.windowsize
    common_len = args.commonlen
    common_threshold = args.commonth
    hamming_threshold = args.hamming_threshold
    z_threshold = args.z_threshold
    n_workers = args.threads or os.cpu_count() or 4
    cluster_batch_size = args.batch_clusters

    index_file = args.fasta + '.bidx'

    # ===================== Stage 1: Build .bidx Binary Index =====================
    if os.path.exists(index_file):
        os.remove(index_file)
    _build_binary_index(args.fasta, index_file, batch_size=1000000)

    # ===================== Stage 2: Parallel Processing =====================
    print(f'Reading CD-HIT cluster file: {args.cluster}')
    total_clusters = count_clusters(args.cluster)
    print(f'Total clusters: {total_clusters}')

    # Clear output file
    with open(args.output, 'w') as f:
        pass

    # Create temp directory (same dir as output)
    temp_dir = args.output + '.tmp.d'
    os.makedirs(temp_dir, exist_ok=True)

    # Single-pass scan of clstr to generate range batches (only (start_idx, end_idx), no read_ids)
    print(f'Scanning clstr, generating range batches ({cluster_batch_size} clusters per batch)...')
    all_ranges = list(clstr_range_generator(args.cluster, cluster_batch_size))
    num_batches = len(all_ranges)
    print(f'Generated {num_batches} batches, ~{cluster_batch_size} clusters per batch')
    print(f'Main process all_ranges memory footprint: {num_batches * sys.getsizeof((0, 1)) / 1e6:.1f} MB')

    # Build complete task args (including batch_idx and temp file path)
    def make_task(batch_idx, range_pair):
        (start_idx, end_idx) = range_pair
        temp_file = os.path.join(temp_dir, f'batch_{batch_idx:05d}.txt')
        return (batch_idx, start_idx, end_idx, args.cluster, args.fasta, index_file,
                temp_file, l_seq, shift, window_size, common_len, common_threshold,
                hamming_threshold, z_threshold)

    task_args_list = [make_task(i, r) for i, r in enumerate(all_ranges)]

    print(f'Starting {n_workers} parallel workers')
    print(f'  - Each worker mmap-reads .bidx (fork COW shares physical pages, no N dict copies)')
    print(f'  - Each task passes only (start_idx, end_idx); worker parses clstr independently')
    print(f'  - Reads fetched on-demand; each batch writes to its own temp file, no file contention')

    try:
        with Pool(processes=n_workers, initializer=worker_init,
                  initargs=(args.fasta, index_file)) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_lifecycle, task_args_list, chunksize=1),
                total=num_batches,
                desc='Processing clusters',
                miniters=max(1, num_batches // 100),
            ))

        total_clusters_processed = sum(r[1] for r in results if r is not None)
        print(f'Done. Processed {total_clusters_processed}/{total_clusters} clusters')

        # ==================== Zero-Copy Merge ====================
        print(f'Zero-copy merging {num_batches} temp files to final output...')
        # Sort by batch_idx (imap_unordered return order is not guaranteed)
        results.sort(key=lambda x: x[0])
        out_fd = os.open(args.output, os.O_WRONLY | os.O_TRUNC)
        total_bytes = 0
        for batch_idx, cluster_count, temp_file in results:
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                with open(temp_file, 'rb') as fin:
                    fin_fd = fin.fileno()
                    file_size = os.fstat(fin_fd).st_size
                    copied = os.sendfile(out_fd, fin_fd, 0, file_size)
                    total_bytes += copied
        os.close(out_fd)
        if total_bytes >= 1e6:
            print(f'Merge complete. Wrote {total_bytes / 1e9:.2f} GB ({total_bytes / 1e6:.2f} MB)')
        elif total_bytes >= 1e3:
            print(f'Merge complete. Wrote {total_bytes / 1e3:.2f} KB ({total_bytes} bytes)')
        else:
            print(f'Merge complete. Wrote {total_bytes} bytes')
    finally:
        # Cleanup temp files and directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Always delete .bidx to avoid using stale index on next run
        if os.path.exists(index_file):
            os.remove(index_file)
            print(f'Deleted index file: {index_file}')


if __name__ == '__main__':
    main()
