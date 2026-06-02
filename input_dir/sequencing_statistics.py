#!/usr/bin/env python3
"""
Sequencing Statistics: Unified BAM analysis script
Combines: per_reference_read_depth, position_error_rates, error_bias, error_bias_kmer
Single BAM read to collect all statistics, outputs: npz, tsv
"""

import argparse
import os
import re
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pysam
import scipy.stats as stats
from scipy.special import gamma as gamma_func

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Supported Distributions ====================
SUPPORTED_DISTRIBUTIONS = ['lognormal', 'gamma', 'normal', 'weibull', 'exponential', 'poisson', 'uniform', 'nbinom', 'beta']


# ==================== Distribution Parameter Validation ====================

def _validate_distribution_params(dist_name: str, params: dict, data: np.ndarray) -> bool:
    """
    Validate whether distribution parameters are valid.

    Args:
        dist_name: Distribution name
        params: Distribution parameters
        data: Data array

    Returns:
        bool: Whether parameters are valid
    """
    import numpy as np
    from scipy.special import gamma as gamma_func

    if params is None:
        return False

    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min if data_max > data_min else 1

    try:
        if dist_name == 'lognormal':
            mu, sigma = params.get('mu'), params.get('sigma')
            if sigma is None or sigma <= 0 or sigma > 100:
                return False
            return True

        elif dist_name == 'gamma':
            k, theta = params.get('k'), params.get('theta')
            if k is None or theta is None or k <= 0 or theta <= 0:
                return False
            if k > 1000 or theta > 1e10:
                return False
            return True

        elif dist_name == 'normal':
            mu, sigma = params.get('mu'), params.get('sigma')
            if sigma is None or sigma <= 0 or sigma > 1e10:
                return False
            return True

        elif dist_name == 'weibull':
            k, lambda_val = params.get('k'), params.get('lambda')
            if k is None or lambda_val is None or k <= 0 or lambda_val <= 0:
                return False
            if k > 100 or lambda_val > 1e10:
                return False
            return True

        elif dist_name == 'exponential':
            lambda_val = params.get('lambda')
            if lambda_val is None or lambda_val <= 0 or lambda_val > 1e10:
                return False
            return True

        elif dist_name == 'poisson':
            # Poisson is not suitable for continuous data or proportion
            return False

        elif dist_name == 'uniform':
            a, b = params.get('a'), params.get('b')
            if a is None or b is None or b <= a:
                return False
            # Check if range is related to data range
            return True

        elif dist_name == 'nbinom':
            r_bio, mu = params.get('r'), params.get('mu')
            if r_bio is None or mu is None or r_bio <= 0 or mu <= 0:
                return False
            # Not suitable for proportion
            return False

        elif dist_name == 'beta':
            alpha, beta_val, scale = params.get('alpha'), params.get('beta'), params.get('scale')
            if alpha is None or beta_val is None or scale is None:
                return False
            if alpha <= 0 or beta_val <= 0 or scale <= 0:
                return False
            # Beta distribution parameters may be valid on proportion data
            return True

        return False
    except Exception:
        return False


def _get_distribution_stats(dist_name: str, params: dict) -> tuple:
    """
    Calculate theoretical mean and CV from distribution parameters.

    Args:
        dist_name: Distribution name
        params: Distribution parameters

    Returns:
        tuple: (mean, cv)
    """
    from scipy.special import gamma as gamma_func

    try:
        if dist_name == 'lognormal':
            mu, sigma = params.get('mu'), params.get('sigma')
            mean = np.exp(mu + sigma**2 / 2)
            var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
            cv = np.sqrt(var) / mean
            return mean, cv

        elif dist_name == 'gamma':
            k, theta = params.get('k'), params.get('theta')
            mean = k * theta
            var = k * theta**2
            cv = np.sqrt(var) / mean
            return mean, cv

        elif dist_name == 'normal':
            mu, sigma = params.get('mu'), params.get('sigma')
            cv = sigma / mu if mu != 0 else 0
            return mu, cv

        elif dist_name == 'weibull':
            k, lambda_val = params.get('k'), params.get('lambda')
            mean = lambda_val * gamma_func(1 + 1/k)
            var = lambda_val**2 * (gamma_func(1 + 2/k) - gamma_func(1 + 1/k)**2)
            cv = np.sqrt(var) / mean
            return mean, cv

        elif dist_name == 'exponential':
            lambda_val = params.get('lambda')
            mean = 1.0 / lambda_val
            cv = 1.0  # Exponential distribution CV is fixed at 1
            return mean, cv

        elif dist_name == 'poisson':
            lam = params.get('lambda')
            cv = 1.0 / np.sqrt(lam) if lam > 0 else 0
            return lam, cv

        elif dist_name == 'uniform':
            a, b = params.get('a'), params.get('b')
            mean = (a + b) / 2
            var = (b - a)**2 / 12
            cv = np.sqrt(var) / mean if mean != 0 else 0
            return mean, cv

        elif dist_name == 'nbinom':
            r_bio, mu = params.get('r'), params.get('mu')
            mean = mu
            var = mu + mu**2 / r_bio
            cv = np.sqrt(var) / mean
            return mean, cv

        elif dist_name == 'beta':
            alpha, beta_val, scale = params.get('alpha'), params.get('beta'), params.get('scale')
            mean = scale * alpha / (alpha + beta_val)
            var = scale**2 * alpha * beta_val / ((alpha + beta_val)**2 * (alpha + beta_val + 1))
            cv = np.sqrt(var) / mean if mean != 0 else 0
            return mean, cv

        return 0, 0
    except Exception:
        return 0, 0


def _fit_distribution_for_aic(dist_name: str, data: np.ndarray) -> tuple:
    """
    Fit distribution and return (log_likelihood, num_params).
    Returns (np.nan, 0) if fitting fails.
    """
    data = data[data > 0]
    if len(data) < 10:
        return np.nan, 0

    try:
        if dist_name == 'lognormal':
            mu, sigma = stats.lognorm.fit(data, floc=0)
            loglik = np.sum(stats.lognorm.logpdf(data, s=sigma, scale=np.exp(mu)))
            return loglik, 2

        elif dist_name == 'gamma':
            a, loc, b = stats.gamma.fit(data, floc=0)
            loglik = np.sum(stats.gamma.logpdf(data, a, loc=loc, scale=b))
            return loglik, 2

        elif dist_name == 'normal':
            mu, sigma = stats.norm.fit(data)
            loglik = np.sum(stats.norm.logpdf(data, mu, sigma))
            return loglik, 2

        elif dist_name == 'weibull':
            c, loc, scale = stats.weibull_min.fit(data, floc=0)
            loglik = np.sum(stats.weibull_min.logpdf(data, c, loc=loc, scale=scale))
            return loglik, 2

        elif dist_name == 'exponential':
            loc, scale = stats.expon.fit(data, floc=0)
            loglik = np.sum(stats.expon.logpdf(data, loc=loc, scale=scale))
            return loglik, 1

        elif dist_name == 'nbinom':
            n, p = stats.nbinom.fit(data, floc=0)
            loglik = np.sum(stats.nbinom.logpmf(data, n, p))
            return loglik, 2

        elif dist_name == 'beta':
            a, b_param, loc, scale = stats.beta.fit(data, floc=0)
            loglik = np.sum(stats.beta.logpdf(data, a, b_param, loc=loc, scale=scale))
            return loglik, 2

        elif dist_name == 'poisson':
            lam = np.mean(data)
            loglik = np.sum(stats.poisson.logpmf(data.astype(int), lam))
            return loglik, 1

        elif dist_name == 'uniform':
            a, b_param = data.min(), data.max()
            loglik = np.sum(stats.uniform.logpdf(data, loc=a, scale=b_param - a))
            return loglik, 2

        return np.nan, 0
    except Exception:
        return np.nan, 0


def _select_best_distribution(data: np.ndarray) -> str:
    """
    Select the best distribution based on AIC.
    AIC = 2 * k - 2 * log_likelihood
    Lower is better.
    """
    aic_scores = {}
    for dist_name in SUPPORTED_DISTRIBUTIONS:
        loglik, k = _fit_distribution_for_aic(dist_name, data)
        if np.isnan(loglik):
            continue
        aic = 2 * k - 2 * loglik
        aic_scores[dist_name] = aic

    if not aic_scores:
        return 'Normal'

    return min(aic_scores, key=aic_scores.get)


# ==================== Distribution Parameter Calculation ====================

def calculate_distribution_params(dist_name: str, mean_depth: float, cv: float) -> dict:
    """
    Calculate distribution parameters from distribution type.

    Args:
        dist_name: Distribution type
        mean_depth: Mean depth
        cv: Coefficient of variation

    Returns:
        dict: {'params': {...}, 'sample_func': callable}
    """
    dist_name = dist_name.lower()

    if dist_name == 'lognormal':
        sigma_sq = np.log(1 + cv**2)
        sigma = np.sqrt(sigma_sq)
        mu = np.log(mean_depth) - sigma_sq / 2
        return {
            'params': {'mu': mu, 'sigma': sigma},
            'sample_func': lambda rng, size: rng.lognormal(mean=mu, sigma=sigma, size=size)
        }

    elif dist_name == 'gamma':
        k = 1.0 / (cv ** 2)
        theta = mean_depth * (cv ** 2)
        return {
            'params': {'k': k, 'theta': theta},
            'sample_func': lambda rng, size: rng.gamma(shape=k, scale=theta, size=size)
        }

    elif dist_name == 'normal':
        sigma = mean_depth * cv
        return {
            'params': {'mu': mean_depth, 'sigma': sigma},
            'sample_func': lambda rng, size: rng.normal(loc=mean_depth, scale=sigma, size=size)
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
            'params': {'k': k, 'lambda': lambda_val},
            'sample_func': lambda rng, size: rng.weibull(k, size=size) * lambda_val
        }

    elif dist_name == 'exponential':
        return {
            'params': {'lambda': 1.0 / mean_depth},
            'sample_func': lambda rng, size: rng.exponential(scale=mean_depth, size=size)
        }

    elif dist_name == 'poisson':
        return {
            'params': {'lambda': mean_depth},
            'sample_func': lambda rng, size: rng.poisson(lam=mean_depth, size=size)
        }

    elif dist_name == 'uniform':
        range_val = np.sqrt(12) * mean_depth * cv
        a = mean_depth - range_val / 2
        b = mean_depth + range_val / 2
        return {
            'params': {'a': a, 'b': b},
            'sample_func': lambda rng, size: rng.uniform(low=a, high=b, size=size)
        }

    elif dist_name == 'nbinom':
        # Biostatistics parameterization: Var = mu + mu^2/r
        # Convert to scipy (n, p) parameterization: n = r, p = r/(r+mu)
        min_cv_sq = 1.0 / mean_depth
        if cv**2 < min_cv_sq:
            cv = np.sqrt(min_cv_sq * 1.1)
        r_bio = 1.0 / (cv**2 - 1.0/mean_depth)  # Dispersion parameter (biostatistics convention)
        p_scipy = r_bio / (r_bio + mean_depth)   # scipy parameterization
        return {
            'params': {'r': r_bio, 'mu': mean_depth},
            'sample_func': lambda rng, size, n=r_bio, p=p_scipy: rng.negative_binomial(n=n, p=p, size=size)
        }

    elif dist_name == 'beta':
        # Normalize to [0,1] then scale
        alpha = 2.0
        beta = 2.0
        scale = mean_depth
        return {
            'params': {'alpha': alpha, 'beta': beta, 'scale': scale},
            'sample_func': lambda rng, size: rng.beta(alpha, beta, size=size) * scale
        }

    else:
        raise ValueError(f"Unsupported distribution type: {dist_name}")

# ==================== Constants ====================
BASES = ['A', 'T', 'C', 'G']
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

# Base encoding: A=0, T=1, C=2, G=3
base2idx = np.zeros(256, dtype=np.int8)
base2idx[ord('A')] = 0
base2idx[ord('T')] = 1
base2idx[ord('C')] = 2
base2idx[ord('G')] = 3


# ==================== Helper Functions ====================

def kmer3_to_idx(s: str) -> int:
    """3-mer string -> 0..63"""
    if len(s) != 3 or any(c not in BASE_TO_IDX for c in s):
        return -1
    return BASE_TO_IDX[s[0]] * 16 + BASE_TO_IDX[s[1]] * 4 + BASE_TO_IDX[s[2]]


def build_position_map(cigar_tuples: List[Tuple], ref_start: int) -> Dict[int, int]:
    """CIGAR -> ref_pos -> read_pos mapping"""
    pos_map = {}
    ref_pos = ref_start
    read_pos = 0
    for op, length in cigar_tuples:
        if op == 0:  # M
            for _ in range(length):
                pos_map[ref_pos] = read_pos
                ref_pos += 1
                read_pos += 1
        elif op == 1:  # I
            read_pos += length
        elif op == 2:  # D
            ref_pos += length
        elif op == 4:  # S
            read_pos += length
    return pos_map


def parse_md_tag(md_tag: str, read_seq: str, ref_start: int,
                 cigar_tuples: List[Tuple]) -> List[Dict]:
    """Parse MD tag, return substitution and deletion errors"""
    if not md_tag:
        return []
    errors = []
    md_string = str(md_tag)
    pos_map = build_position_map(cigar_tuples, ref_start)
    ref_pos_in_md = ref_start
    i = 0
    while i < len(md_string):
        m = re.match(r'(\d+)', md_string[i:])
        if m:
            ref_pos_in_md += int(m.group(1))
            i += len(m.group(0))
            continue
        if md_string[i] in 'ATCGN':
            actual_ref_pos = ref_pos_in_md
            read_pos_in_read = pos_map.get(actual_ref_pos)
            read_base = read_seq[read_pos_in_read] if (read_pos_in_read is not None and read_pos_in_read < len(read_seq)) else None
            errors.append({
                'type': 'sub',
                'ref_pos': actual_ref_pos,
                'ref_base': md_string[i],
                'read_base': read_base,
                'read_pos': read_pos_in_read,
            })
            ref_pos_in_md += 1
            i += 1
            continue
        if md_string[i] == '^':
            i += 1
            deleted = ''
            while i < len(md_string) and md_string[i] in 'ATCGN':
                deleted += md_string[i]
                i += 1
            for j, base in enumerate(deleted):
                errors.append({
                    'type': 'del',
                    'ref_pos': ref_pos_in_md + j,
                    'ref_base': base,
                    'read_base': None,
                    'read_pos': None,
                })
            ref_pos_in_md += len(deleted)
    return errors


# ==================== Main Statistics Class ====================

class SequencingStatistics:
    """Unified statistics collector from BAM file"""

    def __init__(self, ref_fasta: str):
        self.ref_fasta = ref_fasta

        # Load reference sequences
        self.ref_dict = {}
        self.ref_lengths = {}
        self._load_references()

        # Determine sequence length
        self.seq_length = max(self.ref_lengths.values()) if self.ref_lengths else 0

        # ========== 1. Read Depth Statistics ==========
        self.ref_read_counts = defaultdict(int)  # ref_name -> count
        self.total_reads = 0
        self.mapped_reads = 0

        # ========== 2. Position Error Rates ==========
        self.coverage = np.zeros(self.seq_length, dtype=np.int64)
        self.substitution_count = np.zeros(self.seq_length, dtype=np.int64)
        self.insertion_count = np.zeros(self.seq_length, dtype=np.int64)
        self.deletion_count = np.zeros(self.seq_length, dtype=np.int64)

        # ========== 3. Single-base Error Matrix ==========
        self.sub_mat = np.zeros((4, 4), dtype=np.int64)  # reference base -> read base
        self.ins_mat = np.zeros(4, dtype=np.int64)      # inserted base
        self.del_mat = np.zeros(4, dtype=np.int64)       # deleted reference base

        # ========== 4. K-mer Error Matrix (5-mer context, 3-mer key = 64 combinations) ==========
        # Use fixed-shape (64, 4, 4) numpy arrays instead of three-layer nested defaultdict
        # Dimensions: [kmer_idx, ref_base_at_pos4, read/ins/del_base]
        self.kmer_sub_counts = np.zeros((64, 4, 4), dtype=np.int64)
        self.kmer_ins_counts = np.zeros((64, 4, 4), dtype=np.int64)
        self.kmer_del_counts = np.zeros((64, 4, 4), dtype=np.int64)

        # ========== Other statistics ==========
        self.total_bases_all = 0       # Total bases in sequencing file
        self.total_bases_mapped = 0    # Mapped bases
        self.md_empty_count = 0

    def _load_references(self):
        """Load reference sequences from FASTA"""
        if not self.ref_fasta:
            return
        ref_fasta_file = pysam.FastaFile(self.ref_fasta)
        for ref_name in ref_fasta_file.references:
            ref_seq = ref_fasta_file.fetch(ref_name).upper()
            self.ref_dict[ref_name] = ref_seq
            self.ref_lengths[ref_name] = len(ref_seq)
        ref_fasta_file.close()

    def process_read(self, read):
        """Process a single aligned read, update all statistics"""
        if read.is_unmapped:
            return

        self.mapped_reads += 1

        # Get reference info
        if read.reference_id < 0:
            return
        ref_name = self._bam.get_reference_name(read.reference_id)
        if ref_name not in self.ref_dict:
            return

        ref_seq = self.ref_dict[ref_name]
        read_seq = read.query_sequence
        if read_seq is None:
            return

        self.total_bases_mapped += len(read_seq)

        # ========== 1. Read Depth ==========
        self.ref_read_counts[ref_name] += 1

        # ========== Get alignment info ==========
        ref_start = read.reference_start
        cigar_tuples = read.cigartuples or []
        pos_map = build_position_map(cigar_tuples, ref_start)
        md_tag = read.get_tag('MD') if read.has_tag('MD') else None

        # ========== 2. Position Error Rates & 3. Single-base Matrix ==========
        # Coverage and substitution
        if not md_tag:
            self.md_empty_count += 1
            for ref_pos, read_pos in pos_map.items():
                if 0 <= ref_pos < self.seq_length:
                    self.coverage[ref_pos] += 1
        else:
            md_errors = parse_md_tag(md_tag, read_seq, ref_start, cigar_tuples)

            # Coverage
            for ref_pos, read_pos in pos_map.items():
                if 0 <= ref_pos < self.seq_length:
                    self.coverage[ref_pos] += 1

            # Substitution errors
            for err in md_errors:
                if err['type'] != 'sub':
                    continue
                ref_pos = err['ref_pos']
                if ref_pos is None or ref_pos < 0 or ref_pos >= self.seq_length:
                    continue
                self.substitution_count[ref_pos] += 1

                # Single-base substitution matrix
                ref_base = err['ref_base']
                read_base = err['read_base']
                if ref_base and read_base and ref_base != read_base:
                    ref_ord = ord(ref_base)
                    read_ord = ord(read_base)
                    if ref_ord < 256 and read_ord < 256:
                        self.sub_mat[base2idx[ref_ord], base2idx[read_ord]] += 1

        # Insertion and deletion from CIGAR
        if cigar_tuples:
            ref_pos = read.reference_start
            read_pos = 0

            for op, length in cigar_tuples:
                if op == 0:  # M = Match
                    ref_pos += length
                    read_pos += length
                elif op == 1:  # I = Insertion
                    # Insertion: between reference position 3 and position 4
                    ins_ref_pos = ref_pos
                    if 0 <= ins_ref_pos < self.seq_length:
                        self.insertion_count[ins_ref_pos] += length

                    # Single-base insertion matrix
                    for j in range(length):
                        if read_pos + j < len(read_seq):
                            ins_base = read_seq[read_pos + j]
                            if ins_base in "ATCG":
                                self.ins_mat[base2idx[ord(ins_base)]] += 1

                    read_pos += length
                elif op == 2:  # D = Deletion
                    # Deletion: reference positions 4 to 4+length-1
                    for j in range(length):
                        del_pos = ref_pos + j
                        if 0 <= del_pos < self.seq_length:
                            self.deletion_count[del_pos] += 1

                            # Single-base deletion matrix
                            if del_pos < len(ref_seq):
                                del_base = ref_seq[del_pos]
                                if del_base in "ATCG":
                                    self.del_mat[base2idx[ord(del_base)]] += 1

                    ref_pos += length
                elif op == 4:  # S = Soft clip
                    read_pos += length

        # ========== 4. K-mer Error Matrix ==========
        self._process_kmer_errors(ref_seq, read_seq, ref_start, cigar_tuples, md_tag)

    def _process_kmer_errors(self, ref_seq: str, read_seq: str, ref_start: int,
                            cigar_tuples: List[Tuple], md_tag: Optional[str]):
        """Process k-mer context errors (5-mer context)"""
        # Parse MD tag for substitutions and deletions
        md_errors = parse_md_tag(md_tag, read_seq, ref_start, cigar_tuples)
        
        # Parse CIGAR for insertions
        ins_errors = self._parse_cigar_insertions(cigar_tuples, ref_start, read_seq)
        
        # Build position map
        pos_map = build_position_map(cigar_tuples, ref_start)
        
        # ========== Substitution ==========
        for err in md_errors:
            if err['type'] != 'sub':
                continue
            ref_pos = err['ref_pos']
            read_pos_in_read = err.get('read_pos')
            read_base = err.get('read_base')
            if read_base is None or read_pos_in_read is None:
                continue
            
            # Get 3bp context before error position (5-mer positions 1-3: [pos-3, pos-2, pos-1])
            if ref_pos < 3 or ref_pos >= len(ref_seq) - 1:
                continue
            
            ref_5mer = ref_seq[ref_pos - 3:ref_pos + 2]
            if len(ref_5mer) != 5:
                continue
            
            ref_3mer = ref_5mer[:3]
            if any(c not in BASE_TO_IDX for c in ref_3mer):
                continue
            
            kmer_idx = kmer3_to_idx(ref_3mer)
            ref_4th = ref_seq[ref_pos]
            read_4th = read_seq[read_pos_in_read]
            
            if ref_4th not in BASE_TO_IDX or read_4th not in BASE_TO_IDX:
                continue
            
            if ref_4th == read_4th:
                continue
            
            ref4_idx = BASE_TO_IDX[ref_4th]
            read4_idx = BASE_TO_IDX[read_4th]
            self.kmer_sub_counts[kmer_idx, ref4_idx, read4_idx] += 1
        
        # ========== Deletion ==========
        for err in md_errors:
            if err['type'] != 'del':
                continue
            ref_pos = err['ref_pos']
            
            # Get 4-mer context BEFORE deletion (5-mer positions 1-4: [pos-4, pos-3, pos-2, pos-1])
            # Deletion happens at ref_pos, so context is the base BEFORE deletion
            if ref_pos < 4 or ref_pos >= len(ref_seq):
                continue
            
            ref_4mer = ref_seq[ref_pos - 4:ref_pos]
            if len(ref_4mer) != 4:
                continue
            
            ref_3mer = ref_4mer[:3]
            if any(c not in BASE_TO_IDX for c in ref_3mer):
                continue
            
            kmer_idx = kmer3_to_idx(ref_3mer)
            ref_4th = ref_4mer[3]  # Base BEFORE the deletion
            del_base = err['ref_base']  # Deleted base from MD tag
            
            if ref_4th not in BASE_TO_IDX or del_base not in BASE_TO_IDX:
                continue
            
            ref4_idx = BASE_TO_IDX[ref_4th]
            del_idx = BASE_TO_IDX[del_base]
            # deletion: (kmer_idx, ref4_idx, del_idx)
            self.kmer_del_counts[kmer_idx, ref4_idx, del_idx] += 1
        
        # ========== Insertion ==========
        for err in ins_errors:
            if err['type'] != 'ins':
                continue
            ref_pos_after_ins = err['ref_pos']
            # ref_pos_4th is the position AFTER the insertion, so the base BEFORE insertion is ref_pos_4th - 1
            ref_pos_4th = ref_pos_after_ins + 1
            read_pos_in_read = err.get('read_pos')
            
            # Get 4-mer context BEFORE insertion (positions ref_pos_4th-4 to ref_pos_4th-1)
            if ref_pos_4th < 4 or ref_pos_4th > len(ref_seq):
                continue
            
            ref_4mer = ref_seq[ref_pos_4th - 4:ref_pos_4th]
            if len(ref_4mer) != 4:
                continue
            
            ref_3mer = ref_4mer[:3]
            if any(c not in BASE_TO_IDX for c in ref_3mer):
                continue
            
            kmer_idx = kmer3_to_idx(ref_3mer)
            ref_4th = ref_4mer[3]  # Base BEFORE the insertion
            
            if ref_4th not in BASE_TO_IDX:
                continue
            
            if read_pos_in_read is None or read_pos_in_read >= len(read_seq):
                continue
            
            inserted_base = read_seq[read_pos_in_read]
            if inserted_base not in BASE_TO_IDX:
                continue
            
            ref4_idx = BASE_TO_IDX[ref_4th]
            ins_idx = BASE_TO_IDX[inserted_base]  # inserted base index
            # insertion: (kmer_idx, ref4_idx, ins_idx)
            self.kmer_ins_counts[kmer_idx, ref4_idx, ins_idx] += 1

    def _parse_cigar_insertions(self, cigar_tuples: List[Tuple], ref_start: int,
                                read_seq: str) -> List[Dict]:
        """Parse CIGAR to extract insertions"""
        insertions = []
        ref_pos = ref_start
        read_pos = 0
        for op, length in cigar_tuples:
            if op == 0:  # M
                ref_pos += length
                read_pos += length
            elif op == 1:  # I = Insertion
                insertion_ref_pos = ref_pos - 1  # Insertion happens between ref_pos-1 and ref_pos
                if read_pos < len(read_seq):
                    for j in range(length):
                        insertions.append({
                            'type': 'ins',
                            'ref_pos': insertion_ref_pos,
                            'read_base': read_seq[read_pos + j] if read_pos + j < len(read_seq) else None,
                            'read_pos': read_pos + j,
                        })
                read_pos += length
            elif op == 2:  # D
                ref_pos += length
            elif op == 4:  # S
                read_pos += length
        return insertions

            # Check for deletion
            # (handled via MD tag)

    def process_bam(self, bam_file: str):
        """Process entire BAM file, collect all statistics in single pass"""
        self._bam = pysam.AlignmentFile(bam_file, "rb")

        for i, read in enumerate(self._bam):
            self.total_reads += 1

            qlen = read.query_length if read.query_length else (len(read.query_sequence) if read.query_sequence else 0)
            self.total_bases_all += qlen

            self.process_read(read)

            if (i + 1) % 1000000 == 0:
                pass  # progress logging removed

        self._bam.close()

    def calculate_results(self):
        """Calculate all derived statistics"""
        results = {}

        # ========== 1. Read Depth Distribution ==========
        read_counts_array = np.array(list(self.ref_read_counts.values()), dtype=np.float64)
        total_ref_count = len(self.ref_lengths) if self.ref_lengths else None
        num_with_reads = len(read_counts_array)

        # Zero reads count
        if total_ref_count and total_ref_count > 0:
            zero_reads_count = total_ref_count - num_with_reads
            dropout_rate = 100.0 * zero_reads_count / total_ref_count
        else:
            zero_reads_count = 0
            dropout_rate = 0.0

        # Coverage stats
        avg_coverage = np.mean(read_counts_array) if len(read_counts_array) > 0 else 0
        cv = np.std(read_counts_array) / avg_coverage if avg_coverage > 0 else 0

        # Calculate parameters for all supported distributions
        all_dist_params = {}
        for dist_name in SUPPORTED_DISTRIBUTIONS:
            try:
                dist_info = calculate_distribution_params(dist_name, avg_coverage, cv)
                all_dist_params[dist_name] = dist_info['params']
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name} distribution: {e}")
                all_dist_params[dist_name] = None

        # Select best distribution by AIC
        best_dist = _select_best_distribution(read_counts_array)

        results['read_depth'] = {
            'ref_names': list(self.ref_read_counts.keys()),
            'read_counts': read_counts_array,
            'total_ref_count': total_ref_count,
            'num_with_reads': num_with_reads,
            'zero_reads_count': zero_reads_count,
            'dropout_rate': dropout_rate,
            'avg_coverage': avg_coverage,
            'min_coverage': np.min(read_counts_array) if len(read_counts_array) > 0 else 0,
            'max_coverage': np.max(read_counts_array) if len(read_counts_array) > 0 else 0,
            'cv': cv,
            'all_dist_params': all_dist_params,
            'best_distribution': best_dist,
        }

        # ========== 2. Position Error Rates ==========
        coverage_safe = np.maximum(self.coverage, 1)

        substitution_rate = self.substitution_count.astype(np.float64) / coverage_safe
        insertion_rate = self.insertion_count.astype(np.float64) / coverage_safe
        deletion_rate = self.deletion_count.astype(np.float64) / coverage_safe
        total_error_rate = substitution_rate + insertion_rate + deletion_rate

        # Zero coverage positions
        no_coverage = (self.coverage == 0) & (self.deletion_count == 0)
        substitution_rate[no_coverage] = 0.0
        insertion_rate[no_coverage] = 0.0
        deletion_rate[no_coverage] = 0.0
        total_error_rate[no_coverage] = 0.0

        results['position_error'] = {
            'total_error_rate': total_error_rate,
            'substitution_rate': substitution_rate,
            'insertion_rate': insertion_rate,
            'deletion_rate': deletion_rate,
            'coverage': self.coverage,
            'substitution_count': self.substitution_count,
            'insertion_count': self.insertion_count,
            'deletion_count': self.deletion_count,
        }

        # ========== 3. Single-base Error Matrix ==========
        results['error_bias'] = {
            'substitution': self.sub_mat,
            'insertion': self.ins_mat,
            'deletion': self.del_mat,
        }

        # ========== Summary Statistics ==========
        sub_total = np.sum(self.sub_mat)
        ins_total = np.sum(self.ins_mat)
        del_total = np.sum(self.del_mat)
        total_errors = sub_total + ins_total + del_total
        total_bases = np.sum(self.coverage)

        results['summary'] = {
            'total_reads': self.total_reads,
            'mapped_reads': self.mapped_reads,
            'mapping_rate': 100.0 * self.mapped_reads / self.total_reads if self.total_reads > 0 else 0,
            'total_bases_all': self.total_bases_all,
            'total_bases_mapped': self.total_bases_mapped,
            'seq_length': self.seq_length,
            'sub_total': sub_total,
            'ins_total': ins_total,
            'del_total': del_total,
            'total_errors': total_errors,
            'total_bases': total_bases,
            'overall_error_rate': total_errors / total_bases if total_bases > 0 else 0,
            'sub_error_rate': sub_total / total_bases if total_bases > 0 else 0,
            'ins_error_rate': ins_total / total_bases if total_bases > 0 else 0,
            'del_error_rate': del_total / total_bases if total_bases > 0 else 0,
        }

        # ========== 4. K-mer Error Matrix ==========
        results['error_bias_kmer'] = self._convert_kmer_to_arrays()

        return results

    def _convert_kmer_to_arrays(self):
        """
        Calculate normalized probability distributions from numpy arrays.

        Data was written directly into (64, 4, 4) numpy arrays during collection; normalize here.
        """
        # Substitution probabilities: normalize along ref_base axis (axis=1)
        sub_counts = self.kmer_sub_counts.astype(np.float64)
        sub_probs = np.zeros((64, 4, 4), dtype=np.float64)
        for k in range(64):
            row_sums = sub_counts[k].sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            sub_probs[k] = sub_counts[k] / row_sums

        # Insertion probabilities: normalize along ref_base axis (axis=1)
        ins_counts = self.kmer_ins_counts.astype(np.float64)
        ins_probs = np.zeros((64, 4, 4), dtype=np.float64)
        for k in range(64):
            row_sums = ins_counts[k].sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            ins_probs[k] = ins_counts[k] / row_sums

        # Deletion probabilities: normalize along ref_base axis (axis=1)
        del_counts = self.kmer_del_counts.astype(np.float64)
        del_probs = np.zeros((64, 4, 4), dtype=np.float64)
        for k in range(64):
            for ref4_idx in range(4):
                col_sum = del_counts[k, :, ref4_idx].sum()
                if col_sum > 1e-10:
                    del_probs[k, :, ref4_idx] = del_counts[k, :, ref4_idx] / col_sum
                else:
                    del_probs[k, :, ref4_idx] = 0.25  # Uniform if no data

        return {
            'substitution_counts': sub_counts,
            'substitution_probs': sub_probs,
            'insertion_counts': ins_counts,
            'insertion_probs': ins_probs,
            'deletion_counts': del_counts,
            'deletion_probs': del_probs,
        }


# ==================== Output Functions ====================

def save_outputs(results: Dict, output_dir: str, sample_name: str, bam_path: str = None, ref_path: str = None):
    """Save all output files: npz, tsv"""
    if not output_dir:
        output_dir = '.'

    os.makedirs(output_dir, exist_ok=True)

    # ========== 1. Read Depth Distribution ==========
    rd = results['read_depth']

    # NPZ
    npz_path = f"{output_dir}/read_coverage_depth_{sample_name}.npz"
    np.savez_compressed(npz_path,
                        reference=np.array(rd['ref_names'], dtype=object),
                        read_depth=rd['read_counts'].astype(np.int64),
                        avg_coverage=rd['avg_coverage'],
                        cv=rd['cv'],
                        dropout_rate=rd['dropout_rate'])

    # TSV - distribution parameters for all distributions
    tsv_dist_path = f"{output_dir}/read_coverage_depth_distribution_{sample_name}.tsv"
    with open(tsv_dist_path, "w", encoding="utf-8") as f:
        f.write("distribution\tparameter\tvalue\n")
        f.write(f"stats\tavg_coverage\t{rd['avg_coverage']:.6f}\n")
        f.write(f"stats\tcv\t{rd['cv']:.6f}\n")
        f.write(f"stats\tdropout_rate\t{rd['dropout_rate']:.6f}\n")
        f.write(f"stats\tmin_coverage\t{rd['min_coverage']:.6f}\n")
        f.write(f"stats\tmax_coverage\t{rd['max_coverage']:.6f}\n")
        for dist_name in SUPPORTED_DISTRIBUTIONS:
            params = rd['all_dist_params'].get(dist_name)
            if params:
                dist_mean, dist_cv = _get_distribution_stats(dist_name, params)
                f.write(f"{dist_name}\tmean\t{dist_mean:.6f}\n")
                f.write(f"{dist_name}\tcv\t{dist_cv:.6f}\n")
                for param_name, param_value in params.items():
                    f.write(f"{dist_name}\t{param_name}\t{param_value:.6f}\n")

    # TSV - count
    tsv_count_path = f"{output_dir}/read_coverage_depth_count_{sample_name}.tsv"
    with open(tsv_count_path, "w", encoding="utf-8") as f:
        f.write("reference\tread_depth\n")
        for r, c in zip(rd['ref_names'], rd['read_counts']):
            f.write(f"{r}\t{int(c)}\n")

    # ========== 2. Position Error Rates ==========
    pe = results['position_error']

    # NPZ
    npz_path = f"{output_dir}/per_position_error_rates_{sample_name}.npz"
    np.savez_compressed(npz_path,
                        total_error_rate=pe['total_error_rate'],
                        substitution_rate=pe['substitution_rate'],
                        insertion_rate=pe['insertion_rate'],
                        deletion_rate=pe['deletion_rate'],
                        coverage=pe['coverage'],
                        substitution_count=pe['substitution_count'].astype(np.int64),
                        insertion_count=pe['insertion_count'].astype(np.int64),
                        deletion_count=pe['deletion_count'].astype(np.int64))

    # TSV
    tsv_path = f"{output_dir}/per_position_error_rates_{sample_name}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("# Error rates are raw fractional values (multiply by 1000 to get 10^-3 nt^-1 units)\n")
        f.write("position\ttotal_error_rate\tsubstitution_rate\tinsertion_rate\tdeletion_rate\tcoverage\tsubstitution_count\tinsertion_count\tdeletion_count\n")
        for i in range(len(pe['total_error_rate'])):
            f.write(f"{i}\t{pe['total_error_rate'][i]}\t{pe['substitution_rate'][i]}\t{pe['insertion_rate'][i]}\t{pe['deletion_rate'][i]}\t{int(pe['coverage'][i])}\t{int(pe['substitution_count'][i])}\t{int(pe['insertion_count'][i])}\t{int(pe['deletion_count'][i])}\n")

    # ========== 3. Single-base Error Matrix ==========
    em = results['error_bias']

    # NPZ
    npz_path = f"{output_dir}/error_bias_{sample_name}.npz"
    np.savez_compressed(npz_path,
                        substitution=em['substitution'],
                        insertion=em['insertion'],
                        deletion=em['deletion'])

    # TSV
    tsv_path = f"{output_dir}/error_bias_{sample_name}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("## Substitution\n")
        f.write("\t" + "\t".join(BASES) + "\n")
        for ri, ref_base in enumerate(BASES):
            row = "\t".join(str(em['substitution'][ri, ci]) for ci in range(4))
            f.write(f"{ref_base}\t{row}\n")
        f.write("\n## Insertion\n")
        f.write("\tCount\n")
        for i, base in enumerate(BASES):
            f.write(f"{base}\t{int(em['insertion'][i])}\n")
        f.write("\n## Deletion\n")
        f.write("\tCount\n")
        for i, base in enumerate(BASES):
            f.write(f"{base}\t{int(em['deletion'][i])}\n")

    # ========== 4. K-mer Context Error Matrix ==========
    km = results['error_bias_kmer']

    # NPZ
    npz_kmer_path = f"{output_dir}/error_bias_kmer_{sample_name}.npz"
    np.savez_compressed(npz_kmer_path,
                        substitution=km['substitution_probs'],
                        insertion=km['insertion_probs'],
                        deletion=km['deletion_probs'],
                        substitution_counts=km['substitution_counts'],
                        insertion_counts=km['insertion_counts'],
                        deletion_counts=km['deletion_counts'])

    # TSV - human readable (4x1 format for ins/del)
    tsv_kmer_path = f"{output_dir}/error_bias_kmer_{sample_name}.tsv"
    _save_kmer_tsv(km, tsv_kmer_path)

    # TSV - summary (4x16 format)
    tsv_summary_path = f"{output_dir}/error_bias_kmer_summary_{sample_name}.tsv"
    _save_kmer_summary_tsv(km, tsv_summary_path)


def _save_kmer_tsv(km: Dict, output_path: str):
    """Save k-mer error matrix to TSV file (4x1 format for ins/del)"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# K-mer Error Bias (5-mer context)\n\n")

        # Substitution matrix (4x4) for each 3bp
        for kmer_idx in range(64):
            c0 = kmer_idx // 16
            c1 = (kmer_idx // 4) % 4
            c2 = kmer_idx % 4
            kmer3 = BASES[c0] + BASES[c1] + BASES[c2]

            f.write(f"## 3bp: {kmer3}\n")

            # Substitution (4x4)
            sub_data = km['substitution_counts'][kmer_idx]  # (4, 4)
            f.write("### Substitution\n")
            f.write("\t" + "\t".join(BASES) + "\n")
            for i, base in enumerate(BASES):
                row = "\t".join([f"{int(sub_data[i, j])}" for j in range(4)])
                f.write(f"{base}\t{row}\n")

            # Insertion (4x1) - sum over ins_idx
            ins_data = km['insertion_counts'][kmer_idx].sum(axis=1)  # (4,)
            f.write("### Insertion\n")
            f.write("\tCount\n")
            for i, base in enumerate(BASES):
                f.write(f"{base}\t{int(ins_data[i])}\n")

            # Deletion (4x1) - sum over del_idx
            del_data = km['deletion_counts'][kmer_idx].sum(axis=1)  # (4,)
            f.write("### Deletion\n")
            f.write("\tCount\n")
            for i, base in enumerate(BASES):
                f.write(f"{base}\t{int(del_data[i])}\n")

            f.write("\n")


def _save_kmer_summary_tsv(km: Dict, output_path: str):
    """Save k-mer error matrix summary to TSV file (4x16 format)"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# K-mer Error Bias Summary (aggregated over all 5-mers)\n\n")

        # Substitution summary (4x16) - each row repeated 4 times
        sub_counts = km['substitution_counts'].sum(axis=0)  # (4, 4)
        sub_cols = [f"{BASES[r]}→{BASES[c]}" for r in range(4) for c in range(4)]
        f.write("## Substitution Summary\n")
        f.write("\t" + "\t".join(sub_cols) + "\n")
        for r, base in enumerate(BASES):
            row = "\t".join([f"{int(sub_counts[r, c])}" for c in range(4) for _ in range(4)])
            f.write(f"{base}\t{row}\n")

        # Insertion summary (4x16)
        ins_counts = km['insertion_counts'].sum(axis=0)  # (4, 4)
        ins_cols = [f"{BASES[r]}→{BASES[r]}{BASES[i]}" for r in range(4) for i in range(4)]
        f.write("\n## Insertion Summary\n")
        f.write("\t" + "\t".join(ins_cols) + "\n")
        for r, base in enumerate(BASES):
            row = "\t".join([f"{int(ins_counts[r, i])}" for i in range(4) for _ in range(4)])
            f.write(f"{base}\t{row}\n")

        # Deletion summary (4x16)
        del_counts = km['deletion_counts'].sum(axis=0)  # (4, 4)
        del_cols = [f"{BASES[d]}{BASES[r]}→{BASES[r]}" for r in range(4) for d in range(4)]
        f.write("\n## Deletion Summary\n")
        f.write("\t" + "\t".join(del_cols) + "\n")
        for r, base in enumerate(BASES):
            row = "\t".join([f"{int(del_counts[r, d])}" for d in range(4) for _ in range(4)])
            f.write(f"{base}\t{row}\n")


# ==================== Summary Table Output ====================

def save_summary_table(results: Dict, output_dir: str, sample_name: str,
                       synthesis_method: str = "", sequencing_platform: str = ""):
    """
    Save a summary statistics table in plain text format.
    Includes AIC-based best distribution selection.
    """
    if not output_dir:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)

    rd = results['read_depth']
    sm = results['summary']

    total_ref = rd['total_ref_count'] or 0
    num_with_reads = rd['num_with_reads'] or 0
    dropout_rate = (rd['dropout_rate'] or 0.0) / 100.0
    avg_depth = rd['avg_coverage'] or 0.0
    min_cov = int(rd['min_coverage']) if 'min_coverage' in rd else 0
    max_cov = int(rd['max_coverage']) if 'max_coverage' in rd else 0
    best_dist = rd.get('best_distribution', 'Normal')
    best_params = rd['all_dist_params'].get(best_dist)
    if best_params:
        theoretical_mean, theoretical_cv = _get_distribution_stats(best_dist, best_params)
        if theoretical_cv == 0:
            theoretical_cv = rd['cv'] or 0.0
    else:
        theoretical_cv = rd['cv'] or 0.0
    cv = theoretical_cv

    # Error rates as percentages (x 100 of the fractional values)
    sub_rate = sm.get('sub_error_rate', 0.0) * 100
    ins_rate = sm.get('ins_error_rate', 0.0) * 100
    del_rate = sm.get('del_error_rate', 0.0) * 100

    # Header
    header = (
        "SampleID\tSynthesisMethod\tSequencingPlatform\t"
        "NumRefSequences\tNumSeqReads\tRefSeqLength(bp)\t"
        "DropoutRate\tAvgReadDepth\tCV\tReadDepthRange\t"
        "BestDistribution\tSubRate(%)\tInsRate(%)\tDelRate(%)\n"
    )

    row = (
        f"{sample_name}\t{synthesis_method}\t{sequencing_platform}\t"
        f"{total_ref}\t{sm.get('total_reads', 0)}\t{sm.get('seq_length', 0)}\t"
        f"{dropout_rate:.6f}\t{avg_depth:.6f}\t{cv:.6f}\t"
        f"[{min_cov}, {max_cov}]\t"
        f"{best_dist}\t{sub_rate:.6f}\t{ins_rate:.6f}\t{del_rate:.6f}\n"
    )

    output_path = f"{output_dir}/summary_table_{sample_name}.txt"
    # Append if exists, otherwise create
    if os.path.exists(output_path):
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(row)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(row)

    logger.info(f"Summary table saved to {output_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Unified sequencing statistics from BAM file")
    parser.add_argument("--bam", required=True, help="Input BAM file path")
    parser.add_argument("--ref", required=True, help="Reference FASTA file path")
    parser.add_argument("--name", default="sample", help="Sample name (used in output filenames)")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--synthesis", default="", help="Synthesis method (optional, for summary table)")
    parser.add_argument("--platform", default="", help="Sequencing platform (optional, for summary table)")
    args = parser.parse_args()

    output_dir = args.output

    stats = SequencingStatistics(args.ref)
    stats.process_bam(args.bam)

    results = stats.calculate_results()

    save_outputs(results, output_dir, args.name, args.bam, args.ref)
    save_summary_table(results, output_dir, args.name,
                       synthesis_method=args.synthesis, sequencing_platform=args.platform)


if __name__ == "__main__":
    main()
