"""Configuration module."""
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
from .coverage import calculate_depth_distribution_params

logger = logging.getLogger(__name__)


class SynthesisConfig:
    INKJET = "inkjet"
    ELECTROCHEMICAL = "electro"
    PHOTO = "photo"


def get_synthesis_method_display_name(method: str) -> str:
    display_names = {
        SynthesisConfig.INKJET: "Inkjet-based DNA synthesis",
        SynthesisConfig.ELECTROCHEMICAL: "Electrochemical DNA synthesis",
        SynthesisConfig.PHOTO: "Photochemical DNA synthesis"
    }
    return display_names.get(method, method)


def get_synthesis_method_short_name(method: str) -> str:
    if method == SynthesisConfig.INKJET:
        return "inkjet"
    elif method == SynthesisConfig.ELECTROCHEMICAL:
        return "electro"
    elif method == SynthesisConfig.PHOTO:
        return "photo"
    else:
        return "inkjet"


def load_synthesis_config(synthesis_method: str, script_dir: Path,
                         target_seq_length: int = None,
                         use_kmer: bool = False,
                         cli_coverage_params: dict = None,
                         drop_rate: float = 0.0) -> Dict:
    suffix = synthesis_method
    method_name = get_synthesis_method_display_name(synthesis_method)
    input_dir = script_dir / 'input_dir'
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found: {input_dir}\n"
            f"Please create the input_dir directory and place the following .npz files in it:\n"
            f"  - per_position_error_rates_{suffix}.npz\n"
            f"  - error_bias_{suffix}.npz\n"
            f"  - error_bias_kmer_{suffix}.npz  (optional, only needed when using --use-kmer)"
        )
    
    depth_file = input_dir / f"read_coverage_depth_{suffix}.npz"
    if not depth_file.exists():
        raise FileNotFoundError(
            f"Coverage depth parameter file not found: {depth_file}\n"
            f"Hint: You can specify coverage depth distribution directly via CLI parameters such as "
            f"--target-read-depth, --dist, --cv, without needing this file."
        )
    depth_data = np.load(depth_file)
    npz_avg_coverage = float(depth_data['avg_coverage'])
    npz_cv = float(depth_data['cv'])
    npz_dropout_rate = (float(depth_data['dropout_rate']) if 'dropout_rate' in depth_data else 0.0) / 100.0
    
    depth_params = None
    depth_dist_info = None
    if cli_coverage_params is not None:
        mean_depth = cli_coverage_params.get('mean_depth')
        cv = cli_coverage_params.get('cv')
        dist = cli_coverage_params.get('dist', 'gamma')
        depth_dist_info = calculate_depth_distribution_params(
            dist, mean_depth, cv,
            beta_min=cli_coverage_params.get('beta_min'),
            beta_max=cli_coverage_params.get('beta_max'),
        )
    else:
        avg_coverage = npz_avg_coverage
        cv = npz_cv
        depth_params = {'avg_coverage': avg_coverage, 'cv': cv}
        depth_dist_info = calculate_depth_distribution_params('lognormal', avg_coverage, cv)

    position_file = input_dir / f"per_position_error_rates_{suffix}.npz"
    if not position_file.exists():
        raise FileNotFoundError(f"Position error rate file not found: {position_file}")
    position_data = np.load(position_file)
    
    error_matrix_file = input_dir / f"error_bias_{suffix}.npz"
    if not error_matrix_file.exists():
        raise FileNotFoundError(f"Error matrix file not found: {error_matrix_file}")
    error_matrix_data = np.load(error_matrix_file)

    kmer_matrix_data = None
    if use_kmer:
        kmer_path = input_dir / f"error_bias_kmer_{suffix}.npz"
        if not kmer_path.exists():
            raise FileNotFoundError(
                f"kmer error matrix file not found: {kmer_path}\n"
                f"Please run error_matrix_kmer.py to generate this file first, or disable the --use-kmer flag."
            )
        kmer_npz_data = np.load(kmer_path)
        kmer_sub_raw = kmer_npz_data['substitution']
        kmer_ins_raw = kmer_npz_data['insertion']
        kmer_del_raw = kmer_npz_data['deletion']
        kmer_sub = _normalize_kmer_matrix(kmer_sub_raw)
        kmer_ins = _normalize_kmer_matrix(kmer_ins_raw)
        kmer_del = _normalize_kmer_matrix(kmer_del_raw)
        kmer_matrix_data = {
            'substitution': kmer_sub,
            'insertion': kmer_ins,
            'deletion': kmer_del,
        }
    
    config_seq_length = len(position_data['total_error_rate'])
    
    if target_seq_length is not None and target_seq_length != config_seq_length:
        substitution_rate = resize_error_rates_spline(
            position_data['substitution_rate'], target_seq_length
        )
        insertion_rate = resize_error_rates_spline(
            position_data['insertion_rate'], target_seq_length
        )
        deletion_rate = resize_error_rates_spline(
            position_data['deletion_rate'], target_seq_length
        )
        total_error_rate = substitution_rate + insertion_rate + deletion_rate
        position_rates = {
            'total_error_rate': total_error_rate,
            'substitution_rate': substitution_rate,
            'insertion_rate': insertion_rate,
            'deletion_rate': deletion_rate
        }
        seq_length = target_seq_length
    else:
        position_rates = {
            'total_error_rate': position_data['total_error_rate'],
            'substitution_rate': position_data['substitution_rate'],
            'insertion_rate': position_data['insertion_rate'],
            'deletion_rate': position_data['deletion_rate']
        }
        seq_length = config_seq_length
    
    return {
        'error_matrix': {
            'substitution': error_matrix_data['substitution'],
            'insertion': error_matrix_data['insertion'],
            'deletion': error_matrix_data['deletion']
        },
        'kmer': kmer_matrix_data,
        'use_kmer': use_kmer,
        'depth_params': depth_params,
        'depth_dist_info': depth_dist_info,
        'position_rates': position_rates,
        'seq_length': seq_length,
        'method_name': method_name,
        'dropout_rate': npz_dropout_rate
    }


def _normalize_kmer_matrix(raw: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(raw, dtype=np.float64)
    for i in range(64):
        for j in range(4):
            row = raw[i, j, :]
            total = row.sum()
            if total > 0:
                normalized[i, j, :] = row / total
            else:
                normalized[i, j, :] = 0.25
    return normalized


def resize_error_rates_spline(original_rates: np.ndarray, target_length: int) -> np.ndarray:
    original_length = len(original_rates)
    if original_length == target_length:
        return original_rates.copy()
    
    x_old = np.arange(original_length)
    x_new = np.linspace(0, original_length - 1, target_length)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_old, original_rates, bc_type='natural')
    resized_rates = cs(x_new)
    resized_rates = np.maximum(resized_rates, 0)
    resized_rates[0] = original_rates[0]
    resized_rates[-1] = original_rates[-1]
    return resized_rates


def build_substitution_lookup_from_matrix(substitution_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base_to_idx = {65: 0, 84: 1, 67: 2, 71: 3}
    idx_to_base = {0: 65, 1: 84, 2: 67, 3: 71}
    
    sub_choices = np.zeros((256, 3), dtype=np.uint8)
    sub_probs = np.zeros((256, 3), dtype=np.float32)
    
    for base_ascii, ref_idx in base_to_idx.items():
        counts = []
        choices = []
        for alt_idx in range(4):
            if alt_idx != ref_idx:
                counts.append(substitution_matrix[ref_idx, alt_idx])
                choices.append(idx_to_base[alt_idx])
        
        total = sum(counts)
        if total > 0:
            probs = [c / total for c in counts]
        else:
            probs = [1/3, 1/3, 1/3]
        
        prob_sum = sum(probs)
        if prob_sum > 1.0:
            excess = prob_sum - 1.0
            rng = np.random.default_rng(42)
            selected_idx = rng.integers(0, 3)
            probs[selected_idx] = max(0.0, probs[selected_idx] - excess)
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            else:
                probs = [1/3, 1/3, 1/3]
            logger.warning(f"  {chr(base_ascii)}: Probability sum exceeds 1 ({prob_sum:.6f}), corrected")
        
        sub_choices[base_ascii] = np.array(choices, dtype=np.uint8)
        sub_probs[base_ascii] = np.array(probs, dtype=np.float32)
    
    sub_cum_probs = np.cumsum(sub_probs, axis=1).astype(np.float32)
    return sub_choices, sub_cum_probs
