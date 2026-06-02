# DNATerra

DNATerra simulates DNA storage sequencing reads from reference strands using experimentally parameterized and user-adjustable noise profiles. It supports flexible customization of sequencing depth, read-depth distribution patterns controlled by mean depth and coefficient of variation, position-dependent substitution, insertion and deletion rates, and sequence-context-dependent bias associated with GC content, homopolymers and specific motifs. DNATerra also provides parallel read generation, optional read shuffling and ground-truth output, enabling controlled benchmarking and downstream workflow analysis.

## What can it do for DNA storage?

- Use FASTA input and output to connect quickly with upstream encoding tools and downstream clustering, consensus and decoding tools.
- Build end-to-end DNA storage workflows for rapid feasibility tests before wet-lab validation.
- Explore parameter choices across workflow stages, including redundancy, read depth, clustering thresholds, consensus settings and decoding conditions.
- Use ground-truth read-to-reference mappings, CIGAR strings and MD tags to evaluate alignment, clustering and reconstruction accuracy.
- Use ordered reads for controlled feasibility tests, or shuffled reads to evaluate performance under sequencing-like read order.
- Adjust noise parameters independently to test robustness, such as increasing insertion, deletion, substitution, dropout or read-depth variation alone.
- Update noise profiles from new sequencing datasets as synthesis platforms, sequencing devices or experimental protocols change.
- Extrapolate small-scale experimental noise profiles to medium-scale simulations, iterate workflow parameters and prepare for larger-scale DNA storage experiments.

These are typical use cases rather than fixed limits.

## Installation

DNATerra currently targets Linux systems. On Windows, use WSL2 or a Linux environment.

Python dependencies are listed in `requirements.txt`. External command-line tools used by the demo or statistics workflow should be installed as needed.

## Quick Start

Run both normal mode and simple mode:

```bash
bash run_simulation.sh
```

Run one mode only:

```bash
bash run_simulation.sh normal
bash run_simulation.sh simple
```

The input FASTA must use a strict two-line format: one identifier line followed by one sequence line. All reference sequences should have the same length and contain only `A`, `T`, `C` and `G`.

## Main Parameters

| Argument | Meaning |
| --- | --- |
| `-i`, `--input` | Input reference FASTA |
| `-o`, `--output` | Output directory |
| `--method` | Built-in or user-updated noise profile |
| `--target-read-depth` | Mean read depth per reference strand |
| `--drop-rate` | Strand dropout rate |
| `--dist` | Read-depth distribution: `gamma`, `normal`, `lognormal`, `exponential`, `poisson`, `uniform`, `nbinom`, `beta`, `weibull` |
| `--cv` | Coefficient of variation for read depth |
| `--error-rate` | Total IDS rate or separate substitution, insertion and deletion rates, in `10^-3 nt^-1` |
| `--use-kmer` | Use k-mer context bias, `y` or `n` |
| `--shuffle` | Shuffle generated reads, `y` or `n` |
| `--merge-files` | Merge chunk outputs, `y` or `n` |
| `--stats` | Output ground-truth statistics, `y` or `n` |
| `--num-workers` | Number of parallel workers |
| `--chunk-size` | Number of reads processed per chunk |
| `--random-seed` | Random seed |

## Output

Depending on the selected options, DNATerra writes:

- simulated reads in FASTA format, either as per-chunk files or merged FASTA files
- `ref_count.tsv`
- `read_to_ref_ordered.tsv`, including read-to-reference mappings, CIGAR strings and MD tags
- `read_to_ref_shuffled.tsv`, including read-to-reference mappings, CIGAR strings and MD tags when shuffled output is enabled

## Demo

Run the end-to-end demo:

```bash
bash run_demo.sh
```

The script runs encoding, DNATerra simulation, clustering, correction, decoding and verification.

## Performance

DNATerra has been benchmarked at large scale:

| Platform | Throughput | Largest demonstrated run |
| --- | --- | --- |
| Consumer laptop | ~125.9 Mbp/s | 1.1 Tbp in 2.5 hours |
| Single 64-core CPU cluster node | ~1.05 Gbp/s | 100.02 Tbp in 27 hours |

Throughput is reported as generated DNA bases per second.

## Third-party Code

The end-to-end demo includes DNA-Fountain code under `demo/dna_fountain/`. This third-party component is distributed under GPLv3-or-later. See `demo/dna_fountain/COPYING`, `demo/dna_fountain/NOTICE.md` and `THIRD_PARTY_NOTICES.md`.

DNATerra's own code is distributed under the license in `LICENSE`, except for third-party components that carry their own licenses.

## Statistics and Noise-profile Update

To extract statistics from new sequencing data and update a noise profile, edit the paths in the script and run it:

```bash
bash input_dir/self_update_simulator_usage_example.sh
```

The script includes paired-end and single-end examples.

## Citation

Citation details will be added upon publication.
