//! #199 — Two-NN selection cost: sort-based reference vs streaming minima.
//!
//! Baseline reproduces the pre-#199 `estimate_intrinsic_dimension` inner
//! loop: per-sample `Vec<(usize, f64)>` of all non-self distances plus a full
//! sort, parallel across samples with Rayon — identical structure to the
//! removed code. Candidate is the current implementation, which streams the
//! two exact minima with O(1) scratch.
//!
//! Fixture: N=2000 rows × 384 features (CVE-shaped), seed 3407, uniform.
//! Recorded on M4 Pro, Rust stable, release profile.

use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::clustering::ClusteringHeuristic;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::hint::black_box;

const N_SMALL: usize = 2000;
const N_LARGE: usize = 16000;
// CVE-scale (n=335,462) comparison lives in arrowspace-benches
// (`bench/199-twonn-selection`): too heavy for a local cargo bench loop.
const F: usize = 384;
const SEED: u64 = 3407;

fn fixture(n: usize) -> DenseMatrix<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..F).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    DenseMatrix::from_2d_vec(&rows).unwrap()
}

/// Pre-#199 implementation: materialise + sort, parallel across samples.
fn baseline_sort_reference(rows: &DenseMatrix<f64>, n: usize, f: usize, base_seed: u64) -> usize {
    use rayon::prelude::*;

    if n < 10 {
        return f.min(2);
    }
    let sample_size = n.min(500);
    let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(1));
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let sample_indices = &indices[..sample_size];

    let ratios: Vec<f64> = sample_indices
        .par_iter()
        .filter_map(|&i| {
            let row_i = rows.get_row(i);
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d2: f64 = row_i
                        .iterator(0)
                        .zip(rows.get_row(j).iterator(0))
                        .map(|(a, b)| (*a - *b).powi(2))
                        .sum();
                    (j, d2.sqrt())
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if dists.len() >= 2 {
                let d1 = dists[0].1;
                let d2 = dists[1].1;
                if d1 > 1e-12 {
                    return Some(d2 / d1);
                }
            }
            None
        })
        .collect();

    if ratios.is_empty() {
        return f.min(3);
    }
    let mean_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let id = if mean_ratio > 1.001 {
        1.0 / mean_ratio.ln()
    } else {
        f as f64
    };
    (id.round() as usize).clamp(1, f)
}

fn bench_twonn(c: &mut Criterion) {
    let builder = ArrowSpaceBuilder::new();

    for &(label, n, samples) in &[("n2000", N_SMALL, 30usize), ("n16000", N_LARGE, 30)] {
        let dm = fixture(n);

        // Parity guard inside the bench: both paths must agree on this fixture.
        assert_eq!(
            builder.estimate_intrinsic_dimension(&dm, n, F, SEED),
            baseline_sort_reference(&dm, n, F, SEED),
            "bench fixture {label}: streaming and sort-based estimates diverge"
        );

        let mut group = c.benchmark_group(format!("two_nn_selection/{label}"));
        group.sample_size(samples);

        group.bench_with_input(
            BenchmarkId::new("estimate_intrinsic_dimension", "baseline_sort"),
            &dm,
            |b, dm| {
                b.iter(|| black_box(baseline_sort_reference(black_box(dm), n, F, SEED)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("estimate_intrinsic_dimension", "streaming"),
            &dm,
            |b, dm| {
                b.iter(|| {
                    black_box(black_box(&builder).estimate_intrinsic_dimension(
                        black_box(dm),
                        n,
                        F,
                        SEED,
                    ))
                })
            },
        );

        group.finish();
    }
}

criterion_group!(benches, bench_twonn);
criterion_main!(benches);
