//! #199 — streaming Two-NN minima must match the sort-based reference exactly.
//!
//! `oracle_two_smallest` below reproduces the pre-#199 algorithm verbatim:
//! collect all non-self distances as `(usize, f64)` pairs, `sort_by` with
//! `partial_cmp().unwrap()`, read the two smallest values. The streaming
//! implementation (`clustering::two_smallest_distances`) must return
//! bit-identical `(d1, d2)` pairs on every fixture, panic on NaN exactly like
//! the sort's `unwrap` did, and feed an unchanged
//! `estimate_intrinsic_dimension` result.
//!
//! Parity is asserted on `to_bits()`, not `==`, so ±0.0 distinctions cannot
//! hide behind float equality.

use crate::{
    builder::ArrowSpaceBuilder,
    clustering::{ClusteringHeuristic, two_smallest_distances},
};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

// -------------------- Sort-based reference oracle --------------------

/// Pre-#199 per-sample neighbour selection, kept verbatim as the oracle.
fn oracle_two_smallest(rows: &DenseMatrix<f64>, i: usize, n: usize) -> Option<(f64, f64)> {
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
        Some((dists[0].1, dists[1].1))
    } else {
        None
    }
}

fn assert_parity(rows: &DenseMatrix<f64>, n: usize) {
    for i in 0..n {
        let got = two_smallest_distances(rows, i, n).map(|(a, b)| (a.to_bits(), b.to_bits()));
        let want = oracle_two_smallest(rows, i, n).map(|(a, b)| (a.to_bits(), b.to_bits()));
        assert_eq!(
            got, want,
            "row {i}: streaming (d1, d2) differs from sort-based reference"
        );
    }
}

// -------------------- Reference estimate (full-function oracle) --------------------

/// Full pre-#199 `estimate_intrinsic_dimension` replica built on the oracle.
/// The ratio collection order matters: Rayon's indexed `par_iter().collect()`
/// preserves sample order, so the serial replica below sums in the same order
/// and must produce the identical mean ratio bit-for-bit.
fn reference_estimate_intrinsic_dimension(
    rows: &DenseMatrix<f64>,
    n: usize,
    f: usize,
    base_seed: u64,
) -> usize {
    if n < 10 {
        return f.min(2);
    }

    let sample_size = n.min(500);
    let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(1));
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let sample_indices = &indices[..sample_size];

    let ratios: Vec<f64> = sample_indices
        .iter()
        .filter_map(|&i| {
            let (d1, d2) = oracle_two_smallest(rows, i, n)?;
            if d1 > 1e-12 { Some(d2 / d1) } else { None }
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

// -------------------- Fixture helpers --------------------

fn seeded_uniform(n: usize, f: usize, seed: u64) -> DenseMatrix<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..f).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    DenseMatrix::from_2d_vec(&rows).unwrap()
}

// -------------------- Per-sample parity tests --------------------

#[test]
fn streaming_matches_reference_on_seeded_uniform() {
    let dm = seeded_uniform(40, 5, 3407);
    assert_parity(&dm, 40);
}

#[test]
fn streaming_matches_reference_on_exact_duplicate_rows() {
    // Three exact duplicates of row 0, plus distinct rows: distance-0
    // neighbours must count as separate candidates, not collapse.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let base = vec![0.5, -0.25, 0.125];
    rows.push(base.clone());
    rows.push(base.clone());
    rows.push(base.clone());
    for i in 0..12 {
        rows.push(vec![i as f64 * 0.3, 1.0 + i as f64 * 0.1, -2.0]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_parity(&dm, rows.len());
}

#[test]
fn streaming_matches_reference_on_tied_distances() {
    // One-dimensional lattice: every interior row has two neighbours at
    // exactly distance 1.0 — the tie must resolve to (1.0, 1.0) like the sort.
    let rows: Vec<Vec<f64>> = (0..25).map(|i| vec![i as f64]).collect();
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_parity(&dm, rows.len());

    let (d1, d2) = two_smallest_distances(&dm, 12, rows.len()).unwrap();
    assert_eq!(
        (d1.to_bits(), d2.to_bits()),
        (1.0f64.to_bits(), 1.0f64.to_bits())
    );
}

#[test]
fn streaming_matches_reference_on_one_dimensional_data() {
    let dm = seeded_uniform(30, 1, 3407);
    assert_parity(&dm, 30);
}

#[test]
fn streaming_matches_reference_on_tiny_coordinates() {
    // Squares of 1e-200 underflow toward zero; ordering must survive.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for i in 0..15 {
        rows.push(vec![i as f64 * 1e-200, (i % 3) as f64 * 1e-205]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_parity(&dm, rows.len());
}

#[test]
fn streaming_matches_reference_on_huge_coordinates_with_overflow() {
    // Squared differences of 1e200-range coordinates overflow to +inf;
    // sqrt(+inf) = +inf. Infinity distances must sort like the reference.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for i in 0..15 {
        rows.push(vec![i as f64 * 1e200, -(i as f64) * 1e199]);
    }
    // Include some moderate rows so finite and infinite distances mix.
    for i in 0..5 {
        rows.push(vec![i as f64, i as f64 * 2.0]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_parity(&dm, rows.len());
}

#[test]
fn streaming_matches_reference_on_infinite_coordinates() {
    // A row of +inf coordinates is at +inf distance from every finite row.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    rows.push(vec![f64::INFINITY, f64::INFINITY]);
    for i in 0..10 {
        rows.push(vec![i as f64, -i as f64]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_parity(&dm, rows.len());
}

#[test]
fn streaming_returns_none_with_fewer_than_two_candidates() {
    let rows = vec![vec![1.0, 2.0]];
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_eq!(two_smallest_distances(&dm, 0, 1), None);

    let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    assert_eq!(two_smallest_distances(&dm, 0, 2), None);
    assert!(two_smallest_distances(&dm, 0, 3).is_some());
}

// -------------------- NaN contract --------------------

#[test]
#[should_panic(expected = "called `Option::unwrap()`")]
fn oracle_panics_on_nan_distance() {
    // Documents the pre-#199 contract the streaming version must preserve.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    rows.push(vec![f64::NAN, 1.0]);
    for i in 0..10 {
        rows.push(vec![i as f64, -i as f64]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    let _ = oracle_two_smallest(&dm, 1, rows.len());
}

#[test]
#[should_panic(expected = "NaN")]
fn streaming_panics_on_nan_input_distance() {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    rows.push(vec![f64::NAN, 1.0]);
    for i in 0..10 {
        rows.push(vec![i as f64, -i as f64]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    let _ = two_smallest_distances(&dm, 1, rows.len());
}

#[test]
#[should_panic(expected = "NaN")]
fn streaming_panics_on_inf_minus_inf_distance() {
    // inf - inf = NaN: two rows sharing an infinite coordinate produce a NaN
    // distance even though no input value is NaN.
    let mut rows: Vec<Vec<f64>> = Vec::new();
    rows.push(vec![f64::INFINITY, 1.0]);
    rows.push(vec![f64::INFINITY, 2.0]);
    for i in 0..10 {
        rows.push(vec![i as f64, -i as f64]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    let _ = two_smallest_distances(&dm, 0, rows.len());
}

// -------------------- Integrated function parity --------------------

#[test]
fn estimate_intrinsic_dimension_matches_reference_on_line() {
    let mut rows = Vec::new();
    for i in 0..100 {
        let t = i as f64 / 10.0;
        rows.push(vec![t, 2.0 * t, 3.0 * t]);
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    let builder = ArrowSpaceBuilder::new();
    let got = builder.estimate_intrinsic_dimension(&dm, rows.len(), 3, 3407);
    let want = reference_estimate_intrinsic_dimension(&dm, rows.len(), 3, 3407);
    assert_eq!(got, want, "streaming and sort-based estimates diverge");
}

#[test]
fn estimate_intrinsic_dimension_matches_reference_on_seeded_uniform() {
    let dm = seeded_uniform(200, 6, 3407);
    let builder = ArrowSpaceBuilder::new();
    let got = builder.estimate_intrinsic_dimension(&dm, 200, 6, 3407);
    let want = reference_estimate_intrinsic_dimension(&dm, 200, 6, 3407);
    assert_eq!(got, want, "streaming and sort-based estimates diverge");
}

#[test]
fn estimate_intrinsic_dimension_matches_reference_on_duplicates() {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for dup in 0..4 {
        for i in 0..10 {
            rows.push(vec![dup as f64, i as f64 * 0.5]);
        }
    }
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    let builder = ArrowSpaceBuilder::new();
    let got = builder.estimate_intrinsic_dimension(&dm, rows.len(), 2, 3407);
    let want = reference_estimate_intrinsic_dimension(&dm, rows.len(), 2, 3407);
    assert_eq!(got, want, "streaming and sort-based estimates diverge");
}
