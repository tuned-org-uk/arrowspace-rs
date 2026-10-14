//! #197 — deterministic incremental clustering must stay bit-identical when
//! the per-row snapshot clone and Phase-3 re-scan are removed from the
//! serial path.
//!
//! `reference_incremental_clustering` is a verbatim copy of
//! `run_incremental_clustering_with_sampling` as it stood before #197
//! (full centroid-table clone in Phase 1, `nearest_centroid` re-scan in
//! Phase 3), driven serially exactly like deterministic mode. The shipped
//! implementation must produce identical bits for centroids, assignments
//! and counts on every fixture: the optimization may only remove work that
//! is provably redundant in a single-threaded driver, never change a
//! decision, an order of updates, or a floating-point operation.

use crate::builder::ArrowSpaceBuilder;
use crate::clustering::run_incremental_clustering_with_sampling;
use crate::sampling::InlineSampler;
use log::{debug, info, trace, warn};
use rand::{RngExt, SeedableRng};
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::sync::{Arc, Mutex};

const SEED: u64 = 3407;

// -------------------- Verbatim pre-#197 reference --------------------

/// Pre-#197 implementation, kept verbatim (Phase 1 clone + Phase 3
/// re-scan); the driver is the serial loop deterministic mode uses.
#[allow(clippy::too_many_arguments)]
fn reference_incremental_clustering(
    arrowspacebuilder: &ArrowSpaceBuilder,
    rows: &DenseMatrix<f64>,
    nfeatures: usize,
    max_clusters: usize,
    radius: f64,
    sampler: Arc<Mutex<dyn InlineSampler>>,
) -> (DenseMatrix<f64>, Vec<Option<usize>>, Vec<usize>) {
    let nrows = rows.shape().0;

    let centroids = Mutex::new(Vec::<Vec<f64>>::new());
    let counts = Mutex::new(Vec::<usize>::new());
    let assignments = Mutex::new(vec![None; nrows]);

    let process_row = |row_idx: usize| {
        let row: Vec<f64> = rows.get_row(row_idx).iterator(0).copied().collect();
        let row = row.as_slice();

        // PHASE 1: Snapshot and decision
        let cent_snap = {
            let c = centroids.lock().unwrap();
            c.clone()
        };

        let (_snap_best_idx, snap_best_dist_sq) = if cent_snap.is_empty() {
            (0, f64::INFINITY)
        } else {
            let (idx, dist) = crate::clustering::nearest_centroid(row, &cent_snap);
            (idx, dist)
        };

        if arrowspacebuilder.sampling.is_some() {
            let mut smp = sampler.lock().unwrap();
            if !smp.should_keep(row, snap_best_dist_sq, cent_snap.len(), max_clusters) {
                return;
            }
        }

        // PHASE 2: Update phase under lock
        let mut c = centroids.lock().unwrap();
        let mut k = counts.lock().unwrap();
        let mut a = assignments.lock().unwrap();

        #[cfg(test)]
        assert!(
            c.len() >= cent_snap.len(),
            "Row {}: Centroid count went backwards!",
            row_idx
        );

        if c.is_empty() {
            #[cfg(test)]
            assert_eq!(
                cent_snap.len(),
                0,
                "Row {}: Snapshot should be empty",
                row_idx
            );
            #[cfg(test)]
            assert_eq!(
                snap_best_dist_sq,
                f64::INFINITY,
                "Row {}: Distance should be INFINITY for empty",
                row_idx
            );

            c.push(row.to_vec());
            k.push(1);
            a[row_idx] = Some(0);
            return;
        }

        // PHASE 3: Decision based on snapshot distance
        if c.len() < max_clusters && snap_best_dist_sq > (radius * 0.5) {
            let new_idx = c.len();
            c.push(row.to_vec());
            k.push(1);
            a[row_idx] = Some(new_idx);

            #[cfg(test)]
            assert_eq!(c.len(), k.len(), "Row {}: sync", row_idx);
        } else if snap_best_dist_sq <= radius {
            // ASSIGN TO EXISTING CLUSTER — recompute with current centroids
            let (best_idx, current_dist_sq) = crate::clustering::nearest_centroid(row, &c);

            #[cfg(test)]
            assert!(best_idx < c.len(), "Row {}: bad idx", row_idx);

            let k_old = k[best_idx] as f64;
            let k_new = k_old + 1.0;

            #[cfg(test)]
            assert!(k_old > 0.0, "Row {}: zero count", row_idx);

            for j in 0..nfeatures {
                c[best_idx][j] += (row[j] - c[best_idx][j]) / k_new;
            }
            k[best_idx] += 1;
            a[row_idx] = Some(best_idx);

            trace!(
                "ref row {}: assigned to {} (dist²={:.6})",
                row_idx, best_idx, current_dist_sq
            );
        } else {
            // Soft outlier policy
            let (best_idx, current_dist_sq) = crate::clustering::nearest_centroid(row, &c);
            let relax_factor = 1.5;
            let relaxed_radius = radius * relax_factor;

            if current_dist_sq <= relaxed_radius {
                let eta = 0.0;
                if eta > 0.0 {
                    for j in 0..nfeatures {
                        c[best_idx][j] += eta * (row[j] - c[best_idx][j]);
                    }
                }
                k[best_idx] += 1;
                a[row_idx] = Some(best_idx);
            } else {
                return;
            }
        }

        #[cfg(test)]
        {
            assert_eq!(c.len(), k.len(), "Row {}: Final check", row_idx);
            assert!(c.len() <= max_clusters, "Row {}: exceeded cap", row_idx);
        }
    };

    (0..nrows).into_iter().for_each(process_row);

    let final_centroids = centroids.into_inner().unwrap();
    let final_counts = counts.into_inner().unwrap();
    let final_assignments = assignments.into_inner().unwrap();

    let x_out = &final_centroids.len().max(1);
    let mut flat = Vec::<f64>::with_capacity(x_out * nfeatures);
    for c in &final_centroids {
        flat.extend_from_slice(c);
    }

    let centroids_dm: DenseMatrix<f64> = if *x_out > 0 && !final_centroids.is_empty() {
        DenseMatrix::from_iterator(flat.iter().copied(), *x_out, nfeatures, 0)
    } else {
        warn!("No clusters created; returning zero matrix");
        DenseMatrix::from_2d_vec(&vec![vec![0.0_f64; nfeatures]; *x_out]).unwrap()
    };

    debug!(
        "reference clustering produced {} centroids from {} rows",
        final_centroids.len(),
        nrows
    );
    info!(
        "reference incremental clustering done ({} centroids)",
        centroids_dm.shape().0
    );

    (centroids_dm, final_assignments, final_counts)
}

// -------------------- Fixture helpers --------------------

fn clustered_rows(n: usize, f: usize, clusters: usize, jitter: f64) -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let centers: Vec<Vec<f64>> = (0..clusters)
        .map(|_| (0..f).map(|_| rng.random_range(0.0..1.0)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % clusters];
            c.iter()
                .map(|&x| x + (rng.random_range(0.0..1.0) - 0.5) * jitter)
                .collect()
        })
        .collect()
}

fn deterministic_builder() -> ArrowSpaceBuilder {
    // with_seed implies deterministic_clustering (see builder).
    ArrowSpaceBuilder::new().with_seed(SEED)
}

fn dummy_sampler() -> Arc<Mutex<dyn InlineSampler>> {
    Arc::new(Mutex::new(crate::sampling::SimpleRandomSampler::new(
        1.0, 3407,
    )))
}

fn assert_reference_parity(
    rows: &Vec<Vec<f64>>,
    nfeatures: usize,
    max_clusters: usize,
    radius: f64,
    label: &str,
) {
    let dm = DenseMatrix::from_2d_vec(rows).unwrap();
    let builder = deterministic_builder();

    let (got_cent, got_assign, got_counts) = run_incremental_clustering_with_sampling(
        &builder,
        &dm,
        nfeatures,
        max_clusters,
        radius,
        dummy_sampler(),
    );
    let (want_cent, want_assign, want_counts) = reference_incremental_clustering(
        &builder,
        &dm,
        nfeatures,
        max_clusters,
        radius,
        dummy_sampler(),
    );

    assert_eq!(
        got_cent.shape(),
        want_cent.shape(),
        "{label}: centroid matrix shape differs"
    );
    assert_eq!(got_cent.shape().0, want_cent.shape().0);
    for r in 0..got_cent.shape().0 {
        for (g, w) in got_cent
            .get_row(r)
            .iterator(0)
            .zip(want_cent.get_row(r).iterator(0))
        {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: centroid {r} bits differ ({g} vs {w})"
            );
        }
    }
    assert_eq!(got_assign, want_assign, "{label}: assignments differ");
    assert_eq!(got_counts, want_counts, "{label}: counts differ");
}

// -------------------- Parity tests --------------------

#[test]
fn parity_new_cluster_heavy_until_saturation() {
    // Tiny radius: nearly every row creates a cluster until the cap,
    // then soft-outlier assignment and drops take over.
    let rows = clustered_rows(120, 8, 6, 0.2);
    assert_reference_parity(&rows, 8, 10, 0.02, "new-cluster heavy");
}

#[test]
fn parity_assign_heavy_single_cluster() {
    // Huge radius: the first centroid absorbs everything.
    let rows = clustered_rows(150, 6, 5, 0.1);
    assert_reference_parity(&rows, 6, 5, 10.0, "assign heavy");
}

#[test]
fn parity_mixed_branches() {
    let rows = clustered_rows(200, 12, 8, 0.4);
    assert_reference_parity(&rows, 12, 8, 0.5, "mixed branches");
}

#[test]
fn parity_exact_duplicates() {
    let mut rows = Vec::new();
    for dup in 0..6 {
        let base: Vec<f64> = (0..10).map(|i| dup as f64 + i as f64 * 0.01).collect();
        for _ in 0..10 {
            rows.push(base.clone());
        }
    }
    assert_reference_parity(&rows, 10, 6, 0.001, "exact duplicates");
}

#[test]
fn parity_cve_shaped() {
    // Issue #197 fixture shape: K=41, radius 0.7855134010314941.
    let rows = clustered_rows(600, 64, 12, 0.3);
    assert_reference_parity(&rows, 64, 41, 0.785_513_401_031_494_1, "CVE shaped");
}

#[test]
fn parity_single_row() {
    let rows = clustered_rows(1, 4, 1, 0.0);
    assert_reference_parity(&rows, 4, 3, 1.0, "single row");
}

/// #197 timing evidence, interleaved A/B in one process: shipped
/// deterministic path vs the verbatim pre-#197 reference on the issue's
/// CVE-shaped fixture (16 384 x 384, K=41, radius 0.7855134010314941).
/// Run with: cargo test --release --lib -- --ignored timing_incremental --nocapture
#[test]
#[ignore = "timing evidence; run explicitly"]
fn timing_incremental_deterministic_ab() {
    let rows = clustered_rows(16_384, 384, 12, 0.3);
    let dm = DenseMatrix::from_2d_vec(&rows).unwrap();
    let builder = deterministic_builder();
    let max_clusters = 41;
    let radius = 0.785_513_401_031_494_1;

    // Warm-up + parity on a small slice first, so the timing compares
    // equal-behaviour code.
    assert_reference_parity(
        &rows[..200].to_vec(),
        384,
        max_clusters,
        radius,
        "warm-up slice",
    );

    let t0 = std::time::Instant::now();
    let (c_cent, _, _) = run_incremental_clustering_with_sampling(
        &builder,
        &dm,
        384,
        max_clusters,
        radius,
        dummy_sampler(),
    );
    let candidate = t0.elapsed();

    let t1 = std::time::Instant::now();
    let (r_cent, _, _) =
        reference_incremental_clustering(&builder, &dm, 384, max_clusters, radius, dummy_sampler());
    let baseline = t1.elapsed();

    assert_eq!(c_cent.shape(), r_cent.shape());
    println!(
        "incremental clustering 16384x384 K=41 (serial): candidate {candidate:?} vs pre-#197 reference {baseline:?}"
    );
}
