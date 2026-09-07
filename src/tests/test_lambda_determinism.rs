//! Regression tests for issue #170: Eigen λ bit-reproducibility.
//!
//! `deterministic_clustering` (set by `ArrowSpaceBuilder::with_seed`) is
//! documented as making builds reproducible, but the Rayleigh-quotient
//! reduction in `TauMode::compute_rayleigh_quotient_from_matrix` used
//! `par_bridge().sum()`. `par_bridge` batches its source by dynamic
//! work-stealing, so the f64 addends group in a schedule-dependent order —
//! and f64 addition is not associative. When the global rayon pool was busy,
//! two builds of identical input with identical config diverged in the last
//! bits of every λ.
//!
//! These tests saturate the global pool with CPU-bound work while the
//! measured computations run on the same pool, then assert byte-identical
//! (bit-exact) results. Scheduling may reorder *execution*, never the
//! *grouping of the summation*.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use serial_test::serial;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::TriMat;

use crate::builder::{ArrowSpaceBuilder, PipelineKind};
use crate::search::taumode::TauMode;
use crate::tests::init;
use crate::tests::test_data::make_gaussian_hd;

/// Seeded LCG stream so every run exercises identical inputs.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 1_000_000) as f64 / 1_000_000.0
    }
}

/// Fixed sparse Laplacian-like matrix (L = D − A) built from a seeded LCG.
/// Ring + long-range connections, symmetric off-diagonal weights.
fn fixed_laplacian(n: usize) -> sprs::CsMat<f64> {
    let mut lcg = Lcg(0x2545_F491_4F6C_DD1D);
    let mut degrees = vec![0.0f64; n];
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        for step in [1usize, 7, 31] {
            let j = (i + step) % n;
            let w = 0.1 + lcg.next_unit();
            triplets.push((i, j, -w));
            triplets.push((j, i, -w));
            degrees[i] += w;
            degrees[j] += w;
        }
    }
    for i in 0..n {
        triplets.push((i, i, degrees[i]));
    }

    let mut tm = TriMat::new((n, n));
    for (i, j, v) in triplets {
        tm.add_triplet(i, j, v);
    }
    tm.to_csr()
}

/// Fixed query vector for the Rayleigh quotient, mixed magnitudes so the
/// summation grouping matters in the last bits.
fn fixed_vector(n: usize) -> Vec<f64> {
    let mut lcg = Lcg(0xDEAD_BEEF_CAFE_1234);
    (0..n).map(|_| 0.1 + 10.0 * lcg.next_unit()).collect()
}

/// Floods the global rayon pool with CPU-bound tasks from independent
/// threads for the lifetime of the guard, so the code under test runs while
/// the pool is saturated (the condition under which issue #170 manifests).
struct PoolSaturator {
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl PoolSaturator {
    fn start() -> Self {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let stop = Arc::new(AtomicBool::new(false));
        let handles = (0..workers)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        (0..64).into_par_iter().for_each(|_| {
                            let mut x = 0.5f64;
                            for _ in 0..20_000 {
                                x = std::hint::black_box(x * 1.000_000_1 + 1e-9);
                            }
                            std::hint::black_box(x);
                        });
                    }
                })
            })
            .collect();
        PoolSaturator { stop, handles }
    }
}

impl Drop for PoolSaturator {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Unit-level check: identical (matrix, vector) inputs must give a
/// bit-identical Rayleigh quotient on every evaluation, even while the
/// global pool is saturated by unrelated work. `par_bridge().sum()` batches
/// by work-stealing, so this fails on issue #170.
#[test]
#[serial]
fn test_rayleigh_quotient_bit_reproducible_under_pool_saturation() {
    init();

    let n = 200;
    let matrix = fixed_laplacian(n);
    let vector = fixed_vector(n);

    // Reference evaluation on an idle pool.
    let expected = TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &vector);
    assert!(
        expected.is_finite() && expected > 0.0,
        "fixture must produce a non-trivial quotient, got {expected}"
    );

    let _sat = PoolSaturator::start();
    for call in 0..512 {
        let got = TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &vector);
        assert_eq!(
            expected.to_bits(),
            got.to_bits(),
            "Rayleigh quotient not bit-reproducible at call {call} under pool saturation: \
             {expected:e} vs {got:e}"
        );
    }
}

/// Acceptance criterion from issue #170: two independent EigenMaps builds of
/// the same fixed dataset with the same config and seed
/// (`deterministic_clustering = true`) must return byte-identical λ vectors,
/// while the global rayon pool is saturated by unrelated work.
#[test]
#[serial]
fn test_eigen_build_lambda_bit_reproducible_under_pool_saturation() {
    init();

    let rows = make_gaussian_hd(160, 0.5);
    let build_lambdas = || {
        let dense = DenseMatrix::from_2d_vec(&rows).unwrap();
        // Config mirrors the downstream reproduction in #170:
        // eps=0.5, k=5, topk=3, p=2.0, sigma=None, clustering_seed=3407.
        ArrowSpaceBuilder::new()
            .with_lambda_graph(0.5, 5, 3, 2.0, None)
            .with_synthesis(TauMode::Median)
            .with_dims_reduction(false, None)
            .with_inline_sampling(None)
            .with_seed(3407)
            .build_for_persistence(dense, PipelineKind::Eigen)
            .0
            .lambdas()
            .to_vec()
    };

    let _sat = PoolSaturator::start();
    let reference = build_lambdas();
    for build_idx in 1..4 {
        let lambdas = build_lambdas();
        assert_eq!(reference.len(), lambdas.len());
        for (i, (&a, &b)) in reference.iter().zip(lambdas.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "λ[{i}] not byte-identical across builds (build {build_idx}): {a:e} vs {b:e}"
            );
        }
    }
}
