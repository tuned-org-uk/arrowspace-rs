//! #201 — read-out τ must act as a bounded mixing weight: the λτ read-out
//! is a convex combination of bounded Rayleigh energy and edge dispersion.
//!
//! Regression fixture from the median-v1 oracle (arrowspace 0.29.0,
//! source 96995ce): a three-feature Laplacian with one undirected edge
//! (0, 1) of weight 1, isolated feature 2, item [3, 1, 2]. The median of
//! the item's values is τ = 2, and the pre-#201 combination
//! λ = τ·E_b + (1−τ)·G produced the raw λ = 2·(1/8) − 1/2 = **−1/4** —
//! a negative dispersion coefficient. That historical value is bug
//! evidence, not a success criterion; the corrected rule keeps the
//! formula for τ ∈ [0, 1] and converts out-of-domain τ at the
//! combination site:
//!
//! - energy scale  s = max(τ, 0)          (denominator role, finite)
//! - mixing weight w = clamp(τ, 0, 1)     (convex role)
//! - non-finite τ converts to s = 0, w = 0 (λ = G).
//!
//! For τ ∈ [0, 1] every produced bit is unchanged.

use crate::builder::ArrowSpaceBuilder;
use crate::search::taumode::{TAU_FLOOR, TauMode};
use approx::assert_abs_diff_eq;
use sprs::CsMat;
use sprs::TriMat;

/// The median-v1 fixture: one undirected edge (0,1), isolated feature 2.
fn median_v1_laplacian() -> CsMat<f64> {
    let mut tri = TriMat::new((3, 3));
    tri.add_triplet(0, 0, 1.0);
    tri.add_triplet(1, 1, 1.0);
    tri.add_triplet(0, 1, -1.0);
    tri.add_triplet(1, 0, -1.0);
    tri.to_csr()
}

// -------------------- Selector contract --------------------

#[test]
fn selector_median_reads_item_feature_values() {
    // Documents issue item 6: automatic selection reads each item's
    // feature values (not corpus energy statistics), so data scale flows
    // into τ. The unbounded 2.0 is what the combination site must handle.
    let tau = TauMode::select_tau(&[3.0, 1.0, 2.0], TauMode::Median);
    assert_abs_diff_eq!(tau, 2.0, epsilon = 1e-12);
}

// -------------------- Regression: the corrected rule --------------------

#[test]
fn median_v1_raw_lambda_is_non_negative() {
    // Pre-#201 this returned -1/4 (negative dispersion coefficient).
    let lambda =
        TauMode::compute_synthetic_lambda(&[3.0, 1.0, 2.0], None, &median_v1_laplacian(), 2.0);
    assert_abs_diff_eq!(lambda, 0.125, epsilon = 1e-12);
}

#[test]
fn fixed_tau_above_one_stays_in_range() {
    // Fixed(2.0) must not bypass the bounded weight: a selector-only fix
    // is insufficient (issue item 1).
    let lambda =
        TauMode::compute_synthetic_lambda(&[3.0, 1.0, 2.0], None, &median_v1_laplacian(), 2.0);
    assert!((0.0..=1.0).contains(&lambda), "λ={lambda} outside [0,1]");
}

// -------------------- Domain: finite λ ∈ [0,1] for every τ --------------------

#[test]
fn lambda_in_unit_interval_for_all_tau_domains() {
    let graph = median_v1_laplacian();
    for tau in [
        TAU_FLOOR,
        0.25,
        0.5,
        0.999,
        1.0,
        1.5,
        2.0,
        100.0,
        1e9,
        f64::INFINITY,
        -1.0,
        -100.0,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        let lambda = TauMode::compute_synthetic_lambda(&[3.0, 1.0, 2.0], None, &graph, tau);
        assert!(
            lambda.is_finite(),
            "τ={tau}: λ must be finite, got {lambda}"
        );
        assert!(
            (0.0..=1.0).contains(&lambda),
            "τ={tau}: λ={lambda} outside [0,1]"
        );
    }
}

// -------------------- Valid domain: bits unchanged --------------------

#[test]
fn valid_domain_bits_unchanged() {
    // For τ ∈ (0, 1] the corrected rule must reproduce the original
    // formula bit for bit: s = τ and w = τ are the identity there.
    let graph = median_v1_laplacian();
    let item = [3.0, 1.0, 2.0];
    for tau in [TAU_FLOOR, 0.25, 0.5, 1.0] {
        let e_raw = TauMode::compute_rayleigh_quotient_from_matrix(&graph, &item);
        let g_raw = TauMode::compute_item_dispersion(&item, &graph);
        let oracle = tau * (e_raw / (e_raw + tau)) + (1.0 - tau) * g_raw.clamp(0.0, 1.0);
        let got = TauMode::compute_synthetic_lambda(&item, None, &graph, tau);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "τ={tau}: valid-domain bits changed"
        );
    }
}

#[test]
fn zero_tau_yields_dispersion() {
    // E_b = E/(E+0) = 1 for E > 0, weight 0 → λ = G exactly.
    let graph = median_v1_laplacian();
    let item = [3.0, 1.0, 2.0];
    let g = TauMode::compute_item_dispersion(&item, &graph).clamp(0.0, 1.0);
    let lambda = TauMode::compute_synthetic_lambda(&item, None, &graph, 0.0);
    assert_abs_diff_eq!(lambda, g, epsilon = 1e-12);
}

// -------------------- End-to-end: build path uses the same policy --------------------

#[test]
fn build_path_lambdas_stay_in_unit_interval_for_large_scale_data() {
    // Feature values an order of magnitude above 1 push the median (hence
    // τ) far above 1 — the exact regime where the old code produced
    // negative raw λ on the build path.
    let mut rng = rand::rngs::StdRng::seed_from_u64(3407);
    let rows: Vec<Vec<f64>> = (0..40)
        .map(|i| {
            let base = (i % 4) as f64;
            (0..8)
                .map(|_| (base + rng.random_range(0.0..1.0)) * 10.0)
                .collect()
        })
        .collect();
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_synthesis(TauMode::Median)
        .with_seed(3407)
        .build(rows);

    assert_eq!(aspace.lambdas().len(), 40);
    for (i, l) in aspace.lambdas().iter().enumerate() {
        assert!(
            l.is_finite() && (0.0..=1.0).contains(l),
            "item {i}: raw λ={l} outside [0,1]"
        );
    }
}

use rand::{RngExt, SeedableRng};
