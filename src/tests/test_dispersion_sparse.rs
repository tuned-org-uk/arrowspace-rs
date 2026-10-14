//! #200 — sparse directed-edge dispersion must match the dense lookup loop
//! bit for bit.
//!
//! `oracle_dense_dispersion` reproduces the pre-#200 implementation
//! verbatim: every (i, j) feature pair visited in lexicographic order, each
//! adjacency read through `CsMat::get`. The shipped
//! `TauMode::compute_item_dispersion` must return the identical f64 bits on
//! every fixture — the sparse traversal may only change how stored entries
//! are reached, never which terms accumulate in which order.

use crate::search::taumode::TauMode;
use rand::{RngExt, SeedableRng};
use sprs::CsMat;
use sprs::TriMat;

const SEED: u64 = 3407;

// -------------------- Sort-based reference oracle --------------------

/// Pre-#200 implementation, kept verbatim as the oracle.
fn oracle_dense_dispersion(item_vector: &[f64], spectrum: &CsMat<f64>) -> f64 {
    let n_features = item_vector.len();

    let mut edge_energy_sum = 0.0;
    for i in 0..n_features {
        let xi = item_vector[i];
        for (j, &item_j) in item_vector.iter().enumerate() {
            if i != j {
                let lij = spectrum.get(i, j).copied().unwrap_or(0.0);
                let w = (-lij).max(0.0);
                if w > 0.0 {
                    let d = xi - item_j;
                    edge_energy_sum += w * d * d;
                }
            }
        }
    }

    if edge_energy_sum <= 1e-12 {
        return 0.0;
    }

    let mut g_sq_sum = 0.0;
    for i in 0..n_features {
        let xi = item_vector[i];
        for (j, &item_j) in item_vector.iter().enumerate() {
            if i != j {
                let lij = spectrum.get(i, j).copied().unwrap_or(0.0);
                let w = (-lij).max(0.0);
                if w > 0.0 {
                    let d = xi - item_j;
                    let contrib = w * d * d;
                    let share = contrib / edge_energy_sum;
                    g_sq_sum += share * share;
                }
            }
        }
    }

    g_sq_sum.clamp(0.0, 1.0)
}

fn assert_parity(item: &[f64], spectrum: &CsMat<f64>, label: &str) {
    let got = TauMode::compute_item_dispersion(item, spectrum).to_bits();
    let want = oracle_dense_dispersion(item, spectrum).to_bits();
    assert_eq!(
        got, want,
        "{label}: dispersion bits differ from dense oracle"
    );
}

fn rng() -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(SEED)
}

/// Seeded item vector with a mix of magnitudes.
fn item(n: usize) -> Vec<f64> {
    let mut rng = rng();
    (0..n).map(|_| rng.random_range(-2.0..2.0)).collect()
}

// -------------------- Fixture builders --------------------

/// D − A shaped graph: negative symmetric off-diagonals, positive diagonal,
/// some rows left empty (isolated nodes).
fn laplacian_like(n: usize, degree: usize) -> CsMat<f64> {
    let mut rng = rng();
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, rng.random_range(1.0..5.0));
    }
    for i in 0..n {
        for _ in 0..degree {
            let j = rng.random_range(0..n);
            if j != i {
                let w = rng.random_range(0.1..1.0);
                tri.add_triplet(i, j, -w);
                tri.add_triplet(j, i, -w);
            }
        }
    }
    tri.to_csr()
}

// -------------------- Parity tests --------------------

#[test]
fn parity_laplacian_like_odd_features() {
    let n = 17;
    let gl = laplacian_like(n, 4);
    assert_parity(&item(n), &gl, "laplacian-like F=17");
}

#[test]
fn parity_laplacian_like_even_features() {
    let n = 16;
    let gl = laplacian_like(n, 6);
    assert_parity(&item(n), &gl, "laplacian-like F=16");
}

#[test]
fn parity_asymmetric_random_signs() {
    let n = 23;
    let mut rng = rng();
    let mut tri = TriMat::new((n, n));
    for _ in 0..(n * 3) {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let v = rng.random_range(-3.0..3.0);
        tri.add_triplet(i, j, v);
    }
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "asymmetric mixed-sign");
}

#[test]
fn parity_stored_zeros_and_positive_off_diagonals() {
    // Explicit zero entries (get -> Some(0.0), w == 0, skipped) mixed with
    // positive off-diagonals (also skipped: w = (-x).max(0.0) == 0).
    let n = 12;
    let mut rng = rng();
    let mut tri = TriMat::new((n, n));
    for _ in 0..(n * 2) {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        tri.add_triplet(i, j, 0.0);
        tri.add_triplet(j, i, rng.random_range(0.5..2.0)); // positive -> no edge
    }
    // a few real edges so the energy is non-zero
    tri.add_triplet(0, 1, -0.5);
    tri.add_triplet(1, 0, -0.7); // asymmetric weights on purpose
    tri.add_triplet(3, 8, -1.25);
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "stored zeros + positive off-diagonals");
}

#[test]
fn parity_diagonal_only_graph() {
    let n = 9;
    let mut rng = rng();
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, rng.random_range(1.0..4.0));
    }
    let mat = tri.to_csr();
    // i == j is excluded: total energy is zero -> early 0.0 return.
    assert_parity(&item(n), &mat, "diagonal-only");
    assert_eq!(TauMode::compute_item_dispersion(&item(n), &mat), 0.0);
}

#[test]
fn parity_empty_graph() {
    let n = 8;
    let mat: CsMat<f64> = TriMat::new((n, n)).to_csr();
    assert_parity(&item(n), &mat, "empty graph");
    assert_eq!(TauMode::compute_item_dispersion(&item(n), &mat), 0.0);
}

#[test]
fn parity_isolated_nodes() {
    // laplacian_like can leave rows empty; force several by degree 0 rows.
    let n = 14;
    let mut rng = rng();
    let mut tri = TriMat::new((n, n));
    tri.add_triplet(0, 1, -1.0);
    tri.add_triplet(1, 0, -1.0);
    tri.add_triplet(2, 3, -2.0);
    tri.add_triplet(3, 2, -0.5);
    for i in 4..n {
        tri.add_triplet(i, i, rng.random_range(1.0..2.0));
    }
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "isolated nodes");
}

#[test]
fn parity_energy_just_below_and_above_threshold() {
    // Two edges with tiny w: w*d*d terms land near the 1e-12 early-return.
    let n = 6;
    let mut tri = TriMat::new((n, n));
    tri.add_triplet(0, 1, -1e-7);
    tri.add_triplet(1, 0, -1e-7);
    let tiny = tri.to_csr();
    let x = vec![1.0, 1.0001, 2.0, 3.0, 4.0, 5.0];
    assert_parity(&x, &tiny, "energy below threshold");
    assert_eq!(TauMode::compute_item_dispersion(&x, &tiny), 0.0);

    let mut tri = TriMat::new((n, n));
    tri.add_triplet(0, 1, -1e-3);
    tri.add_triplet(1, 0, -1e-3);
    tri.add_triplet(2, 3, -4e-4);
    let small = tri.to_csr();
    assert_parity(&x, &small, "energy just above threshold");
    assert!(TauMode::compute_item_dispersion(&x, &small) > 0.0);
}

#[test]
fn parity_infinite_edge_weights() {
    // lij = -inf -> w = +inf: energy = inf, shares become inf/inf = NaN.
    // Both paths must produce the identical NaN bits.
    let n = 5;
    let mut tri = TriMat::new((n, n));
    tri.add_triplet(0, 1, f64::NEG_INFINITY);
    tri.add_triplet(1, 0, f64::NEG_INFINITY);
    tri.add_triplet(2, 3, f64::NEG_INFINITY);
    tri.add_triplet(3, 2, f64::NEG_INFINITY);
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "infinite weights");
}

#[test]
fn parity_nan_edge_entries_are_skipped() {
    // (-NaN).max(0.0) == 0.0 per f64::max NaN handling: NaN entries are
    // silent no-edges on both paths.
    let n = 7;
    let mut tri = TriMat::new((n, n));
    tri.add_triplet(0, 1, f64::NAN);
    tri.add_triplet(1, 0, f64::NAN);
    tri.add_triplet(2, 4, -0.75);
    tri.add_triplet(4, 2, -0.75);
    tri.add_triplet(5, 6, -1.5);
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "NaN entries");
}

#[test]
fn parity_spectrum_larger_than_item() {
    // Only the leading n_features × n_features block may contribute.
    let n = 10;
    let mut rng = rng();
    let mut tri = TriMat::new((24, 24));
    for _ in 0..60 {
        let i = rng.random_range(0..24);
        let j = rng.random_range(0..24);
        tri.add_triplet(i, j, rng.random_range(-2.0..2.0));
    }
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "24x24 spectrum, 10-feature item");
}

#[test]
fn parity_spectrum_smaller_than_item() {
    // get() out of range returns None -> treated as 0.0 by the oracle;
    // the sparse path must agree (missing entries contribute nothing).
    let n = 10;
    let mut rng = rng();
    let mut tri = TriMat::new((6, 6));
    for _ in 0..14 {
        let i = rng.random_range(0..6);
        let j = rng.random_range(0..6);
        tri.add_triplet(i, j, rng.random_range(-2.0..2.0));
    }
    let mat = tri.to_csr();
    assert_parity(&item(n), &mat, "6x6 spectrum, 10-feature item");
}

#[test]
fn parity_csc_storage() {
    // CSC input takes whatever path the implementation chooses — the result
    // must still match the get()-based oracle, which handles both formats.
    let n = 13;
    let csc = laplacian_like(n, 3).to_csc();
    assert_parity(&item(n), &csc, "CSC storage");
}

#[test]
fn parity_item_with_nonfinite_values() {
    // inf - inf = NaN inside d; NaN weights of the item propagate
    // identically through both paths.
    let n = 9;
    let mat = laplacian_like(n, 3);
    let mut x = item(n);
    x[2] = f64::INFINITY;
    x[5] = f64::NEG_INFINITY;
    x[7] = f64::NAN;
    assert_parity(&x, &mat, "non-finite item values");
}

#[test]
fn parity_larger_seeded_fixture() {
    let n = 40;
    let mat = laplacian_like(n, 8);
    assert_parity(&item(n), &mat, "40x40 seeded");
}

#[test]
fn parity_real_builder_graph() {
    // End-to-end: an actual builder-produced Laplacian, item projected the
    // same way the read-out does (identity projection here).
    use crate::builder::ArrowSpaceBuilder;
    let mut rng = rng();
    let rows: Vec<Vec<f64>> = (0..30)
        .map(|_| (0..12).map(|_| rng.random_range(0.0..1.0)).collect())
        .collect();
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.5, 6, 3, 2.0, None)
        .with_seed(SEED)
        .build(rows);
    assert_parity(&item(gl.shape().1), &gl.matrix, "builder-produced gl");
}
