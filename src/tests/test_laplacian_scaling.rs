//! Pins the wall-clock cost of `build_laplacian_matrix` (issue #141).
//!
//! `_build_adjacency` only consumes `CosinePair::query_row_top_k`, which
//! recomputes distances from `samples` and row norms. The eager
//! `CosinePair::with_top_k` constructor additionally runs a Theta(n^2)
//! `init()` pair scan whose outputs (`distances`, `neighbours`) are never
//! read - pure waste that dominated builds beyond ~10k items.
//!
//! The budget test below is release-gated: debug-build timings carry no
//! signal. The brute-force oracle next to it is profile-independent and
//! guards edge semantics across implementation changes.

use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::tests::init;
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Deterministic xorshift* generator, seed fixed by convention (3407).
struct XorShift(u64);

impl XorShift {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        ((x >> 11) as f64) / (1u64 << 53) as f64
    }

    /// `n` unit-normalised vectors in R^d drawn from `n_clusters` random
    /// directions with small intra-cluster jitter, mirroring the seeded
    /// latent-basin generator used to report issue #141.
    fn latent_basins(
        n: usize,
        d: usize,
        n_clusters: usize,
        jitter: f64,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let mut rng = XorShift(seed);
        let centroids: Vec<Vec<f64>> = (0..n_clusters)
            .map(|_| (0..d).map(|_| rng.next_f64() - 0.5).collect())
            .collect();

        (0..n)
            .map(|i| {
                let c = &centroids[i % n_clusters];
                let mut v: Vec<f64> = c
                    .iter()
                    .map(|&x| x + jitter * (rng.next_f64() - 0.5))
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect()
    }
}

fn params(eps: f64, topk: usize) -> GraphParams {
    GraphParams {
        eps,
        k: 10,
        topk,
        p: 2.0,
        sigma: Some(0.1),
        normalise: true,
        sparsity_check: false,
    }
}

fn laplacian_from_items(items: &[Vec<f64>], p: &GraphParams) -> GraphLaplacian {
    // build_laplacian_matrix graphs the ROWS of the matrix it receives
    // (callers pass F x N when features are the nodes); passing item rows
    // directly makes items the nodes.
    let m = DenseMatrix::<f64>::from_2d_vec(&items.to_vec()).unwrap();
    build_laplacian_matrix(m, p, Some(items.len()), false)
}

/// Issue #141 reproduction: at this shape the eager Theta(n^2) init scan
/// alone costs multiple seconds on the reference host; the lazy path plus a
/// single merged kNN pass finishes far below the budget. Failure here means
/// an all-pairs precomputation crept back into the adjacency builder.
#[test]
#[cfg(not(debug_assertions))]
fn laplacian_build_at_2000x256_stays_under_quadratic_init_budget() {
    use std::time::{Duration, Instant};

    init();

    let items = XorShift::latent_basins(2000, 256, 40, 0.05, 3407);
    let p = params(0.5, 8);

    let start = Instant::now();
    let gl = laplacian_from_items(&items, &p);
    let elapsed = start.elapsed();

    assert_eq!(gl.nnodes, 2000);
    assert!(
        gl.matrix.nnz() > 2000,
        "degenerate graph: {} nnz on 2000 nodes, eps/sigma misconfigured",
        gl.matrix.nnz()
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "laplacian build took {:?} for 2000x256; all-pairs CosinePair init is back",
        elapsed
    );
}

/// Oracle independent of smartcore: expected Laplacian entries computed by
/// brute-force cosine kNN over the raw items. Guards edge semantics of the
/// builder across the CosinePair lazy switch and any future rewrite.
#[test]
fn laplacian_edges_match_brute_force_cosine_knn_oracle() {
    init();

    let items = XorShift::latent_basins(60, 12, 8, 0.05, 3407);
    let eps = 0.4;
    let topk = 4;
    let p = params(eps, topk);
    // normalise=true routes items through StandardScaler before graph build;
    // use raw magnitudes so the oracle sees exactly what the graph saw.
    let p_raw = GraphParams {
        normalise: false,
        ..p
    };

    let gl = laplacian_from_items(&items, &p_raw);

    let n = items.len();
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let dist = |a: usize, b: usize| -> f64 {
        let (na, nb) = (
            dot(&items[a], &items[a]).sqrt(),
            dot(&items[b], &items[b]).sqrt(),
        );
        if na == 0.0 || nb == 0.0 {
            return f64::MAX;
        }
        1.0 - dot(&items[a], &items[b]) / (na * nb)
    };

    let sigma = p_raw.sigma.unwrap_or(1.0);
    let weight = |dist: f64| 1.0 / (1.0 + (dist / sigma).powf(p_raw.p));

    // Directed candidate edges per row after eps filtering.
    let directed: Vec<std::collections::HashSet<usize>> = (0..n)
        .map(|i| {
            let mut knn: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, dist(i, j)))
                .collect();
            // query_row_top_k(topk+1) already EXCLUDES the self row and
            // returns topk+1 real neighbours; the builder filters by eps.
            knn.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            knn.truncate(topk + 1);
            knn.into_iter()
                .filter(|&(_, dij)| dij <= eps && weight(dij) > 1e-12)
                .map(|(j, _)| j)
                .collect()
        })
        .collect();

    // L = D - A with A symmetrised by max-weight: an off-diagonal entry
    // i->j exists when EITHER direction proposed the edge.
    let symmetric: Vec<std::collections::HashSet<usize>> = (0..n)
        .map(|i| {
            let mut s: std::collections::HashSet<usize> = directed[i].clone();
            for (src, row) in directed.iter().enumerate() {
                if row.contains(&i) {
                    s.insert(src);
                }
            }
            s
        })
        .collect();

    for (i, expected) in symmetric.iter().enumerate() {
        for &j in expected {
            assert!(gl.get(i, j) < 0.0, "expected edge {i}->{j} missing from L");
        }
        for j in 0..n {
            if i == j || expected.contains(&j) {
                continue;
            }
            assert!(gl.get(i, j) >= 0.0, "unexpected edge {i}->{j} present in L");
        }
    }
}
