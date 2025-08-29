//! Lambda-proximity graph builder and Rayleigh quotient
//!
//! - Build symmetric ε-graph over nodes using λ proximity: |λ_i - λ_j| ≤ eps
//! - Optional per-node k-sparsification by smallest |Δλ|
//! - Weights from a monotone kernel of the λ-gap: w_ij = 1 / (1 + (|Δλ|/σ)^p)
//! - Laplacian: off-diagonals -w_ij, diagonal = degree
//! - Stable CSR with deterministic ordering by (|Δλ| asc, index asc)

use crate::graph_factory::GraphLaplacian;

/// Computes the Euclidean norm (L2) without allocating.
#[inline]
pub fn norm(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Rayleigh quotient x^T L x / x^T x for Laplacian L (CSR).
pub fn rayleigh_lambda(gl: &GraphLaplacian, x: &[f64]) -> f64 {
    assert!(!x.is_empty(), "vector cannot be empty");
    let den: f64 = x.iter().map(|&xi| xi * xi).sum();
    if den <= 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for i in 0..gl.nnodes {
        let xi = x[i];
        let start = gl.rows[i];
        let end = gl.rows[i + 1];
        let s: f64 = (start..end).map(|idx| gl.vals[idx] * x[gl.cols[idx]]).sum();
        num += xi * s;
    }
    num / den
}

/// Build symmetric λ-threshold graph (epsilon-graph) with optional k-sparsification.
/// - lambdas: per-node λ values of length n
/// - eps: edge if |λ_i - λ_j| ≤ eps
/// - k: optional cap of neighbors per node after thresholding (keep smallest |Δλ|)
/// - p: kernel exponent (>0) for weight: w_ij = 1 / (1 + (|Δλ|/σ)^p), with σ = eps or median gap
/// - sigma_override: if Some(σ), use this scale instead of eps.
///
/// Deterministic ordering by (|Δλ| asc, j asc).
pub fn build_lambda_graph(
    lambdas: &Vec<f64>,
    eps: f64,
    k: Option<usize>,
    p: f64,
    sigma_override: Option<f64>,
) -> GraphLaplacian {
    let n = lambdas.len();
    let mut rows: Vec<usize> = Vec::with_capacity(n + 1);
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    rows.push(0);

    if n == 0 {
        return GraphLaplacian {
            rows,
            cols,
            vals,
            nnodes: 0,
        };
    }

    let sigma = sigma_override.unwrap_or_else(|| eps.max(1e-12));
    let cap = k.unwrap_or(usize::MAX);

    // Collect candidate neighbors per node: within eps
    for i in 0..n {
        // Gather (j, |Δλ|) for j != i with |Δλ| ≤ eps
        let mut nbrs: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, (lambdas[i] - lambdas[j]).abs()))
            .filter(|&(_, d)| d <= eps)
            .collect();

        // Sort by (|Δλ| asc, j asc)
        nbrs.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if nbrs.len() > cap {
            nbrs.truncate(cap);
        }

        // Compute degree as sum of symmetric weights later; for now store off-diagonals
        // Weight kernel: w = 1 / (1 + (d/sigma)^p), d = |Δλ|
        let mut degree = 0.0f64;
        for &(j, d) in &nbrs {
            let w = 1.0 / (1.0 + (d / sigma).powf(p));
            cols.push(j);
            vals.push(-w);
            degree += w;
        }

        // Diagonal last
        cols.push(i);
        vals.push(degree);
        rows.push(cols.len());
    }

    // Symmetrize by union: ensure for any (i->j) there is corresponding (j->i) with same magnitude.
    // Since we emitted rows independently, we now enforce symmetry by reconstructing adjacency and re-emitting CSR.
    // Convert CSR to adjacency (positive weights) first.
    let mut adj = vec![std::collections::BTreeMap::<usize, f64>::new(); n];
    for i in 0..n {
        let (s, e) = (rows[i], rows[i + 1]);
        let mut _deg = 0.0;
        for idx in s..e {
            let j = cols[idx];
            let v = vals[idx];
            if j == i {
                // ignore diagonal on this pass; will recompute after symmetrization
                _deg = v;
            } else {
                let w = (-v).max(0.0);
                if w > 0.0 {
                    *adj[i].entry(j).or_insert(0.0) += w;
                }
            }
        }
    }
    // Symmetrize union
    for i in 0..n {
        let keys: Vec<_> = adj[i].keys().copied().collect();
        for j in keys {
            let w = *adj[i].get(&j).unwrap_or(&0.0);
            if w > 0.0 {
                let back = adj[j].entry(i).or_insert(0.0);
                if *back == 0.0 {
                    *back = w;
                }
            }
        }
    }

    // Re-emit CSR deterministically by ascending column
    let mut rows2 = Vec::with_capacity(n + 1);
    let mut cols2 = Vec::new();
    let mut vals2 = Vec::new();
    rows2.push(0);
    for i in 0..n {
        let mut degree = 0.0f64;
        for (&j, &w) in adj[i].iter() {
            if i == j || w <= 0.0 {
                continue;
            }
            cols2.push(j);
            vals2.push(-w);
            degree += w;
        }
        cols2.push(i);
        vals2.push(degree);
        rows2.push(cols2.len());
    }

    GraphLaplacian {
        rows: rows2,
        cols: cols2,
        vals: vals2,
        nnodes: n,
    }
}
