use std::collections::{BTreeMap};

/// CSR Laplacian: off-diagonals are negative weights (-w_ij), diagonal stores degree (sum of incident positive weights).
#[derive(Clone, Debug)]
pub struct GraphLaplacian {
    pub rows: Vec<usize>, // CSR row pointer (len = nnodes+1)
    pub cols: Vec<usize>, // CSR column indices (len = nnz)
    pub vals: Vec<f64>,   // CSR values (len = nnz)
    pub nnodes: usize,
}

/// Graph factory: all construction ultimately uses the λτ-graph built from data.
///
/// High-level policy:
/// - The base graph for any pipeline is a λ-proximity Laplacian over items (columns),
///   derived from the provided data matrix (rows are feature signals).
/// - Ensembles vary λτ-graph parameters (k, eps) and/or overlay hypergraph operations.
pub struct GraphFactory;

impl GraphFactory {
    /// Build a λ-proximity Laplacian over items from item vectors (row-major):
    /// - items: Vec<Vec<f64>> where each inner Vec is an item with F features (N×F format).
    /// - eps: connect items i,j if |λ_i - λ_j| ≤ eps.
    /// - k: optional per-item neighbor cap after thresholding (keep smallest |Δλ|).
    /// - p: kernel exponent (>0) for weight: w_ij = 1 / (1 + (|Δλ|/σ)^p).
    /// - sigma_override: optional weight scale σ; default σ = eps (with floor 1e-12).
    ///
    /// Steps:
    /// 1) Build a temporary Laplacian over items from item vector similarities.
    /// 2) Transpose to feature-major format internally for Rayleigh computation.
    /// 3) For each feature f, compute Rayleigh λ_f = x_f^T L x_f / (x_f^T x_f).
    /// 4) Aggregate to per-item λ_i as weighted mean of λ_f with weights |x_f[i]|.
    /// 5) Build ε-graph on items using |λ_i - λ_j| with k-pruning and kernel weights.
    pub fn build_lambda_graph(
        items: &Vec<Vec<f64>>, // N×F: N items, each with F features
        eps: f64,
        k: usize,
        p: f64,
        sigma_override: Option<f64>,
    ) -> GraphLaplacian {
        let n_items = items.len();
        let n_features = items[0].len();
        assert!(
            n_items >= 2 && n_features >= 1,
            "graph should have at least two items and one feature"
        );
        for item in items {
            assert_eq!(
                item.len(),
                items[0].len(),
                "All items must have identical number of features"
            );
        }

        // 1) Build a temporary symmetric similarity graph over items using cosine similarity
        let tmp = GraphFactory::build_item_similarity_laplacian_from_items(items);

        // 2) Transpose items to feature-major format for Rayleigh computation
        // Convert from N×F (items×features) to F×N (features×items)
        let mut feature_rows = vec![vec![0.0; n_items]; n_features];
        for (item_idx, item_vec) in items.iter().enumerate() {
            for (feature_idx, &value) in item_vec.iter().enumerate() {
                feature_rows[feature_idx][item_idx] = value;
            }
        }

        // 3) Compute per-feature Rayleigh λ_f using the item similarity Laplacian
        let mut lambda_feature: Vec<f64> = Vec::with_capacity(n_features);
        for feature_row in &feature_rows {
            let den: f64 = feature_row.iter().map(|&v| v * v).sum();
            if den <= 0.0 {
                lambda_feature.push(0.0);
                continue;
            }

            let mut num = 0.0;
            for i in 0..n_items {
                let xi = feature_row[i];
                let start = tmp.rows[i];
                let end = tmp.rows[i + 1];
                let s: f64 = (start..end)
                    .map(|idx| tmp.vals[idx] * feature_row[tmp.cols[idx]])
                    .sum();
                num += xi * s;
            }
            lambda_feature.push(num / den);
        }

        // 4) Aggregate to per-item λ_i with weights |feature_value|
        let mut lambda_item = vec![0.0f64; n_items];
        let mut lambda_wsum = vec![0.0f64; n_items];
        for (feature_idx, &lambda_f) in lambda_feature.iter().enumerate() {
            for (item_idx, &feature_value) in feature_rows[feature_idx].iter().enumerate() {
                let w = feature_value.abs();
                lambda_item[item_idx] += lambda_f * w;
                lambda_wsum[item_idx] += w;
            }
        }

        for i in 0..n_items {
            lambda_item[i] = if lambda_wsum[i] > 0.0 {
                lambda_item[i] / lambda_wsum[i]
            } else {
                0.0
            };
        }

        // 5) Build ε-graph over items using |Δλ| (union-symmetrized)
        GraphFactory::build_lambda_eps_graph_from_items(&lambda_item, eps, k, p, sigma_override)
    }

    /// Build ε-graph over items from per-item λ values, with optional k-pruning and kernel weighting.
    fn build_lambda_eps_graph_from_items(
        lambdas: &[f64],
        eps: f64,
        k: usize,
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
        let cap = k;

        // Local row-wise neighbor candidates then symmetrize by union
        let mut adj = vec![BTreeMap::<usize, f64>::new(); n];

        for i in 0..n {
            let mut nbrs: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, (lambdas[i] - lambdas[j]).abs()))
                .filter(|&(_, d)| d <= eps)
                .collect();

            nbrs.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });

            if nbrs.len() > cap {
                nbrs.truncate(cap);
            }

            for (j, d) in nbrs {
                let w = 1.0 / (1.0 + (d / sigma).powf(p));
                if w > 0.0 {
                    *adj[i].entry(j).or_insert(0.0) += w;
                    // do not add reverse here; symmetrize after loop for union
                }
            }
        }

        // Symmetrize by union
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

        // Emit CSR deterministically by ascending column
        for i in 0..n {
            let mut degree = 0.0f64;
            for (&j, &w) in adj[i].iter() {
                if i == j || w <= 0.0 {
                    continue;
                }
                cols.push(j);
                vals.push(-w);
                degree += w;
            }
            cols.push(i);
            vals.push(degree);
            rows.push(cols.len());
        }

        GraphLaplacian {
            rows,
            cols,
            vals,
            nnodes: n,
        }
    }

    /// Build a temporary Laplacian over items using cosine similarity between item vectors.
    /// Input: items as N×F matrix (each row is an item vector)
    /// Output: N×N Laplacian with nodes = items
    fn build_item_similarity_laplacian_from_items(items: &Vec<Vec<f64>>) -> GraphLaplacian {
        let n_items = items.len();
        let _n_features = items[0].len();

        // Precompute norms for all items
        let norms: Vec<f64> = items
            .iter()
            .map(|item| (item.iter().map(|&x| x * x).sum::<f64>()).sqrt())
            .collect();

        let mut rows = Vec::with_capacity(n_items + 1);
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        rows.push(0);

        for i in 0..n_items {
            let mut degree = 0.0f64;
            for j in 0..n_items {
                if i == j {
                    continue;
                }

                // Compute cosine similarity between items i and j
                let denom = norms[i] * norms[j];
                let sim = if denom > 0.0 {
                    let dot: f64 = items[i]
                        .iter()
                        .zip(items[j].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    (dot / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                let w = sim.max(0.0); // keep only non-negative similarity
                if w > 0.0 {
                    cols.push(j);
                    vals.push(-w); // off-diagonal is -w_ij
                    degree += w;
                }
            }

            // diagonal entry (degree)
            cols.push(i);
            vals.push(degree);
            rows.push(cols.len());
        }

        GraphLaplacian {
            rows,
            cols,
            vals,
            nnodes: n_items,
        }
    }
}
