//! # Builds a graph Laplacian matrix from a collection of high-dimensional feature vectors

//!
//! ## Algorithm Overview
//!
//! 1. **Normalization** (optional): Normalizes each item to unit L2 norm if `params.normalise` is true
//! 2. **Similarity computation**: Computes cosine similarities between all pairs of items
//! 3. **k-NN graph construction**: For each item, retains only the k most similar neighbors within distance threshold `eps`
//! 4. **Weight assignment**: Applies kernel weighting: `w = 1 / (1 + (distance/sigma)^p)`
//! 5. **Symmetrization**: Makes the adjacency matrix symmetric by adding reverse edges
//! 6. **Laplacian construction**: Builds L = D - A where D is degree matrix and A is adjacency matrix
//!
//! ## Compute Laplacian Complexity
//! 1. **Build fastpair (CosinePair) structure**: `O(n × d × log n)`
//! 2. **k-NN queries**: `O(n × k × log n × d)`
//! - n queries (one per item)
//! - Each query returns k neighbors
//! - Each neighbor evaluation: `O(d)` for distance computation
//! - Tree traversal: `O(log n)` expected depth
//!
//! **Total: `O(n × d × log n + n × k × d × log n)` = `O(n × k × d × log n)`**
//!
//! ## Speedup Factor
//! Compared to `O(n_2)`: `n / (k × log n)`
//!
//! For typical values:
//! - **n = 10,000 items, k = 10 neighbors, d = 384 features**
//! - **Old**: 10,000² × 384 = **3.84 × 10¹⁰** operations
//! - **New**: 10,000 × 10 × 384 × log₂(10,000) ≈ **5.1 × 10⁷** operations
//! - **Speedup**: ~**750x faster!**

use crate::graph::{GraphLaplacian, GraphParams};

use smartcore::algorithm::neighbour::cosinepair::CosinePair;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::preprocessing::numerical::{StandardScaler, StandardScalerParameters};
use smartcore::api::{UnsupervisedEstimator, Transformer};

use sprs::{CsMat, TriMat};
use std::collections::BTreeMap;
use rayon::prelude::*;
use log::{debug, info, trace};

/// Builds a graph Laplacian matrix from a collection of high-dimensional vectors
///
/// This function constructs a k-nearest neighbor graph based on cosine similarity between items,
/// then converts it to a symmetric Laplacian matrix suitable for spectral analysis. The resulting
/// Laplacian encodes local similarity relationships and can be used for dimensionality reduction,
/// clustering, and spectral indexing.
///
/// # Parameters
///
/// * `items` - Vector of feature vectors, where `items[i]` is a d-dimensional vector representing item i.
///   All items must have identical dimensionality. Modified in-place if normalization is enabled.
/// * `params` - Graph construction parameters:
/// * `n_items`` of the original dataset (in case to need the computation of a second-order laplacian)
///
/// # Returns
///
/// `GraphLaplacian`
///
/// # Complexity
///
/// * **Time**: O(n × k × d × log n) where n = number of items, d = feature dimension
/// * **Space**: O(n × d + n × k) for input data and sparse output Laplacian matrix
///
/// The sparse representation makes this approach suitable for much larger datasets.
///
/// # Similarity Measure
///
/// Uses **rectified cosine distance**: `distance = 1 - max(0, cosine_similarity)`
/// * Cosine similarity ∈ [-1, 1] → Distance ∈ [0, 2]  
/// * Only non-negative similarities (distance ≤ 1) contribute to positive weights
/// * Items with negative cosine similarity are effectively disconnected
///
/// # Panics
///
/// * If `items` is empty or contains fewer than 2 items
/// * If items have inconsistent dimensions
/// * If any item has fewer than 2 features
/// * If matrix construction fails due to memory constraints
///
/// # Examples
///
/// ```
/// use arrowspace::laplacian::build_laplacian_matrix;
/// use arrowspace::graph::GraphParams;
/// use smartcore::linalg::basic::arrays::{Array, Array1, Array2};
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// // Create sample data: 4 items with 3 features each
/// let items = vec![
/// vec![1.0, 0.0, 0.0], // Item 0
/// vec![0.8, 0.6, 0.0], // Item 1 (similar to 0)
/// vec![0.0, 1.0, 0.0], // Item 2
/// vec![0.0, 0.0, 1.0], // Item 3
/// ];
///
/// let params = GraphParams {
/// eps: 0.5, // Accept neighbors with distance ≤ 0.5
/// k: 3, // At most 3 neighbors per item considered
/// topk: 3,
/// p: 2.0, // Quadratic kernel
/// sigma: Some(0.1), // Kernel bandwidth
/// normalise: true, // Normalize to unit vectors
/// };
///
/// let laplacian = build_laplacian_matrix(
///     DenseMatrix::from_2d_vec(&items).unwrap().transpose(), &params, None);
/// assert_eq!(laplacian.nnodes, 4);
/// assert_eq!(laplacian.matrix.shape(), (3, 3));
/// println!("{:?}", laplacian);
/// ```
///
/// # Performance Notes
///
/// * **Parallelization**: k-NN computation is parallelized across items using rayon
/// * **Memory usage**: Stores sparse n×n matrix using CSR format
/// * **Preprocessing**: Optional normalisation and norm precomputation minimise repeated calculations
pub fn build_laplacian_matrix(
    transposed: DenseMatrix<f64>,       // matrix to compute the Laplacian
    params: &GraphParams,          // requested params from the graph
    // n_items of the original dataset (in case to need the computation of L(FxN))
    n_items: Option<usize>,
) -> GraphLaplacian {
    let (d, n) = transposed.shape();
    assert!(n >= 2 && d >= 2, "items should be at least of shape (2,2): ({},{})", d, n);

    info!("Building Laplacian matrix for {} items with {} features", n, d);
    debug!(
        "Graph parameters: eps={}, k={}, p={}, sigma={:?}, normalise={}",
        params.eps, params.k, params.p, params.sigma, params.normalise
    );

    // Step 1: Conditional normalization based on params.normalise flag
    let mut items = if params.normalise {
        debug!("Normalizing items to unit norm");
        let scaler = StandardScaler::fit(&transposed, StandardScalerParameters::default()).unwrap();
        let scaled = scaler.transform(&transposed).unwrap();
        trace!("Items normalized successfully");
        scaled
    } else {
        debug!("Skipping normalization - using raw item magnitudes");
        transposed
    };

    let triplets = _main_laplacian(&mut items, params);

    let sparse_matrix: CsMat<f64> = triplets.to_csr();
    let graph_laplacian = GraphLaplacian { 
        matrix: sparse_matrix, 
        nnodes: match n_items {
            Some(n_items) => n_items,
            None => n
        },
        graph_params: params.clone() 
    };

    info!("Successfully built sparse Laplacian matrix ({}x{}) with {} non-zeros", 
          n, n, graph_laplacian.matrix.nnz());
    graph_laplacian
}

/// Laplacian main body
/// Provide the main steps of computation for Laplacian(items)
fn _main_laplacian(
    items: &mut DenseMatrix<f64>,
    params: &GraphParams
) -> sprs::TriMatBase<Vec<usize>, Vec<f64>> {
    let n = items.shape().0;

    // Step 2: Build CosinePair structure - O(n × d × log n)
    info!("Building CosinePair data structure");
    #[allow(clippy::unnecessary_mut_passed)]
    let fastpair = CosinePair::new(items).unwrap();
    debug!("CosinePair structure built for {} items", n);

    // Step 3: k-NN queries - O(n × k × d × log n)
    info!("Computing k-NN with CosinePair: k={}", params.topk + 1);
    let adj_rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let item: Vec<f64> =  items.get_row(i).iterator(0).copied().collect();
            let neighbors = &fastpair.query(&item, params.topk + 1).unwrap();
            neighbors
                .iter()
                .filter_map(|(distance, j)| {
                    if i != *j && *distance <= params.eps {
                        let weight = 1.0
                            / (1.0
                                + (distance / params.sigma.unwrap_or(1.0))
                                    .powf(params.p));
                        if weight > 1e-12 {
                            Some((*j, weight))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    debug!("Built adjacency rows for {} items", n);

    // Step 4: Symmetrise adjacency
    trace!("Symmetrizing adjacency matrix");

    // Use parallel collection first, then sequential symmetrization
    let all_edges: Vec<(usize, usize, f64)> = adj_rows
        .par_iter()
        .enumerate()
        .flat_map(|(i, row)| {
            // Convert to owned data for parallel processing
            row.par_iter().map(move |&(j, w)| (i, j, w)).collect::<Vec<_>>()
        })
        .collect();

    // Sequential symmetrization to avoid race conditions
    let mut edge_map: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();

    // Add all edges in both directions
    for (i, j, w) in all_edges {
        // Add edge i -> j
        edge_map.insert((i, j), w);
        // Add edge j -> i (symmetrization)
        edge_map.insert((j, i), w);
    }

    // Convert back to adjacency lists
    let mut sym: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for ((i, j), w) in edge_map {
        if i != j { // Skip diagonal entries for now
            sym[i].push((j, w));
        }
    }

    // Sort adjacency lists in parallel (this DOES work with rayon)
    sym.par_iter_mut().for_each(|row| {
        row.sort_unstable_by_key(|&(j, _)| j);
    });


    // Step 5: Build sparse Laplacian matrix L = D - A using triplet format
    info!("Converting adjacency to sparse Laplacian matrix");
    let mut triplets: sprs::TriMatBase<Vec<usize>, Vec<f64>> = TriMat::new((n, n));
    let mut total_edges = 0;

    for (i, s) in sym.iter().enumerate() {
        let degree: f64 = s.iter().map(|&(_j, w)| w).sum();
        triplets.add_triplet(i, i, degree);

        for &(j, w) in s {
            if i != j {
                triplets.add_triplet(i, j, -w);
                if i < j {
                    total_edges += 1;
                }
            }
        }
    }

    debug!("Laplacian matrix has {} total edges", total_edges);

    triplets
}

/// Alternative version that builds adjacency first, then converts to Laplacian
/// Useful for debugging or when you need access to the adjacency matrix
pub fn build_laplacian_matrix_with_adjacency(
    items: &[Vec<f64>],
    params: &GraphParams,
) -> (GraphLaplacian, CsMat<f64>) {
    let n_items = items.len();
    if n_items < 2 {
        panic!("Matrix too small")
    }

    info!("Building Laplacian with adjacency matrix output for {} items", n_items);

    let adjacency_matrix = build_adjacency_matrix(items, params);

    debug!("Converting adjacency to Laplacian matrix");
    let mut laplacian_triplets = TriMat::new((n_items, n_items));

    for i in 0..n_items {
        let degree: f64 = adjacency_matrix.outer_view(i).unwrap().iter().map(|(_, &w)| w).sum();
        laplacian_triplets.add_triplet(i, i, degree);

        for (j, &weight) in adjacency_matrix.outer_view(i).unwrap().iter() {
            if i != j {
                laplacian_triplets.add_triplet(i, j, -weight);
            }
        }
    }

    let laplacian_matrix = laplacian_triplets.to_csr();
    let graph_laplacian = GraphLaplacian {
        matrix: laplacian_matrix,
        nnodes: n_items,
        graph_params: params.clone(),
    };

    info!("Successfully built Laplacian with adjacency matrix");
    (graph_laplacian, adjacency_matrix)
}

/// Helper function to build just the adjacency matrix
fn build_adjacency_matrix(
    items: &[Vec<f64>],
    params: &GraphParams,
) -> CsMat<f64> {
    let n_items = items.len();
    debug!("Building adjacency matrix for {} items", n_items);

    let norms: Vec<f64> = items
        .iter()
        .map(|item| (item.iter().map(|&x| x * x).sum::<f64>()).sqrt())
        .collect();
    trace!("Precomputed norms for all items");

    let mut adj = vec![BTreeMap::<usize, f64>::new(); n_items];
    let sigma = params.sigma.unwrap_or_else(|| params.eps.max(1e-12));
    debug!("Using sigma={} for adjacency computation", sigma);

    for i in 0..n_items {
        let mut candidates: Vec<(usize, f64, f64)> = Vec::new();
        for j in 0..n_items {
            if i == j {
                continue;
            }

            let denom = norms[i] * norms[j];
            let cosine_sim = if denom > 1e-15 {
                let dot: f64 =
                    items[i].iter().zip(items[j].iter()).map(|(a, b)| a * b).sum();
                (dot / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let distance = 1.0 - cosine_sim.max(0.0);
            if distance <= params.eps {
                let normalized_dist = distance / sigma;
                let weight = 1.0 / (1.0 + normalized_dist.powf(params.p));
                if weight > 1e-15 {
                    candidates.push((j, distance, weight));
                }
            }
        }

        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let results_k: usize = params.topk;
        if candidates.len() > results_k {
            candidates.truncate(results_k);
        }

        if i % 50 == 0 {
            trace!(
                "Item {} has {} candidates within eps threshold",
                i,
                candidates.len()
            );
        }

        for (j, _dist, weight) in candidates {
            adj[i].insert(j, weight);
        }
    }

    trace!("Symmetrizing adjacency matrix");
    for i in 0..n_items {
        let keys: Vec<_> = adj[i].keys().copied().collect();
        for j in keys {
            let w = *adj[i].get(&j).unwrap_or(&0.0);
            if w > 1e-15 {
                let back_entry = adj[j].entry(i).or_insert(0.0);
                if *back_entry < 1e-15 {
                    *back_entry = w;
                }
            }
        }
    }

    trace!("Converting to sparse CSR format");
    let mut triplets = TriMat::new((n_items, n_items));
    for (i, ad) in adj.iter().enumerate() {
        for (&j, &weight) in ad.iter() {
            if i != j && weight > 1e-15 {
                triplets.add_triplet(i, j, weight);
            }
        }
    }

    let adjacency_matrix = triplets.to_csr();
    debug!("Successfully built sparse adjacency matrix with {} non-zeros", adjacency_matrix.nnz());
    adjacency_matrix
}
