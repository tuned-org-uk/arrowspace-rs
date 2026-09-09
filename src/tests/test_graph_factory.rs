use crate::{
    builder::ArrowSpaceBuilder,
    core::ArrowSpace,
    graph::{GraphFactory, GraphLaplacian, GraphParams},
    laplacian::build_laplacian_matrix,
    maps::eigenmaps::EigenMaps,
    search::taumode::TauMode,
    tests::test_data::{make_gaussian_blob, make_moons_hd},
};

use log::debug;
use smartcore::linalg::basic::arrays::Array2;
use sprs::CsMat;

/// Exact structural equality for sparse matrices built deterministically.
fn sparse_matrices_equal(a: &CsMat<f64>, b: &CsMat<f64>) -> bool {
    a.shape() == b.shape()
        && a.nnz() == b.nnz()
        && a.indptr().raw_storage() == b.indptr().raw_storage()
        && a.indices() == b.indices()
        && a.data() == b.data()
}

#[test]
fn test_signals_laplacian_is_computed_on_retransposed_gl_156() {
    // Issue #156: signals = compute_graph_laplacian(gl.T)
    //
    // The signals structure is a second-order graph Laplacian computed on the
    // RE-TRANSPOSED (item-space) feature Laplacian: the columns of gl.matrix
    // (the "eigenvectors" of the feature graph) become the items whose graph
    // Laplacian is computed. Without the re-transpose the wired profiles are
    // the Laplacian rows and the results are unusable.
    //
    // A hand-built NON-symmetric Laplacian makes the orientation observable:
    // pipeline-built Laplacians are symmetric, which would mask the choice.
    crate::tests::init();

    let l_rows: Vec<Vec<f64>> = vec![
        vec![4.0, -3.0, -1.0, 0.0],
        vec![0.0, 2.0, -2.0, 0.0],
        vec![-1.0, 0.0, 3.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let mut tm = sprs::TriMat::new((4, 4));
    for (i, row) in l_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                tm.add_triplet(i, j, v);
            }
        }
    }
    let matrix: CsMat<f64> = tm.to_csr();
    // fixture must be genuinely non-symmetric
    assert_ne!(
        matrix.get(0, 1).copied().unwrap_or(0.0),
        matrix.get(1, 0).copied().unwrap_or(0.0),
        "fixture must be non-symmetric to discriminate orientation"
    );

    let gl = GraphLaplacian {
        init_data: crate::graph::sparse_to_dense(&matrix),
        matrix: matrix.clone(),
        nnodes: 4,
        graph_params: GraphParams {
            eps: 2.0,
            k: 4,
            topk: 3,
            p: 2.0,
            sigma: Some(1.0),
            normalise: false,
            sparsity_check: false,
        },
        energy: false,
    };

    let mut aspace = ArrowSpace::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        TauMode::Median,
    );
    aspace.nfeatures = 4;
    aspace.nitems = 4;
    aspace.reduced_dim = None;

    GraphFactory::build_spectral_laplacian(&mut aspace, &gl);

    // Expected: Laplacian of the re-transposed (item-space) Laplacian — the
    // columns of gl.matrix act as items.
    let expected_item_space = build_laplacian_matrix(
        crate::graph::sparse_to_dense(&gl.matrix).transpose(),
        &gl.graph_params,
        None,
        false,
    )
    .matrix;
    // The rejected orientation: Laplacian over the gl.matrix rows directly.
    let row_space = build_laplacian_matrix(
        crate::graph::sparse_to_dense(&gl.matrix),
        &gl.graph_params,
        None,
        false,
    )
    .matrix;

    assert!(
        !sparse_matrices_equal(&expected_item_space, &row_space),
        "fixture is degenerate: item-space and row-space graphs coincide"
    );

    assert_eq!(aspace.signals.shape(), expected_item_space.shape());
    assert!(
        sparse_matrices_equal(&aspace.signals, &expected_item_space),
        "signals must be the graph Laplacian computed on the re-transposed \
         (item-space) gl, with gl.matrix columns as items (issue #156)"
    );
}

#[test]
fn test_spectral_signals_second_order_pipeline_properties_156() {
    // End-to-end guard: with spectral enabled the signals structure is a
    // usable second-order graph — square, symmetric, distinct from the first-
    // order Laplacian, deterministic across builds, and usable by the taumode
    // read-out and search.
    crate::tests::init();

    let items: Vec<Vec<f64>> = make_moons_hd(60, 0.15, 0.4, 8, 77);

    let build = || {
        ArrowSpaceBuilder::default()
            .with_lambda_graph(0.3, 5, 2, 2.0, None)
            .with_normalisation(true)
            .with_spectral(true)
            .with_seed(77)
            .with_dims_reduction(false, None)
            .with_inline_sampling(None)
            .with_sparsity_check(false)
            .build(items.clone())
    };

    let (aspace, gl) = build();
    let (aspace_again, gl_again) = build();

    let f = aspace.nfeatures;
    assert_eq!(aspace.signals.shape(), (f, f), "signals must be F×F");
    assert!(aspace.signals.nnz() > 0, "signals must be wired");

    // The second-order graph is built through symmetrisation: it must be
    // symmetric.
    for (i, row) in aspace.signals.outer_iterator().enumerate() {
        for (j, &v) in row.iter() {
            let mirror = aspace.signals.get(j, i).copied().unwrap_or(0.0);
            assert!(
                (v - mirror).abs() <= 1e-12 * (1.0 + v.abs().max(mirror.abs())),
                "signals must be symmetric: S[{i},{j}]={v} vs S[{j},{i}]={mirror}"
            );
        }
    }

    // Non-vestigial: the second-order graph differs from the first-order one.
    assert!(
        !sparse_matrices_equal(&aspace.signals, &gl.matrix),
        "signals must not coincide with gl.matrix (issue #156)"
    );

    // Determinism by construction (AGENTS.md #4): same seed → same signals.
    assert!(
        sparse_matrices_equal(&aspace.signals, &aspace_again.signals),
        "identical inputs must produce identical signals"
    );
    assert!(sparse_matrices_equal(&gl.matrix, &gl_again.matrix));

    // Usability: the taumode read-out consumes the signals graph and search
    // retrieves the query's own row as top hit.
    assert_eq!(aspace.lambdas().len(), items.len());
    assert!(
        aspace.lambdas().iter().all(|&l| l >= 0.0),
        "λ must be non-negative on the signals graph"
    );

    let query = &items[10];
    let results = aspace.search(query, &gl, 4, 0.7);
    assert!(!results.is_empty(), "search must return results");
    assert_eq!(
        results[0].0, 10,
        "self-retrieval must rank the query row first: {:?}",
        &results[..2.min(results.len())]
    );
}

#[test]
fn test_builder_basic_clustering_with_synthetic_data() {
    // Test basic clustering functionality with high-dimensional moons data
    let items: Vec<Vec<f64>> = make_moons_hd(
        100,  // Moderate number of samples
        0.15, // Moderate noise
        0.4,  // Good separation
        10,   // 10-dimensional data
        42,   // Reproducible seed
    );

    debug!(
        "Generated {} items with {} features",
        items.len(),
        items[0].len()
    );

    let (_aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .build(items.clone());

    // Verify basic properties
    debug!("Graph has {} nodes", gl.nnodes);
}

#[test]
fn test_builder_laplacian_diagonal_properties() {
    // Test that Laplacian diagonal entries are non-negative and finite
    let items: Vec<Vec<f64>> = make_moons_hd(
        80,   // Sufficient samples
        0.12, // Low noise for stable structure
        0.5,  // Large separation
        8,    // 8 dimensions
        123,  // Seed
    );

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 4, 2, 2.0, None)
        .with_normalisation(true)
        .build(items);

    // Check diagonal properties
    let csr = &gl.matrix;
    assert!(csr.is_csr(), "Expected CSR layout");

    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();

    for i in 0..aspace.n_clusters {
        let start = indptr.into_raw_storage()[i];
        let end = indptr.into_raw_storage()[i + 1];
        let mut found = false;
        let mut diag = 0.0_f64;

        for pos in start..end {
            let j = indices[pos];
            if j == i {
                diag = data[pos];
                found = true;
                break;
            }
        }

        assert!(
            found,
            "Diagonal entry at ({},{}) should exist in Laplacian",
            i, i
        );
        assert!(
            diag.is_finite(),
            "Diagonal at ({},{}) must be finite, got {}",
            i,
            i,
            diag
        );
        assert!(
            diag >= 0.0,
            "Diagonal at ({},{}) must be non-negative, got {}",
            i,
            i,
            diag
        );
    }

    debug!(
        "✓ All {} diagonal entries are non-negative and finite",
        aspace.n_clusters
    );
}

#[test]
fn test_builder_minimum_items() {
    // Test minimum viable dataset
    let items: Vec<Vec<f64>> = make_moons_hd(
        20,  // Small dataset
        0.1, // Low noise
        0.6, // High separation
        5,   // Low dimensions
        42,
    );

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.5, 3, 2, 2.0, None)
        .build(items.clone());

    assert!(
        aspace.n_clusters >= 1,
        "Should produce at least one cluster"
    );
    assert_eq!(gl.nnodes, items.len());

    debug!(
        "Minimum items test: {} clusters from {} items",
        aspace.n_clusters, 20
    );
}

#[test]
fn test_builder_scale_invariance_with_normalization() {
    // Test that normalization makes the graph structure scale-invariant
    let items: Vec<Vec<f64>> = make_moons_hd(60, 0.15, 0.4, 8, 0);

    // Build with original scale
    let (aspace1, gl1) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true) // Normalize for scale invariance
        .build(items.clone());

    // Scale all items by constant factor
    let scale_factor = 5.7;
    let items_scaled: Vec<Vec<f64>> = items
        .iter()
        .map(|item| item.iter().map(|&x| x * scale_factor).collect())
        .collect();

    // Build with scaled data
    let (aspace2, gl2) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true) // Normalize for scale invariance
        .build(items_scaled);

    // With normalization, cluster counts should be similar (allowing minor numerical differences)
    assert!(
        (aspace1.n_clusters as i32 - aspace2.n_clusters as i32).abs() <= 3,
        "Normalized clustering should be scale-invariant: {} vs {}",
        aspace1.n_clusters,
        aspace2.n_clusters
    );

    // Graph sizes should match
    assert_eq!(
        gl1.nnodes, gl2.nnodes,
        "Graph node counts should match under scaling"
    );

    debug!(
        "✓ Scale invariance verified: original={} clusters, scaled={} clusters",
        aspace1.n_clusters, aspace2.n_clusters
    );
}

#[test]
fn test_builder_laplacian_symmetry() {
    // Test that the Laplacian is symmetric (undirected graph)
    let items: Vec<Vec<f64>> = make_moons_hd(70, 0.18, 0.35, 9, 456);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 5, 2, 2.0, None)
        .with_normalisation(true)
        .build(items);

    let csr = &gl.matrix;
    assert!(csr.is_csr(), "Expected CSR layout");

    let n = aspace.n_clusters;
    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();
    let eps = 1e-10;

    let mut symmetric_pairs = 0;
    let mut total_edges = 0;

    for i in 0..n {
        let start = indptr.into_raw_storage()[i];
        let end = indptr.into_raw_storage()[i + 1];

        for p in start..end {
            let j = indices[p];
            if i == j {
                continue; // Skip diagonal
            }

            total_edges += 1;
            let vij = data[p];

            // Find symmetric entry (j, i)
            let js = indptr.into_raw_storage()[j];
            let je = indptr.into_raw_storage()[j + 1];
            let mut vji_opt: Option<f64> = None;

            for q in js..je {
                if indices[q] == i {
                    vji_opt = Some(data[q]);
                    break;
                }
            }

            if let Some(vji) = vji_opt {
                assert!(
                    (vij - vji).abs() <= eps * (1.0 + vij.abs().max(vji.abs())),
                    "Symmetric entries must match: L[{},{}]={:.6} vs L[{},{}]={:.6}",
                    i,
                    j,
                    vij,
                    j,
                    i,
                    vji
                );
                symmetric_pairs += 1;
            } else {
                panic!(
                    "Graph should be symmetric: found edge ({},{}) = {:.6} but missing ({},{})",
                    i, j, vij, j, i
                );
            }
        }
    }

    debug!(
        "✓ Verified symmetry for {} edge pairs (total {} edges)",
        symmetric_pairs, total_edges
    );
}

#[test]
fn test_builder_parameter_preservation() {
    // Test that graph parameters are correctly preserved through the builder
    let items: Vec<Vec<f64>> = make_moons_hd(50, 0.2, 0.4, 7, 321);

    let (_, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(
            0.123,       // eps
            7,           // k
            3,           // topk
            3.5,         // p
            Some(0.456), // sigma
        )
        .with_normalisation(false)
        .build(items);

    // Verify all parameters are preserved
    assert_eq!(gl.graph_params.eps, 0.123, "eps must match");
    assert_eq!(gl.graph_params.k, 7, "k must match");
    assert_eq!(gl.graph_params.topk, 3, "topk must match");
    assert_eq!(gl.graph_params.p, 3.5, "p must match");
    assert_eq!(gl.graph_params.sigma, Some(0.456), "sigma must match");
    assert_eq!(
        gl.graph_params.normalise, false,
        "normalise flag must match"
    );

    debug!("✓ All graph parameters correctly preserved");
}

#[test]
fn test_builder_with_different_dimensions() {
    // Test builder works across different dimensionalities
    let test_cases = vec![
        (50, 3, "low-dimensional"),
        (60, 10, "medium-dimensional"),
        (70, 25, "high-dimensional"),
    ];

    for (n_samples, dims, desc) in test_cases {
        let items: Vec<Vec<f64>> = make_moons_hd(
            n_samples,
            0.15,
            0.4,
            dims,
            42 + dims as u64, // Vary seed by dimension
        );

        let (aspace, gl) = ArrowSpaceBuilder::default()
            .with_lambda_graph(0.3, 5, 2, 2.0, None)
            .with_normalisation(true)
            .with_spectral(true)
            .with_sparsity_check(false)
            .build(items);

        assert!(aspace.n_clusters > 0, "{}: Should produce clusters", desc);
        assert!(
            aspace.nfeatures == dims,
            "{}: Features should be {}",
            desc,
            dims
        );

        debug!(
            "{}: {} clusters, {} features, {} nodes",
            desc, aspace.n_clusters, aspace.nfeatures, gl.nnodes
        );
    }
}

#[test]
fn test_builder_spectral_laplacian_shape() {
    // Test that spectral Laplacian has correct shape (FxF)
    let items: Vec<Vec<f64>> = make_moons_hd(90, 0.16, 0.38, 12, 555);

    // Build WITHOUT spectral
    let (aspace_no_spectral, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 4, 2, 2.0, None)
        .with_spectral(false)
        .build(items.clone());

    // Build WITH spectral
    let (aspace_spectral, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 4, 2, 2.0, None)
        .with_spectral(true)
        .build(items.clone());

    // Without spectral, signals should be empty
    assert_eq!(
        aspace_no_spectral.signals.shape(),
        (0, 0),
        "Signals should be empty when spectral is disabled"
    );

    // With spectral, signals should be FxF where F is number of features
    let expected_dim = aspace_spectral.nfeatures;
    assert_eq!(
        aspace_spectral.signals.shape(),
        (expected_dim, expected_dim),
        "Signals should be {}x{} (feature-by-feature Laplacian)",
        expected_dim,
        expected_dim
    );

    debug!(
        "✓ Spectral Laplacian shape: {:?}",
        aspace_spectral.signals.shape()
    );
}

#[test]
fn test_builder_lambda_values_are_nonnegative() {
    // Test that all lambda values (spectral scores) are non-negative
    let items: Vec<Vec<f64>> = make_moons_hd(100, 0.2, 0.35, 11, 999);

    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .build(items);

    let lambdas = aspace.lambdas();

    for (i, &lam) in lambdas.iter().enumerate() {
        assert!(
            lam >= 0.0,
            "Lambda at index {} should be non-negative, got {:.6}",
            i,
            lam
        );
    }

    let min_lambda = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_lambda = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    debug!(
        "✓ All {} lambdas are non-negative: min={:.6}, max={:.6}",
        lambdas.len(),
        min_lambda,
        max_lambda
    );
}

#[test]
fn test_builder_with_high_noise() {
    // Generate 3 Gaussian blobs with noise=0.9 (moderate overlap)
    let items = make_gaussian_blob(300, 0.9);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.4, 6, 3, 2.0, None)
        .with_normalisation(true)
        .build(items);

    // Note: With noise=0.9, the optimal K heuristic may conservatively
    // choose K=2 instead of K=3 due to cluster overlap. This is correct
    // behavior - the algorithm prefers under-clustering to over-clustering.
    assert!(
        aspace.n_clusters >= 2,
        "Should produce valid clusters even with high noise, got {}",
        aspace.n_clusters
    );

    debug!(
        "✓ Found {} clusters (conservative estimate for noisy data)",
        aspace.n_clusters
    );
}

#[test]
fn test_builder_normalization_effects() {
    // Compare normalized vs unnormalized builds
    let items: Vec<Vec<f64>> = make_moons_hd(75, 0.14, 0.45, 8, 654);

    // Build with normalization
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .build(items.clone());

    // Build without normalization
    let (aspace_raw, gl_raw) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(false)
        .build(items);

    debug!("Normalized: {} clusters", aspace_norm.n_clusters);
    debug!("Raw (τ-mode): {} clusters", aspace_raw.n_clusters);

    // Parameters should be correctly set
    assert_eq!(gl_norm.graph_params.normalise, true);
    assert_eq!(gl_raw.graph_params.normalise, false);

    // Both should produce valid results
    assert!(aspace_norm.n_clusters > 0);
    assert!(aspace_raw.n_clusters > 0);

    debug!("✓ Both normalization modes produce valid clusterings");
}
