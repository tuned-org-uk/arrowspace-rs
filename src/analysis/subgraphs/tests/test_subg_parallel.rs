use crate::analysis::subgraphs::sg_from_centroids::recluster_centroids;
use crate::analysis::subgraphs::{
    CentroidGraphParams, MotiveConfig, SubgraphConfig, SubgraphsCentroid, SubgraphsMotive,
};
use crate::builder::ArrowSpaceBuilder;
use crate::maps::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::tests::test_data::make_gaussian_cliques_multi;

use log::debug;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashSet;

/// Test that recluster_centroids produces deterministic results across multiple runs
#[test]
fn test_recluster_centroids_deterministic() {
    crate::init();

    let data = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0],
        vec![2.5, 2.5, 2.5],
        vec![3.5, 3.5, 3.5],
    ];
    let centroids = DenseMatrix::from_2d_vec(&data).unwrap();
    let k = 4;

    // Run multiple times and verify identical results
    let (labels1, new_centroids1) = recluster_centroids(&centroids, k, None);
    let (labels2, new_centroids2) = recluster_centroids(&centroids, k, None);
    let (labels3, new_centroids3) = recluster_centroids(&centroids, k, None);

    assert_eq!(labels1, labels2, "Labels should be deterministic");
    assert_eq!(labels1, labels3, "Labels should be deterministic");

    // Check that centroids are numerically identical
    for i in 0..new_centroids1.shape().0 {
        for j in 0..new_centroids1.shape().1 {
            assert_eq!(
                *new_centroids1.get((i, j)),
                *new_centroids2.get((i, j)),
                "Centroid values should be deterministic"
            );
            assert_eq!(
                *new_centroids1.get((i, j)),
                *new_centroids3.get((i, j)),
                "Centroid values should be deterministic"
            );
        }
    }
}

/// Verifies that `spot_subg_motives` is fully deterministic across repeated calls.
///
/// ## Non-determinism sources (fixed in spot_motives_energy):
///
/// 1. **Step 4 — greedy tie-breaking**: when two candidates share the same
///    triangle-gain score, `best_u` was set by whichever node `HashSet`
///    iteration happened to yield last. Fix: iterate candidates in sorted order.
///
/// 2. **Step 7 — item dedup input order**: `item_sets` was produced by
///    `sc_results.par_iter()`, whose delivery order varies between runs.
///    The sequential Jaccard dedup then evicted different items on tie.
///    Fix: collect to sorted `Vec<Vec<usize>>` before dedup.
///
/// After both fixes, this test must pass on every run with no flakiness.
#[test]
fn test_motif_subgraphs_parallel_correctness() {
    crate::init();

    let rows = make_gaussian_cliques_multi(200, 0.3, 6, 100, 999);
    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.35, 12, 8, 2.0, None)
        .with_seed(999)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let p = EnergyParams::new(&builder);
    let (aspace, gl) = builder.build_energy(rows, p);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 18,
            min_triangles: 3,
            min_clust: 0.35,
            max_motif_size: 30,
            max_sets: 60,
            jaccard_dedup: 0.65,
        },
        rayleigh_max: None,
        min_size: 5,
    };

    let subgraphs1 = gl
        .try_spot_subg_motives(&aspace, &cfg)
        .expect("energy build must satisfy energy-mode requirements");
    let subgraphs2 = gl
        .try_spot_subg_motives(&aspace, &cfg)
        .expect("energy build must satisfy energy-mode requirements");

    // --- Structural invariants: must hold independently on each run ---
    let f_parent = gl.init_data.shape().0;
    for (run, subgraphs) in [&subgraphs1, &subgraphs2].iter().enumerate() {
        for (i, sg) in subgraphs.iter().enumerate() {
            let (f_sg, x_centroids) = sg.laplacian.init_data.shape();
            assert_eq!(
                f_sg, f_parent,
                "run={run} sg={i}: feature dim must match parent"
            );
            assert_eq!(
                sg.laplacian.nnodes, x_centroids,
                "run={run} sg={i}: nnodes must equal initdata column count"
            );
            assert_eq!(
                sg.node_indices.len(),
                x_centroids,
                "run={run} sg={i}: node_indices length must equal nnodes"
            );
            assert!(
                sg.item_indices.is_some(),
                "run={run} sg={i}: energy subgraphs must carry item_indices"
            );
            assert!(
                sg.laplacian.nnodes >= 2,
                "run={run} sg={i}: subgraph needs at least 2 centroids"
            );
        }
    }

    // --- Full determinism: count, content and order must be identical ---
    assert_eq!(
        subgraphs1.len(),
        subgraphs2.len(),
        "run1={} subgraphs vs run2={} — non-determinism detected",
        subgraphs1.len(),
        subgraphs2.len()
    );

    for (i, (sg1, sg2)) in subgraphs1.iter().zip(subgraphs2.iter()).enumerate() {
        assert_eq!(
            sg1.node_indices, sg2.node_indices,
            "sg={i}: node_indices differ between runs"
        );
        assert_eq!(
            sg1.laplacian.nnodes, sg2.laplacian.nnodes,
            "sg={i}: nnodes differs between runs"
        );
        // item_indices content must also match (both sorted in production fix)
        assert_eq!(
            sg1.item_indices, sg2.item_indices,
            "sg={i}: item_indices differ between runs"
        );
    }

    println!(
        "✓ Determinism confirmed: {} subgraphs, all node/item indices identical",
        subgraphs1.len()
    );
}

/// Test that centroid hierarchy is deterministic with parallelization.
///
/// Verifies three levels of identity across two independent builds:
/// 1. Structural shape  — same number of levels and subgraphs per level.
/// 2. Node identity     — same centroid indices at every node.
/// 3. Parent mapping    — same label assignments propagated down the tree.
///
/// `recluster_centroids` uses modulo-based label assignment (label[i] = i % k),
/// so full bit-for-bit identity is expected — not just count equality.
#[test]
fn test_centroid_hierarchy_parallel_determinism() {
    crate::init();

    let rows = make_gaussian_cliques_multi(150, 0.3, 5, 100, 888);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_seed(888)
        .build(rows);

    let params = CentroidGraphParams {
        k: 4,
        topk: 4,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(123),
        min_centroids: 3,
        max_depth: 2,
    };

    let hierarchy1 = gl.build_centroid_hierarchy(&aspace, params.clone());
    let hierarchy2 = gl.build_centroid_hierarchy(&aspace, params.clone());

    // --- Level 1: structural shape ---
    assert_eq!(
        hierarchy1.levels.len(),
        hierarchy2.levels.len(),
        "Number of hierarchy levels must be identical across runs"
    );
    assert_eq!(
        hierarchy1.count_subgraphs(),
        hierarchy2.count_subgraphs(),
        "Total subgraph count must be identical across runs"
    );

    // --- Level 2: per-level node counts and node identity ---
    // count_only is insufficient: same count with different centroids
    // would still represent a non-deterministic result.
    for (depth, (level1, level2)) in hierarchy1
        .levels
        .iter()
        .zip(hierarchy2.levels.iter())
        .enumerate()
    {
        assert_eq!(
            level1.len(),
            level2.len(),
            "Node count at depth {depth} must match"
        );

        for (node_idx, (node1, node2)) in level1.iter().zip(level2.iter()).enumerate() {
            // Centroid node indices must be identical.
            assert_eq!(
                node1.graph.node_indices, node2.graph.node_indices,
                "node_indices differ at depth={depth}, node={node_idx}"
            );

            // Parent label assignments must be identical.
            // parentmap[i] is the child cluster ID centroid i was mapped to;
            // any divergence here means the re-clustering path is non-deterministic.
            assert_eq!(
                node1.parent_map, node2.parent_map,
                "parentmap differs at depth={depth}, node={node_idx}"
            );

            // Graph shape invariants must hold on both runs.
            let (f1, x1) = node1.graph.laplacian.init_data.shape();
            let (f2, x2) = node2.graph.laplacian.init_data.shape();
            assert_eq!(
                (f1, x1),
                (f2, x2),
                "initdata shape differs at depth={depth}, node={node_idx}"
            );
            assert_eq!(
                node1.graph.laplacian.nnodes, node2.graph.laplacian.nnodes,
                "nnodes differs at depth={depth}, node={node_idx}"
            );
        }
    }

    // --- Level 3: root item-to-centroid index mapping ---
    // root_indices[c] is the list of original item indices assigned to centroid c.
    // Two builds with identical params must produce the same mapping.
    assert_eq!(
        hierarchy1.root.root_indices, hierarchy2.root.root_indices,
        "Root item-to-centroid mapping must be identical across runs"
    );

    println!(
        "✓ Determinism verified: {} levels, {} total subgraphs",
        hierarchy1.levels.len(),
        hierarchy1.count_subgraphs()
    );
}

/// Test that recluster_centroids means are computed correctly
#[test]
fn test_recluster_centroids_means_correctness() {
    crate::init();

    let data = vec![
        vec![0.0, 0.0],   // index 0 → cluster 0 % 2 = 0
        vec![1.0, 1.0],   // index 1 → cluster 1 % 2 = 1
        vec![0.5, 0.5],   // index 2 → cluster 2 % 2 = 0
        vec![1.5, 1.5],   // index 3 → cluster 3 % 2 = 1
        vec![0.25, 0.25], // index 4 → cluster 4 % 2 = 0
    ];
    let centroids = DenseMatrix::from_2d_vec(&data).unwrap();
    let k = 2;

    let (labels, new_centroids) = recluster_centroids(&centroids, k, None);

    // Check labels assignment (modulo-based: i % k)
    assert_eq!(
        labels,
        vec![0, 1, 0, 1, 0],
        "Labels should follow modulo assignment"
    );

    // Cluster 0: indices 0, 2, 4
    // Mean = ((0.0, 0.0) + (0.5, 0.5) + (0.25, 0.25)) / 3
    let expected_c0_0 = (0.0 + 0.5 + 0.25) / 3.0;
    let expected_c0_1 = (0.0 + 0.5 + 0.25) / 3.0;

    // Cluster 1: indices 1, 3
    // Mean = ((1.0, 1.0) + (1.5, 1.5)) / 2
    let expected_c1_0 = (1.0 + 1.5) / 2.0;
    let expected_c1_1 = (1.0 + 1.5) / 2.0;

    let tolerance = 1e-10;

    // Check cluster 0 mean
    let actual_c0_0 = *new_centroids.get((0, 0));
    let actual_c0_1 = *new_centroids.get((0, 1));
    assert!(
        (actual_c0_0 - expected_c0_0).abs() < tolerance,
        "Cluster 0 dim 0: expected {}, got {}",
        expected_c0_0,
        actual_c0_0
    );
    assert!(
        (actual_c0_1 - expected_c0_1).abs() < tolerance,
        "Cluster 0 dim 1: expected {}, got {}",
        expected_c0_1,
        actual_c0_1
    );

    // Check cluster 1 mean
    let actual_c1_0 = *new_centroids.get((1, 0));
    let actual_c1_1 = *new_centroids.get((1, 1));
    assert!(
        (actual_c1_0 - expected_c1_0).abs() < tolerance,
        "Cluster 1 dim 0: expected {}, got {}",
        expected_c1_0,
        actual_c1_0
    );
    assert!(
        (actual_c1_1 - expected_c1_1).abs() < tolerance,
        "Cluster 1 dim 1: expected {}, got {}",
        expected_c1_1,
        actual_c1_1
    );

    debug!("Cluster 0 mean: ({:.6}, {:.6})", actual_c0_0, actual_c0_1);
    debug!("Cluster 1 mean: ({:.6}, {:.6})", actual_c1_0, actual_c1_1);
}

/// Test that parallel motif processing doesn't lose or duplicate motifs
#[test]
fn test_motif_subgraphs_no_loss_or_duplication() {
    crate::init();

    let rows = make_gaussian_cliques_multi(250, 0.25, 8, 50, 777);
    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.35, 14, 10, 2.0, None)
        .with_seed(777)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let p = EnergyParams::new(&builder);
    let (aspace, gl) = builder.build_energy(rows, p);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 20,
            min_triangles: 3,
            min_clust: 0.35,
            max_motif_size: 35,
            max_sets: 70,
            jaccard_dedup: 0.6,
        },
        rayleigh_max: None,
        min_size: 6,
    };

    let subgraphs = gl
        .try_spot_subg_motives(&aspace, &cfg)
        .expect("energy build must satisfy energy-mode requirements");

    if subgraphs.is_empty() {
        debug!("No subgraphs extracted; skipping duplication check");
        return;
    }

    // Check that all subgraphs have unique node_indices sets
    // Convert to sorted Vec for comparison (HashSet doesn't implement Hash)
    let mut seen_sets = HashSet::new();
    for sg in &subgraphs {
        let mut node_vec = sg.node_indices.clone();
        node_vec.sort_unstable();
        assert!(
            seen_sets.insert(node_vec.clone()),
            "Found duplicate subgraph with same node_indices: {:?}",
            node_vec
        );
    }

    // Check that all subgraphs pass basic invariants
    for sg in &subgraphs {
        assert!(sg.laplacian.nnodes >= 2, "Must have at least 2 centroids");
        assert_eq!(sg.laplacian.nnodes, sg.node_indices.len());
        assert!(
            sg.item_indices.is_some(),
            "Energy mode must have item_indices"
        );
    }
}

/// Test thread safety with concurrent hierarchy builds
#[test]
fn test_concurrent_hierarchy_builds() {
    use std::sync::Arc;
    use std::thread;

    crate::init();

    let rows = make_gaussian_cliques_multi(100, 0.3, 4, 100, 555);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_seed(555)
        .build(rows);

    let aspace = Arc::new(aspace);
    let gl = Arc::new(gl);

    let params = CentroidGraphParams {
        k: 3,
        topk: 3,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(999),
        min_centroids: 3,
        max_depth: 2,
    };
    let params = Arc::new(params);

    // Spawn multiple threads building hierarchies concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let aspace = Arc::clone(&aspace);
            let gl = Arc::clone(&gl);
            let params = Arc::clone(&params);

            thread::spawn(move || {
                let hierarchy = gl.build_centroid_hierarchy(&aspace, (*params).clone());
                hierarchy.count_subgraphs()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should get same result
    let first = results[0];
    for count in &results {
        assert_eq!(
            *count, first,
            "All concurrent builds should produce same hierarchy size"
        );
    }
}

/// Stress test: large parallel workload
#[test]
fn test_parallel_stress_large_dataset() {
    crate::init();

    let rows = make_gaussian_cliques_multi(500, 0.2, 10, 100, 1234);
    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.3, 16, 12, 2.0, None)
        .with_seed(1234)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let p = EnergyParams::new(&builder);
    let (aspace, gl) = builder.build_energy(rows, p);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 25,
            min_triangles: 4,
            min_clust: 0.3,
            max_motif_size: 40,
            max_sets: 100,
            jaccard_dedup: 0.55,
        },
        rayleigh_max: None,
        min_size: 8,
    };

    // Should complete without hanging or panicking
    let subgraphs = gl
        .try_spot_subg_motives(&aspace, &cfg)
        .expect("energy build must satisfy energy-mode requirements");

    // Basic sanity checks
    for sg in &subgraphs {
        assert!(sg.laplacian.nnodes >= 2);
        assert_eq!(sg.laplacian.nnodes, sg.node_indices.len());
        let (f_dim, x_dim) = sg.laplacian.init_data.shape();
        assert_eq!(x_dim, sg.laplacian.nnodes);
        assert!(f_dim > 0);
    }

    debug!(
        "Stress test passed: extracted {} subgraphs from 500-item dataset",
        subgraphs.len()
    );
}
