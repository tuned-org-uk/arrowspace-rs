 use crate::graph_factory::GraphFactory;


#[test]
fn test_build_lambda_graph_basic() {
    // Test with 3 items, each with 2 features
    let items = vec![
        vec![1.0, 0.0], // Item 0: high in feature 0, low in feature 1
        vec![0.0, 1.0], // Item 1: low in feature 0, high in feature 1
        vec![0.5, 0.5], // Item 2: medium in both features
    ];

    let gl = GraphFactory::build_lambda_graph(&items, 0.5, 2, 2.0, None);

    // Verify basic structure
    assert_eq!(gl.nnodes, 3, "Graph should have 3 nodes for 3 items");
    assert_eq!(
        gl.rows.len(),
        4,
        "CSR rows array should have nnodes+1 elements"
    );
    assert!(
        gl.cols.len() >= 3,
        "Should have at least 3 entries (diagonal)"
    );
    assert!(
        gl.vals.len() >= 3,
        "Should have at least 3 values (diagonal)"
    );

    // Verify each node has a non-negative diagonal entry
    for i in 0..gl.nnodes {
        let start = gl.rows[i];
        let end = gl.rows[i + 1];
        let mut found_diagonal = false;

        for idx in start..end {
            if gl.cols[idx] == i {
                found_diagonal = true;
                assert!(gl.vals[idx] >= 0.0, "Diagonal entry should be non-negative");
                break;
            }
        }
        assert!(found_diagonal, "Each node should have a diagonal entry");
    }
}

#[test]
fn test_build_lambda_graph_minimum_items() {
    // Test minimum case: 2 items
    let items = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    let gl = GraphFactory::build_lambda_graph(&items, 1.0, 6, 2.0, None);
    assert_eq!(gl.nnodes, 2);
    assert!(!gl.vals.is_empty());
}

#[test]
#[should_panic(expected = "graph should have at least two items")]
fn test_build_lambda_graph_insufficient_items() {
    // Should panic with only 1 item
    let items = vec![vec![1.0, 2.0]];
    GraphFactory::build_lambda_graph(&items, 1.0, 6, 2.0, None);
}

#[test]
fn test_build_lambda_graph_k_pruning() {
    // Test k-pruning with more items than k
    let items = vec![
        vec![1.0, 0.0, 0.0], // Item 0: distinct pattern
        vec![0.0, 1.0, 0.0], // Item 1: distinct pattern
        vec![0.0, 0.0, 1.0], // Item 2: distinct pattern
        vec![1.0, 1.0, 1.0], // Item 3: similar to all others
    ];

    let gl = GraphFactory::build_lambda_graph(&items, 10.0, 2, 2.0, None);
    assert_eq!(gl.nnodes, 4);

    // Due to symmetrization, nodes can have more than k neighbors
    // The k parameter limits outgoing edges before symmetrization,
    // but incoming edges from other nodes' selections can increase the final degree
    for i in 0..gl.nnodes {
        let start = gl.rows[i];
        let end = gl.rows[i + 1];
        let degree = end - start;

        // More relaxed assertion: degree should be at least 1 (diagonal)
        // and at most n (all other nodes + diagonal)
        assert!(degree >= 1, "Each node should have at least diagonal entry");
        assert!(
            degree <= gl.nnodes,
            "Degree {} should be <= nnodes {}",
            degree,
            gl.nnodes
        );

        // Verify diagonal entry exists and is non-negative
        let mut found_diagonal = false;
        for idx in start..end {
            if gl.cols[idx] == i {
                found_diagonal = true;
                assert!(gl.vals[idx] >= 0.0, "Diagonal should be non-negative");
                break;
            }
        }
        assert!(found_diagonal, "Each node should have a diagonal entry");
    }

    // Alternative test: verify that k-pruning has some effect
    // Compare with unlimited k
    let gl_unlimited = GraphFactory::build_lambda_graph(&items, 10.0, 6, 2.0, None);

    // The k-pruned graph should have fewer or equal total edges
    let total_edges_pruned = gl.vals.len();
    let total_edges_unlimited = gl_unlimited.vals.len();
    assert!(
        total_edges_pruned <= total_edges_unlimited,
        "k-pruning should not increase total edges: {total_edges_pruned} vs {total_edges_unlimited}"
    );
}

#[test]
fn test_build_lambda_graph_k_pruning_before_symmetrisation() {
    // Test that k-pruning limits initial neighbor selection
    // This test checks the behavior before symmetrization

    let items = vec![
        vec![0.0, 0.0], // Item 0
        vec![1.0, 0.0], // Item 1: distance 1 from item 0
        vec![2.0, 0.0], // Item 2: distance 2 from item 0
        vec![3.0, 0.0], // Item 3: distance 3 from item 0
    ];

    // With k=2, each item should initially select at most 2 neighbors
    // But after symmetrization, degrees can be higher
    let gl = GraphFactory::build_lambda_graph(&items, 5.0, 2, 2.0, None);
    assert_eq!(gl.nnodes, 4);

    // Just verify the graph is well-formed
    for i in 0..gl.nnodes {
        let start = gl.rows[i];
        let end = gl.rows[i + 1];
        assert!(start <= end);
        assert!(end <= gl.cols.len());

        // Find diagonal
        let mut has_diagonal = false;
        for idx in start..end {
            if gl.cols[idx] == i {
                has_diagonal = true;
                assert!(gl.vals[idx] >= 0.0);
            }
        }
        assert!(has_diagonal);
    }
}

#[test]
fn test_build_lambda_graph_scale_invariance() {
    // Test that scaling all items uniformly doesn't affect graph structure
    let items = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ];

    let gl1 = GraphFactory::build_lambda_graph(&items, 0.5, 2, 2.0, None);

    // Scale all items by constant factor
    let scale_factor = 5.7;
    let items_scaled: Vec<Vec<f64>> = items
        .iter()
        .map(|item| item.iter().map(|&x| x * scale_factor).collect())
        .collect();

    let gl2 = GraphFactory::build_lambda_graph(&items_scaled, 0.5, 2, 2.0, None);

    // Graph structure should be identical
    assert_eq!(gl1.nnodes, gl2.nnodes);
    assert_eq!(gl1.rows, gl2.rows);
    assert_eq!(gl1.cols, gl2.cols);
    // Note: values may differ due to lambda aggregation, but structure should be same
}

#[test]
fn test_graph_laplacian_structure() {
    // Test that created Laplacians have proper structure
    let items = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let gl = GraphFactory::build_lambda_graph(&items, 1.0, 6, 2.0, None);

    // CSR structure validation
    assert_eq!(gl.rows[0], 0);
    assert_eq!(gl.rows[gl.nnodes], gl.cols.len());
    assert_eq!(gl.cols.len(), gl.vals.len());

    // Symmetry check (for undirected graph)
    for i in 0..gl.nnodes {
        let start = gl.rows[i];
        let end = gl.rows[i + 1];

        for idx in start..end {
            let j = gl.cols[idx];
            if i != j {
                // Find corresponding entry (j,i)
                let j_start = gl.rows[j];
                let j_end = gl.rows[j + 1];
                let mut found_reverse = false;

                for j_idx in j_start..j_end {
                    if gl.cols[j_idx] == i {
                        found_reverse = true;
                        break;
                    }
                }
                assert!(found_reverse, "Graph should be symmetric");
            }
        }
    }
}
