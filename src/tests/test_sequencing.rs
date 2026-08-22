use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::TriMat;

use crate::analysis::sequencing::{sequence_by_graph, sequence_by_lambda};
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;

/// Build a GraphLaplacian directly from Laplacian triplets (i, j, value).
/// Sequencing operates on `matrix` alone, so `init_data` stays empty.
fn laplacian_from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> GraphLaplacian {
    let mut tm = TriMat::new((n, n));
    for &(i, j, v) in triplets {
        tm.add_triplet(i, j, v);
    }
    GraphLaplacian {
        init_data: DenseMatrix::new(0, 0, vec![], true).unwrap(),
        matrix: tm.to_csr(),
        nnodes: n,
        graph_params: GraphParams {
            eps: 0.9,
            k: 4,
            topk: 4,
            p: 2.0,
            sigma: None,
            normalise: false,
            sparsity_check: false,
        },
        energy: false,
    }
}

/// Path-graph Laplacian for nodes 0-1-2-3-4 with unit edge weights.
fn path_laplacian(n: usize) -> GraphLaplacian {
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
        triplets.push((i, i, degree));
        if i > 0 {
            triplets.push((i, i - 1, -1.0));
            triplets.push((i - 1, i, -1.0));
        }
    }
    laplacian_from_triplets(n, &triplets)
}

#[test]
fn test_sequence_by_lambda_orders_ascending_and_descending() {
    let lambdas = [0.5, 0.1, 0.9];

    let asc = sequence_by_lambda(&lambdas, false);
    assert_eq!(asc.order, vec![1, 0, 2]);
    assert_eq!(asc.positions, vec![0.1, 0.5, 0.9]);
    assert_eq!(asc.components, 1);

    let desc = sequence_by_lambda(&lambdas, true);
    assert_eq!(desc.order, vec![2, 0, 1]);
    assert_eq!(desc.positions, vec![0.9, 0.5, 0.1]);
}

#[test]
fn test_sequence_by_lambda_is_permutation_and_deterministic() {
    let lambdas = [0.7, 0.3, 0.9, 0.1];
    let a = sequence_by_lambda(&lambdas, false);
    let b = sequence_by_lambda(&lambdas, false);

    assert_eq!(a.order, b.order);
    assert_eq!(a.positions, b.positions);

    let mut sorted = a.order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
}

#[test]
fn test_sequence_by_graph_walks_path_monotonically() {
    let gl = path_laplacian(5);
    let seq = sequence_by_graph(&gl);

    // A path graph has one connected component...
    assert_eq!(seq.components, 1);
    // ...and seriation must recover the chain exactly.
    assert_eq!(seq.order, vec![0, 1, 2, 3, 4]);

    for step in seq.order.windows(2) {
        let d = step[1] as isize - step[0] as isize;
        assert!(
            d.abs() == 1,
            "consecutive steps must be graph neighbours, got {step:?}"
        );
    }
    // Positions are discovery depths along the walk.
    assert_eq!(seq.positions, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sequence_by_graph_is_deterministic_and_reversible_on_path() {
    let gl = path_laplacian(6);
    let a = sequence_by_graph(&gl);
    let b = sequence_by_graph(&gl);

    assert_eq!(a.order, b.order);
    assert_eq!(a.positions, b.positions);

    // The reverse chain is an equally valid seriation; either direction must
    // be a monotone walk over the path.
    for w in a.order.windows(2) {
        let d = w[1] as isize - w[0] as isize;
        assert_eq!(d.abs(), 1);
    }
}

#[test]
fn test_sequence_by_graph_keeps_disjoint_cliques_contiguous() {
    // Complete triangle {0,1,2} (weight 1) disjoint from pair {3,4}.
    let triplets = [
        (0, 0, 2.0),
        (0, 1, -1.0),
        (1, 0, -1.0),
        (1, 1, 2.0),
        (0, 2, -1.0),
        (2, 0, -1.0),
        (1, 2, -1.0),
        (2, 1, -1.0),
        (2, 2, 2.0),
        (3, 3, 1.0),
        (3, 4, -1.0),
        (4, 3, -1.0),
        (4, 4, 1.0),
    ];
    let gl = laplacian_from_triplets(5, &triplets);
    let seq = sequence_by_graph(&gl);

    assert_eq!(seq.components, 2);

    // Larger component first; blocks are contiguous.
    let head: std::collections::HashSet<usize> = seq.order[..3].iter().copied().collect();
    let tail: std::collections::HashSet<usize> = seq.order[3..].iter().copied().collect();
    assert_eq!(head, [0, 1, 2].into_iter().collect());
    assert_eq!(tail, [3, 4].into_iter().collect());

    // No interleaving across components anywhere in the order.
    let mut seen_pair = false;
    for &n in &seq.order {
        if n >= 3 {
            seen_pair = true;
        } else {
            assert!(!seen_pair, "components must not interleave");
        }
    }
}

#[test]
fn test_sequence_by_graph_edgeless_laplacian_yields_identity_order() {
    let gl = laplacian_from_triplets(3, &[(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)]);
    let seq = sequence_by_graph(&gl);

    assert_eq!(seq.components, 3);
    assert_eq!(seq.order, vec![0, 1, 2]);
    assert!(seq.positions.iter().all(|&p| p == 0.0));
}

#[test]
fn test_sequence_by_graph_handles_isolated_nodes_attached_to_component() {
    // Clique {0,1} plus isolated nodes 2 and 3.
    let triplets = [
        (0, 0, 1.0),
        (0, 1, -1.0),
        (1, 0, -1.0),
        (1, 1, 1.0),
        (2, 2, 0.0),
        (3, 3, 0.0),
    ];
    let gl = laplacian_from_triplets(4, &triplets);
    let seq = sequence_by_graph(&gl);

    assert_eq!(seq.components, 3);
    // Connected component (size 2) precedes singletons.
    assert_eq!(seq.order[..2], [0, 1]);
}

#[test]
fn test_sequence_by_graph_on_pipeline_laplacian_is_permutation() {
    let items = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.8, 0.6, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.8, 0.6],
        vec![0.0, 0.0, 1.0],
    ];
    let params = GraphParams {
        eps: 0.5,
        k: 3,
        topk: 2,
        p: 2.0,
        sigma: Some(0.1),
        normalise: false,
        sparsity_check: true,
    };
    let gl = build_laplacian_matrix(
        DenseMatrix::<f64>::from_2d_vec(&items).unwrap().transpose(),
        &params,
        None,
        false,
    );

    let seq = sequence_by_graph(&gl);

    // Valid permutation of the Laplacian nodes, deterministic across calls.
    let mut sorted = seq.order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..seq.order.len()).collect::<Vec<_>>());

    let again = sequence_by_graph(&gl);
    assert_eq!(seq.order, again.order);
    assert_eq!(seq.positions, again.positions);
}
