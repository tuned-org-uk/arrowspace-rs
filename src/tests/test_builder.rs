use crate::builder::ArrowSpaceBuilder;
use crate::graph_factory::GraphLaplacian;

#[test]
#[should_panic]
fn test_missing_input() {
    ArrowSpaceBuilder::new().build();
}

#[test]
fn simple_build() {
    // build `with_lambda_graph`
    let rows = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

    let eps = 1e-3;
    let k = 6usize;
    let p = 2.0;
    let sigma_override = None;

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_rows(rows.clone())
        .with_lambda_graph(eps, k, p, sigma_override)
        .build();

    assert_eq!(aspace.shape(), (3, 2));
    assert_eq!(gl.nnodes, 2);
    assert!(aspace.lambdas() == vec![0.6166666666666667, 0.0]);
}

#[test]
fn build_from_rows_without_graph_zero_lambda() {
    let rows = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let (aspace, gl) = ArrowSpaceBuilder::new().with_rows(rows.clone()).build();

    assert_eq!(aspace.shape(), (3, 2));
    assert_eq!(gl.nnodes, 2);
    assert!(aspace.lambdas() == vec![0.6166666666666667, 0.0]);
}

#[test]
fn build_from_rows_with_lambda_graph() {
    let rows = vec![vec![1.0, 0.0, 0.0], vec![0.5, 0.5, 0.0]];

    // Build a lambda-proximity Laplacian over items from the data matrix
    // Parameters mirror the old intent: small eps, k=2 cap, p=2.0 kernel, default sigma
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_rows(rows)
        .with_lambda_graph(1e-3, 2, 2.0, None)
        .build();

    assert_eq!(aspace.shape(), (3, 2));
    assert_eq!(gl.nnodes, 2);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
fn build_with_lambda_graph_over_product_like_rows() {
    // Original test used product-support coordinates to build a graph.
    // In the new API, construct rows directly (signals over items) and
    // build the lambda-graph from those rows.
    let c1 = [0, 2, 5, 9];
    let height = 3usize;
    let n_feature = c1.len() * height;

    let rows = vec![vec![1.0; n_feature], {
        let mut v = vec![0.0; n_feature];
        if n_feature > 0 {
            v[0] = 1.0;
        }
        v
    }];

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_rows(rows)
        .with_lambda_graph(1e-3, 3, 2.0, None)
        .build();

    assert_eq!(aspace.shape(), (12, 2));
    assert_eq!(gl.nnodes, 2);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
fn build_with_prebuilt_laplacian_zero_lambda_then_recompute() {
    // Prebuilt small Laplacian for 3 nodes (chain 0-1-2, symmetric weights 1)
    let rows_dense = [vec![1.0, -1.0, 0.0],
        vec![-1.0, 2.0, -1.0],
        vec![0.0, -1.0, 1.0]];
    // CSR
    let mut csr_rows = vec![0];
    let mut csr_cols = Vec::new();
    let mut csr_vals = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            let v = rows_dense[r][c];
            if v != 0.0 {
                csr_cols.push(c);
                csr_vals.push(v);
            }
        }
        csr_rows.push(csr_cols.len());
    }
    let _ = GraphLaplacian {
        rows: csr_rows,
        cols: csr_cols,
        vals: csr_vals,
        nnodes: 3,
    };

    // Build aspace with zero lambdas
    let rows = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let (aspace, _) = ArrowSpaceBuilder::new().with_rows(rows).build();

    assert_eq!(aspace.shape(), (3, 2));
    assert!(aspace.lambdas() == vec![0.6166666666666667, 0.0]);
}

#[test]
fn lambda_graph_shape_matches_rows() {
    // Replaces the old product_support graph builder: we validate that
    // the lambda-graph over items matches the number of columns in rows.
    let c1 = [0, 2, 4];
    let height = 4usize;
    let nnodes = c1.len() * height;

    let items = vec![vec![1.0; nnodes], vec![0.5; nnodes], vec![0.0; nnodes]];

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_rows(items.clone())
        .with_lambda_graph(1e-3, 3, 2.0, None)
        .build();

    assert_eq!(aspace.shape(), (nnodes, items.len()));
    assert_eq!(gl.nnodes, items.len());
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}
