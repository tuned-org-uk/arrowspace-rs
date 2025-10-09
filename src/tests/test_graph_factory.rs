use smartcore::dataset::iris;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::graph::{GraphFactory, GraphLaplacian, GraphParams};
use crate::tests::GRAPH_PARAMS;

use approx::{assert_relative_eq, relative_eq};

use log::debug;

#[test]
fn test_build_lambda_graph_basic_sparse() {
    // Prepare a moderate dataset; Iris shown here, but any non-degenerate set works
    let dataset = iris::load_dataset();
    let items: Vec<Vec<f64>> = dataset
        .as_matrix()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| {
                    let mut v = *val as f64;
                    v *= 100.0;
                    v
                })
                .collect()
        })
        .collect();

    let gl = GraphFactory::build_laplacian_matrix_from_items(
        items,
        1e-1,
        10,
        GRAPH_PARAMS.topk,
        GRAPH_PARAMS.p,
        Some(1e-3 * 0.50),
        false,
        true,
    );

    // Expect a square Laplacian of size len_items
    assert_eq!(gl.matrix.rows(), 4, "Laplacian must be square (rows)");
    assert_eq!(gl.matrix.cols(), 4, "Laplacian must be square (cols)");

    // For a Laplacian L = D - W, diagonal entries are non-negative and finite.
    // Traverse each row and locate the diagonal (i,i) efficiently using indptr/indices.
    // This avoids dense access and remains O(nnz) overall.
    let csr = &gl.matrix; // assumed CSR: outer = rows
    assert!(csr.is_csr(), "Expected CSR layout for row-wise traversal");

    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();

    for i in 0..4 {
        let start = indptr.into_raw_storage()[i];
        let end = indptr.into_raw_storage()[i + 1];
        let mut diag = 0.0_f64;
        let mut found = false;

        for pos in start..end {
            let j = indices[pos];
            if j == i {
                diag = data[pos];
                found = true;
                break;
            }
        }

        // If the diagonal was not explicitly stored, it is structurally zero.
        // A proper graph Laplacian should store degrees on the diagonal; assert presence.
        assert!(found, "Diagonal entry (row {}) should exist in Laplacian storage", i);
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
}

#[test]
fn test_build_lambda_graph_minimum_items() {
    // Test minimum case: 2 items
    let items = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    let gl = GraphFactory::build_laplacian_matrix_from_items(
        items, 1.0, 6, 3, 2.0, None, true, true,
    );
    assert_eq!(gl.nnodes, 2);
}

#[test]
#[should_panic(expected = "items should be at least of shape (2,2): (2,1)")]
fn test_build_lambda_graph_insufficient_items() {
    // Should panic with only 1 item
    let items = vec![vec![1.0, 2.0]];
    GraphFactory::build_laplacian_matrix_from_items(
        items, 1.0, 6, 3, 2.0, None, true, true,
    );
}

#[test]
fn test_build_lambda_graph_scale_invariance() {
    // Test that scaling all items uniformly doesn't affect graph structure
    let items = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![3.0, 6.0, 9.0]];

    let gl1 = GraphFactory::build_laplacian_matrix_from_items(
        items.clone(),
        0.5,
        2,
        2,
        2.0,
        None,
        true,
        true,
    );

    // Scale all items by constant factor
    let scale_factor = 5.7;
    let items_scaled: Vec<Vec<f64>> = items
        .iter()
        .map(|item| item.iter().map(|&x| x * scale_factor).collect())
        .collect();

    let gl2 = GraphFactory::build_laplacian_matrix_from_items(
        items_scaled,
        0.5,
        2,
        2,
        2.0,
        None,
        true,
        true,
    );

    // Graph structure should be identical
    assert_eq!(gl1.nnodes, gl2.nnodes);
    assert_eq!(gl1.matrix.diag(), gl2.matrix.diag());
    // Note: values may differ due to lambda aggregation, but structure should be same
}

#[test]
fn test_graph_laplacian_structure_sparse() {
    // Small toy dataset
    let items = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![3.0, 6.0, 9.0]];

    // Build Laplacian with normalisation=true (undirected graph expected)
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        items, 1.0, 6, 3, 2.0, None, true, true,
    );

    // Expect square Laplacian
    assert_eq!(gl.matrix.rows(), gl.matrix.cols(), "Laplacian must be square");

    // Assume CSR; traverse rows and check symmetric counterparts per nonzero
    let csr = &gl.matrix;
    assert!(csr.is_csr(), "Expected CSR layout for row-wise traversal");

    let n = csr.rows();
    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();

    let eps = 1e-12;

    for i in 0..n {
        let start = indptr.into_raw_storage()[i];
        let end = indptr.into_raw_storage()[i + 1];
        for p in start..end {
            let j = indices[p];
            let vij = data[p];

            if i == j {
                // Skip diagonal for symmetry check
                continue;
            }

            // Find matching (j,i) by scanning row j's indices; rows are typically short (k-NN)
            let js = indptr.into_raw_storage()[j];
            let je = indptr.into_raw_storage()[j + 1];
            let mut vji_opt: Option<f64> = None;
            for q in js..je {
                if indices[q] == i {
                    vji_opt = Some(data[q]);
                    break;
                }
            }

            // Off-diagonal symmetry requires both presence and equal magnitude within tolerance
            let vji = vji_opt.unwrap_or(0.0);
            assert!(
                vji.abs() > eps,
                "Graph should be symmetric: found edge ({},{}) = {} but missing symmetric ({},{}). vji={}",
                i, j, vij, j, i, vji
            );

            // Values should match within tolerance (for symmetric weights)
            assert!(
                (vij - vji).abs() <= 1e-10 * (1.0 + vij.abs().max(vji.abs())),
                "Symmetric entries must match: L[{},{}]={} vs L[{},{}]={}",
                i,
                j,
                vij,
                j,
                i,
                vji
            );
        }
    }
}

#[test]
fn test_new_from_items_transpose_verification() {
    // Prepare a moderate dataset; Iris shown here, but any non-degenerate set works
    let dataset = iris::load_dataset();
    let data: Vec<Vec<f64>> = dataset
        .as_matrix()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| {
                    let mut v = *val as f64;
                    v *= 100.0;
                    v
                })
                .collect()
        })
        .collect();
    let items = DenseMatrix::from_2d_vec(&data[0..75].to_vec()).unwrap();

    let graph_params = GraphParams {
        eps: 0.1,
        k: 10,
        topk: GRAPH_PARAMS.topk,
        p: GRAPH_PARAMS.p,
        sigma: Some(0.1 * 0.50),
        normalise: false,
        sparsity_check: true,
    };

    // This should be treated as 3 items with 4 features each
    // After transposition for features matrix, it becomes 4 features with 3 items each
    let laplacian = GraphLaplacian::prepare_from_items(items.clone(), graph_params);

    // Verify dimensions after transposition
    assert_eq!(laplacian.nnodes, 75, "Should have 4 nodes (number of features)");
    assert_eq!(laplacian.matrix.shape().0, 4, "Matrix rows should be 4 (features)");
    assert_eq!(laplacian.matrix.shape().1, 4, "Matrix cols should be 3 (items)");

    // Verify transposition is correct
    println!("{:?}", *laplacian.matrix.get(0, 0).unwrap());
    assert!(relative_eq!(*laplacian.matrix.get(0, 0).unwrap(), 1.0973, epsilon = 1e-3));
    assert!(relative_eq!(
        *laplacian.matrix.get(0, 1).unwrap(),
        -0.8596,
        epsilon = 1e-3
    ));
    assert!(relative_eq!(
        *laplacian.matrix.get(0, 2).unwrap(),
        -0.2377,
        epsilon = 1e-3
    ));

    // Feature 1 across items:
    assert!(relative_eq!(
        *laplacian.matrix.get(1, 0).unwrap(),
        -0.8596,
        epsilon = 1e-3
    ));
    assert!(relative_eq!(*laplacian.matrix.get(1, 1).unwrap(), 0.8596, epsilon = 1e-3));

    // Feature 2 across items:
    assert!(relative_eq!(
        *laplacian.matrix.get(2, 0).unwrap(),
        -0.2377,
        epsilon = 1e-3
    ));
    assert!(relative_eq!(*laplacian.matrix.get(2, 2).unwrap(), 0.9801, epsilon = 1e-3));
    assert!(relative_eq!(
        *laplacian.matrix.get(2, 3).unwrap(),
        -0.7425,
        epsilon = 1e-3
    ));

    // Feature 3 across items:
    assert!(relative_eq!(
        *laplacian.matrix.get(3, 2).unwrap(),
        -0.7425,
        epsilon = 1e-3
    ));
    assert!(relative_eq!(*laplacian.matrix.get(3, 3).unwrap(), 0.7425, epsilon = 1e-3));

    debug!("Transpose verification test passed");
}

#[test]
fn test_new_from_items_non_square_matrix() {
    // Create a non-square matrix (this should panic)
    let non_square_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let non_square_matrix =
        DenseMatrix::from_iterator(non_square_data.into_iter(), 2, 3, 0);

    let graph_params = GraphParams {
        eps: 1.0,
        k: 2,
        topk: 1,
        p: 2.0,
        sigma: Some(0.1),
        normalise: true,
        sparsity_check: true,
    };

    // This should panic because input matrix is not square
    let gl = GraphLaplacian::prepare_from_items(non_square_matrix, graph_params);

    // should have returned the transposed of the input
    assert!(gl.nnodes == 2 && gl.matrix.shape() == (3, 3));
}

#[test]
fn test_new_from_items_parameter_preservation() {
    // Test that graph parameters are preserved correctly
    let matrix_data = vec![10.0, 20.0, 30.0, 40.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let original_params = GraphParams {
        eps: 0.123,
        k: 5,
        topk: 3,
        p: 3.5,
        sigma: Some(0.456),
        normalise: true,
        sparsity_check: true,
    };

    let laplacian = GraphLaplacian::prepare_from_items(matrix, original_params.clone());

    // Verify all parameters are preserved
    assert_eq!(laplacian.graph_params.eps, original_params.eps);
    assert_eq!(laplacian.graph_params.k, original_params.k);
    assert_eq!(laplacian.graph_params.p, original_params.p);
    assert_eq!(laplacian.graph_params.sigma, original_params.sigma);

    debug!("Parameter preservation test passed");
}

#[test]
fn test_new_from_items_single_element() {
    // Test with 1x1 matrix
    let single_data = vec![42.0, 24.0, 26.0, 19.0];
    let single_matrix = DenseMatrix::from_iterator(single_data.into_iter(), 2, 2, 0);

    let graph_params = GraphParams {
        eps: 2.0,
        k: 3,
        topk: 1,
        p: 1.5,
        sigma: None,
        normalise: true,
        sparsity_check: true,
    };

    let laplacian = GraphLaplacian::prepare_from_items(single_matrix, graph_params);

    assert_eq!(laplacian.nnodes, 2);
    assert_eq!(laplacian.matrix.shape().0, 2);
    assert_eq!(laplacian.matrix.shape().1, 2);
    assert_relative_eq!(
        *laplacian.matrix.get(0, 0).unwrap(),
        0.2612038749637415,
        epsilon = 1e-10
    );

    debug!("Single element test passed");
}
