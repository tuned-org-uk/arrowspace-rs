use crate::graph_factory::{GraphFactory, GraphLaplacian};
use crate::dimensional::{DimensionalOps, ArrowDimensionalOps};
use crate::operators::rayleigh_lambda;


#[test]
fn cantor_product_and_boxdim() {
    let c1 = DimensionalOps::make_cantor_1d(3, 1.0 / 3.0, 3usize.pow(5));
    assert!(!c1.is_empty() && c1.len() < 243);

    let supp = DimensionalOps::make_product_support(&c1, 16);
    assert_eq!(supp.len(), c1.len() * 16);

    let pts: Vec<(i32, i32)> = supp.iter().map(|&(r, c)| (r as i32, c as i32)).collect();
    let d = DimensionalOps::box_count_dimension(&pts, &[1, 3, 9, 27, 81]);
    assert!(d.is_some());
}

#[test]
fn debug_laplacian_type() {
    use crate::core::ArrowSpace;
    use crate::graph_factory::GraphLaplacian;

    // Helper builds a λτ-graph from item data (row-major: items = rows, features = cols)
    // Items are identical across features to ensure the graph is connected.
    let lambda_graph = |items: Vec<Vec<f64>>, eps: f64, k: usize, p: f64| -> GraphLaplacian {
        crate::graph_factory::GraphFactory::build_lambda_graph(&items, eps, k, p, None)
    };

    // 1) Build a small λτ-graph over two items using two identical item rows (row-major).
    //    items: N×F where each inner Vec is one item (row). Here N=2, F=2.
    let items_row_major = vec![
        vec![1.0, 1.0], // item 0
        vec![2.0, 2.0], // item 1
    ];
    let gl = lambda_graph(items_row_major, 1.0, 1, 2.0);

    // Basic CSR/Laplacian sanity: shape and diagonal presence
    assert_eq!(gl.nnodes, 2, "Graph must have 2 nodes for 2 items");
    assert_eq!(
        gl.rows.len(),
        gl.nnodes + 1,
        "CSR row ptr length must be nnodes+1"
    );
    assert_eq!(
        gl.cols.len(),
        gl.vals.len(),
        "CSR cols/vals length mismatch"
    );
    for i in 0..gl.nnodes {
        let s = gl.rows[i];
        let e = gl.rows[i + 1];
        assert!(e > s, "Each row must have at least the diagonal entry");
        let mut has_diag = false;
        for idx in s..e {
            let j = gl.cols[idx];
            let v = gl.vals[idx];
            if j == i {
                has_diag = true;
                assert!(v >= 0.0, "Diagonal (degree) must be non-negative");
            } else {
                assert!(v < 0.0, "Off-diagonal must be negative (-w_ij)");
            }
        }
        assert!(has_diag, "Diagonal entry missing at row {i}");
    }

    // 2) Build ArrowSpace with feature rows (column-major internally).
    //    ArrowSpace::from_items expects items as N×F but stores column-major (features as rows).
    //    To test Rayleigh on feature signals, pass each signal as its own row vector.
    //
    // Constant feature (same value on all items) should yield near-zero Rayleigh on connected graphs.
    let constant_feature = vec![vec![1.0, 1.0], vec![0.0, 0.0]]; // length = nnodes
    let mut aspace_const = ArrowSpace::from_items(constant_feature.clone());
    aspace_const.recompute_lambdas(&gl);
    let lambda_constant = aspace_const.lambdas()[0];

    // Alternating/high-frequency feature over the same two items.
    let alternating_feature = vec![vec![1.0, -1.0], vec![1.0, -1.0]];
    let mut aspace_alt = ArrowSpace::from_items(alternating_feature.clone());
    aspace_alt.recompute_lambdas(&gl);
    let lambda_alt = aspace_alt.lambdas()[0];

    println!("Constant feature: x = {constant_feature:?}");
    println!("Alternating feature: x = {alternating_feature:?}");
    println!("Constant vector lambda: {lambda_constant:?}");
    println!("Alternating vector lambda: {lambda_alt:?}");

    // 3) Assertions: constant ≈ 0, alternating > constant
    assert!(
        lambda_constant < 0.1,
        "Constant vector should have low lambda"
    );
    assert!(
        lambda_alt > lambda_constant,
        "Alternating should have higher lambda"
    );

    // 4) Optional: show how to access items vs. features per the memory layout note.
    //    ArrowSpace stores features as rows; to inspect an item (column) use get_item(i).
    //    This demonstrates the column extraction path that aligns with the reminder.
    let item0 = aspace_const.get_item(0);
    let item1 = aspace_const.get_item(1);
    println!("Item 0 (column) from ArrowSpace: {:?}", item0.item);
    println!("Item 1 (column) from ArrowSpace: {:?}", item1.item);
}

/// Explanation for each test case:
/// - Constant vector lambda should be near zero since it's in the Laplacian's nullspace.
/// - Alternating vector (higher frequency) produces higher lambda (non-negative).
/// - Random vector lambda is always non-negative.
/// - A conservative upper bound uses 2*max_degree for unnormalized Laplacians.

#[test]
fn arrowspace_rayleigh_lambda_valid() {
    // Four-item λτ-graph built from two feature rows
    let gl =  GraphFactory::build_lambda_graph(
        &vec![vec![0.9, 0.1, 0.2, 0.8], vec![0.1, 0.9, 0.8, 0.2]],
        1e-3,
        3,
        2.0,
        None
    );

    // All-ones vector: tests nullspace
    let row_const = vec![1.0, 1.0, 1.0, 1.0];
    let lambda_const = rayleigh_lambda(&gl, &row_const);
    println!("lambda_const: {lambda_const}");
    assert!(
        lambda_const.abs() < 1e-6,
        "Constant vector lambda should be near zero"
    );

    // Alternating sign vector
    let row_alt = vec![1.0, -1.0, 1.0, -1.0];
    let lambda_alt = rayleigh_lambda(&gl, &row_alt);
    println!("lambda_alt: {lambda_alt}");
    assert!(lambda_alt >= 0.0, "Lambda must be non-negative");

    // Random real vector
    let row_rnd = vec![0.4, 0.3, -0.2, -0.5];
    let lambda_rnd = rayleigh_lambda(&gl, &row_rnd);
    println!("lambda_rnd: {lambda_rnd}");
    assert!(lambda_rnd >= 0.0, "Lambda must be non-negative");

    // Conservative upper bound via max diagonal (degree)
    let mut max_deg = 0.0f64;
    for i in 0..gl.nnodes {
        let s = gl.rows[i];
        let e = gl.rows[i + 1];
        for idx in s..e {
            if gl.cols[idx] == i {
                max_deg = max_deg.max(gl.vals[idx]);
                break;
            }
        }
    }
    assert!(
        lambda_alt <= 2.0 * max_deg + 1e-6,
        "Lambda unexpectedly large for alternating vector"
    );
}

#[test]
fn arrowspace_rayleigh_lambda_valid_alt() {
    let gl =  GraphFactory::build_lambda_graph(
        &vec![vec![0.7, 0.3, 0.2, 0.6], vec![0.3, 0.7, 0.6, 0.2]],
        1e-3,
        3,
        2.0,
        None
    );

    // All-ones vector: tests nullspace
    let row_const = vec![1.0, 1.0, 1.0, 1.0];
    let lambda_const = rayleigh_lambda(&gl, &row_const);
    println!("lambda_const: {lambda_const:?}");
    assert!(
        lambda_const.abs() < 1e-6,
        "Constant vector lambda should be near zero"
    );

    // Alternating sign vector
    let row_alt = vec![1.0, -1.0, 1.0, -1.0];
    let lambda_alt = rayleigh_lambda(&gl, &row_alt);
    println!("lambda_alt: {lambda_alt:?}");
    assert!(lambda_alt >= 0.0, "Lambda must be non-negative");

    // Random real vector
    let row_rnd = vec![0.4, 0.3, -0.2, -0.5];
    let lambda_rnd = rayleigh_lambda(&gl, &row_rnd);
    println!("lambda_rnd: {lambda_rnd:?}");
    assert!(lambda_rnd >= 0.0, "Lambda must be non-negative");

    // Bound via 2*max_degree
    let mut max_deg = 0.0f64;
    for i in 0..gl.nnodes {
        let s = gl.rows[i];
        let e = gl.rows[i + 1];
        for idx in s..e {
            if gl.cols[idx] == i {
                max_deg = max_deg.max(gl.vals[idx]);
                break;
            }
        }
    }
    assert!(
        lambda_alt <= 2.0 * max_deg + 1e-6,
        "Lambda unexpectedly large for alternating vector"
    );
}
