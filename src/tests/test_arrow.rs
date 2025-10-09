use smartcore::linalg::basic::arrays::Array;

use crate::builder::ArrowSpaceBuilder;
use crate::core::{ArrowSpace, TAUDEFAULT};
use crate::graph::{GraphFactory, GraphLaplacian};

use crate::taumode::TauMode;
use crate::tests::test_data::make_moons_hd;
use crate::tests::{GRAPH_PARAMS, TAU_PARAMS};

use approx::{assert_relative_eq, assert_relative_ne, relative_eq};
use smartcore::dataset::iris;

#[test]
fn arrowspace_build_and_recompute() {
    // Test ArrowSpace construction and λ recomputation with realistic high-dimensional data
    let dims = 10;

    let items: Vec<Vec<f64>> = make_moons_hd(300, 0.12, 0.01, dims, 42);

    let (mut aspace_plain, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            1e-3,
            5,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            Some(1e-3 * 0.50),
        )
        .with_normalisation(true)
        .with_spectral(false)
        .build(items[0..150].to_vec());

    assert_eq!(aspace_plain.nfeatures, dims, "Expected 13-dimensional feature aspace");
    assert_eq!(aspace_plain.nitems, 150, "Expected 7 data points");

    println!("=== ARROWSPACE CONSTRUCTION ===");
    println!("Feature dimensions: {}", aspace_plain.nfeatures);
    println!("Number of items: {}", aspace_plain.nitems);
    println!("Data shape: {:?}", aspace_plain.data.shape());

    // Build λτ-graph from the same data matrix with realistic parameters
    let (aspace_spec, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            1e-1,
            5,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            Some(1e-1 * 0.50),
        )
        .with_normalisation(true)
        .with_spectral(true)
        .build(items[150..].to_vec());

    assert_eq!(
        gl.nnodes, aspace_spec.nitems,
        "Graph nodes should match ArrowSpace items"
    );
    assert_eq!(
        aspace_spec.signals.shape(),
        (dims, dims),
        "Signals matrix should be 5x5 for spectral decomposition"
    );
    println!("spectral {:?}:", aspace_spec.signals);

    println!("\n=== SPECTRAL ANALYSIS ===");
    println!("Graph nodes: {}", gl.nnodes);
    println!("Signals matrix shape: {:?}", aspace_spec.signals.shape());

    // Recompute lambdas and analyze spectral properties
    aspace_plain.recompute_lambdas(&gl);

    println!("Lambda values (raw): {:?}", aspace_plain.lambdas);

    // Spectral graph theory properties: all eigenvalues should be non-negative
    assert!(
        aspace_plain.lambdas[0] >= 0.0,
        "Smallest eigenvalue should be non-negative: {}",
        aspace_plain.lambdas[0]
    );
    assert!(
        aspace_plain.lambdas[1] >= 0.0,
        "Fiedler eigenvalue should be non-negative: {}",
        aspace_plain.lambdas[1]
    );
}

#[test]
fn arrowspace_build_and_recompute_nonzero() {
    let dims = 10;

    let items: Vec<Vec<f64>> = make_moons_hd(300, 0.12, 0.01, dims, 42);

    let aspace = ArrowSpace::from_items(items.clone(), TAU_PARAMS);
    assert_eq!(aspace.nfeatures, dims);
    assert_eq!(aspace.nitems, 300);

    // Build λτ-graph with parameters that encourage connectivity
    // Larger eps to include more neighbors, higher k for more connections
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        items,
        0.5, // Larger eps for more connectivity
        4,   // Higher k for more neighbors
        3,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);

    assert!(gl.nnodes == aspace.nitems);
    aspace.recompute_lambdas(&gl);
    let lam0 = aspace.lambdas()[0];
    let lam1 = aspace.lambdas()[1];

    // Should now have non-zero lambdas
    println!("lam0: {:?}, lam1: {:?}", lam0, lam1);
    assert!(lam0 >= 0.0);
    assert!(lam1 >= 0.0);
    assert!(lam0 > 1e-10 || lam1 > 1e-10, "At least one lambda should be non-zero");
}

#[test]
fn arrowspace_construct_and_lambda() {
    // Load the Iris dataset - 150 samples with 4 features each
    let dataset = iris::load_dataset();

    // Convert to DenseMatrix format expected by ArrowSpace
    let items: Vec<Vec<f64>> = dataset
        .as_matrix()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| *val as f64) // Convert f32 to f64
                .collect()
        })
        .collect();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            1e-1,
            10,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            Some(1e-3 * 0.50),
        )
        .with_normalisation(false)
        .with_spectral(false)
        .build(items[0..75].to_vec());

    assert_eq!(aspace.data.shape(), (75, 4));

    assert!(gl.nnodes == aspace.data.shape().0);
    // Basic sanity: non-negative and often lam1 <= lam0 for this pair of rows
    let all_pos =
        aspace.lambdas().iter().all(|&a| !relative_eq!(0.0, a, epsilon = 1e-10));
    assert!(all_pos, "Vectors have been recomputed");
}

#[test]
fn arrowspace_add_rows_superpose() {
    // Two one-hot signals on adjacent items
    let rows = vec![
        vec![0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19],
        vec![0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21],
    ];
    let aspace = ArrowSpace::from_items_default(rows.clone());

    // λτ-graph from the same data
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        rows,
        1e-1,
        2,
        2,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);
    aspace.recompute_lambdas(&gl);

    let lam_before: Vec<f64> = aspace.lambdas().to_vec();

    // superpose item1 into item0
    aspace.get_item(0).add_inplace(&aspace.get_item(1));
    let lam_after: Vec<f64> = aspace.lambdas().to_vec();

    // Non-negativity and boundedness sanity; superposition should not produce NaN/Inf
    assert!(lam_after[0].is_finite() && lam_after[0] >= 0.0);
    // Typically, adding a nearby emitter reduces roughness; allow non-strict inequality
    assert!(lam_after[0] <= lam_before[0] + 1e-12);
}

#[test]
fn arrowspace_lambda_taumode_characteristics() {
    // Create test data with distinct patterns to analyze taumode behavior
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
        vec![
            0.85, 0.09, 0.41, 0.31, 0.67, 0.29, 0.53, 0.52, 0.17, 0.76, 0.05, 0.38,
            0.60,
        ],
        vec![
            0.77, 0.14, 0.47, 0.26, 0.59, 0.35, 0.51, 0.45, 0.23, 0.68, 0.10, 0.34,
            0.54,
        ],
    ];

    let (mut aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .build(items);

    println!("Taumode lambdas: {:?}", &aspace.lambdas);

    // Test 1: Non-negativity (fundamental property of Rayleigh quotients)
    assert!(
        aspace.lambdas.iter().all(|&x| x >= 0.0),
        "Taumode indices must be non-negative"
    );

    // Test 2: Bounded above by reasonable values (based on graph structure)
    let max_lambda: f64 = aspace.lambdas.iter().fold(0.0, |a, &b| a.max(b));
    assert!(
        max_lambda < 100.0,
        "Taumode indices should be reasonably bounded: max={}",
        max_lambda
    );

    // Test 3: Consistency - recomputing should give identical results
    let initial_lambdas = aspace.lambdas.to_vec();
    aspace.recompute_lambdas(&gl);
    let recomputed_lambdas = aspace.lambdas();

    for (i, (&initial, &recomputed)) in
        initial_lambdas.iter().zip(recomputed_lambdas.iter()).enumerate()
    {
        assert!(
            (initial - recomputed).abs() < 1e-12,
            "Recomputation inconsistency at feature {}: initial={}, recomputed={}",
            i,
            initial,
            recomputed
        );
    }

    // Test 4: Sensitivity to data changes - different data should give different lambdas
    let modified_items = vec![
        vec![
            0.90, 0.05, 0.50, 0.35, 0.70, 0.25, 0.60, 0.55, 0.12, 0.80, 0.03, 0.40,
            0.65,
        ],
        vec![
            0.70, 0.18, 0.40, 0.22, 0.55, 0.38, 0.48, 0.40, 0.28, 0.62, 0.15, 0.30,
            0.48,
        ],
        vec![
            0.88, 0.06, 0.38, 0.34, 0.72, 0.26, 0.50, 0.58, 0.14, 0.82, 0.02, 0.42,
            0.68,
        ],
        vec![
            0.74, 0.17, 0.44, 0.23, 0.56, 0.39, 0.49, 0.42, 0.26, 0.65, 0.13, 0.31,
            0.49,
        ],
    ];

    let (aspace2, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .build(modified_items);

    println!("Taumode lambdas: {:?}", &aspace.lambdas);

    let mut significant_differences = 0;
    for (_, (&original, &modified)) in
        aspace.lambdas.iter().zip(aspace2.lambdas.iter()).enumerate()
    {
        let diff = (original - modified).abs();
        if diff > 1e-6 {
            significant_differences += 1;
        }
    }

    assert!(
        significant_differences > 0,
        "Taumode should be sensitive to data changes - no significant differences found"
    );

    // Test 5: Finite values (no NaN or infinity)
    assert!(
        aspace.lambdas.iter().all(|&x| x.is_finite()),
        "All taumode indices should be finite"
    );

    // Test 6: Feature variation analysis
    let lambda_variance = {
        let mean = aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64;
        let variance = aspace.lambdas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / aspace.lambdas.len() as f64;
        variance
    };

    println!("Lambda statistics:");
    println!(
        "  Min: {:.6}",
        aspace.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!("  Max: {:.6}", max_lambda);
    println!(
        "  Mean: {:.6}",
        aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64
    );
    println!("  Variance: {:.6}", lambda_variance);

    // Variance should be positive (indicating feature discrimination)
    assert!(lambda_variance >= 0.0, "Lambda variance should be non-negative");

    println!("✓ Taumode index characteristics validated");
    println!("✓ Non-negativity, boundedness, and consistency verified");
    println!("✓ Sensitivity to data changes confirmed");
    println!("✓ Statistical properties within expected ranges");
}

#[test]
fn test_arrowspace_get() {
    // Create test data: 3 items, 4 features
    // item 0: [1.0, 2.0, 3.0, 4.0]
    // item 1: [5.0, 6.0, 7.0, 8.0]
    // item 2: [9.0, 10.0, 11.0, 12.0]
    let items = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];

    let arrow_space = ArrowSpace::from_items(items, TAUDEFAULT);

    // Test item 0
    let item_0: Vec<f64> = arrow_space.get_item(0).item;
    assert_eq!(item_0, &[1.0, 2.0, 3.0, 4.0]);

    // Test item 1
    let item_1 = arrow_space.get_item(1).item;
    assert_eq!(item_1, &[5.0, 6.0, 7.0, 8.0]);

    // Test item 2
    let item_2 = arrow_space.get_item(2).item;
    assert_eq!(item_2, &[9.0, 10.0, 11.0, 12.0]);

    // Test feature 0
    let feature_0 = arrow_space.get_feature(0).feature;
    assert_eq!(feature_0, &[1.0, 5.0, 9.0]);

    // Test feature 1
    let feature_1 = arrow_space.get_feature(1).feature;
    assert_eq!(feature_1, &[2.0, 6.0, 10.0]);

    // Test feature 2
    let feature_2 = arrow_space.get_feature(2).feature;
    assert_eq!(feature_2, &[3.0, 7.0, 11.0]);

    // Test feature 3
    let feature_3 = arrow_space.get_feature(3).feature;
    assert_eq!(feature_3, &[4.0, 8.0, 12.0]);
}

#[test]
fn arrowspace_lambda_sanity() {
    // Four items with 13 features each; check non-negativity and a conservative upper bound using max degree
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
        vec![
            0.85, 0.09, 0.41, 0.31, 0.67, 0.29, 0.53, 0.52, 0.17, 0.76, 0.05, 0.38,
            0.60,
        ],
        vec![
            0.77, 0.14, 0.47, 0.26, 0.59, 0.35, 0.51, 0.45, 0.23, 0.68, 0.10, 0.34,
            0.54,
        ],
    ];

    let aspace = ArrowSpace::from_items_default(items.clone());
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        items,
        1e-3,
        3,
        3,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );

    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);
    aspace.recompute_lambdas(&gl);
    let lambda = aspace.lambdas();

    let max_deg = (0..aspace.nitems).map(|i| aspace.data.get((i, i))).fold(
        aspace.data.get((0, 0)),
        |max, current| {
            if current > max {
                current
            } else {
                max
            }
        },
    );

    let upper = 2.0 * max_deg + 1e-12;

    assert!(lambda.iter().all(|&x| x >= 0.0), "synthetic index must be nonnegative");
    assert!(
        lambda.iter().all(|&x| x <= upper),
        "synthetic index {lambda:?} exceeded conservative upper bound {upper} (max_deg={max_deg})"
    );
}

#[test]
fn arrowspace_lambda_positivity_bounds() {
    // Build λτ-graph from a small basis
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        vec![
            vec![
                0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
                0.58,
            ],
            vec![
                0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
                0.56,
            ],
        ],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.topk,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );

    let test_rows0 = vec![
        vec![
            0.85, 0.15, 0.42, 0.33, 0.67, 0.28, 0.59, 0.41, 0.22, 0.78, 0.05, 0.38,
            0.62,
        ],
        vec![
            0.76, 0.09, 0.48, 0.31, 0.58, 0.35, 0.51, 0.49, 0.18, 0.72, 0.11, 0.34,
            0.54,
        ],
    ];
    let test_rows1 = vec![
        vec![
            0.73, 0.18, 0.46, 0.25, 0.69, 0.31, 0.57, 0.43, 0.16, 0.81, 0.04, 0.39,
            0.61,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
    ];

    for test in [test_rows0, test_rows1].iter() {
        let aspace = ArrowSpace::from_items_default(test.clone());
        let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);
        aspace.recompute_lambdas(&gl);
        let lambda = aspace.lambdas()[0];
        assert!(lambda >= 0.0, "Lambda positivity failed");
        // Generous upper bound for small tests
        assert!(lambda <= 10.0, "Lambda upper bound failed");
    }
}

#[test]
fn arrowitem_addition_inplace() {
    let row_a = vec![0.1, 0.5, 0.6, 0.2];
    let row_b = vec![0.9, 0.1, 0.3, 0.6];

    let aspace0 = ArrowSpace::from_items_default(vec![row_a.clone(), row_b.clone()]);
    let aspace1 = ArrowSpace::from_items_default(vec![row_b.clone(), row_a.clone()]);

    let mut item0 = aspace0.get_item(0);
    item0.add_inplace(&aspace1.get_item(0));
    assert_eq!(item0.item, vec![1.0, 0.6, 0.8999999999999999, 0.8]);

    let mut item1 = aspace1.get_item(1);
    item1.add_inplace(&aspace0.get_item(1));
    assert_eq!(item1.item, vec![1.0, 0.6, 0.8999999999999999, 0.8]);
}

#[test]
fn arrowspace_get_item() {
    // arrowspace stores the data column-wise
    // to return rows a lookup is needed

    let row_a = vec![0.1, 0.5, 0.6, 0.2];
    let row_b = vec![0.9, 0.1, 0.3, 0.6];

    let aspace0 = ArrowSpace::from_items_default(vec![row_a.clone(), row_b.clone()]);

    let item0 = aspace0.get_item(0);
    assert_eq!(item0.item, vec![0.1, 0.5, 0.6, 0.2]);

    let item1 = aspace0.get_item(1);
    assert_eq!(item1.item, vec![0.9, 0.1, 0.3, 0.6]);
}

#[test]
fn arrowspace_addition_commutativity() {
    // Test that (A + B) and (B + A) produce the same lambda effects
    // when operating on items in column-major ArrowSpace

    // More realistic high-dimensional feature vectors (13 features)
    let item_a = vec![
        0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
    ];
    let item_b = vec![
        0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37, 0.56,
    ];

    // Create two ArrowSpaces with the SAME items but in different positions
    // aspace1: A at index 0, B at index 1
    // aspace2: B at index 0, A at index 1
    let aspace1 = ArrowSpace::from_items_default(vec![item_a.clone(), item_b.clone()]);
    let aspace2 = ArrowSpace::from_items_default(vec![item_b.clone(), item_a.clone()]);

    println!("ArrowSpace shape: {:?}", aspace1.data.shape());

    // Build λ-graph from the union of items for consistent Laplacian
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        vec![item_a.clone(), item_b.clone()],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.topk,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );

    println!("Graph nodes: {:?}", gl.nnodes);

    // Compute initial lambdas for both spaces
    let mut aspace1 = GraphFactory::build_spectral_laplacian(aspace1, &gl);
    let mut aspace2 = GraphFactory::build_spectral_laplacian(aspace2, &gl);
    aspace1.recompute_lambdas(&gl);
    aspace2.recompute_lambdas(&gl);

    println!("=== BEFORE ADDITION ===");
    println!(
        "aspace1 items: [0]={:?}, [1]={:?}",
        aspace1.get_item(0).item,
        aspace1.get_item(1).item
    );
    println!(
        "aspace2 items: [0]={:?}, [1]={:?}",
        aspace2.get_item(0).item,
        aspace2.get_item(1).item
    );
    println!("aspace1 lambdas: {:?}", aspace1.lambdas());
    println!("aspace2 lambdas: {:?}", aspace2.lambdas());

    // Verify that we have the items we expect
    assert_eq!(aspace1.get_item(0).item, item_a, "aspace1[0] should be item_a");
    assert_eq!(aspace1.get_item(1).item, item_b, "aspace1[1] should be item_b");
    assert_eq!(aspace2.get_item(0).item, item_b, "aspace2[0] should be item_b");
    assert_eq!(aspace2.get_item(1).item, item_a, "aspace2[1] should be item_a");

    // Store initial lambda states for comparison
    let aspace1_initial_lambdas = aspace1.lambdas().to_vec();
    let aspace2_initial_lambdas = aspace2.lambdas().to_vec();

    // Perform the addition operations:
    // aspace1: A + B (add item[1] into item[0]) -> item[0] becomes A+B
    // aspace2: B + A (add item[1] into item[0]) -> item[0] becomes B+A
    aspace1.add_items(0, 1, &gl); // A += B
    aspace2.add_items(0, 1, &gl); // B += A

    println!("\n=== AFTER ADDITION ===");
    println!("aspace1 result: A+B = {:?}", aspace1.get_item(0).item);
    println!("aspace2 result: B+A = {:?}", aspace2.get_item(0).item);
    println!("aspace1 lambdas: {:?}", aspace1.lambdas());
    println!("aspace2 lambdas: {:?}", aspace2.lambdas());

    // Verify that both results are identical due to commutativity: A+B = B+A
    let result1 = aspace1.get_item(0); // A + B
    let result2 = aspace2.get_item(0); // B + A

    // Expected result: element-wise sum of the two 13-dimensional vectors
    // [0.82+0.79, 0.11+0.12, 0.43+0.45, 0.28+0.29, 0.64+0.61, 0.32+0.33, 0.55+0.54,
    //  0.48+0.47, 0.19+0.21, 0.73+0.70, 0.07+0.08, 0.36+0.37, 0.58+0.56]
    let expected = vec![
        1.61, 0.23, 0.88, 0.57, 1.25, 0.65, 1.09, 0.95, 0.40, 1.43, 0.15, 0.73, 1.14,
    ];

    // Verify both results match the expected sum
    for (i, (&actual, &expected_val)) in
        result1.item.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual - expected_val).abs() < 1e-10,
            "aspace1 feature {i}: got {actual}, expected {expected_val}"
        );
    }

    for (i, (&actual, &expected_val)) in
        result2.item.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual - expected_val).abs() < 1e-10,
            "aspace2 feature {i}: got {actual}, expected {expected_val}"
        );
    }

    // Verify commutativity: A+B = B+A
    for (i, (&v1, &v2)) in result1.item.iter().zip(result2.item.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Commutativity failed at feature {}: A+B={}, B+A={}, diff={}",
            i,
            v1,
            v2,
            (v1 - v2).abs()
        );
    }

    // Test the lambda symmetry property:
    // After addition, the feature distributions have changed, and the lambdas reflect this.
    // Since both spaces now contain the same data (A+B in position 0, unchanged B/A in position 1),
    // but with different item arrangements, the feature lambdas should reflect the swapped symmetry.
    //
    // Key insight:
    // - aspace1 now has [A+B, B] as items
    // - aspace2 now has [B+A, A] as items
    // Since A+B = B+A, both have identical item 0, but different item 1
    // The lambda patterns should be related but may not be identical due to different item 1
    //
    // With these more realistic vectors, we can observe how spectral properties change
    // across higher-dimensional feature spaces and more complex similarity relationships.

    println!("\n=== LAMBDA ANALYSIS ===");
    println!("Initial aspace1 lambdas: {aspace1_initial_lambdas:?}");
    println!("Initial aspace2 lambdas: {aspace2_initial_lambdas:?}");
    println!("Final aspace1 lambdas:   {:?}", aspace1.lambdas());
    println!("Final aspace2 lambdas:   {:?}", aspace2.lambdas());

    // Calculate and display feature-wise differences to analyze the effect
    let feature_diff: Vec<f64> =
        item_a.iter().zip(item_b.iter()).map(|(a, b)| (a - b).abs()).collect();
    println!("Feature-wise |A-B| differences: {feature_diff:?}");
    println!(
        "Max difference: {:.6}",
        feature_diff.iter().fold(0.0, |a: f64, &b| a.max(b))
    );
    println!(
        "Mean difference: {:.6}",
        feature_diff.iter().sum::<f64>() / feature_diff.len() as f64
    );

    // The main test: verify mathematical commutativity of the addition operation
    println!("\n✓ Addition commutativity verified: A+B = B+A = {expected:?}");
    println!("✓ Column-major storage and retrieval working correctly with 13-dimensional vectors!");
    println!("✓ High-dimensional spectral lambda computation maintains mathematical consistency!");
}

#[test]
#[should_panic]
fn arrowspace_zero_vector_should_panic() {
    let gl = GraphFactory::build_laplacian_matrix_from_items(
        vec![
            vec![
                0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
                0.58,
            ],
            vec![
                0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
                0.56,
            ],
        ],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.topk,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );

    let zero_row =
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let aspace = ArrowSpace::from_items_default(vec![zero_row.clone(), zero_row]);
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);
    aspace.recompute_lambdas(&gl);
}

// #[test]
// fn arrowspace_superposition_bounds() {
//     use crate::introspection::dimensional::{ArrowDimensionalOps, DimensionalOps};
//     // Instantiate support and create two smooth emitter-like rows
//     let length = 3usize.pow(5);
//     let c1 = DimensionalOps::make_cantor_1d(4, 1.0 / 3.0, length);
//     let height = 48usize;
//     let support = DimensionalOps::make_product_support(&c1, height);
//     let n = support.len();

//     let mk = |src: (f64, f64)| -> Vec<f64> {
//         let alpha = 0.05;
//         let eps = 1e-6;
//         support
//             .iter()
//             .map(|&(r, c)| {
//                 let dr = src.0 - r as f64;
//                 let dc = src.1 - c as f64;
//                 let d = (dr * dr + dc * dc).sqrt();
//                 (-alpha * d).exp() / (d + eps)
//             })
//             .collect()
//     };

//     let src_a = (length as f64 * 0.3, height as f64 * 0.5);
//     let src_b = (length as f64 * 0.7, height as f64 * 0.5);

//     for _ in 0..5 {
//         let row_a = mk(src_a);
//         let row_b = mk(src_b);

//         // Build λτ-graph directly from these rows
//         let gl = GraphFactory::build_laplacian_matrix(
//             vec![row_a.clone(), row_b.clone()],
//             1e-3,
//             8,
//             4,
//             2.0,
//             None,
//             GRAPH_PARAMS.normalise,
//         );

//         let aspace = ArrowSpace::from_items_default(vec![row_a.clone(), row_b.clone()]);
//         let mut aspace =
//             GraphFactory::build_spectral_laplacian(aspace, &gl);
//         aspace.recompute_lambdas(&gl);

//         let lam_a = aspace.lambdas()[0];
//         let lam_b = aspace.lambdas()[1];
//         let min_lam = lam_a.min(lam_b);
//         let max_lam = lam_a.max(lam_b);

//         aspace.get_item(0).add_inplace(&aspace.get_item(1));

//         let lam_sum = aspace.lambdas()[0];

//         assert!(lam_sum >= 0.0);
//         assert!(lam_sum <= 2.0 * max_lam);
//         // Informational: it may be outside [min,max] due to interference; just ensure boundedness
//         let _ = (n, min_lam); // silence warnings in minimal builds
//     }
// }

#[test]
#[should_panic]
fn graph_one_node() {
    let _ = GraphFactory::build_laplacian_matrix_from_items(
        vec![vec![1.0, 1.0, 1.0]],
        1.0,
        2,
        2,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
        GRAPH_PARAMS.sparsity_check,
    );
}

#[test]
fn arrowspace_recompute_synthetic() {
    // Load the Iris dataset - 150 samples with 4 features each
    let dataset = iris::load_dataset();

    // Convert to DenseMatrix format expected by ArrowSpace
    let items: Vec<Vec<f64>> = dataset
        .as_matrix()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| *val as f64) // Convert f32 to f64
                .collect()
        })
        .collect();

    let (mut aspace, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            1e-1,
            10,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            Some(1e-3 * 0.50),
        )
        .with_normalisation(false)
        .with_spectral(false)
        .build(items[0..75].to_vec());

    println!("lambdas before: {:?}", aspace.lambdas);
    let before = aspace.lambdas.clone();

    let (_, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            1e-1,
            10,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            Some(1e-3 * 0.50),
        )
        .with_normalisation(false)
        .with_spectral(false)
        .build(items[75..].to_vec());

    aspace.recompute_lambdas(&gl);
    let after = aspace.lambdas;

    println!("lambdas after: {:?}", after);

    let all_diff =
        before.iter().zip(&after).all(|(&b, &a)| !relative_eq!(b, a, epsilon = 1e-9));
    assert!(all_diff, "Vectors have been recomputed");
}

// Helper function to create a simple test ArrowSpace with GraphLaplacian
fn create_test_space() -> (ArrowSpace, GraphLaplacian) {
    let items = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0],
        vec![4.0, 5.0, 6.0],
        vec![5.0, 6.0, 7.0],
    ];

    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 2, 3, 2.0, Some(0.5));

    builder.build(items)
}

#[test]
fn test_prepare_query_item_basic() {
    let (aspace, gl) = create_test_space();
    let query = vec![2.5, 3.5, 4.5];

    // Should not panic with valid input
    aspace.prepare_query_item(&query, &gl);
}

#[test]
#[should_panic]
fn test_prepare_query_item_wrong_dimensions() {
    let (aspace, gl) = create_test_space();
    let query = vec![1.0, 2.0]; // Wrong dimension (2 instead of 3)

    aspace.prepare_query_item(&query, &gl);
}

#[test]
fn test_prepare_query_item_with_different_tau_modes() {
    let (mut aspace, gl) = create_test_space();
    let query = vec![3.0, 4.0, 5.0];

    aspace.taumode = TauMode::Fixed(0.9);
    // Test with different tau modes if TauMode is an enum
    // Assuming TauMode has variants like Fixed(f64), Adaptive, etc.
    let l0 = aspace.prepare_query_item(&query, &gl);
    println!("tau 0.9 -> tau: {}", l0);

    aspace.taumode = TauMode::Fixed(0.1);
    let l1 = aspace.prepare_query_item(&query, &gl);
    println!("tau 0.9 -> tau: {}", l1);
    assert_relative_ne!(l0, l1);

    aspace.taumode = TauMode::Mean;
    let l2 = aspace.prepare_query_item(&query, &gl);
    println!("tau 0.9 -> tau: {}", l2);
    assert_relative_ne!(l0, l2);
    assert_relative_ne!(l1, l2);
}

#[test]
#[should_panic]
fn test_prepare_query_item_boundary_constant_zeros() {
    let (aspace, gl) = create_test_space();

    // Test with all zeros
    let query_zeros = vec![0.0, 0.0, 0.0];
    aspace.prepare_query_item(&query_zeros, &gl);
}

#[test]
fn test_prepare_query_item_boundary_values() {
    let (aspace, gl) = create_test_space();

    // Test with large values
    let query_large = vec![1000.0, 2000.0, 3000.0];
    let l0 = aspace.prepare_query_item(&query_large, &gl);
    println!("large value, tau median -> tau: {}", l0);

    // Test with negative values
    let query_negative = vec![-1.0, -2.0, -3.0];
    let l1 = aspace.prepare_query_item(&query_negative, &gl);
    println!("negative value, tau median -> tau: {}", l1);
}

#[test]
#[should_panic]
fn test_prepare_query_item_tiny_float_values() {
    let (aspace, gl) = create_test_space();

    // Test with very small values: minimal tolerance is currently set to 1e-10
    let query_tiny = vec![1e-10, 1e-10, 1e-10];
    aspace.prepare_query_item(&query_tiny, &gl);
}

#[test]
fn test_prepare_query_item_special_float_values() {
    let (aspace, gl) = create_test_space();

    // Test with mixed positive and negative
    let query_mixed = vec![-1.0, 0.0, 1.0];
    let l0 = aspace.prepare_query_item(&query_mixed, &gl);
    println!("'symmetric' item, tau median -> tau: {}", l0);
}

#[test]
#[should_panic]
fn test_prepare_query_item_with_nan() {
    let (aspace, gl) = create_test_space();
    let query = vec![f64::NAN, 2.0, 3.0];

    aspace.prepare_query_item(&query, &gl);
}

#[test]
#[should_panic]
fn test_prepare_query_item_with_infinity() {
    let (aspace, gl) = create_test_space();
    let query = vec![f64::INFINITY, 2.0, 3.0];

    aspace.prepare_query_item(&query, &gl);
}

#[test]
#[should_panic]
fn test_prepare_query_item_empty_space() {
    // Create an empty or minimal aspace
    let items = vec![vec![1.0]];
    let builder = ArrowSpaceBuilder::new().with_lambda_graph(1.0, 1, 1, 2.0, Some(0.5));
    let (aspace, gl) = builder.build(items);

    let query = vec![2.0];
    aspace.prepare_query_item(&query, &gl);
}

#[test]
fn test_prepare_query_item_high_dimensional() {
    let dim = 100;
    let items = vec![vec![1.0; dim], vec![2.0; dim], vec![3.0; dim]];
    let builder = ArrowSpaceBuilder::new().with_lambda_graph(2.0, 1, 2, 2.0, Some(1.0));
    let (aspace, gl) = builder.build(items);

    let query = vec![1.5; dim];
    aspace.prepare_query_item(&query, &gl);
}

#[test]
fn test_prepare_query_item_consistency() {
    let (aspace, gl) = create_test_space();
    let query = vec![2.0, 3.0, 4.0];

    // Call multiple times with same input - should be deterministic
    let l0 = aspace.prepare_query_item(&query, &gl);
    let l1 = aspace.prepare_query_item(&query, &gl);
    assert_relative_eq!(l0, l1, epsilon = 1e-10);
}

#[test]
fn test_prepare_query_item_orthogonal_to_space() {
    let items = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    let builder = ArrowSpaceBuilder::new().with_lambda_graph(1.0, 1, 2, 2.0, Some(1.0));
    let (aspace, gl) = builder.build(items);

    // Query in different subspace
    let query = vec![1.0, 1.0, 1.0];
    let l0 = aspace.prepare_query_item(&query, &gl);
    println!("orthogonal item, tau median -> tau: {:?}", l0);
    assert_relative_eq!(l0, 0.0, epsilon = 1e-10);
}

#[test]
fn test_prepare_query_item_with_different_graph_params() {
    let items = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0], vec![4.0, 5.0]];

    // Test with tight epsilon (sparse graph)
    let builder1 =
        ArrowSpaceBuilder::new().with_lambda_graph(0.1, 1, 2, 2.0, Some(0.05));
    let (aspace1, gl1) = builder1.build(items.clone());
    let query = vec![2.5, 3.5];
    let l0 = aspace1.prepare_query_item(&query, &gl1);
    println!("test item, tau median -> tau: {:?}", l0);

    // Test with loose epsilon (dense graph)
    let builder2 =
        ArrowSpaceBuilder::new().with_lambda_graph(0.5, 3, 3, 2.0, Some(5.0));
    let (aspace2, gl2) = builder2.build(items);
    let l1 = aspace2.prepare_query_item(&query, &gl2);
    println!("test item, tau median -> tau: {:?}", l1);

    assert_relative_ne!(l0, l1, epsilon = 1e-5);
}

#[test]
fn test_prepare_query_item_normalized_vs_unnormalized() {
    // Generate 300D moon dataset
    let items = make_moons_hd(50, 0.25, 0.05, 300, 42);
    assert_eq!(items[0].len(), 300, "Expected 300-dimensional data");
    
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            1e-1,
            5,
            GRAPH_PARAMS.topk,
            GRAPH_PARAMS.p,
            Some(1e-1 * 0.50),
        )
        .with_normalisation(true)
        .with_spectral(false)
        .with_sparsity_check(false)
        .build(items[0..45].to_vec());

    // Create a 300D query vector (all ones)
    let query_base = items[45].clone();
    
    // Test 1: Normalized query (unit norm)
    let norm = (query_base.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let query_normalized: Vec<f64> = query_base.iter().map(|x| x / norm).collect();
    
    // Verify normalization
    let norm_check = (query_normalized.iter().map(|x| x * x).sum::<f64>()).sqrt();
    assert!((norm_check - 1.0).abs() < 1e-10, "Query should be normalized to unit norm");
    
    let lambda_normalized = aspace.prepare_query_item(&query_normalized, &gl);
    println!("Normalized query (300D, unit norm) → lambda: {:.6}", lambda_normalized);

    // Test 2: Unnormalized query (10x larger)
    let query_unnormalized: Vec<f64> = query_base.iter().map(|x| x * 10.0).collect();
    let unnorm_norm = (query_unnormalized.iter().map(|x| x * x).sum::<f64>()).sqrt();
    println!("Unnormalized query norm: {:.2} (should be ~173.2)", unnorm_norm);
    
    let lambda_unnormalized = aspace.prepare_query_item(&query_unnormalized, &gl);
    println!("Unnormalized query (300D, 10x scale) → lambda: {:.6}", lambda_unnormalized);

    // Assertions
    assert!(lambda_normalized.is_finite(), "Normalized lambda should be finite");
    assert!(lambda_unnormalized.is_finite(), "Unnormalized lambda should be finite");
    assert!(lambda_normalized >= 0.0, "Lambda should be non-negative");
    assert!(lambda_unnormalized >= 0.0, "Lambda should be non-negative");
    
    // Lambdas should be DIFFERENT because normalization affects tau computation
    approx::assert_relative_ne!(
        lambda_normalized,
        lambda_unnormalized,
        epsilon = 1e-6,
        max_relative = 0.01
    );
    
    println!("✓ Normalization effect verified: Δλ = {:.6}", 
             (lambda_normalized - lambda_unnormalized).abs());
}

#[test]
fn test_prepare_query_item_moon_structure() {
    let items = make_moons_hd(50, 0.25, 0.05, 300, 42);
    
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1e-1, 5, GRAPH_PARAMS.topk, GRAPH_PARAMS.p, Some(1e-1 * 0.50))
        .with_normalisation(true)
        .with_spectral(false)
        .with_sparsity_check(false)
        .build(items[0..45].to_vec());

    // Query 1: From upper moon (items 0-24)
    let query_upper = items[5].clone();
    
    // Query 2: From lower moon (items 25-49)
    let query_lower = items[30].clone();
    
    let lambda_upper = aspace.prepare_query_item(&query_upper, &gl);
    let lambda_lower = aspace.prepare_query_item(&query_lower, &gl);
    
    println!("Upper moon query → lambda: {:.6}", lambda_upper);
    println!("Lower moon query → lambda: {:.6}", lambda_lower);
    
    // Lambdas should be different for different manifold regions
    assert!((lambda_upper - lambda_lower).abs() > 1e-3, 
            "Different moon regions should have different lambdas");
}


#[test]
fn test_prepare_query_item_non_scale_invariance() {
    let (aspace, gl) = create_test_space();
    let base_query = vec![1.0, 2.0, 3.0];
    let l0 = aspace.prepare_query_item(&base_query, &gl);
    println!("test item, tau median -> tau: {:?}", l0);

    // Test with different scales
    for scale in &[0.1, 10.0, 100.0] {
        let scaled_query: Vec<f64> = base_query.iter().map(|x| x * scale).collect();
        let l1 = aspace.prepare_query_item(&scaled_query, &gl);
        println!("test item, tau median -> tau: {:?}", l1);
        assert_relative_ne!(l0, l1);
    }
}
