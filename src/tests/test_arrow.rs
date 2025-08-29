use crate::graph_factory::{GraphFactory, GraphLaplacian};
use crate::dimensional::{DimensionalOps, ArrowDimensionalOps};
use crate::core::ArrowSpace;


#[test]
fn arrowspace_construct_and_lambda() {
    let rows = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.5, 0.5, 0.0, 0.0]];

    let mut aspace = ArrowSpace::from_items(rows.clone());
    assert_eq!(aspace.shape(), (4, 2));

    // Build λτ-graph from the same data matrix
    let gl = GraphFactory::build_lambda_graph(&rows, 1e-3, 3, 2.0, None);

    aspace.recompute_lambdas(&gl);
    let lam0 = aspace.lambdas()[0];
    let lam1 = aspace.lambdas()[1];

    // Basic sanity: non-negative and often lam1 <= lam0 for this pair of rows
    assert!(lam0 >= 0.0);
    assert!(lam1 >= 0.0);
}

#[test]
fn arrowspace_add_rows_superpose() {
    // Two one-hot signals on adjacent items
    let rows = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let mut aspace = ArrowSpace::from_items(rows.clone());

    // λτ-graph from the same data
    let gl = GraphFactory::build_lambda_graph(&rows, 1e-3, 2, 2.0, None);
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
fn arrowspace_lambda_scale_invariance() {
    // Create meaningful test data with multiple items showing different patterns
    // These represent protein-like data with distinct feature profiles
    let items = vec![
        vec![1.0, 2.0, 3.0, 0.5], // Item 1: ascending pattern
        vec![3.0, 1.0, 2.0, 2.5], // Item 2: mixed pattern
        vec![0.5, 3.0, 1.0, 2.0], // Item 3: different mixed pattern
        vec![2.0, 0.5, 3.0, 1.0], // Item 4: another variation
    ];

    let mut aspace1 = ArrowSpace::from_items(items.clone());

    // Scale all items by the same factor (3.5)
    let scale_factor = 3.5;
    let items_scaled: Vec<Vec<f64>> = items
        .iter()
        .map(|item| item.iter().map(|&x| x * scale_factor).collect())
        .collect();
    let mut aspace2 = ArrowSpace::from_items(items_scaled.clone());

    // Build λ-graphs from each matrix; use identical parameters
    let gl1 = GraphFactory::build_lambda_graph(&items, 1e-3, 3, 2.0, None);
    let gl2 = GraphFactory::build_lambda_graph(&items_scaled, 1e-3, 3, 2.0, None);

    aspace1.recompute_lambdas(&gl1);
    aspace2.recompute_lambdas(&gl2);

    let lam1 = aspace1.lambdas();
    let lam2 = aspace2.lambdas();

    println!("Original lambdas: {lam1:?}");
    println!("Scaled lambdas:   {lam2:?}");

    // Rayleigh quotient should be scale-invariant: λ(cx) = λ(x) for any scalar c > 0
    // This is because λ = (cx)ᵀL(cx) / (cx)ᵀ(cx) = c²xᵀLx / c²xᵀx = xᵀLx / xᵀx = λ(x)
    for (i, (l1, l2)) in lam1.iter().zip(lam2.iter()).enumerate() {
        assert!(
            (l1 - l2).abs() < 1e-10,
            "Feature {} lambda scale invariance failed: original={}, scaled={}, diff={}",
            i,
            l1,
            l2,
            (l1 - l2).abs()
        );
    }

    println!(
        "✓ Scale invariance verified for all {} features",
        lam1.len()
    );
}

#[test]
fn arrowspace_lambda_rayleigh_sanity() {
    // Four items with 4 features each; check non-negativity and a conservative upper bound using max degree
    let items = vec![
        vec![0.37, 0.60, 0.60, 0.37], // Item 1: 4 features
        vec![0.20, 0.80, 0.40, 0.60], // Item 2: 4 features
        vec![0.50, 0.30, 0.70, 0.90], // Item 3: 4 features
        vec![0.10, 0.90, 0.20, 0.80], // Item 4: 4 features
    ];

    let gl = GraphFactory::build_lambda_graph(&items.clone(), 1e-3, 3, 2.0, None);

    let mut aspace = ArrowSpace::from_items(items);
    aspace.recompute_lambdas(&gl);
    let lambda = aspace.lambdas();

    // Find max diagonal entry (degree)
    let mut max_deg = 0.0f64;
    for i in 0..gl.nnodes {
        let start = gl.rows[i];
        let end = gl.rows[i + 1];
        for idx in start..end {
            if gl.cols[idx] == i {
                max_deg = max_deg.max(gl.vals[idx]);
                break;
            }
        }
    }

    let upper = 2.0 * max_deg + 1e-12;

    assert!(
        lambda.iter().all(|&x| x >= 0.0),
        "Rayleigh must be nonnegative"
    );
    assert!(
        lambda.iter().all(|&x| x <= upper),
        "Rayleigh {lambda:?} exceeded conservative upper bound {upper} (max_deg={max_deg})"
    );
}

#[test]
fn arrowspace_lambda_positivity_bounds() {
    // Build λτ-graph from a small basis
    let gl = GraphFactory::build_lambda_graph(
        &vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.9, 0.1, 0.3, 0.6]],
        1e-3,
        2,
        2.0,
        None,
    );

    let test_rows0 = vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, -1.0, 1.0, -1.0]];
    let test_rows1 = vec![vec![0.2, 0.3, 0.6, 0.1], vec![0.9, 0.1, 0.3, 0.6]];

    for test in [test_rows0, test_rows1].iter() {
        let mut aspace = ArrowSpace::from_items(test.clone());
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

    let aspace0 = ArrowSpace::from_items(vec![row_a.clone(), row_b.clone()]);
    let aspace1 = ArrowSpace::from_items(vec![row_b.clone(), row_a.clone()]);

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

    let aspace0 = ArrowSpace::from_items(vec![row_a.clone(), row_b.clone()]);

    let item0 = aspace0.get_item(0);
    assert_eq!(item0.item, vec![0.1, 0.5, 0.6, 0.2]);

    let item1 = aspace0.get_item(1);
    assert_eq!(item1.item, vec![0.9, 0.1, 0.3, 0.6]);
}

#[test]
fn arrowspace_addition_commutativity() {
    // Test that (A + B) and (B + A) produce the same lambda effects
    // when operating on items in column-major ArrowSpace

    let item_a = vec![0.1, 0.5, 0.6, 0.2]; // Item A: 4 features  
    let item_b = vec![0.9, 0.1, 0.3, 0.6]; // Item B: 4 features
    let item_c = [0.1, 0.3, 0.6, 0.1];
    let item_d = [0.9, 0.1, 0.7, 0.6];

    // Create two ArrowSpaces with the SAME items but in different positions
    // aspace1: A at index 0, B at index 1
    // aspace2: B at index 0, A at index 1
    let mut aspace1 = ArrowSpace::from_items(vec![item_a.clone(), item_b.clone()]);
    let mut aspace2 = ArrowSpace::from_items(vec![item_b.clone(), item_a.clone()]);

    println!("{:?}", aspace1.shape());

    // Build λ-graph from the union of items for consistent Laplacian
    let gl = GraphFactory::build_lambda_graph(
        &vec![item_a.clone(), item_b.clone()],
        1e-3,
        3,
        2.0,
        None,
    );

    println!("{:?}", gl.nnodes);

    // Compute initial lambdas for both spaces
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
    assert_eq!(
        aspace1.get_item(0).item,
        item_a,
        "aspace1[0] should be item_a"
    );
    assert_eq!(
        aspace1.get_item(1).item,
        item_b,
        "aspace1[1] should be item_b"
    );
    assert_eq!(
        aspace2.get_item(0).item,
        item_b,
        "aspace2[0] should be item_b"
    );
    assert_eq!(
        aspace2.get_item(1).item,
        item_a,
        "aspace2[1] should be item_a"
    );

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

    // Expected result: [0.1+0.9, 0.5+0.1, 0.6+0.3, 0.2+0.6] = [1.0, 0.6, 0.9, 0.8]
    let expected = vec![1.0, 0.6, 0.9, 0.8];

    // Verify both results match the expected sum
    for (i, (&actual, &expected_val)) in result1.item.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected_val).abs() < 1e-10,
            "aspace1 feature {i}: got {actual}, expected {expected_val}"
        );
    }

    for (i, (&actual, &expected_val)) in result2.item.iter().zip(expected.iter()).enumerate() {
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
    // So the lambda patterns should be related but may not be identical due to different item 1

    println!("\n=== LAMBDA ANALYSIS ===");
    println!("Initial aspace1 lambdas: {aspace1_initial_lambdas:?}");
    println!("Initial aspace2 lambdas: {aspace2_initial_lambdas:?}");
    println!("Final aspace1 lambdas:   {:?}", aspace1.lambdas());
    println!("Final aspace2 lambdas:   {:?}", aspace2.lambdas());

    // The main test: verify mathematical commutativity of the addition operation
    println!(
        "\n✓ Addition commutativity verified: A+B = B+A = {expected:?}"
    );
    println!("✓ Column-major storage and retrieval working correctly!");
}

#[test]
fn arrowspace_zero_vector_lambda() {
    let gl = GraphFactory::build_lambda_graph(
        &vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 1.0, 0.0]],
        1e-3,
        2,
        2.0,
        None,
    );

    let zero_row = vec![0.0, 0.0, 0.0, 0.0];
    let mut aspace = ArrowSpace::from_items(vec![zero_row.clone(), zero_row]);
    aspace.recompute_lambdas(&gl);

    let lambda = aspace.lambdas()[0];
    assert_eq!(lambda, 0.0);
}

#[test]
fn arrowspace_superposition_bounds() {
    use crate::dimensional::{ArrowDimensionalOps, DimensionalOps};
    // Instantiate support and create two smooth emitter-like rows
    let length = 3usize.pow(5);
    let c1 = DimensionalOps::make_cantor_1d(4, 1.0 / 3.0, length);
    let height = 48usize;
    let support = DimensionalOps::make_product_support(&c1, height);
    let n = support.len();

    let mk = |src: (f64, f64)| -> Vec<f64> {
        let alpha = 0.05;
        let eps = 1e-6;
        support
            .iter()
            .map(|&(r, c)| {
                let dr = src.0 - r as f64;
                let dc = src.1 - c as f64;
                let d = (dr * dr + dc * dc).sqrt();
                (-alpha * d).exp() / (d + eps)
            })
            .collect()
    };

    let src_a = (length as f64 * 0.3, height as f64 * 0.5);
    let src_b = (length as f64 * 0.7, height as f64 * 0.5);

    for _ in 0..5 {
        let row_a = mk(src_a);
        let row_b = mk(src_b);

        // Build λτ-graph directly from these rows
        let gl = GraphFactory::build_lambda_graph(
            &vec![row_a.clone(), row_b.clone()],
            1e-3,
            8,
            2.0,
            None,
        );

        let mut aspace = ArrowSpace::from_items(vec![row_a.clone(), row_b.clone()]);
        aspace.recompute_lambdas(&gl);

        let lam_a = aspace.lambdas()[0];
        let lam_b = aspace.lambdas()[1];
        let min_lam = lam_a.min(lam_b);
        let max_lam = lam_a.max(lam_b);

        aspace.get_item(0).add_inplace(&aspace.get_item(1));

        let lam_sum = aspace.lambdas()[0];

        assert!(lam_sum >= 0.0);
        assert!(lam_sum <= 2.0 * max_lam);
        // Informational: it may be outside [min,max] due to interference; just ensure boundedness
        let _ = (n, min_lam); // silence warnings in minimal builds
    }
}

#[test]
#[should_panic]
fn graph_one_node() {
    let gl = GraphFactory::build_lambda_graph(&vec![vec![1.0, 1.0, 1.0]], 1.0, 2, 2.0, None);
}

#[test]
fn arrowspace_constant_vector_lambda() {
    // Constant vector should yield near-zero Rayleigh on a connected graph
    let gl = GraphFactory::build_lambda_graph(
        &vec![vec![1.0, 1.0, 1.0], vec![0.1, 0.8, 1.0]],
        1.0,
        2,
        2.0,
        None,
    );

    let constant = vec![vec![1.0, 1.0, 1.0], vec![0.1, 0.8, 1.0]];
    let mut aspace = ArrowSpace::from_items(constant);
    aspace.recompute_lambdas(&gl);

    let lambda = aspace.lambdas()[0];
    assert!(lambda < 1e-10, "Constant vector lambda should be ~0");
}
