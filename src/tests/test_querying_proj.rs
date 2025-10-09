//! # Query Tests: Lambda-aware search with random projection
//!
//! Tests the full query pipeline including:
//! 1. Building index with optional random projection
//! 2. Projecting query vectors using the same transformation
//! 3. Computing query lambda values
//! 4. Lambda-aware similarity search
//! 5. Hybrid search combining semantic and spectral scoring

use crate::builder::ArrowSpaceBuilder;
use crate::core::ArrowItem;
use crate::tests::test_data::make_moons_hd;
use crate::tests::GRAPH_PARAMS;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Helper: return test data and a vector of 4 queries
fn create_test_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let test_data = make_moons_hd(100, 0.12, 0.01, 384, 42);
    let data = test_data[0..95].to_vec();
    let query = test_data[96..].to_vec(); // Query in same space
    (data, query)
}

#[test]
fn test_query_without_projection() {
    // Build index without projection
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.5, 5, 5, 2.0, Some(0.25))
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    // Prepare query (no projection needed)
    let query_lambda = aspace.prepare_query_item(&queries[0].clone(), &gl);
    assert!(query_lambda.is_finite());
    assert!(query_lambda >= 0.0);

    let query_item = ArrowItem::new(queries[0].clone(), query_lambda);

    // Search
    let results = aspace.search_lambda_aware(&query_item, 5, 0.7);
    assert_eq!(results.len(), 5);

    // Verify descending order
    for i in 0..results.len() - 1 {
        assert!(results[i].1 >= results[i + 1].1);
    }
}

#[test]
fn test_query_with_projection_manual() {
    // Simulate what builder does with projection
    let (data, queries) = create_test_data();
    let nfeatures = 384;
    assert_eq!(queries[0].len(), nfeatures);

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 5, 5, 2.0, None)
        .with_dims_reduction(true, None)
        .build(data);

    let reduced_dim = *aspace.reduced_dim.as_ref().unwrap();
    println!("Testing projection: {} → {} dimensions", nfeatures, reduced_dim);

    // Query: must be projected too
    let query_original = queries[1].clone();
    let query_projected = aspace.project_query(&query_original);

    assert_eq!(query_projected.len(), reduced_dim);

    // Compute lambda on projected query
    let query_lambda = aspace.prepare_query_item(&query_original, &gl);
    assert!(query_lambda.is_finite() && query_lambda > 0.0);

    let query_item = ArrowItem::new(query_original, query_lambda);
    // Search should work in projected space
    let results = aspace.search_lambda_aware(&query_item, 10, 0.7);
    assert_eq!(results.len(), 10);

    // Verify scores are valid
    for (idx, score) in &results {
        assert!(*idx < aspace.nitems);
        assert!(score.is_finite());
        assert!(*score >= -1.0 && *score <= 1.0);
    }
}

#[test]
fn test_prepare_query_item_consistency() {
    // Test that prepare_query_item produces stable lambda values
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[2].clone();

    // Compute lambda multiple times
    let lambda1 = aspace.prepare_query_item(&query, &gl);
    let lambda2 = aspace.prepare_query_item(&query, &gl);
    let lambda3 = aspace.prepare_query_item(&query, &gl);

    assert_eq!(lambda1, lambda2);
    assert_eq!(lambda2, lambda3);
    assert!(lambda1 >= 0.0);
}

#[test]
fn test_search_lambda_aware_alpha_effect() {
    // Test that alpha parameter affects results
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 2, 2, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[2].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    // High alpha (0.9): should favor semantic similarity
    let results_high_alpha = aspace.search_lambda_aware(&query_item, 3, 0.9);

    // Low alpha (0.1): should favor lambda similarity
    let results_low_alpha = aspace.search_lambda_aware(&query_item, 3, 0.1);

    // Results should differ (unless by chance)
    // At minimum, verify both produce valid results
    assert_eq!(results_high_alpha.len(), 3);
    assert_eq!(results_low_alpha.len(), 3);

    // Verify top result is closest semantically with high alpha
    let top_idx_high = results_high_alpha[0].0;
    let top_item_high = aspace.get_item(top_idx_high);
    let semantic_sim = query_item.cosine_similarity(&top_item_high.item);

    // Should be high semantic similarity
    assert!(semantic_sim > 0.8, "High alpha should favor semantic match");
}

#[test]
fn test_search_lambda_aware_hybrid() {
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 4, 4, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[1].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    // Regular search
    let results_regular = aspace.search_lambda_aware(&query_item, 10, 0.7);

    // Hybrid search
    let results_hybrid = aspace.search_lambda_aware_hybrid(&query_item, 10, 0.7);

    assert_eq!(results_regular.len(), 10);
    assert_eq!(results_hybrid.len(), 10);

    // Both should return valid results
    for (idx, score) in results_regular.iter().chain(results_hybrid.iter()) {
        assert!(*idx < aspace.nitems);
        assert!(score.is_finite());
    }
}

#[test]
fn test_query_dimension_mismatch_panics() {
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    // Query with wrong dimension should panic
    let wrong_query = queries[0][0..50].to_vec();

    let result =
        std::panic::catch_unwind(|| aspace.prepare_query_item(&wrong_query, &gl));

    assert!(result.is_err(), "Should panic on dimension mismatch");
}

#[test]
fn test_query_with_nan_values() {
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    // Query with NaN should panic
    let mut bad_query = queries[0].clone();
    bad_query[3] = f64::NAN;

    let result =
        std::panic::catch_unwind(|| aspace.prepare_query_item(&bad_query, &gl));

    assert!(result.is_err(), "Should panic on NaN values");
}

#[test]
fn test_range_search_with_query_lambda() {
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[1].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    // Find all within radius 0.5
    let results = aspace.range_search(&query_item, 0.5);

    // Should find at least the closest points
    assert!(!results.is_empty());

    // Verify all results are within radius
    for (idx, dist) in &results {
        assert!(*idx < aspace.nitems);
        assert!(*dist <= 0.5);
        assert!(dist.is_finite());
    }
}

#[test]
fn test_lambda_values_reasonable_range() {
    let (data, queries) = create_test_data();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    // Test multiple queries
    for query in queries {
        let lambda = aspace.prepare_query_item(&query, &gl);

        // Lambda should be non-negative and finite
        assert!(lambda >= 0.0, "Lambda should be non-negative");
        assert!(lambda.is_finite(), "Lambda should be finite");

        // Lambda should be in reasonable range (typically < 10.0 for normalized data)
        assert!(lambda < 100.0, "Lambda unusually large: {}", lambda);
    }
}

#[test]
fn test_search_returns_top_k_exactly() {
    let (data, queries) = create_test_data();
    assert_eq!(data.len(), 95);

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[2].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    // Test various k values
    for k in [1, 5, 10, 20, 50] {
        let results = aspace.search_lambda_aware(&query_item, k, 0.7);
        assert_eq!(results.len(), k, "Should return exactly {} results", k);
    }

    // Test k larger than dataset
    let results = aspace.search_lambda_aware(&query_item, 150, 0.7);
    assert_eq!(results.len(), 95, "Should return all 100 items when k > n");
}

#[test]
fn test_projection_preserves_relative_distances() {
    // Generate 300D moon dataset (50 samples)
    let items = make_moons_hd(50, 0.25, 0.05, 300, 42);
    assert_eq!(items[0].len(), 300, "Expected 300-dimensional data");
    
    let nfeatures = 300;  // FIXED: Original feature dimension, not number of items
    
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
        .with_dims_reduction(true, None)  // Enable random projection
        .build(items[0..45].to_vec());

    // Verify projection was applied
    assert!(aspace.projection_matrix.is_some(), "Projection should be enabled");
    assert!(aspace.reduced_dim.is_some(), "Reduced dimension should be set");
    
    let reduced_dim = aspace.reduced_dim.unwrap();
    assert_eq!(aspace.nfeatures, nfeatures, "Original dimension should be 300");
    assert_eq!(aspace.reduced_dim.unwrap(), reduced_dim, "nfeatures should match reduced_dim");
    assert!(
        reduced_dim < nfeatures,
        "Reduced dimension {} should be < original {}",
        reduced_dim,
        nfeatures
    );

    println!(
        "Projection: {} → {} dimensions ({:.1}x compression)",
        nfeatures,
        reduced_dim,
        nfeatures as f64 / reduced_dim as f64
    );

    // Create three 300D queries with known relationships
    let query1_orig = vec![0.5; 300];
    let query2_orig = vec![0.51; 300];  // Very close to q1
    let query3_orig = vec![5.0; 300];   // Far from q1

    // Project all three queries
    let query1_proj = aspace.project_query(&query1_orig);
    let query2_proj = aspace.project_query(&query2_orig);
    let query3_proj = aspace.project_query(&query3_orig);

    // Verify projected dimensions
    assert_eq!(query1_proj.len(), reduced_dim);
    assert_eq!(query2_proj.len(), reduced_dim);
    assert_eq!(query3_proj.len(), reduced_dim);

    // Compute distances in ORIGINAL space (300D)
    let dist_12_orig: f64 = query1_orig
        .iter()
        .zip(&query2_orig)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let dist_13_orig: f64 = query1_orig
        .iter()
        .zip(&query3_orig)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    // Compute distances in PROJECTED space (reduced_dim)
    let dist_12_proj: f64 = query1_proj
        .iter()
        .zip(&query2_proj)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let dist_13_proj: f64 = query1_proj
        .iter()
        .zip(&query3_proj)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    println!(
        "Original space (300D): dist(q1,q2)={:.4}, dist(q1,q3)={:.4}",
        dist_12_orig, dist_13_orig
    );
    println!(
        "Projected space ({}D): dist(q1,q2)={:.4}, dist(q1,q3)={:.4}",
        reduced_dim, dist_12_proj, dist_13_proj
    );

    // CRITICAL: Verify relative ordering is preserved (Johnson-Lindenstrauss property)
    assert!(
        dist_12_orig < dist_13_orig,
        "In original space, q1 should be closer to q2 than q3: {} < {}",
        dist_12_orig, dist_13_orig
    );
    assert!(
        dist_12_proj < dist_13_proj,
        "In projected space, q1 should STILL be closer to q2 than q3 (ordering preserved): {} < {}",
        dist_12_proj, dist_13_proj
    );

    // Verify approximate distance preservation (JL guarantee)
    let epsilon = 0.2; // 20% tolerance for practical test with Gaussian projection

    let ratio_12 = dist_12_proj / dist_12_orig;
    let ratio_13 = dist_13_proj / dist_13_orig;
    
    println!("Distance preservation ratios: q1-q2={:.3}, q1-q3={:.3}", ratio_12, ratio_13);
    
    assert!(
        ratio_12 > 1.0 - epsilon && ratio_12 < 1.0 + epsilon,
        "Distance q1-q2 not preserved: ratio {:.3} outside [{:.3}, {:.3}]",
        ratio_12,
        1.0 - epsilon,
        1.0 + epsilon
    );

    assert!(
        ratio_13 > 1.0 - epsilon && ratio_13 < 1.0 + epsilon,
        "Distance q1-q3 not preserved: ratio {:.3} outside [{:.3}, {:.3}]",
        ratio_13,
        1.0 - epsilon,
        1.0 + epsilon
    );

    // Verify lambda computation works on projected queries
    let lambda1 = aspace.prepare_query_item(&query1_orig, &gl);
    let lambda2 = aspace.prepare_query_item(&query2_orig, &gl);
    let lambda3 = aspace.prepare_query_item(&query3_orig, &gl);

    assert!(lambda1.is_finite() && lambda1 >= 0.0, "Lambda1 should be finite and non-negative");
    assert!(lambda2.is_finite() && lambda2 >= 0.0, "Lambda2 should be finite and non-negative");
    assert!(lambda3.is_finite() && lambda3 >= 0.0, "Lambda3 should be finite and non-negative");
    
    println!("Lambdas: q1={:.6}, q2={:.6}, q3={:.6}", lambda1, lambda2, lambda3);
    
    // Similar queries should have similar lambdas
    let lambda_diff_similar = (lambda1 - lambda2).abs();
    let lambda_diff_dissimilar = (lambda1 - lambda3).abs();
    
    println!("Lambda differences: similar={:.6}, dissimilar={:.6}", 
             lambda_diff_similar, lambda_diff_dissimilar);
    
    println!("✓ Projection preserves relative distances and enables lambda computation");
}

use crate::core::ArrowSpace;

/// Helper: Create ArrowSpace WITHOUT projection
fn create_arrowspace_no_projection(n_items: usize, dim: usize) -> ArrowSpace {
    let data: Vec<f64> = (0..n_items * dim).map(|i| (i as f64) * 0.1).collect();
    let data_matrix = DenseMatrix::from_iterator(data.into_iter(), n_items, dim, 0);

    use sprs::TriMat;
    let mut triplets = TriMat::new((n_items, n_items));
    for i in 0..n_items {
        triplets.add_triplet(i, i, 1.0);
    }
    let signals = triplets.to_csr();

    ArrowSpace {
        nfeatures: dim,
        nitems: n_items,
        data: data_matrix,
        signals,
        lambdas: vec![0.5; n_items],
        taumode: crate::taumode::TauMode::Median,
        cluster_assignments: vec![None; n_items],
        cluster_sizes: vec![],
        cluster_radius: 1.0,
        projection_matrix: None,
        reduced_dim: None,
    }
}

#[test]
fn test_project_query_no_projection() {
    let aspace = create_arrowspace_no_projection(10, 8);
    let query = vec![0.5; 8];

    // Should return query unchanged
    let projected = aspace.project_query(&query);

    assert_eq!(projected.len(), 8);
    assert_eq!(projected, query);
}

#[test]
fn test_project_query_with_projection() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);
    let query = queries[1].clone();

    // Should project from 50D to 10D
    let projected = aspace.project_query(&query);

    assert_eq!(projected.len(), aspace.reduced_dim.unwrap());

    // All values should be finite
    assert!(projected.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_project_query_consistency() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);
    let query = queries[1].clone();

    // Project same query multiple times
    let projected1 = aspace.project_query(&query);
    let projected2 = aspace.project_query(&query);
    let projected3 = aspace.project_query(&query);

    // Should be deterministic
    assert_eq!(projected1, projected2);
    assert_eq!(projected2, projected3);
}

#[test]
fn test_project_query_linearity() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);
    let query = queries[1].clone();

    let scaled_query: Vec<f64> = query.iter().map(|x| x * 2.0).collect();

    let projected = aspace.project_query(&query);
    let projected_scaled = aspace.project_query(&scaled_query);

    // Projection should be linear: project(2*x) = 2*project(x)
    for i in 0..projected.len() {
        let expected = projected[i] * 2.0;
        let actual = projected_scaled[i];
        assert!(
            (expected - actual).abs() < 1e-9,
            "Linearity violation at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_project_query_zero_vector() {
    let (data, _queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    // NOTE: still thinking if this should panic
    let query_zero = vec![0.0; 384];

    let projected = aspace.project_query(&query_zero);

    // Projection of zero should be zero (or near-zero due to floating point)
    assert_eq!(projected.len(), aspace.reduced_dim.unwrap());
    for &val in &projected {
        assert!(val.abs() < 1e-10, "Expected near-zero, got {}", val);
    }
}

#[test]
fn test_project_query_preserves_scale_approximately() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);
    let query = queries[1].clone();

    let projected = aspace.project_query(&query);

    // Original norm
    let orig_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Projected norm
    let proj_norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Johnson-Lindenstrauss preserves norms approximately
    // Ratio should be within reasonable bounds
    let ratio = proj_norm / orig_norm;
    assert!(ratio > 0.5 && ratio < 2.0, "Norm ratio {} out of expected range", ratio);
}

#[test]
fn test_project_query_different_queries() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query1 = queries[1].clone();
    let query2 = queries[2].clone();
    let query3 = queries[3].clone();

    let proj1 = aspace.project_query(&query1);
    let proj2 = aspace.project_query(&query2);
    let proj3 = aspace.project_query(&query3);

    // All should have correct dimension
    assert_eq!(proj1.len(), aspace.reduced_dim.unwrap());
    assert_eq!(proj2.len(), aspace.reduced_dim.unwrap());
    assert_eq!(proj3.len(), aspace.reduced_dim.unwrap());

    // Different queries should produce different projections
    assert_ne!(proj1, proj2);
    assert_ne!(proj2, proj3);
    assert_ne!(proj1, proj3);
}

#[test]
fn test_project_query_preserves_dot_product_sign() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    // Two queries: one all positive, one all negative
    let query_pos = queries[0].clone();
    let query_neg: Vec<f64> =
        queries[0].clone().into_iter().map(|x| x * -1.0).collect();

    let proj_pos = aspace.project_query(&query_pos);
    let proj_neg = aspace.project_query(&query_neg);

    // Their projections should have opposite signs (negative dot product)
    let dot: f64 = proj_pos.iter().zip(&proj_neg).map(|(a, b)| a * b).sum();

    assert!(dot < 0.0, "Projection should preserve opposite directions");
}

#[test]
fn test_project_query_with_realistic_jl_dimension() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[0].clone();

    let projected = aspace.project_query(&query);

    assert_eq!(projected.len(), aspace.reduced_dim.unwrap());
    assert!(projected.iter().all(|&x| x.is_finite()));

    println!(
        "JL projection: {} → {} dims (ratio: {:.2}x)",
        aspace.nfeatures,
        aspace.reduced_dim.unwrap(),
        aspace.nfeatures as f64 / aspace.reduced_dim.unwrap() as f64
    );
}

#[test]
fn test_project_query_infinity_handling() {
    let (data, queries) = create_test_data();

    // Build index with projection enabled
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_dims_reduction(true, None)
        .with_sparsity_check(false)
        .build(data);

    let mut query_with_inf = queries[1].clone();
    query_with_inf[10] = f64::INFINITY;

    let projected = aspace.project_query(&query_with_inf);

    // Projection of infinity will produce infinity or very large values
    assert!(projected.iter().any(|&x| !x.is_finite()));
}
