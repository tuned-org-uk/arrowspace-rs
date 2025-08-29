 use crate::core::ArrowSpace;
 use crate::taumode::{TAU_FLOOR, select_tau, TauMode, compute_synthetic_lambdas};
 use crate::graph_factory::GraphFactory;

#[test]
fn test_select_tau_fixed() {
    // Valid fixed tau
    let energies = vec![0.1, 0.5, 1.0];
    assert_eq!(select_tau(&energies, TauMode::Fixed(0.3)), 0.3);

    // Invalid fixed tau should return floor
    assert_eq!(select_tau(&energies, TauMode::Fixed(-0.1)), TAU_FLOOR);
    assert_eq!(select_tau(&energies, TauMode::Fixed(0.0)), TAU_FLOOR);
    assert_eq!(select_tau(&energies, TauMode::Fixed(f64::NAN)), TAU_FLOOR);
    assert_eq!(
        select_tau(&energies, TauMode::Fixed(f64::INFINITY)),
        TAU_FLOOR
    );
}

#[test]
fn test_select_tau_mean() {
    // Normal case
    let energies = vec![1.0, 2.0, 3.0];
    let expected_mean = 2.0;
    assert!((select_tau(&energies, TauMode::Mean) - expected_mean).abs() < 1e-12);

    // With NaN/Inf values - should filter them out
    let energies_with_nan = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 2.0];
    let expected_filtered_mean = 2.0; // (1.0 + 3.0 + 2.0) / 3
    assert!(
        (select_tau(&energies_with_nan, TauMode::Mean) - expected_filtered_mean).abs() < 1e-12
    );

    // All NaN/Inf should return floor
    let all_invalid = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    assert_eq!(select_tau(&all_invalid, TauMode::Mean), TAU_FLOOR);

    // Empty array should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(select_tau(&empty, TauMode::Mean), TAU_FLOOR);
}

#[test]
fn test_select_tau_median() {
    // Odd number of elements
    let energies_odd = vec![3.0, 1.0, 2.0];
    assert_eq!(select_tau(&energies_odd, TauMode::Median), 2.0);

    // Even number of elements
    let energies_even = vec![1.0, 2.0, 3.0, 4.0];
    let expected_median = 2.5; // (2.0 + 3.0) / 2
    assert!((select_tau(&energies_even, TauMode::Median) - expected_median).abs() < 1e-12);

    // Single element
    let single = vec![5.0];
    assert_eq!(select_tau(&single, TauMode::Median), 5.0);

    // With NaN/Inf - should filter them out
    let with_invalid = vec![f64::NAN, 1.0, 3.0, f64::INFINITY, 2.0];
    assert_eq!(select_tau(&with_invalid, TauMode::Median), 2.0);

    // All invalid should return floor
    let all_invalid = vec![f64::NAN, f64::INFINITY];
    assert_eq!(select_tau(&all_invalid, TauMode::Median), TAU_FLOOR);

    // Empty should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(select_tau(&empty, TauMode::Median), TAU_FLOOR);
}

#[test]
fn test_select_tau_percentile() {
    let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // 0th percentile (minimum)
    assert_eq!(select_tau(&energies, TauMode::Percentile(0.0)), 1.0);

    // 100th percentile (maximum)
    assert_eq!(select_tau(&energies, TauMode::Percentile(1.0)), 5.0);

    // 50th percentile (median)
    assert_eq!(select_tau(&energies, TauMode::Percentile(0.5)), 3.0);

    // Out of bounds percentiles should be clamped
    assert_eq!(select_tau(&energies, TauMode::Percentile(-0.1)), 1.0);
    assert_eq!(select_tau(&energies, TauMode::Percentile(1.5)), 5.0);

    // Empty array should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(select_tau(&empty, TauMode::Percentile(0.5)), TAU_FLOOR);
}

#[test]
fn test_select_tau_floor_enforcement() {
    // Very small positive values should be preserved if above floor
    let small_positive = vec![TAU_FLOOR * 2.0];
    assert_eq!(select_tau(&small_positive, TauMode::Mean), TAU_FLOOR * 2.0);

    // Values below floor should be raised to floor
    let below_floor = vec![TAU_FLOOR / 2.0];
    assert_eq!(select_tau(&below_floor, TauMode::Mean), TAU_FLOOR);

    // Zero should be raised to floor
    let zero = vec![0.0];
    assert_eq!(select_tau(&zero, TauMode::Mean), TAU_FLOOR);
}

#[test]
fn test_compute_synthetic_lambdas_basic() {
    // Create a simple 2x2 ArrowSpace (2 features, 2 items)
    let items = vec![
        vec![1.0, 0.0], // Item 0: high in feature 0, low in feature 1
        vec![0.0, 1.0], // Item 1: low in feature 0, high in feature 1
    ];

    let mut aspace = ArrowSpace::from_items(items.clone());

    // Create a simple graph (items similarity)
    let gl = GraphFactory::build_lambda_graph(&items, 1e-3, 2, 2.0, None);

    // Apply synthetic lambda computation
    let alpha = 0.7;
    let tau_mode = TauMode::Fixed(0.1);
    compute_synthetic_lambdas(&mut aspace, &gl, alpha, tau_mode);

    let lambdas = aspace.lambdas();
    assert_eq!(lambdas.len(), 2); // Should have lambda for each feature

    // All lambdas should be finite and non-negative
    assert!(lambdas.iter().all(|&l| l.is_finite() && l >= 0.0));

    // Lambdas should be bounded between 0 and 1 due to the bounded transform
    assert!(lambdas.iter().all(|&l| l <= 1.0));
}

#[test]
fn test_compute_synthetic_lambdas_different_alpha() {
    let items = vec![
        vec![2.0, 1.0, 0.5],
        vec![0.5, 2.0, 1.0],
        vec![1.0, 0.5, 2.0],
    ];

    let gl = GraphFactory::build_lambda_graph(&items, 1e-3, 3, 2.0, None);

    // Test with alpha = 1.0 (pure energy term)
    let mut aspace1 = ArrowSpace::from_items(items.clone());
    compute_synthetic_lambdas(&mut aspace1, &gl, 1.0, TauMode::Fixed(0.1));
    let lambdas1 = aspace1.lambdas().to_vec();

    // Test with alpha = 0.0 (pure dispersion term)
    let mut aspace2 = ArrowSpace::from_items(items.clone());
    compute_synthetic_lambdas(&mut aspace2, &gl, 0.0, TauMode::Fixed(0.1));
    let lambdas2 = aspace2.lambdas().to_vec();

    // Test with alpha = 0.5 (balanced)
    let mut aspace3 = ArrowSpace::from_items(items);
    compute_synthetic_lambdas(&mut aspace3, &gl, 0.5, TauMode::Fixed(0.1));
    let lambdas3 = aspace3.lambdas().to_vec();

    // All should be different (unless edge case)
    assert_eq!(lambdas1.len(), lambdas2.len());
    assert_eq!(lambdas2.len(), lambdas3.len());

    // All should be finite and bounded
    for lambdas in [&lambdas1, &lambdas2, &lambdas3] {
        assert!(
            lambdas
                .iter()
                .all(|&l| l.is_finite() && (0.0..=1.0).contains(&l))
        );
    }
}

#[test]
fn test_compute_synthetic_lambdas_zero_vectors() {
    // Test with zero vectors (should handle gracefully)
    let items = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

    let mut aspace = ArrowSpace::from_items(items.clone());
    let gl = GraphFactory::build_lambda_graph(&items, 1e-3, 2, 2.0, None);

    compute_synthetic_lambdas(&mut aspace, &gl, 0.7, TauMode::Median);

    let lambdas = aspace.lambdas();
    // Zero vectors should produce zero synthetic lambdas
    assert!(lambdas.iter().all(|&l| l == 0.0));
}

// #[test]
// fn test_compute_synthetic_lambdas_constant_vectors() {
//     // Test with constant vectors (should have low energy on connected graphs)
//     let items = vec![
//         vec![1.0, 1.0, 1.0],
//         vec![2.0, 2.0, 2.0],
//     ];

//     let mut aspace = ArrowSpace::from_items(items.clone());
//     let gl = GraphFactory::build_lambda_graph(&items, 1e-3, Some(2), 2.0, None);

//     compute_synthetic_lambdas(&mut aspace, &gl, 0.7, TauMode::Median);

//     let lambdas = aspace.lambdas();
//     println!("{:?}", lambdas);
//     // Constant vectors should have very low synthetic lambdas
//     assert!(lambdas.iter().all(|&l| l < 0.1));
// }

#[test]
fn test_compute_synthetic_lambdas_different_tau_modes() {
    let items = vec![
        vec![1.0, 2.0, 3.0],
        vec![3.0, 1.0, 2.0],
        vec![2.0, 3.0, 1.0],
    ];

    let gl = GraphFactory::build_lambda_graph(&items, 1e-3, 3, 2.0, None);

    // Test different tau modes
    let tau_modes = vec![
        TauMode::Fixed(0.1),
        TauMode::Mean,
        TauMode::Median,
        TauMode::Percentile(0.25),
    ];

    for tau_mode in tau_modes {
        let mut aspace = ArrowSpace::from_items(items.clone());
        compute_synthetic_lambdas(&mut aspace, &gl, 0.7, tau_mode);

        let lambdas = aspace.lambdas();

        // All modes should produce valid results
        assert!(
            lambdas
                .iter()
                .all(|&l| l.is_finite() && (0.0..=1.0).contains(&l))
        );
        assert_eq!(lambdas.len(), 3); // One lambda per feature
    }
}

#[test]
#[should_panic]
fn test_compute_synthetic_lambdas_graph_mismatch_panics() {
    let items = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let aspace = ArrowSpace::from_items(items.clone());

    // Create a graph with wrong number of features
    let wrong_items = vec![vec![1.0], vec![2.0], vec![3.0]]; // 3 items instead of 2
    let wrong_gl = GraphFactory::build_lambda_graph(&wrong_items, 1e-3, 2, 2.0, None);

    // This should panic due to node count mismatch
    compute_synthetic_lambdas(&mut aspace.clone(), &wrong_gl, 0.7, TauMode::Median);
}

#[test]
fn test_tau_floor_constant() {
    // Verify TAU_FLOOR is a reasonable small positive value
    assert!(TAU_FLOOR > 0.0);
    assert!(TAU_FLOOR < 1e-6);
    assert!(TAU_FLOOR.is_finite());
}

#[test]
fn test_synthetic_lambda_properties() {
    // Test that synthetic lambdas satisfy expected mathematical properties
    let items = vec![
        vec![1.0, 0.5, 0.0],
        vec![0.0, 0.5, 1.0],
        vec![0.5, 1.0, 0.5],
    ];

    let mut aspace = ArrowSpace::from_items(items.clone());
    let gl = GraphFactory::build_lambda_graph(&items, 1e-3, 3, 2.0, None);

    compute_synthetic_lambdas(&mut aspace, &gl, 0.8, TauMode::Median);
    let lambdas = aspace.lambdas();

    // Properties that should hold:
    // 1. All lambdas are in [0, 1] due to bounded transform
    assert!(lambdas.iter().all(|&l| (0.0..=1.0).contains(&l)));

    // 2. Lambdas are deterministic (same input -> same output)
    let mut aspace2 = ArrowSpace::from_items(items);
    compute_synthetic_lambdas(&mut aspace2, &gl, 0.8, TauMode::Median);
    let lambdas2 = aspace2.lambdas();

    for (l1, l2) in lambdas.iter().zip(lambdas2.iter()) {
        assert!(
            (l1 - l2).abs() < 1e-12,
            "Synthetic lambdas should be deterministic"
        );
    }

    // 3. All values are finite
    assert!(lambdas.iter().all(|l| l.is_finite()));
}
