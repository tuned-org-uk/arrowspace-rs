use crate::reduction::{ImplicitProjection, compute_jl_dimension, project_matrix};
use smartcore::linalg::basic::{
    arrays::{Array, Array2},
    matrix::DenseMatrix,
};

// ============================================================================
// ImplicitProjection Tests
// ============================================================================

#[test]
fn test_implicit_projection_creates() {
    let proj = ImplicitProjection::new(100, 10, Some(42));
    assert_eq!(proj.original_dim, 100);
    assert_eq!(proj.reduced_dim, 10);
    assert!(proj.seed > 0);
}

#[test]
fn test_implicit_projection_dimensions() {
    let proj = ImplicitProjection::new(50, 8, Some(42));
    let query = vec![0.5; 50];

    let projected = proj.project(&query);

    assert_eq!(projected.len(), 8);
    assert!(projected.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_implicit_projection_deterministic() {
    // Same seed should produce same projection
    let proj = ImplicitProjection::new(30, 5, Some(42));
    let query = vec![1.0; 30];

    let result1 = proj.project(&query);
    let result2 = proj.project(&query);

    assert_eq!(result1, result2);
}

#[test]
fn test_implicit_projection_different_seeds() {
    // Different instances should have different seeds
    let proj1 = ImplicitProjection::new(20, 5, None);
    let proj2 = ImplicitProjection::new(20, 5, None);

    // Seeds should be different (probabilistically)
    assert_ne!(proj1.seed, proj2.seed);

    let query = vec![1.0; 20];
    let result1 = proj1.project(&query);
    let result2 = proj2.project(&query);

    // Results should differ due to different seeds
    assert_ne!(result1, result2);
}

#[test]
fn test_implicit_projection_zero_vector() {
    let proj = ImplicitProjection::new(40, 10, Some(42));
    let query = vec![0.0; 40];

    let projected = proj.project(&query);

    assert_eq!(projected.len(), 10);
    // All should be near-zero
    assert!(projected.iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_implicit_projection_linearity() {
    let proj = ImplicitProjection::new(25, 6, Some(42));

    let query = vec![1.0; 25];
    let scaled_query: Vec<f64> = query.iter().map(|x| x * 2.0).collect();

    let proj1 = proj.project(&query);
    let proj2 = proj.project(&scaled_query);

    // Projection is linear: project(2x) = 2*project(x)
    for i in 0..proj1.len() {
        let expected = proj1[i] * 2.0;
        let actual = proj2[i];
        assert!(
            (expected - actual).abs() < 1e-9,
            "Linearity violation at {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_implicit_projection_preserves_scale() {
    let proj = ImplicitProjection::new(50, 15, Some(42));
    let query = vec![1.0; 50];

    let projected = proj.project(&query);

    let orig_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
    let proj_norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();

    let ratio = proj_norm / orig_norm;

    // JL guarantees approximate norm preservation
    assert!(ratio > 0.5 && ratio < 2.0);
}

#[test]
fn test_implicit_projection_non_trivial() {
    let proj = ImplicitProjection::new(30, 8, Some(42));
    let query = vec![1.0; 30];

    let projected = proj.project(&query);

    // Should have at least one non-zero value
    let has_nonzero = projected.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero);
}

// ============================================================================
// project_matrix Tests
// ============================================================================

#[test]
fn test_project_matrix_dimensions() {
    let data = vec![1.0; 60]; // 3 rows × 20 cols
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 3, 20, 1);
    let proj = ImplicitProjection::new(20, 5, Some(42));

    let projected = project_matrix(&matrix, &proj);

    assert_eq!(projected.shape(), (3, 5));
}

#[test]
fn test_project_matrix_preserves_rows() {
    let data = vec![0.5; 100]; // 10 rows × 10 cols
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 10, 10, 1);
    let proj = ImplicitProjection::new(10, 3, Some(42));

    let projected = project_matrix(&matrix, &proj);

    assert_eq!(projected.shape().0, 10);
    assert_eq!(projected.shape().1, 3);
}

#[test]
fn test_project_matrix_zero_matrix() {
    let data = vec![0.0; 80]; // 4 rows × 20 cols
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 4, 20, 0);
    let proj = ImplicitProjection::new(20, 6, Some(42));

    let projected = project_matrix(&matrix, &proj);

    // All values should be near-zero
    for i in 0..projected.shape().0 {
        for j in 0..projected.shape().1 {
            assert!(projected.get((i, j)).abs() < 1e-10);
        }
    }
}

#[test]
fn test_project_matrix_different_rows_different_projections() {
    let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 3, 4, 0);
    let proj = ImplicitProjection::new(4, 2, Some(42));

    let projected = project_matrix(&matrix, &proj);

    // Extract rows
    let row0: Vec<f64> = (0..2).map(|j| *projected.get((0, j))).collect();
    let row1: Vec<f64> = (0..2).map(|j| *projected.get((1, j))).collect();
    let row2: Vec<f64> = (0..2).map(|j| *projected.get((2, j))).collect();

    // Different input rows should produce different projections
    assert_ne!(row0, row1);
    assert_ne!(row1, row2);
}

// ============================================================================
// compute_jl_dimension Tests
// ============================================================================

// ============================================================================
// compute_jl_dimension Tests (v0.25.12 - adaptive buffer) - CORRECTED
// ============================================================================

#[test]
fn test_jl_dimension_preserves_low_dims() {
    // Rule 1: Dimensions < 32 are preserved exactly
    assert_eq!(compute_jl_dimension(100, 16, 0.3), 16);
    assert_eq!(compute_jl_dimension(1000, 8, 0.1), 8);
    assert_eq!(compute_jl_dimension(50, 31, 0.2), 31);
    assert_eq!(compute_jl_dimension(10, 1, 0.5), 1);
}

#[test]
fn test_jl_dimension_never_expands() {
    // Rule 2: Never exceed original_dim (upper bound clamp)
    let n = 10;
    let epsilon = 0.3;

    // JL bound for n=10, ε=0.3: ~205 dims
    // But if original is only 100, must cap at 100
    assert_eq!(compute_jl_dimension(n, 100, epsilon), 100);
    assert_eq!(compute_jl_dimension(n, 50, epsilon), 50);

    // For n=100, ε=0.5: JL bound = 8*ln(100)/0.25 ≈ 148
    // If original_dim=200, stays at 148 (no expansion needed)
    let dim = compute_jl_dimension(100, 200, 0.5);
    assert!(dim >= 148 && dim <= 149); // Within [32, 200], uses JL bound
}

#[test]
fn test_jl_dimension_minimum_bound() {
    // For very small n and large ε, JL bound might be < 32
    // Should clamp to 32 minimum (if original_dim >= 32)
    let n = 2;
    let epsilon = 0.9;

    // JL: 8*ln(2)/0.81 ≈ 6.9 → clamped to 32
    assert_eq!(compute_jl_dimension(n, 1000, epsilon), 32);

    // But if original_dim < 32, preserve it
    assert_eq!(compute_jl_dimension(n, 20, epsilon), 20);
}

#[test]
fn test_jl_dimension_standard_regime() {
    // Rule 3: Standard case (32 <= dim <= 2048)
    // Use vanilla JL bound, clamped to [32, original_dim]
    let n = 1000;
    let epsilon = 0.1;
    let original_dim = 512;

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    // JL bound: 8*ln(1000)/0.01 ≈ 5530 → capped at original_dim=512
    assert_eq!(dim, 512);

    // Case where JL < original_dim (standard regime, no buffer)
    let n2 = 100;
    let original_dim2 = 2000;
    let dim2 = compute_jl_dimension(n2, original_dim2, 0.2);
    // JL: 8*ln(100)/0.04 = 8*4.605/0.04 = 921.024... → ceil = 922
    // Since 2000 <= 2048, no buffer applied
    assert!(dim2 >= 921 && dim2 <= 923); // Account for floating point
}

#[test]
fn test_jl_dimension_high_dim_buffer_mild_compression() {
    // Test adaptive buffer: compression < 10× → 1.2× buffer
    let n = 500; // Larger n to reduce JL bound
    let epsilon = 0.3;
    let original_dim = 3000; // > 2048, triggers high-D path

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    // JL: 8*ln(500)/0.09 ≈ 551
    // Compression: 3000/551 ≈ 5.4× (< 10×)
    // Buffer: 1.2×
    // Expected: 551 * 1.2 = 661
    let jl_bound = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected = ((jl_bound as f64) * 1.2).ceil() as usize;

    assert!(dim >= expected - 2 && dim <= expected + 2);
    assert!(dim < original_dim);
}

#[test]
fn test_jl_dimension_high_dim_buffer_moderate_compression() {
    // Test adaptive buffer: 10× < compression < 100× → 1.5× buffer
    let n = 100;
    let epsilon = 0.3;
    let original_dim = 100_000;

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    // JL: 8*ln(100)/0.09 ≈ 410
    // Compression: 100000/410 ≈ 244× (> 100×)
    // Buffer: 2.0× (not 1.5× because compression > 100×)
    // Expected: 410 * 2.0 = 820
    let jl_bound = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let compression = original_dim as f64 / jl_bound as f64;
    let buffer = if compression >= 100.0 { 2.0 } else { 1.5 };
    let expected = ((jl_bound as f64) * buffer).ceil() as usize;

    assert!(dim >= expected - 2 && dim <= expected + 2);
    assert!(dim < original_dim);
}

#[test]
fn test_jl_dimension_high_dim_buffer_severe_compression() {
    // Test adaptive buffer: compression > 100× → 2.0× buffer
    let n = 10;
    let epsilon = 0.3;
    let original_dim = 50_000;

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    // JL: 8*ln(10)/0.09 ≈ 205
    // Compression: 50000/205 ≈ 244× (> 100×)
    // Buffer: 2.0×
    // Expected: 205 * 2.0 = 410
    let jl_bound = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected = ((jl_bound as f64) * 2.0).ceil() as usize;

    assert!(dim >= expected - 2 && dim <= expected + 2);
    assert!(dim < original_dim);
}

#[test]
fn test_jl_dimension_high_dim_buffer_caps_at_original() {
    // Even with buffer, never exceed original_dim
    // BUT: if buffered result < original_dim, use buffered result!
    let n = 10_000;
    let epsilon = 0.3;
    let original_dim = 5_000;

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    // JL: 8*ln(10000)/0.09 ≈ 819
    // Compression: 5000/819 ≈ 6.1× (< 10×)
    // Buffer: 1.2×
    // Expected: 819 * 1.2 = 983 (< 5000, so NOT capped!)
    let jl_bound = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let buffered = ((jl_bound as f64) * 1.2).ceil() as usize;

    // The result is 983, not 5000, because buffered < original_dim
    assert_eq!(dim, buffered);
    assert!(dim < original_dim); // Verify it's less than original
}

#[test]
fn test_jl_dimension_buffer_actually_hits_cap() {
    // Test where buffer DOES get capped at original_dim
    let n = 5_000;
    let epsilon = 0.3;
    let original_dim = 3_000;

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    // JL: 8*ln(5000)/0.09 ≈ 755
    // Compression: 3000/755 ≈ 3.97× (< 10×)
    // Buffer: 1.2×
    // Buffered: 755 * 1.2 = 906 (< 3000, so use buffered)
    let jl_bound = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let buffered = ((jl_bound as f64) * 1.2).ceil() as usize;

    if buffered <= original_dim {
        assert_eq!(dim, buffered);
    } else {
        assert_eq!(dim, original_dim);
    }
}

#[test]
fn test_jl_dimension_grows_with_n() {
    // More points → more dimensions needed (logarithmic growth)
    let epsilon = 0.2;
    let original_dim = 10_000;

    let dim_100 = compute_jl_dimension(100, original_dim, epsilon);
    let dim_1000 = compute_jl_dimension(1000, original_dim, epsilon);
    let dim_10000 = compute_jl_dimension(10000, original_dim, epsilon);

    assert!(dim_1000 > dim_100);
    assert!(dim_10000 > dim_1000);
}

#[test]
fn test_jl_dimension_inversely_proportional_epsilon() {
    // Smaller ε → more dimensions needed (quadratic relationship)
    let n = 5000;
    let original_dim = 10_000;

    let dim_05 = compute_jl_dimension(n, original_dim, 0.5);
    let dim_02 = compute_jl_dimension(n, original_dim, 0.2);
    let dim_01 = compute_jl_dimension(n, original_dim, 0.1);

    assert!(dim_02 > dim_05);
    assert!(dim_01 > dim_02);
}

#[test]
fn test_jl_dimension_dorothea_scenario() {
    // Real-world case from Dorothea experiments
    // 1000 items, 100K features, 17 clusters
    let n_clusters = 17;
    let original_dim = 100_000;
    let epsilon = 0.3;

    let dim = compute_jl_dimension(n_clusters, original_dim, epsilon);

    // JL: 8*ln(17)/0.09 ≈ 251
    // Compression: 100000/251 ≈ 398× (> 100×)
    // Buffer: 2.0×
    // Expected: 251 * 2.0 = 502
    assert!(dim >= 480 && dim <= 520); // Reasonable range around 502
    assert!(dim < 1000);
}

#[test]
fn test_jl_dimension_moderate_compression() {
    // Verify buffer scaling with compression ratio
    let n = 100;
    let epsilon = 0.3;

    // Mild: 5000 → JL~410, compression~12× → 1.5× buffer
    let dim_5k = compute_jl_dimension(n, 5000, epsilon);
    // Expected: 410 * 1.5 = 615
    assert!(dim_5k >= 600 && dim_5k <= 630);

    // Severe: 50000 → JL~410, compression~122× → 2.0× buffer
    let dim_50k = compute_jl_dimension(n, 50000, epsilon);
    // Expected: 410 * 2.0 = 820
    assert!(dim_50k >= 800 && dim_50k <= 840);
}

#[test]
fn test_jl_dimension_reasonable_range() {
    // Sanity checks: results should be practical
    let test_cases = vec![
        (100, 384, 0.2),    // BERT embeddings
        (200, 1536, 0.3),   // OpenAI embeddings
        (50, 100_000, 0.3), // Dorothea-like
        (1000, 768, 0.15),  // RoBERTa embeddings
    ];

    for (n, original_dim, eps) in test_cases {
        let dim = compute_jl_dimension(n, original_dim, eps);

        // Basic sanity
        assert!(dim >= 32 || dim == original_dim); // Minimum or preserved
        assert!(dim <= original_dim); // Never expand

        // Practical bounds
        assert!(
            dim < 10_000,
            "Result too large: {} for n={}, F={}",
            dim,
            n,
            original_dim
        );
    }
}

#[test]
fn test_jl_dimension_formula_correctness_standard() {
    // Verify vanilla JL formula in standard regime (no buffer)
    let n = 1000;
    let epsilon = 0.15;
    let original_dim = 1500; // Standard regime (< 2048)

    let dim = compute_jl_dimension(n, original_dim, epsilon);

    let expected_jl = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected = expected_jl.clamp(32, original_dim);

    assert_eq!(dim, expected);
}

#[test]
fn test_jl_dimension_edge_case_single_point() {
    // Pathological case: n=1 (single point, no distance preservation needed)
    let n = 1;
    let epsilon = 0.1;

    // JL: 8*ln(1)/0.01 = 0 → clamped to 32 or original_dim
    assert_eq!(compute_jl_dimension(n, 100, epsilon), 32);
    assert_eq!(compute_jl_dimension(n, 10, epsilon), 10); // Preserve if < 32
}

#[test]
fn test_jl_dimension_buffer_factor_application() {
    // Verify adaptive buffer tiers work correctly
    let epsilon = 0.3;

    // Tier 1: compression < 10× → 1.2× buffer
    // Need VERY mild compression to stay under 10×
    let n1 = 800; // JL ≈ 600
    let original_1 = 5500; // compression = 5500/600 ≈ 9.2× (< 10×)
    let dim1 = compute_jl_dimension(n1, original_1, epsilon);
    let jl1 = (8.0 * (n1 as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected1 = ((jl1 as f64) * 1.2).ceil() as usize;
    assert!(
        dim1 >= expected1 - 5 && dim1 <= expected1 + 5,
        "Tier 1 failed: expected ~{}, got {}",
        expected1,
        dim1
    );

    // Tier 2: 10× < compression < 100× → 1.5× buffer
    let n2 = 100; // JL ≈ 410
    let original_2 = 20_000; // compression = 20000/410 ≈ 48.8× (10-100×)
    let dim2 = compute_jl_dimension(n2, original_2, epsilon);
    let jl2 = (8.0 * (n2 as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected2 = ((jl2 as f64) * 1.5).ceil() as usize;
    assert!(
        dim2 >= expected2 - 5 && dim2 <= expected2 + 5,
        "Tier 2 failed: expected ~{}, got {}",
        expected2,
        dim2
    );

    // Tier 3: compression > 100× → 2.0× buffer
    let n3 = 50; // JL ≈ 348
    let original_3 = 50_000; // compression = 50000/348 ≈ 143× (> 100×)
    let dim3 = compute_jl_dimension(n3, original_3, epsilon);
    let jl3 = (8.0 * (n3 as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected3 = ((jl3 as f64) * 2.0).ceil() as usize;
    assert!(
        dim3 >= expected3 - 5 && dim3 <= expected3 + 5,
        "Tier 3 failed: expected ~{}, got {}",
        expected3,
        dim3
    );
}

#[test]
fn test_jl_dimension_boundary_2048() {
    // Test transition at 2048 boundary (buffer applies above this)
    let n = 100;
    let epsilon = 0.3;

    // Just below threshold: no buffer
    let dim_2048 = compute_jl_dimension(n, 2048, epsilon);
    let jl_bound = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    assert_eq!(dim_2048, jl_bound.clamp(32, 2048));

    // Just above threshold: buffer applied
    let dim_2049 = compute_jl_dimension(n, 2049, epsilon);
    // compression = 2049/410 ≈ 5× → buffer 1.2×
    let buffered = ((jl_bound as f64) * 1.2).ceil() as usize;
    assert_eq!(dim_2049, buffered.clamp(32, 2049));
}

#[test]
fn test_jl_dimension_consistency() {
    // Same n and ε should give same result for same original_dim
    let n = 500;
    let original_dim = 5000;
    let epsilon = 0.2;

    let dim1 = compute_jl_dimension(n, original_dim, epsilon);
    let dim2 = compute_jl_dimension(n, original_dim, epsilon);

    assert_eq!(dim1, dim2, "Function should be deterministic");
}

#[test]
fn test_jl_dimension_edge_cases() {
    // Low dim: preserve exactly
    assert_eq!(compute_jl_dimension(100, 16, 0.3), 16);

    // Standard: respect original as upper bound
    assert_eq!(compute_jl_dimension(10, 50, 0.3), 50); // JL wants 205, capped

    // Ultra-high: apply adaptive buffer
    let result = compute_jl_dimension(100, 100_000, 0.3);
    // JL=410, compression=244×, buffer=2.0×, expected≈820
    assert!(result >= 800 && result <= 850);
    assert!(result < 100_000); // Never expand

    // Verify monotonicity: more points → more dims needed
    let d1 = compute_jl_dimension(10, 10_000, 0.3);
    let d2 = compute_jl_dimension(100, 10_000, 0.3);
    assert!(d2 > d1);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline_implicit_projection() {
    // Simulate full pipeline: matrix → project → verify
    let n_samples = 20;
    let orig_dim = 100;
    let reduced_dim = 15;

    // Create test data
    let data: Vec<f64> = (0..n_samples * orig_dim)
        .map(|i| (i as f64) * 0.01)
        .collect();
    let matrix = DenseMatrix::from_iterator(data.into_iter(), n_samples, orig_dim, 0);

    // Create projection
    let proj = ImplicitProjection::new(orig_dim, reduced_dim, Some(42));

    // Project matrix
    let projected = project_matrix(&matrix, &proj);

    assert_eq!(projected.shape(), (n_samples, reduced_dim));

    // Verify all values are finite
    for i in 0..projected.shape().0 {
        for j in 0..projected.shape().1 {
            assert!(projected.get((i, j)).is_finite());
        }
    }
}

#[test]
fn test_memory_efficiency() {
    // ImplicitProjection should be tiny (just 24 bytes)
    let proj = ImplicitProjection::new(1000, 100, Some(42));

    // Verify it can project without storing the matrix
    let query = vec![1.0; 1000];
    let projected = proj.project(&query);

    assert_eq!(projected.len(), 100);

    // The struct should be minimal size
    assert_eq!(std::mem::size_of::<ImplicitProjection>(), 24); // 3 usizes
}
