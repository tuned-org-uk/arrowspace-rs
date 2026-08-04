//! # Fallible query path tests (issue #121)
//!
//! Verifies that `try_prepare_query_item` and `try_search_lambda_aware` return
//! typed `ArrowSpaceError` variants instead of panicking, and that the existing
//! infallible wrappers preserve their panic behaviour for backward compatibility.

use crate::builder::ArrowSpaceBuilder;
use crate::core::{ArrowItem, ArrowSpace};
use crate::error::ArrowSpaceError;
use crate::graph::GraphLaplacian;
use crate::tests::{init, test_data::make_gaussian_hd};

/// Build a small eigen-mode index suitable for exercising the query path.
fn build_test_index(n: usize) -> (ArrowSpace, GraphLaplacian) {
    let data = make_gaussian_hd(n, 0.6);
    ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_sparsity_check(false)
        .with_dims_reduction(false, None)
        .with_seed(42)
        .build(data)
}

// ------------------------------------------------------------------
// ArrowSpaceError type
// ------------------------------------------------------------------

#[test]
fn test_error_display_messages_are_informative() {
    let e = ArrowSpaceError::NonFiniteQuery;
    let s = e.to_string();
    assert!(s.to_lowercase().contains("non-finite"), "got: {s}");

    let e = ArrowSpaceError::DegenerateLambda { raw: 0.0 };
    let s = e.to_string();
    assert!(s.to_lowercase().contains("degenerate"), "got: {s}");

    let e = ArrowSpaceError::DimensionMismatch {
        expected: 64,
        got: 32,
    };
    let s = e.to_string();
    assert!(s.to_lowercase().contains("dimension"), "got: {s}");
    assert!(s.contains("64"), "expected dim in message: {s}");
    assert!(s.contains("32"), "got dim in message: {s}");

    let e = ArrowSpaceError::EmptyItems;
    let s = e.to_string();
    assert!(s.to_lowercase().contains("empty"), "got: {s}");
}

#[test]
fn test_error_implements_std_error() {
    fn assert_error<E: std::error::Error>(_: &E) {}
    let e = ArrowSpaceError::NonFiniteQuery;
    assert_error(&e);
}

// ------------------------------------------------------------------
// try_prepare_query_item — error paths
// ------------------------------------------------------------------

#[test]
fn test_try_prepare_query_item_nan_returns_non_finite_err() {
    init();
    let (aspace, gl) = build_test_index(99);

    let mut bad_query = vec![0.0; aspace.nfeatures];
    bad_query[3] = f64::NAN;

    let result = aspace.try_prepare_query_item(&bad_query, &gl);
    assert!(
        matches!(result, Err(ArrowSpaceError::NonFiniteQuery)),
        "expected NonFiniteQuery, got {result:?}"
    );
}

#[test]
fn test_try_prepare_query_item_inf_returns_non_finite_err() {
    init();
    let (aspace, gl) = build_test_index(99);

    let mut bad_query = vec![0.0; aspace.nfeatures];
    bad_query[7] = f64::INFINITY;

    let result = aspace.try_prepare_query_item(&bad_query, &gl);
    assert!(
        matches!(result, Err(ArrowSpaceError::NonFiniteQuery)),
        "expected NonFiniteQuery, got {result:?}"
    );
}

#[test]
fn test_try_prepare_query_item_dimension_mismatch_returns_err() {
    init();
    let (aspace, gl) = build_test_index(99);

    let wrong_query = vec![0.0; aspace.nfeatures / 2];

    let result = aspace.try_prepare_query_item(&wrong_query, &gl);
    match result {
        Err(ArrowSpaceError::DimensionMismatch {
            expected,
            got,
        }) => {
            assert_eq!(expected, aspace.nfeatures);
            assert_eq!(got, aspace.nfeatures / 2);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn test_try_prepare_query_item_zero_vector_returns_degenerate_lambda() {
    init();
    let (aspace, gl) = build_test_index(99);

    // A zero vector produces raw_lambda = 0.0 inside compute_synthetic_lambda
    let zero_query = vec![0.0; aspace.nfeatures];

    let result = aspace.try_prepare_query_item(&zero_query, &gl);
    match result {
        Err(ArrowSpaceError::DegenerateLambda { raw }) => {
            assert!(raw.abs() < 1e-12, "raw should be ~0, got {raw}");
        }
        other => panic!("expected DegenerateLambda, got {other:?}"),
    }
}

// ------------------------------------------------------------------
// try_prepare_query_item — success path
// ------------------------------------------------------------------

#[test]
fn test_try_prepare_query_item_valid_query_returns_ok() {
    init();
    let (aspace, gl) = build_test_index(99);

    let query = aspace.get_item(0).item.clone();

    let result = aspace.try_prepare_query_item(&query, &gl);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let lambda = result.unwrap();
    assert!(lambda.is_finite(), "lambda should be finite");
    assert!(lambda >= 0.0, "lambda should be non-negative, got {lambda}");
}

#[test]
fn test_try_prepare_query_item_matches_prepare_query_item() {
    init();
    let (aspace, gl) = build_test_index(99);

    let query = aspace.get_item(3).item.clone();

    let fallible = aspace.try_prepare_query_item(&query, &gl).unwrap();
    let infallible = aspace.prepare_query_item(&query, &gl);

    assert!(
        (fallible - infallible).abs() < 1e-12,
        "try_ and infallible should match: {fallible} vs {infallible}"
    );
}

// ------------------------------------------------------------------
// try_search_lambda_aware — error paths
// ------------------------------------------------------------------

#[test]
fn test_try_search_lambda_aware_unprepared_query_returns_err() {
    init();
    let (aspace, _gl) = build_test_index(99);

    // Query with lambda = 0.0 (caller forgot to prepare)
    let query = ArrowItem::new(&aspace.get_item(0).item, 0.0);

    let result = aspace.try_search_lambda_aware(&query, 5, 0.7);
    match result {
        Err(ArrowSpaceError::DegenerateLambda { raw }) => {
            assert!(raw.abs() < 1e-12, "raw should be ~0, got {raw}");
        }
        other => panic!("expected DegenerateLambda, got {other:?}"),
    }
}

#[test]
fn test_try_search_lambda_aware_dimension_mismatch_returns_err() {
    init();
    let (aspace, _gl) = build_test_index(99);

    // Wrong-dimension query with non-zero lambda
    let short_query = ArrowItem::new(&[0.5, 0.5], 0.3);

    let result = aspace.try_search_lambda_aware(&short_query, 5, 0.7);
    match result {
        Err(ArrowSpaceError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, aspace.nfeatures);
            assert_eq!(got, 2);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

// ------------------------------------------------------------------
// try_search_lambda_aware — success path
// ------------------------------------------------------------------

#[test]
fn test_try_search_lambda_aware_prepared_query_returns_results() {
    init();
    let (aspace, gl) = build_test_index(99);

    let lambda = aspace.prepare_query_item(&aspace.get_item(0).item, &gl);
    let query = ArrowItem::new(&aspace.get_item(0).item, lambda);

    let result = aspace.try_search_lambda_aware(&query, 5, 0.7);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let results = result.unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_try_search_lambda_aware_matches_search_lambda_aware() {
    init();
    let (aspace, gl) = build_test_index(99);

    let lambda = aspace.prepare_query_item(&aspace.get_item(2).item, &gl);
    let query = ArrowItem::new(&aspace.get_item(2).item, lambda);

    let fallible = aspace.try_search_lambda_aware(&query, 5, 0.7).unwrap();
    let infallible = aspace.search_lambda_aware(&query, 5, 0.7);

    assert_eq!(fallible.len(), infallible.len());
    for (a, b) in fallible.iter().zip(infallible.iter()) {
        assert_eq!(a.0, b.0, "index mismatch");
        assert!((a.1 - b.1).abs() < 1e-12, "score mismatch: {} vs {}", a.1, b.1);
    }
}

// ------------------------------------------------------------------
// degenerate_lambda_count
// ------------------------------------------------------------------

#[test]
fn test_degenerate_lambda_count_normal_index_is_small() {
    init();
    let (aspace, _gl) = build_test_index(99);

    let count = aspace.degenerate_lambda_count();
    // After normalisation the minimum is 0.0, so at most a handful are ~0.
    assert!(
        count < aspace.nitems / 10,
        "normal index should have few degenerate lambdas, got {count}/{}",
        aspace.nitems
    );
}

// ------------------------------------------------------------------
// Backward compatibility — infallible wrappers still panic
// ------------------------------------------------------------------

#[test]
#[should_panic]
fn test_prepare_query_item_still_panics_on_nan() {
    init();
    let (aspace, gl) = build_test_index(99);

    let mut bad_query = vec![0.0; aspace.nfeatures];
    bad_query[3] = f64::NAN;

    let _ = aspace.prepare_query_item(&bad_query, &gl);
}

#[test]
#[should_panic]
fn test_prepare_query_item_still_panics_on_zero_vector() {
    init();
    let (aspace, gl) = build_test_index(99);

    let zero_query = vec![0.0; aspace.nfeatures];

    let _ = aspace.prepare_query_item(&zero_query, &gl);
}

#[test]
#[should_panic]
fn test_search_lambda_aware_still_panics_on_unprepared() {
    init();
    let (aspace, _gl) = build_test_index(99);

    let query = ArrowItem::new(&aspace.get_item(0).item, 0.0);

    let _ = aspace.search_lambda_aware(&query, 5, 0.7);
}
