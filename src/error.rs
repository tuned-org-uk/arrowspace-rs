//! Error types for fallible ArrowSpace operations.
//!
//! These errors represent **recoverable input conditions** — not invariant
//! violations — and are returned by the `try_*` family of methods so that FFI
//! bindings can surface typed exceptions instead of relying on Rust panics
//! (which cross language boundaries as opaque `PanicException`).
//!
//! The infallible wrappers (`prepare_query_item`, `search_lambda_aware`, …)
//! delegate to the `try_*` variants and `.expect()` on `Err`, preserving their
//! original panic behaviour for existing callers.

use std::fmt;

/// Recoverable error conditions arising from the query path.
///
/// # When each variant occurs
///
/// | Variant | Trigger |
/// |---------|---------|
/// | `DegenerateLambda` | A query or item has a spectral score (λ) of ~0, making λ-aware ranking undecidable. Usually a sign of a mis-tuned `eps` or an out-of-context query. |
/// | `NonFiniteQuery` | The query vector contains `NaN` or `±Infinity`. |
/// | `DimensionMismatch` | The query vector's length differs from the index's feature count. |
/// | `EmptyItems` | The item matrix passed to the builder is empty. |
#[derive(Clone, Debug, PartialEq)]
pub enum ArrowSpaceError {
    /// The spectral score (lambda) computed for a query — or stored on a query
    /// item — is approximately zero, so lambda-aware similarity cannot produce
    /// a meaningful ranking.
    ///
    /// `raw` is the offending lambda value so callers can decide whether it was
    /// a stored zero (caller forgot to call `prepare_query_item`) or a computed
    /// zero (the graph Laplacian maps the query to its null space).
    DegenerateLambda {
        /// The degenerate lambda value that triggered the error.
        raw: f64,
    },

    /// The query vector contains one or more non-finite values (`NaN`,
    /// `+Infinity`, `-Infinity`).
    NonFiniteQuery,

    /// The query vector's dimensionality does not match the index's feature
    /// count.
    DimensionMismatch {
        /// Expected dimension (the index's `nfeatures`).
        expected: usize,
        /// Actual dimension of the query vector.
        got: usize,
    },

    /// The item matrix passed to the builder is empty.
    EmptyItems,
}

impl fmt::Display for ArrowSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateLambda { raw } => write!(
                f,
                "degenerate lambda (raw={raw:.6}): lambda is ~0.0, \
                 check the eps parameter for the builder — every dataset has an \
                 optimal eps. The query item may also be out of context for the \
                 dataset (undecidable)."
            ),
            Self::NonFiniteQuery => {
                write!(f, "query item contains non-finite values (NaN or Infinity)")
            }
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} features, got {got}"
            ),
            Self::EmptyItems => write!(f, "items cannot be empty"),
        }
    }
}

impl std::error::Error for ArrowSpaceError {}
