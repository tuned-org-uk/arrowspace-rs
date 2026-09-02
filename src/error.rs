//! Error types for fallible ArrowSpace operations.
//!
//! These errors represent **recoverable input conditions** — not invariant
//! violations — and are returned by the `try_*` family of methods so that FFI
//! bindings can surface typed exceptions instead of relying on Rust panics
//! (which cross language boundaries as opaque `PanicException`).
//!
//! The infallible wrappers (`prepare_query_item`, `search_lambda_aware`, …)
//! delegate to the `try_*` variants and `.expect()` on `Err`, preserving their
//! original panic behaviour for existing callers. Since #153 they are
//! `#[deprecated]`: the `try_*` variants are the primary documented entry
//! points, and the wrappers remain only as a backward-compatibility surface
//! for downstream callers that have not migrated yet.

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
/// | `InvalidConfig` | The builder configuration is invalid for the requested pipeline (e.g. energy build without dims reduction). |
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

    /// The builder configuration is invalid for the requested pipeline.
    ///
    /// Triggered by `try_build_energy` when the builder lacks dims reduction
    /// or carries the (experimental, unimplemented) spectral flag. Replaces
    /// the `assert!`/`panic!` the energy build path used before (#155) so
    /// misconfiguration surfaces as a typed error instead of a process abort.
    InvalidConfig {
        /// What is wrong with the configuration, as a human-readable fragment.
        reason: &'static str,
    },

    /// The operation requires an EnergyMaps build, but the target lacks
    /// energy-mode bookkeeping.
    ///
    /// Triggered by `try_spot_motives_energy` / `try_spot_subg_motives` when
    /// the Laplacian was not built via `build_energy`, or the index carries no
    /// `sub_centroids` / `centroid_map`. Running energy motif detection on an
    /// EigenMaps build would silently operate on feature-space nodes and
    /// mislabel them as item indices (issue #161), so the operation refuses
    /// instead of degrading.
    EnergyModeRequired {
        /// What is missing, as a human-readable fragment (e.g. `"energy build
        /// (use build_energy)"`, `"sub_centroids"`, `"centroid_map"`).
        missing: &'static str,
    },
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
            Self::InvalidConfig { reason } => {
                write!(f, "invalid configuration: {reason}")
            }
            Self::EnergyModeRequired { missing } => write!(
                f,
                "energy mode required: missing {missing}. Build via \
                 EnergyMapsBuilder::build_energy so sub_centroids and \
                 centroid_map are populated; running energy motif detection \
                 on an EigenMaps build would mislabel feature-space indices \
                 as item indices."
            ),
        }
    }
}

impl std::error::Error for ArrowSpaceError {}
