//! ArrowSpace: enhanced with search-specific zero-copy operations.
//!
//! This module provides two core abstractions for working with row-major numeric
//! data in search/graph contexts:
//!
//! - ArrowItem: an owned row with convenience methods (norm, dot, cosine_similarity,
//!   lambda-aware similarity), in-place arithmetic, and iterator access.
//! - ArrowSpace: a dense, row-major, zero-copy container of rows with per-row
//!   spectral score `lambda`, supporting row views (immutable/mutable), iteration,
//!   and search utilities.
//!
//! Design goals:
//! - Zero-copy access to rows for performance-critical routines.
//! - Iterator-first APIs for cache-friendly, allocation-free operations.
//! - Spectral-aware scoring via Rayleigh quotient against a Graph Laplacian.
//!
//!
//! Zero-copy mutate a row using a mutable view and update its lambda from a graph:
//!
//!
//! Run documentation tests with `cargo test --doc`; Rustdoc extracts code blocks
//! and executes them as tests, ensuring examples stay correct over time.
//!
//! # Panics
//!
//! - Indexing functions panic on out-of-bounds row/column indices.
//! - Arithmetic between mismatched row lengths panics.
//!
//! # Performance
//!
//! - Row accessors favor zero-copy slices/views; prefer `row_view`/`row_view_mut`
//!   over `get_row` when allocation must be avoided.
//! - Batch operations rely on iterators to minimize bounds checks and enable
//!   vectorization opportunities.
//!
//! # Testing examples
//!
//! Rustdoc preprocesses examples: it injects the crate, wraps code in `fn main`
//! if missing, and allows common lints to reduce boilerplate. Keep examples
//! small and focused; add hidden setup lines with `#` when needed so that examples
//! compile while showing only the essential lines to readers.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::{BinaryHeap, HashSet};
use std::fmt::Debug;

use approx::relative_eq;
use rayon::prelude::*;
use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;

use crate::builder::ConfigValue;
use crate::graph::GraphLaplacian;
use crate::reduction::ImplicitProjection;
use crate::search::sorted_index::SortedLambdas;
use crate::search::taumode::TauMode;

// Add logging
use log::{debug, info, trace, warn};

/// A single owned row with an associated spectral score `lambda`.
///
/// ArrowItem provides iterator-based, allocation-free primitives (norm, dot,
/// cosine similarity, Euclidean distance) and in-place arithmetic. It is useful
/// both as a convenience handle returned by `ArrowSpace::get_row` and as a
/// standalone value in query-time computations.
///
/// # Examples
///
/// Construct, compute similarity, and scale in place:
///
/// ```
/// use arrowspace::core::ArrowItem;
///
/// let mut a = ArrowItem::new(vec![1.0, 2.0, 3.0].as_ref(), 0.5);
/// let b = vec![1.0, 0.0, 1.0];
///
/// let cos = a.cosine_similarity(&b);
/// assert!(cos.is_finite());
///
/// a.scale(2.0);
/// assert_eq!(a.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct ArrowItem {
    pub item: Vec<f64>,
    pub lambda: f64,
}

/// A structure representing a feature-column
///  just the data for now but will be useful for index building
#[derive(Clone, Debug)]
pub struct ArrowFeature {
    pub feature: Vec<f64>,
}

impl ArrowItem {
    /// Creates a new ArrowItem from owned data.
    /// This just store the vector with a placeholder lambda, to compute the
    ///  lambda (Rayleigh quotient) use `new_with_graph` or precompute lambda
    ///  and pass it to this method.
    ///
    /// Prefer passing already-allocated vectors to avoid extra copies.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let r = ArrowItem::new(vec![0.0, 1.0].as_ref(), 0.3);
    /// assert_eq!(r.len(), 2);
    /// ```
    #[inline]
    pub fn new(item: &[f64], lambda: f64) -> Self {
        trace!(
            "Creating ArrowItem with {} dimensions, lambda: {:.6}",
            item.len(),
            lambda
        );
        Self {
            item: item.to_vec(),
            lambda,
        }
    }

    /// Returns the length (dimensionality) of the row.
    #[inline]
    pub fn len(&self) -> usize {
        self.item.len()
    }

    /// Returns true if the row has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.item.is_empty()
    }

    /// Lambda component similarity (spectral proximity).
    ///
    /// Computes `sim_λ = 1 − min(|λ_a − λ_b|, 1.0)`, which maps identical
    /// lambdas to 1.0 and lambdas differing by ≥ 1.0 to 0.0.
    /// This is the form used in both search entry points and reconciles the
    /// paper shorthand `sim_λ = λτ_q − λτ_i` (a signed difference) with the
    /// bounded proximity score actually computed here.
    #[inline]
    pub fn lambda_component_similarity(&self, other: &ArrowItem) -> f64 {
        let lambda_diff = (self.lambda - other.lambda).abs();
        1.0 - lambda_diff.min(1.0)
    }

    /// Combined lambda-aware similarity.
    ///
    /// Blends cosine (semantic) similarity and spectral proximity using a
    /// convex combination controlled by `alpha`:
    ///
    /// ```text
    /// score = α · cos(q, i) + (1 − α) · sim_λ(q, i)
    /// ```
    ///
    /// where `sim_λ = 1 − min(|λ_q − λ_i|, 1.0)` (see
    /// [`lambda_component_similarity`]).
    ///
    /// This is the same unsigned blend used by `search_lambda_aware_hybrid`
    /// (`alpha * cosine + beta * lambda_component`).  Both entry points are
    /// therefore consistent for all cosine values, including the negative
    /// half-space.
    ///
    /// `alpha` weights semantic similarity (cosine); `(1 − alpha)` weights
    /// spectral proximity.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 0.0].as_ref(), 0.5);
    /// let b = ArrowItem::new(vec![1.0, 0.0].as_ref(), 0.6);
    /// let s = a.lambda_similarity(&b, 0.7);
    /// assert!(s >= 0.0 && s <= 1.0);
    /// ```
    #[inline]
    pub fn lambda_similarity(&self, other: &ArrowItem, alpha: f64) -> f64 {
        assert_eq!(
            self.item.len(),
            other.item.len(),
            "items should be of the same length"
        );
        let cosine_sim = self.cosine_similarity(&other.item);
        let lambda_sim = self.lambda_component_similarity(other);

        // Unsigned convex blend: score = α·cos + (1−α)·sim_λ
        // Equivalent to the hybrid path: alpha * cosine + beta * lambda_component
        let result = alpha * cosine_sim + (1.0 - alpha) * lambda_sim;

        trace!(
            "Lambda similarity: semantic={:.6}, lambda={:.6}, combined={:.6}",
            cosine_sim, lambda_sim, result
        );

        result
    }

    /// Computes the dot product with another row without allocating.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 2.0, 3.0].as_ref(), 0.0);
    /// let b = ArrowItem::new(vec![4.0, 5.0, 6.0].as_ref(), 0.0);
    /// assert_eq!(a.dot(&b), 32.0);
    /// ```
    #[inline]
    pub fn dot(&self, other: &ArrowItem) -> f64 {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        let result = self
            .item
            .iter()
            .zip(other.item.iter())
            .map(|(a, b)| a * b)
            .sum();
        trace!("Computed dot product: {:.6}", result);
        result
    }

    /// Computes the Euclidean norm (L2) without allocating.
    #[inline]
    pub fn norm(a: &[f64]) -> f64 {
        let result = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        trace!("Computed norm: {:.6}", result);
        result
    }

    /// Computes cosine similarity, guarding against zero vectors.
    ///
    /// Returns 0.0 if either vector has zero norm.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 0.0].as_ref(), 0.0);
    /// let b = vec![0.0, 1.0];
    /// assert!((a.cosine_similarity(&b) - 0.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn cosine_similarity(&self, other: &[f64]) -> f64 {
        let denom = ArrowItem::norm(&self.item) * ArrowItem::norm(other);
        let result = if denom > 0.0 {
            self.dot(&ArrowItem::new(other, 0.0)) / denom
        } else {
            warn!("Zero vector encountered in cosine similarity computation");
            0.0
        };
        trace!("Computed cosine similarity: {:.6}", result);
        result
    }

    /// Computes Euclidean distance without allocation.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 1.0].as_ref(), 0.0);
    /// let b = ArrowItem::new(vec![4.0, 5.0].as_ref(), 0.0);
    /// assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn euclidean_distance(&self, other: &ArrowItem) -> f64 {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        let result = self
            .item
            .iter()
            .zip(other.item.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        trace!("Computed Euclidean distance: {:.6}", result);
        result
    }

    /// Adds another row element-wise in-place.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[inline]
    pub fn add_inplace(&mut self, other: &ArrowItem) {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        trace!("Adding vectors in-place");
        self.item
            .iter_mut()
            .zip(other.item.iter())
            .for_each(|(a, b)| *a += *b);
    }

    /// Multiplies element-wise in-place by another row.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[inline]
    pub fn mul_inplace(&mut self, other: &ArrowItem) {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        trace!("Multiplying vectors element-wise in-place");
        self.item
            .iter_mut()
            .zip(other.item.iter())
            .for_each(|(a, b)| *a *= *b);
    }

    /// Scales all elements by a scalar in place.
    #[inline]
    pub fn scale(&mut self, scalar: f64) {
        trace!("Scaling vector by {:.6}", scalar);
        self.item.iter_mut().for_each(|x| *x *= scalar);
    }

    /// Immutable iterator over elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.item.iter()
    }

    /// Mutable iterator over elements.
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.item.iter_mut()
    }
}

/// Scored item for min-heap (keeps top-k by popping smallest)
#[derive(Debug, Clone, Copy)]
struct ScoredItem {
    index: usize,
    score: f64,
}

impl PartialEq for ScoredItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredItem {}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap (smallest score at top)
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// A dense, row-major matrix of f64 with per-row spectral scores (`lambda`).
///
/// ArrowSpace stores all data in a flattened row-major `Vec<f64>` and maintains
/// a parallel `lambdas` array. It exposes allocation-free row views and
/// search-oriented operations that recompute spectral scores on mutation.
///
/// # Construction
///
/// - `from_rows` builds from a `Vec<Vec<f64>>`, validating consistent width.
///
///
/// # Panics
///
/// - Constructors panic if row lengths are inconsistent or lambda length mismatches.
/// - Indexing methods panic on out-of-bound indices.
///
/// # Performance
///
#[derive(Clone, Debug)]
pub struct ArrowSpace {
    pub nfeatures: usize, // F: original dimensions
    pub nitems: usize,
    pub data: DenseMatrix<f64>,        // NxF raw data
    pub signals: CsMat<f64>,           // Laplacian(Transpose(FfxFn))
    pub lambdas: Vec<f64>,             // N lambdas (every lambda is a lambda for an item-row)
    pub lambdas_sorted: SortedLambdas, // sorted by lambda ascending
    pub taumode: TauMode,              // tau_mode as in select_tau_mode

    // lambdas normalisation
    pub min_lambdas: f64,
    pub max_lambdas: f64,
    pub(crate) range_lambdas: f64,

    pub n_clusters: usize,
    /// Cluster assignment per original row (N entries, each in 0..X or None for outliers)
    pub cluster_assignments: Vec<Option<usize>>,
    /// Cluster sizes (X entries)
    pub cluster_sizes: Vec<usize>,
    /// Squared distance threshold used during clustering
    pub cluster_radius: f64,

    // Projection data: dims reduction data (needed to prepare the query vector)
    pub projection_matrix: Option<ImplicitProjection>, // F × r (if projection was used)
    pub reduced_dim: Option<usize>, // r (reduced dimension, None if no projection)
    pub extra_reduced_dim: bool,    // optional extra dimensionality reduction for energymaps

    // energymaps specific
    pub centroid_map: Option<Vec<usize>>, // Maps item_idx -> centroid_idx
    pub sub_centroids: Option<DenseMatrix<f64>>,
    pub subcentroid_lambdas: Option<Vec<f64>>,

    /// Pre-computed L2 norms for tie-breaking (energy mode)
    ///
    /// Computed during build to accelerate cosine similarity in search.
    /// Only used when items have identical lambdas (same subcentroid).
    pub item_norms: Option<Vec<f64>>,
}

pub const TAUDEFAULT: TauMode = TauMode::Median;

impl Default for ArrowSpace {
    fn default() -> Self {
        debug!("Creating default ArrowSpace");
        Self {
            nfeatures: 0,
            nitems: 0,
            data: DenseMatrix::new(0, 0, Vec::new(), true).unwrap(),
            signals: sprs::CsMat::zero((0, 0)),
            lambdas: Vec::new(),
            lambdas_sorted: SortedLambdas::new(),
            // lambdas normalisation
            min_lambdas: -1.0,
            max_lambdas: -1.0,
            range_lambdas: -1.0,
            // enable synthetic λ with Median τ by default
            taumode: TAUDEFAULT,
            // Clustering defaults
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            // projection
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            // energymaps
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }
}

impl ArrowSpace {
    /// Returns an empty space from the initial data
    pub(crate) fn new(items: Vec<Vec<f64>>, taumode: TauMode) -> Self {
        assert!(!items.is_empty(), "items cannot be empty");
        assert!(
            items.len() > 1,
            "cannot create a arrowspace of one arrow only"
        );
        let n_items = items.len(); // Number of items (columns in final layout)
        let n_features = items[0].len(); // Number of features (rows in final layout)
        Self {
            nfeatures: n_features,
            nitems: n_items,
            data: DenseMatrix::from_2d_vec(&items).unwrap(),
            signals: sprs::CsMat::zero((0, 0)), // will be computed later
            lambdas: vec![0.0; n_items],        // will be computed later
            lambdas_sorted: SortedLambdas::new(),
            // lambdas normalisation
            min_lambdas: -1.0,
            max_lambdas: -1.0,
            range_lambdas: -1.0,
            taumode,
            // Clustering defaults
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            // projection
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            // energymaps
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }

    /// Convenience method to generate a temporary `ArrowSpace` to reproject vectors
    pub fn empty_with_projection(
        proj_data: HashMap<String, ConfigValue>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        debug!(
            "ArrowSpace::empty_with_projection called with nrows={}, ncols={}",
            nrows, ncols
        );

        let extra_reduced = proj_data["extra_reduced_dim"].as_bool().unwrap();
        debug!("extra_reduced_dim from proj_data: {}", extra_reduced);
        assert!(
            extra_reduced == false,
            "Reconstructing with extra dim reduction is not implemented yet"
        );

        let has_projection = proj_data["pj_mtx_original_dim"].as_usize().is_some();
        debug!("projection present in proj_data: {}", has_projection);

        let mut aspace = Self::default();
        aspace.nitems = nrows;
        aspace.nfeatures = ncols;

        if has_projection {
            let original_dim = proj_data["pj_mtx_original_dim"]
                .as_usize()
                .expect("pj_mtx_original_dim must be usize when projection is present");
            let reduced_dim = proj_data["pj_mtx_reduced_dim"]
                .as_usize()
                .expect("pj_mtx_reduced_dim must be usize when projection is present");
            let seed = proj_data["pj_mtx_seed"]
                .as_u64()
                .expect("pj_mtx_seed must be u64 when projection is present");

            info!(
                "Reconstructing ImplicitProjection: original_dim={}, reduced_dim={}, seed={}",
                original_dim, reduced_dim, seed
            );

            aspace.projection_matrix = Some(ImplicitProjection {
                original_dim,
                reduced_dim,
                seed,
            });
            aspace.reduced_dim = Some(reduced_dim);
            aspace.extra_reduced_dim = extra_reduced;
        } else {
            warn!(
                "empty_with_projection called without projection metadata; \
                returning ArrowSpace without projection_matrix"
            );
        }

        debug!(
            "ArrowSpace::empty_with_projection created ArrowSpace \
            with nitems={}, nfeatures={}, reduced_dim={:?}",
            aspace.nitems, aspace.nfeatures, aspace.reduced_dim
        );

        aspace
    }

    /// Recreates an ArrowSpace from a aspace configuration HashMap.
    ///
    /// This method reconstructs a workable ArrowSpace with all properties set
    /// from the builder configuration, but with an empty data matrix.
    ///
    /// Returns:
    /// A fully configured ArrowSpace with empty data
    pub fn from_config(config: HashMap<String, ConfigValue>) -> Self {
        let nitems = config
            .get("nitems")
            .and_then(|v| v.as_usize())
            .expect("from_config: missing nitems");
        let nfeatures = config
            .get("nfeatures")
            .and_then(|v| v.as_usize())
            .expect("from_config: missing nfeatures");

        debug!(
            "ArrowSpace::from_config called (nitems={}, nfeatures={})",
            nitems, nfeatures
        );

        // --- Projection matrix ---
        let projection_matrix = if let (
            Some(ConfigValue::OptionUsize(Some(original_dim))),
            Some(ConfigValue::OptionUsize(Some(reduced_dim))),
            Some(ConfigValue::OptionU64(Some(seed))),
        ) = (
            config.get("pj_mtx_original_dim"),
            config.get("pj_mtx_reduced_dim"),
            config.get("pj_mtx_seed"),
        ) {
            info!(
                "ArrowSpace::from_config: projection matrix used: original_dim={}, reduced_dim={}",
                original_dim, reduced_dim
            );
            Some(ImplicitProjection {
                original_dim: *original_dim,
                reduced_dim: *reduced_dim,
                seed: *seed,
            })
        } else {
            debug!("ArrowSpace::from_config: projection matrix not used");
            None
        };
        let reduced_dim = match config.get("pj_mtx_reduced_dim") {
            Some(ConfigValue::OptionUsize(Some(d))) => Some(*d),
            _ => None,
        };
        let extra_reduced_dim = config
            .get("extra_reduced_dim")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // --- Tau mode (synthesis) ---
        let taumode = config
            .get("taumode")
            .and_then(|v| match v {
                ConfigValue::TauMode(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // --- Clustering ---
        let n_clusters = config
            .get("n_clusters")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let cluster_radius = config
            .get("cluster_radius")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        info!(
            "ArrowSpace::from_config: n_clusters={}, cluster_radius={}",
            n_clusters, cluster_radius
        );

        // --- Empty data and auxiliary fields ---
        let data = DenseMatrix::new(0, 0, Vec::new(), true).unwrap();
        let signals = sprs::CsMat::zero((0, 0));
        let lambdas = vec![0.0; nitems];
        let lambdas_sorted = SortedLambdas::new();

        let aspace = ArrowSpace {
            nfeatures,
            nitems,
            data,
            signals,
            lambdas,
            lambdas_sorted,
            // Normalization fields
            min_lambdas: -1.0,
            max_lambdas: -1.0,
            range_lambdas: -1.0,
            taumode,
            n_clusters,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius,
            // Projection
            projection_matrix,
            reduced_dim,
            extra_reduced_dim,
            // Energy-maps related fields
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        };

        debug!(
            "ArrowSpace::from_config created ArrowSpace: nitems={}, nfeatures={}, n_clusters={}, reduced_dim={:?}, extra_reduced_dim={}",
            aspace.nitems,
            aspace.nfeatures,
            aspace.n_clusters,
            aspace.reduced_dim,
            aspace.extra_reduced_dim
   