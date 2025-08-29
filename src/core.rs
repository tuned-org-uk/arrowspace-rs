//! Arrow and ArrowSpace: enhanced with search-specific zero-copy operations.
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
//! # Examples
//!
//! Create a small ArrowSpace and compute cosine similarity between rows:
//!
//! ```
//! use arrowspace::core::{ArrowItem, ArrowSpace};
//!
//! let aspace = ArrowSpace::from_items(
//!     vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]
//! );
//!
//! let a = aspace.get_feature(0);
//! let b = aspace.get_feature(1);
//! ```
//!
//! Zero-copy mutate a row using a mutable view and update its lambda from a graph:
//!
//! ```ignore
//! use arrowspace::core::ArrowSpace;
//! use arrowspace::operators::build_knn_graph;
//!
//! // Build a toy space and a small KNN graph.
//! let mut aspace = ArrowSpace::from_items(vec![vec![1.0, 2.0, 3.0]], vec![0.0]);
//! let edges =vec![(0,0), (1,0), (0,1)];
//! let gl = build_knn_graph(&, 2, 2.0, None);
//!
//! // Scale in-place without extra allocation.
//! {
//!     let mut rv = aspace.feature_view_mut(0);
//!     for x in rv.iter_mut() { *x *= 2.0; }
//! }
//! aspace.recompute_lambdas(&gl);
//! assert!(aspace.lambdas()[0].is_nan());
//! ```
//!
//! Run documentation tests with `cargo test --doc`; Rustdoc extracts code blocks
//! and executes them as tests, ensuring examples stay correct over time[3][6].
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
//! if missing, and allows common lints to reduce boilerplate[3][11]. Keep examples
//! small and focused; add hidden setup lines with `#` when needed so that examples
//! compile while showing only the essential lines to readers[3][4][8].

use std::fmt::Debug;

use crate::graph_factory::GraphLaplacian;
use crate::operators::rayleigh_lambda;

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
/// let mut a = ArrowItem::new(vec![1.0, 2.0, 3.0], 0.5);
/// let b = ArrowItem::new(vec![1.0, 0.0, 1.0], 1.2);
///
/// let cos = a.cosine_similarity(&b);
/// assert!(cos.is_finite());
///
/// a.scale(2.0);
/// assert_eq!(a.len(), 3);
/// ```
///
/// Iterate without copying:
///
/// ```
/// use arrowspace::core::ArrowItem;
///
/// let r = ArrowItem::new(vec![1.0, 2.0, 3.0], 0.0);
/// let s: f64 = r.iter().copied().sum();
/// assert_eq!(s, 6.0);
/// ```
///
/// # Panics
///
/// - `dot`, `cosine_similarity`, and `euclidean_distance` panic if lengths differ.
///
/// # Complexity
///
/// - `norm`: O(n)
/// - `dot`: O(n)
/// - `cosine_similarity`: O(n)
/// - `euclidean_distance`: O(n)
#[derive(Clone, Debug)]
pub struct ArrowItem {
    pub item: Vec<f64>,
    pub lambda: f64,
}

#[derive(Clone, Debug)]
pub struct ArrowFeature {
    pub data: Vec<f64>,
}

impl ArrowItem {
    /// Creates a new ArrowItem from owned data and its spectral score.
    ///
    /// Prefer passing already-allocated vectors to avoid extra copies.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let r = ArrowItem::new(vec![0.0, 1.0], 0.3);
    /// assert_eq!(r.len(), 2);
    /// ```
    #[inline]
    pub fn new(item: Vec<f64>, lambda: f64) -> Self {
        Self { item, lambda }
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
    /// let a = ArrowItem::new(vec![1.0, 2.0, 3.0], 0.0);
    /// let b = ArrowItem::new(vec![4.0, 5.0, 6.0], 0.0);
    /// assert_eq!(a.dot(&b), 32.0);
    /// ```
    #[inline]
    pub fn dot(&self, other: &ArrowItem) -> f64 {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        self.item
            .iter()
            .zip(other.item.iter())
            .map(|(a, b)| a * b)
            .sum()
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
    /// let a = ArrowItem::new(vec![1.0, 0.0], 0.0);
    /// let b = ArrowItem::new(vec![0.0, 1.0], 0.0);
    /// assert!((a.cosine_similarity(&b) - 0.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn cosine_similarity(&self, other: &ArrowItem) -> f64 {
        use crate::operators::norm;
        let denom = norm(&self.item) * norm(&other.item);
        if denom > 0.0 {
            self.dot(other) / denom
        } else {
            0.0
        }
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
    /// let a = ArrowItem::new(vec![1.0, 1.0], 0.0);
    /// let b = ArrowItem::new(vec![4.0, 5.0], 0.0);
    /// assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn euclidean_distance(&self, other: &ArrowItem) -> f64 {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        self.item
            .iter()
            .zip(other.item.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Combines semantic (cosine) similarity and lambda proximity.
    ///
    /// `alpha` weights semantic similarity; `beta` weights lambda proximity
    /// defined as `1 / (1 + |lambda_a - lambda_b|)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 0.0], 0.5);
    /// let b = ArrowItem::new(vec![1.0, 0.0], 0.6);
    /// let s = a.lambda_similarity(&b, 0.7, 0.3);
    /// assert!(s <= 1.0 && s >= 0.0);
    /// ```
    #[inline]
    pub fn lambda_similarity(&self, other: &ArrowItem, alpha: f64, beta: f64) -> f64 {
        assert_eq!(
            self.item.len(),
            other.item.len(),
            "items should be of the same length"
        );
        let semantic_sim = self.cosine_similarity(other);
        let lambda_sim = 1.0 / (1.0 + (self.lambda - other.lambda).abs());
        alpha * semantic_sim + beta * lambda_sim
    }

    /// Adds another row element-wise in-place.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[inline]
    pub fn add_inplace(&mut self, other: &ArrowItem) {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
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
        self.item
            .iter_mut()
            .zip(other.item.iter())
            .for_each(|(a, b)| *a *= *b);
    }

    /// Scales all elements by a scalar in place.
    #[inline]
    pub fn scale(&mut self, scalar: f64) {
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
#[derive(Clone, Debug, Default)]
pub struct ArrowSpace {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<f64>,    // column-major flattened
    pub lambdas: Vec<f64>, // row-major (every lambda is a lambda for an item-column)
}

impl ArrowSpace {
    /// Builds from a vector of equally-sized rows and per-row lambdas.
    ///
    /// # Panics
    ///
    /// - If `rows` is empty.
    /// - If rows have differing lengths.
    /// - If `lambdas.len() != rows.len()`.
    #[inline]
    pub fn from_items(items: Vec<Vec<f64>>) -> Self {
        assert!(!items.is_empty(), "items cannot be empty");
        assert!(
            items.len() > 1,
            "cannot create a arrowspace of one arrow only"
        );
        let n_items = items.len(); // Number of items (columns in final layout)
        let n_features = items[0].len(); // Number of features (rows in final layout)

        assert!(
            items.iter().all(|item| item.len() == n_features),
            "All items must have same number of features"
        );

        // Convert from items (N×F) to column-major storage (F×N)
        // In column-major: data[col * nrows + row] = matrix[row][col]
        // Here: data[item * n_features + feature] = feature_value_for_item
        let mut data = Vec::with_capacity(n_features * n_items);

        // Initialize with zeros, then fill column by column (item by item)
        data.resize(n_features * n_items, 0.0);

        for (item_idx, item_features) in items.iter().enumerate() {
            for (feature_idx, &feature_value) in item_features.iter().enumerate() {
                // Column-major indexing: data[col * nrows + row]
                // Here: col=item_idx, row=feature_idx
                data[item_idx * n_features + feature_idx] = feature_value;
            }
        }

        // Initialize feature lambdas (will be recomputed later)
        let items_lambdas = vec![0.0; n_items];

        Self {
            nrows: n_features, // Features are rows
            ncols: n_items,    // Items are columns
            data,              // Column-major storage
            lambdas: items_lambdas,
        }
    }

    /// Returns (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        // column-major: (number of features, number of items)
        (self.nrows, self.ncols)
    }

    /// Returns a shared reference to all lambdas.
    #[inline]
    pub fn lambdas(&self) -> &[f64] {
        self.lambdas.as_ref()
    }

    /// Returns an owned ArrowItem copy of the requested row.
    ///
    /// Prefer zero-copy `row_view` for performance-sensitive paths.
    ///
    /// # Panics
    ///
    /// Panics if `row >= nrows`.
    #[inline]
    pub fn get_feature(&self, i: usize) -> ArrowFeature {
        assert!(i < self.nrows, "Row index out of bounds");
        let start = i * self.ncols;
        let end = start + self.ncols;
        ArrowFeature {
            data: self.data[start..end].to_vec(),
        }
    }

    /// Returns a zero-copy mutable view of the requested feature (row).
    ///
    /// # Panics
    /// Panics if `feature >= n_features`.
    #[inline]
    pub fn get_feature_mut(&mut self, feature: usize) -> ArrowFeature {
        assert!(feature < self.nrows, "Feature index out of bounds");
        let start = feature * self.ncols;
        let end = start + self.ncols;
        let (head, _) = self.data.split_at_mut(end);
        let feature_slice = &mut head[start..end];
        ArrowFeature {
            data: feature_slice.to_vec(),
        }
    }

    /// Extracts an owned ArrowItem for the requested item (column).
    ///
    /// This reconstructs item `i` by collecting its values across all features.
    ///
    /// # Panics
    /// Panics if `item >= n_items`.
    #[inline]
    pub fn get_item(&self, item: usize) -> ArrowItem {
        assert!(item < self.ncols, "Item index out of bounds");
        let mut item_values = Vec::with_capacity(self.nrows);

        // Extract column `item` from column-major storage
        for feature_idx in 0..self.nrows {
            // Column-major indexing: data[col * nrows + row]
            // Here: col=item, row=feature_idx
            let value = self.data[item * self.nrows + feature_idx];
            item_values.push(value);
        }

        ArrowItem::new(item_values, self.lambdas[item])
    }

    /// Adds item `b` into item `a` in-place and recomputes feature lambdas.
    ///
    /// This method:
    /// 1. Extracts item `a` and item `b` as complete ArrowItem vectors
    /// 2. Performs element-wise addition: item_a += item_b  
    /// 3. Writes the result back to the column-major matrix
    /// 4. Recomputes feature lambdas
    #[inline]
    pub fn add_items(&mut self, a: usize, b: usize, gl: &GraphLaplacian) {
        assert!(
            a < self.ncols && b < self.ncols,
            "Item indices out of bounds: a={}, b={}, ncols={}",
            a,
            b,
            self.ncols
        );
        assert_eq!(
            gl.nnodes, self.ncols,
            "Laplacian nodes must match number of items"
        );

        // Extract both items as complete ArrowItem vectors
        let mut item_a = self.get_item(a);
        let item_b = self.get_item(b);

        // Perform the addition: item_a += item_b
        item_a.add_inplace(&item_b);

        // Write the result back to column `a` in column-major storage
        for (feature_idx, &new_value) in item_a.item.iter().enumerate() {
            // Column-major indexing: data[col * nrows + row]
            // Here: col=a, row=feature_idx
            self.data[a * self.nrows + feature_idx] = new_value;
        }

        // Recompute lambdas for all features since item values changed
        self.recompute_lambdas(gl);
    }

    /// Multiplies item `a` element-wise by item `b` and recomputes feature lambdas.
    #[inline]
    pub fn mul_items(&mut self, a: usize, b: usize, gl: &GraphLaplacian) {
        assert!(
            a < self.ncols && b < self.ncols,
            "Item indices out of bounds: a={}, b={}, ncols={}",
            a,
            b,
            self.ncols
        );
        assert_eq!(
            gl.nnodes, self.ncols,
            "Laplacian nodes must match number of items"
        );

        // Extract both items as complete ArrowItem vectors
        let mut item_a = self.get_item(a);
        let item_b = self.get_item(b);

        // Perform the multiplication: item_a *= item_b
        item_a.mul_inplace(&item_b);

        // Write the result back to column `a` in the column-major matrix
        for (feature_idx, &new_value) in item_a.item.iter().enumerate() {
            let feature_row_start = feature_idx * self.ncols;
            self.data[feature_row_start + a] = new_value;
        }

        // Recompute lambdas for all features since item values changed
        self.recompute_lambdas(gl);
    }

    /// Scales item `a` by a scalar value and recomputes feature lambdas.
    #[inline]
    pub fn scale_item(&mut self, a: usize, scalar: f64, gl: &GraphLaplacian) {
        assert!(
            a < self.ncols,
            "Item index out of bounds: a={}, ncols={}",
            a,
            self.ncols
        );
        assert_eq!(
            gl.nnodes, self.ncols,
            "Laplacian nodes must match number of items"
        );

        // Extract item as complete ArrowItem vector
        let mut item_a = self.get_item(a);

        // Perform the scaling: item_a *= scalar
        item_a.scale(scalar);

        // Write the result back to column `a` in the column-major matrix
        for (feature_idx, &new_value) in item_a.item.iter().enumerate() {
            let feature_row_start = feature_idx * self.ncols;
            self.data[feature_row_start + a] = new_value;
        }

        // Recompute lambdas for all features since item values changed
        self.recompute_lambdas(gl);
    }

    /// Recomputes all feature lambdas using the provided Graph Laplacian.
    ///
    /// The Laplacian must have nodes equal to the number of items.
    #[inline]
    pub fn recompute_lambdas(&mut self, gl: &GraphLaplacian) {
        assert_eq!(
            gl.nnodes, self.ncols,
            "Laplacian nodes must match number of items"
        );

        for i in 0..self.ncols {
            let item = self.get_item(i);
            self.lambdas[i] = rayleigh_lambda(gl, &item.item);
        }
    }

    /// Lambda-aware top-k search against an ArrowItem query.
    ///
    /// Returns indices and scores sorted descending by similarity.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::{ArrowItem, ArrowSpace};
    ///
    /// let aspace = ArrowSpace::from_items(
    ///     vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0]]
    /// );
    /// let q = ArrowItem::new(vec![1.0, 0.1], 0.5);
    /// let res = aspace.search_lambda_aware(&q, 2, 0.7, 0.3);
    /// assert_eq!(res.len(), 2);
    /// assert!(res[1].1 >= 0.0);
    /// ```
    #[inline]
    pub fn search_lambda_aware(
        &self,
        query: &ArrowItem,
        k: usize,
        alpha: f64,
        beta: f64,
    ) -> Vec<(usize, f64)> {
        let mut results: Vec<_> = (0..self.ncols)
            .map(|i| {
                let item = self.get_item(i);
                let similarity = query.lambda_similarity(&item, alpha, beta);
                (i, similarity)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }

    /// Range search by Euclidean distance within `radius`.
    ///
    /// Returns (row_index, distance) pairs for all rows whose distance is
    /// ≤ `radius` from the query.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::{ArrowItem, ArrowSpace};
    ///
    /// let aspace = ArrowSpace::from_items(
    ///     vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]
    /// );
    /// let q = ArrowItem::new(vec![0.5, 0.0], 0.0);
    /// let hits = aspace.range_search(&q, 0.6);
    /// ```
    #[inline]
    pub fn range_search(&self, query: &ArrowItem, radius: f64) -> Vec<(usize, f64)> {
        (0..self.ncols)
            .filter_map(|i| {
                let item = self.get_item(i);
                let distance = query.euclidean_distance(&item);
                if distance <= radius {
                    Some((i, distance))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Computes a pairwise cosine similarity submatrix for selected indices.
    ///
    /// This allocates an output matrix of size `indices.len() x indices.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowSpace;
    ///
    /// let aspace = ArrowSpace::from_items(
    ///     vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![1.0, 1.0, 0.0]]
    /// );
    /// let sims = aspace.pairwise_similarities(&[0,1,2]);
    /// assert_eq!(sims.len(), 3);
    /// assert_eq!(sims.len(), 3);
    /// ```
    #[inline]
    pub fn pairwise_similarities(&self, indices: &[usize]) -> Vec<Vec<f64>> {
        indices
            .iter()
            .map(|&i| {
                let item_i = self.get_item(i);
                indices
                    .iter()
                    .map(|&j| {
                        let item_j = self.get_item(j);
                        item_i.cosine_similarity(&item_j)
                    })
                    .collect()
            })
            .collect()
    }

    /// Returns a borrowed feature slice (immutable).
    #[inline]
    pub fn iter_feature(&self, feature: usize) -> &[f64] {
        assert!(feature < self.nrows, "Feature index out of bounds");
        let start = feature * self.ncols;
        let end = start + self.ncols;
        &self.data[start..end]
    }

    /// Iterates a single item (column) as references across features.
    #[inline]
    pub fn iter_item(&self, item: usize) -> impl Iterator<Item = &f64> {
        assert!(item < self.ncols, "Item index out of bounds");
        (0..self.nrows).map(move |f| &self.data[f * self.ncols + item])
    }

    /// Update the lambdas with new synthetic values
    pub fn update_lambdas(&mut self, new_lambdas: Vec<f64>) {
        assert_eq!(
            new_lambdas.len(),
            self.lambdas.len(),
            "New lambdas length must match existing lambdas length"
        );
        self.lambdas = new_lambdas;
    }
}

// Flattened AsRef/AsMut for ArrowSpace
impl AsRef<[f64]> for ArrowSpace {
    #[inline]
    fn as_ref(&self) -> &[f64] {
        &self.data
    }
}
impl AsMut<[f64]> for ArrowSpace {
    #[inline]
    fn as_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

// Iterate all elements by reference (feature-major)
impl<'a> IntoIterator for &'a ArrowSpace {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// Iterate all elements mutably (feature-major)
impl<'a> IntoIterator for &'a mut ArrowSpace {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}
