//! Fractal support utilities with zero-copy operations.
//!
//! This module provides utilities for constructing 1D Cantor-like supports,
//! building Cartesian product supports, and estimating fractal dimension via
//! box-counting (log–log regression).
//!
//! Design goals:
//! - Iterator-first, allocation-conscious implementations.
//! - Deterministic behavior across runs for reproducible results.
//! - Simple composability for downstream geometry/graph pipelines.
//!
//! # Examples
//!
//! Generate a 1D Cantor-like support and estimate a box-counting dimension for a line:
//!
//! ```
//! use arrowspace::dimensional::{ArrowDimensionalOps, DimensionalOps};
//!
//! // A small 1D Cantor-like set over a discrete grid of length 81.
//! let keep = DimensionalOps::make_cantor_1d(2, 0.33, 81);
//!
//! // Points sampled on a line (dimension near 1).
//! let pts = [(0,0), (1,1), (2,2), (3,3)];
//! let scales = [1, 2, 3, 1];
//! let dim = DimensionalOps::box_count_dimension(&pts, &scales);
//! assert!(dim.unwrap() > 0.0);
//! ```
//!
//! Build a Cartesian product (support × range) without allocations beyond the output:
//!
//! ```
//! use arrowspace::dimensional::{ArrowDimensionalOps, DimensionalOps};
//! let one_d = vec![0,2];
//! let prod = DimensionalOps::make_product_support(&one_d, 3);
//! ```
//!
//! Run documentation tests with `cargo test --doc` to keep examples correct over time[15][2][5].
//!
//! # Panics
//!
//! - `make_cantor_1d` panics if `gamma ∉ (0,1)`.
//!
//! # Notes
//!
//! - Box-counting regression uses natural logarithms on positive scales only.
//!   Invalid inputs (empty points/scales or degenerate regression) yield `None`.

use std::collections::HashSet;

/// Operations for generating discrete fractal supports and estimating dimension.
///
/// Methods are provided as associated functions via the trait to allow mocking
/// or swapping implementations in tests or specialized backends.
///
/// # Examples
///
/// ```
/// use arrowspace::dimensional::{ArrowDimensionalOps, DimensionalOps};
///
/// let support = DimensionalOps::make_cantor_1d(1, 0.33, 27);
/// assert!(support.iter().all(|&i| i < 27));
/// ```
pub trait ArrowDimensionalOps {
    /// Generates a discrete 1D Cantor-like support over `[0, length)`.
    ///
    /// Starting from a single segment `[0,length)`, iteratively remove a centered
    /// middle portion of relative size `gamma` (rounded), keeping left/right
    /// subsegments. The process stops after `iters` steps or when segments become
    /// too small to split further.
    ///
    /// Returns sorted, deduplicated indices to keep.
    ///
    /// # Arguments
    ///
    /// - `iters`: number of removal iterations.
    /// - `gamma`: middle proportion to remove each iteration; must satisfy `0<gamma<1`.
    /// - `length`: total discrete length of the universe `[0,length)`.
    ///
    /// # Returns
    ///
    /// A sorted `Vec<usize>` of kept indices in `[0,length)`. Empty if `length==0`.
    ///
    /// # Panics
    ///
    /// Panics if `gamma <= 0.0` or `gamma >= 1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::dimensional::{ArrowDimensionalOps, DimensionalOps};
    /// let keep = DimensionalOps::make_cantor_1d(2, 0.33, 81);
    /// assert!(!keep.is_empty());
    /// assert!(keep.len() < 81);
    /// ```
    fn make_cantor_1d(iters: usize, gamma: f64, length: usize) -> Vec<usize>;

    /// Builds a Cartesian product support `c1 × {0..len2}`.
    ///
    /// For each `i ∈ c1`, produces pairs `(i, j)` for `j=0..len2-1`.
    ///
    /// # Arguments
    ///
    /// - `c1`: sorted (not required) indices from the first axis.
    /// - `len2`: size of the second axis.
    ///
    /// # Returns
    ///
    /// A `Vec<(usize, usize)>` of length `c1.len() * len2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::dimensional::{ArrowDimensionalOps, DimensionalOps};
    /// let a = vec![5, 2];
    /// let prod = DimensionalOps::make_product_support(&a, 3);
    /// ```
    fn make_product_support(c1: &[usize], len2: usize) -> Vec<(usize, usize)>;

    /// Estimates box-counting (Minkowski–Bouligand) dimension by log–log regression.
    ///
    /// For each scale `s ∈ scales`, points are assigned to boxes of side `s` using
    /// integer division, i.e., box key is `(x.div_euclid(s), y.div_euclid(s))`.
    /// The number of occupied boxes `N(s)` is counted; a linear regression is fit
    /// to `(ln(1/s), ln N(s))`, and the slope is returned.
    ///
    /// Returns `None` if inputs are invalid or regression is degenerate.
    ///
    /// # Arguments
    ///
    /// - `points`: collection of `(x,y)` integer points.
    /// - `scales`: positive integer box sizes; non-positive scales are ignored.
    ///
    /// # Returns
    ///
    /// `Some(slope)` if at least 2 valid scales remain and the regression is well-posed;
    /// otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::dimensional::{ArrowDimensionalOps, DimensionalOps};
    /// // Sampled along a line: dimension should be near 1 with enough data.
    /// let pts = [(0,0), (1,1), (2,2), (3,3), (4,4)];
    /// let scales = [1, 2, 4, 5];
    /// let dim = DimensionalOps::box_count_dimension(&pts, &scales);
    /// assert!(dim.is_some());
    /// ```
    ///
    /// # Edge cases
    ///
    /// - Empty `points` or `scales` → `None`.
    /// - All invalid scales or singular design matrix → `None`.
    fn box_count_dimension(points: &[(i32, i32)], scales: &[usize]) -> Option<f64>;
}

/// Default implementation of ArrowDimensionalOps.
pub struct DimensionalOps;

impl ArrowDimensionalOps for DimensionalOps {
    #[inline]
    fn make_cantor_1d(iters: usize, gamma: f64, length: usize) -> Vec<usize> {
        if length == 0 {
            return vec![];
        }
        assert!(gamma > 0.0 && gamma < 1.0, "gamma must be in (0,1)");

        let mut segs: Vec<(usize, usize)> = vec![(0, length)];
        for _ in 0..iters {
            let next: Vec<_> = segs
                .iter()
                .flat_map(|&(s, e)| {
                    let len = e - s;
                    if len < 3 {
                        vec![(s, e)]
                    } else {
                        let mid_len = ((gamma * len as f64).round() as usize).clamp(1, len - 2);
                        let left_len = (len - mid_len) / 2;
                        let right_start = s + left_len + mid_len;

                        vec![(s, s + left_len), (right_start, e)]
                            .into_iter()
                            .filter(|(start, end)| end > start)
                            .collect()
                    }
                })
                .collect();
            if next.is_empty() {
                break;
            }
            segs = next;
        }

        let mut kept: Vec<_> = segs.iter().flat_map(|(s, e)| *s..*e).collect();
        kept.sort_unstable();
        kept.dedup();
        kept
    }

    #[inline]
    fn make_product_support(c1: &[usize], len2: usize) -> Vec<(usize, usize)> {
        c1.iter()
            .flat_map(|&i| (0..len2).map(move |j| (i, j)))
            .collect()
    }

    #[inline]
    fn box_count_dimension(points: &[(i32, i32)], scales: &[usize]) -> Option<f64> {
        if points.is_empty() || scales.is_empty() {
            return None;
        }

        let scale_data: Vec<_> = scales
            .iter()
            .filter(|&&s| s > 0)
            .map(|&s| {
                let boxes: HashSet<_> = points
                    .iter()
                    .map(|&(x, y)| (x.div_euclid(s as i32), y.div_euclid(s as i32)))
                    .collect();
                let n = boxes.len().max(1);
                ((1.0 / s as f64).ln(), (n as f64).ln())
            })
            .collect();

        if scale_data.len() < 2 {
            return None;
        }

        // Least squares regression using iterators
        let n = scale_data.len() as f64;
        let sx: f64 = scale_data.iter().map(|(x, _)| x).sum();
        let sy: f64 = scale_data.iter().map(|(_, y)| y).sum();
        let sxx: f64 = scale_data.iter().map(|(x, _)| x * x).sum();
        let sxy: f64 = scale_data.iter().map(|(x, y)| x * y).sum();

        let denom = n * sxx - sx * sx;
        if denom.abs() < 1e-12 {
            return None;
        }

        Some((n * sxy - sx * sy) / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cantor_generation() {
        let cantor = DimensionalOps::make_cantor_1d(2, 0.33, 81);
        assert!(!cantor.is_empty());
        assert!(cantor.len() < 81); // Should be subset
        assert!(cantor.iter().all(|&x| x < 81)); // All within bounds
    }

    #[test]
    fn test_product_support() {
        let c1 = vec![0, 2, 4];
        let support = DimensionalOps::make_product_support(&c1, 3);
        assert_eq!(support.len(), 9); // 3 * 3
        assert!(support.contains(&(0, 0)));
        assert!(support.contains(&(4, 2)));
    }

    #[test]
    fn test_box_count_dimension() {
        let points = vec![(0, 0), (1, 1), (2, 2), (3, 3)]; // Line
        let scales = vec![1, 2, 3, 4];

        if let Some(dim) = DimensionalOps::box_count_dimension(&points, &scales) {
            assert!(dim > 0.0 && dim < 3.0); // Reasonable dimension
        }
    }
}
