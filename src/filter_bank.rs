//! Discrete Rayleigh filter-bank features.
//!
//! This module does not compute eigenvectors or dense spectral embeddings. It
//! decomposes the graph-vector Rayleigh energy into a short, flat band vector by
//! binning sparse Laplacian edge-energy contributions. The band energies sum to
//! the same Rayleigh numerator/denominator used by the λτ read-out.

use serde::{Deserialize, Serialize};
use sprs::CsMat;

use crate::reduction::ImplicitProjection;

/// Band boundary policy for [`FilterBankConfig`].
///
/// Boundaries live in the normalised interval `[0, 1]`. Uniform bands split the
/// interval into equal-width bands. Custom bands must start at 0, end at 1, and
/// contain `num_bands + 1` strictly increasing values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BandEdges {
    Uniform,
    Custom(Vec<f64>),
}

/// Optional filter-bank configuration.
///
/// `num_bands` is the length of each per-item output vector. The default build
/// path leaves the filter bank disabled; use
/// [`ArrowSpaceBuilder::with_filter_bank`](crate::builder::ArrowSpaceBuilder::with_filter_bank)
/// to opt in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterBankConfig {
    pub num_bands: usize,
    pub band_edges: BandEdges,
}

impl FilterBankConfig {
    pub fn new(num_bands: usize, band_edges: BandEdges) -> Self {
        let cfg = Self {
            num_bands,
            band_edges,
        };
        cfg.validate();
        cfg
    }

    pub fn uniform(num_bands: usize) -> Self {
        Self::new(num_bands, BandEdges::Uniform)
    }

    pub fn custom(edges: Vec<f64>) -> Self {
        let num_bands = edges.len().saturating_sub(1);
        Self::new(num_bands, BandEdges::Custom(edges))
    }

    pub fn validate(&self) {
        assert!(self.num_bands > 0, "filter bank needs at least one band");
        match &self.band_edges {
            BandEdges::Uniform => {}
            BandEdges::Custom(edges) => {
                assert_eq!(
                    edges.len(),
                    self.num_bands + 1,
                    "custom filter-bank edges must have num_bands + 1 entries"
                );
                assert!(
                    (edges[0] - 0.0).abs() <= 1e-12,
                    "custom filter-bank edges must start at 0.0"
                );
                assert!(
                    (edges[edges.len() - 1] - 1.0).abs() <= 1e-12,
                    "custom filter-bank edges must end at 1.0"
                );
                for pair in edges.windows(2) {
                    assert!(
                        pair[0].is_finite() && pair[1].is_finite() && pair[0] < pair[1],
                        "custom filter-bank edges must be finite and strictly increasing"
                    );
                }
            }
        }
    }

    pub fn edges(&self) -> Vec<f64> {
        self.validate();
        match &self.band_edges {
            BandEdges::Uniform => (0..=self.num_bands)
                .map(|i| i as f64 / self.num_bands as f64)
                .collect(),
            BandEdges::Custom(edges) => edges.clone(),
        }
    }
}

/// Query-side λτ plus optional band-energy coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedQuery {
    pub lambda: f64,
    pub band_energies: Vec<f64>,
}

/// Project a vector when the index uses an implicit projection.
pub(crate) fn project_item_for_graph(
    item_vector: &[f64],
    projection_matrix: Option<ImplicitProjection>,
) -> Vec<f64> {
    match projection_matrix {
        Some(proj) if item_vector.len() == proj.original_dim => proj.project(item_vector),
        Some(proj) if item_vector.len() == proj.reduced_dim => item_vector.to_owned(),
        None => item_vector.to_owned(),
        Some(proj) => {
            panic!(
                "Check the projection pipeline, item seems neither projected nor unprojected. \n\
                 input item len: {:?} \
                 projection matrix is set: {} \
                 projection matrix original dims: {} \
                 projection matrix reduced dims: {}",
                item_vector.len(),
                true,
                proj.original_dim,
                proj.reduced_dim
            )
        }
    }
}

/// Compute per-band Rayleigh energies for one vector.
///
/// The output is scale-invariant: scaling the vector by a non-zero constant
/// leaves the band energies unchanged. The sum of all bands equals
/// `x^T L x / x^T x` for a symmetric graph Laplacian.
pub fn compute_band_energies_from_matrix(
    matrix: &CsMat<f64>,
    vector: &[f64],
    config: &FilterBankConfig,
) -> Vec<f64> {
    assert_eq!(matrix.rows(), matrix.cols(), "Matrix must be square");
    assert_eq!(
        matrix.rows(),
        vector.len(),
        "Matrix rows {} must match vector length {}. Matrix shape: {:?}",
        matrix.rows(),
        vector.len(),
        matrix.shape()
    );

    let edges = config.edges();
    let mut bands = vec![0.0; config.num_bands];
    let denominator: f64 = vector.iter().map(|&x| x * x).sum();
    if denominator <= 1e-12 {
        return bands;
    }

    let mut max_contribution = 0.0_f64;
    for (i, row) in matrix.outer_iterator().enumerate() {
        for (j, &lij) in row.iter() {
            if j <= i {
                continue;
            }
            let w = (-lij).max(0.0);
            if w <= 0.0 {
                continue;
            }
            let d = vector[i] - vector[j];
            max_contribution = max_contribution.max(w * d * d);
        }
    }

    if max_contribution <= 1e-12 {
        return bands;
    }

    for (i, row) in matrix.outer_iterator().enumerate() {
        for (j, &lij) in row.iter() {
            if j <= i {
                continue;
            }
            let w = (-lij).max(0.0);
            if w <= 0.0 {
                continue;
            }
            let d = vector[i] - vector[j];
            let contribution = w * d * d;
            if contribution <= 0.0 {
                continue;
            }
            let normalised = (contribution / max_contribution).clamp(0.0, 1.0);
            let band_idx = band_index(normalised, &edges);
            bands[band_idx] += contribution / denominator;
        }
    }

    bands
}

fn band_index(value: f64, edges: &[f64]) -> usize {
    let last_band = edges.len() - 2;
    if value >= 1.0 {
        return last_band;
    }
    for band in 0..last_band {
        if value >= edges[band] && value < edges[band + 1] {
            return band;
        }
    }
    last_band
}
