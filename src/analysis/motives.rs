//! Graph motif detection via triangle density and spectral cohesion.
//!
//! This module provides efficient triangle-based motif spotting on sparse graph Laplacians,
//! leveraging local clustering coefficients and optional Rayleigh-quotient validation
//! to surface cohesive, low-boundary subgraphs and near-cliques.
//!
//! # Overview
//!
//! - **Motives trait**: Public API for motif detection on any graph structure.
//! - **MotiveConfig**: Tunable parameters for seeding, expansion, and deduplication.
//! - **Zero-copy adjacency**: Iterates Laplacian off-diagonals on the fly; no separate matrix.
//! - **Triangle seeding**: Seeds from nodes with high triangle counts and clustering ≥ threshold.
//! - **Greedy expansion**: Grows motifs by maximizing triangle gain per added node.
//! - **Rayleigh validation**: Optional spectral check to keep sets cohesive and low-cut.
//!
//! # Usage
//!
//! ```ignore
//! use arrowspace::graph::GraphLaplacian;
//! use arrowspace::analysis::motives::{Motives, MotiveConfig};
//!
//! let gl: GraphLaplacian = /* ... */;
//! let cfg = MotiveConfig {
//!     top_l: 16,
//!     min_triangles: 3,
//!     min_clust: 0.5,
//!     max_motif_size: 24,
//!     max_sets: 128,
//!     jaccard_dedup: 0.8,
//! };
//! // Feature-space ensembles (nodes of this Laplacian, as built):
//! let motifs = gl.try_spot_motives_featurespace(&cfg)?;
//! // Item-space motifs on an EigenMaps build (needs the ArrowSpace index):
//! let item_motifs = gl.try_spot_motives_eigen(&aspace, &cfg)?;
//! ```
//!
//! # References
//!
//! - Scalable motif-aware clustering: <https://arxiv.org/abs/1606.06235>
//! - Local clustering coefficient: <https://en.wikipedia.org/wiki/Clustering_coefficient>
//! - Cheeger inequality & spectral cuts: MIT OCW Lecture Notes

use crate::graph::GraphLaplacian;
use log::{debug, info};
use rayon::prelude::*;
use smartcore::linalg::basic::arrays::{Array, Array2};
use std::collections::HashSet;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for motif detection.
#[derive(Clone, Debug)]
pub struct MotiveConfig {
    /// Prune to top-L strongest neighbors per node (from Laplacian).
    pub top_l: usize,
    /// Minimum triangle count to seed a motif.
    pub min_triangles: usize,
    /// Minimum local clustering coefficient C_i to seed a motif.
    pub min_clust: f64,
    /// Maximum size (number of nodes) per motif during greedy expansion.
    pub max_motif_size: usize,
    /// Limit on number of returned motif sets.
    pub max_sets: usize,
    /// Jaccard similarity threshold for deduplication (0..=1).
    pub jaccard_dedup: f64,
}

impl Default for MotiveConfig {
    fn default() -> Self {
        Self {
            top_l: 16,
            min_triangles: 2,
            min_clust: 0.4,
            max_motif_size: 32,
            max_sets: 256,
            jaccard_dedup: 0.8,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public trait
// ──────────────────────────────────────────────────────────────────────────────

/// Trait for detecting graph motifs (triangles, near-cliques) via local density and spectral cohesion.
///
/// # Node spaces
///
/// The pipeline hands out two different Laplacians, and motif ids live in the
/// node space of the graph they were detected on (issue #165):
///
/// - The EigenMaps / energy-bootstrap `GraphLaplacian.matrix` is **F×F over
///   feature dimensions** — its node ids are feature indices.
/// - Item-space motifs are detected on the centroid (or subcentroid) graph and
///   expanded to **item indices** via the index bookkeeping
///   (`cluster_assignments` / `centroid_map`): use
///   [`Motives::try_spot_motives_eigen`] or [`Motives::try_spot_motives_energy`].
pub trait Motives {
    /// Spot motifs on this Laplacian's own nodes (deprecated, feature-space).
    ///
    /// On a pipeline-built EigenMaps (or energy bootstrap) graph the returned
    /// ids are **feature-dimension indices** of the F×F Laplacian, not item
    /// indices, even though `GraphLaplacian::nnodes` reports the item count
    /// (issue #165). For item-space ids use [`Motives::try_spot_motives_eigen`]
    /// (eigen track) or [`Motives::try_spot_motives_energy`] (energy track);
    /// this exact feature-space behaviour, as a fallible call, is
    /// [`Motives::try_spot_motives_featurespace`].
    ///
    /// # Arguments
    ///
    /// - `cfg`: Configuration for seeding, expansion, and filtering.
    ///
    /// # Algorithm
    ///
    /// 1. Build top-L neighbor lists per node by iterating Laplacian off-diagonals.
    /// 2. Count triangles per node and compute local clustering coefficient C_i = 2T_i / (k_i(k_i-1)).
    /// 3. Seed from nodes meeting `min_triangles` and `min_clust` thresholds, sorted by triangle count descending.
    /// 4. Greedily expand each seed by adding neighbors that maximize triangle gain with existing motif members.
    /// 5. Optional: enforce Rayleigh quotient on indicator ≤ `rayleigh_max` to keep motifs cohesive.
    /// 6. Deduplicate sets with Jaccard similarity ≥ `jaccard_dedup`.
    ///
    /// # Performance
    ///
    /// - Time: O(n · L²) for triangle enumeration, O(seeds · expansion) for greedy growth.
    /// - Space: O(n · L) for neighbor lists; no separate adjacency matrix.
    ///
    /// # References
    ///
    /// - Triangle-based clustering: <https://arxiv.org/abs/1606.06235>
    /// - Local clustering: <https://en.wikipedia.org/wiki/Clustering_coefficient>
    /// - Rayleigh quotient & cuts: MIT OCW, Cheeger inequality notes
    #[deprecated(
        since = "0.27.4",
        note = "returns this Laplacian's node ids as-is (feature-space ids on pipeline builds); use try_spot_motives_eigen for item-space motifs, or try_spot_motives_featurespace for the same feature-space behaviour as a fallible call"
    )]
    fn spot_motives_eigen(&self, cfg: &MotiveConfig) -> Vec<Vec<usize>>;

    /// Fallible, explicitly-named twin of the deprecated
    /// [`Motives::spot_motives_eigen`]: runs the identical detection on this
    /// Laplacian's own nodes and reproduces its output exactly.
    ///
    /// # Node space
    ///
    /// The returned ids are **node ids of `self.matrix`**. On pipeline-built
    /// EigenMaps graphs that matrix is the F×F bootstrap Laplacian, so the ids
    /// enumerate **feature dimensions, not items** (issue #165). Use this
    /// variant for feature-space analysis (dimension ensembles); use
    /// [`Motives::try_spot_motives_eigen`] for item-space motifs.
    ///
    /// Never fails on a well-formed Laplacian — the `Result` keeps the
    /// `try_*` surface uniform and leaves room for validation without a
    /// future signature change.
    fn try_spot_motives_featurespace(
        &self,
        cfg: &MotiveConfig,
    ) -> Result<Vec<Vec<usize>>, crate::error::ArrowSpaceError>;

    /// Item-space motif spotting for the EigenMaps track (issue #165),
    /// mirroring [`Motives::try_spot_motives_energy`]:
    ///
    /// 1. Rebuild the X×X Laplacian over cluster centroids from the index's
    ///    own rows (`aspace.data`) and the pipeline's item→cluster
    ///    assignment — the same clustered structure the bootstrap graph was
    ///    assembled from, no raw-data bypass. (Centroid coordinates are
    ///    reconstructed rather than read from `gl.init_data`: see issue
    ///    #167 for the `clustered_dm` layout defect that makes `init_data`
    ///    columns unusable as coordinates today.)
    /// 2. Spot motifs on the centroid graph.
    /// 3. Expand each centroid set to **item indices** via
    ///    `ArrowSpace.cluster_assignments`, then deduplicate.
    ///
    /// Requirements (enforced — returns
    /// [`ArrowSpaceError::EigenModeRequired`](crate::error::ArrowSpaceError::EigenModeRequired)
    /// instead of degrading):
    /// - `self.energy` must be false (built via `build`; EnergyMaps graphs
    ///   already have the finer-grained [`Motives::try_spot_motives_energy`])
    /// - `aspace.n_clusters >= 2`
    /// - `aspace.cluster_assignments` carries one entry per raw-data row and
    ///   **every** entry is `Some(c)` with `c < n_clusters` — outliers or
    ///   out-of-range ids would silently drop items from the projection, so
    ///   the call refuses instead
    /// - `aspace.data` is present with a feature axis of at least 2
    /// - every centroid is non-empty after the assignment validation
    ///
    /// The returned sets are item indices in `0..aspace.nitems`; each motif is
    /// a union of whole clusters (every item assigned to a detected centroid
    /// is included).
    fn try_spot_motives_eigen(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &MotiveConfig,
    ) -> Result<Vec<Vec<usize>>, crate::error::ArrowSpaceError>;

    /// EnergyMaps-aware motif spotting:
    /// 1) Spot motifs on the subcentroid Laplacian (self).
    /// 2) Map each subcentroid-set to original item indices via ArrowSpace.centroid_map.
    /// 3) Deduplicate and return item-index motifs.
    ///
    /// Requirements (enforced — the method returns
    /// [`ArrowSpaceError::EnergyModeRequired`](crate::error::ArrowSpaceError::EnergyModeRequired)
    /// instead of degrading):
    /// - self.energy must be true (built via build_energy)
    /// - aspace.sub_centroids must be Some, so motif detection runs in
    ///   subcentroid space rather than over the F×F bootstrap Laplacian
    ///   (whose nodes enumerate features)
    /// - aspace.centroid_map must be Some(Vec<usize>) mapping item -> subcentroid index
    ///
    /// Use this fallible variant; the panicking twin
    /// [`Motives::spot_motives_energy`] is deprecated since 0.27.3.
    fn try_spot_motives_energy(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &crate::analysis::motives::MotiveConfig,
    ) -> Result<Vec<Vec<usize>>, crate::error::ArrowSpaceError>;

    /// Panicking twin of [`Motives::try_spot_motives_energy`].
    #[deprecated(
        since = "0.27.3",
        note = "use try_spot_motives_energy; this panics on non-energy builds"
    )]
    fn spot_motives_energy(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &crate::analysis::motives::MotiveConfig,
    ) -> Vec<Vec<usize>>;

    /// Check if a given set of nodes forms a clique in the graph.
    ///
    /// Returns `true` if all pairs in `set` are connected.
    fn is_clique(&self, set: &HashSet<usize>) -> bool;

    /// Compute the Rayleigh quotient R_L(1_S) = (1_S^T L 1_S) / (1_S^T 1_S) for an indicator vector of `set`.
    ///
    /// Low values indicate cohesive, low-boundary subgraphs.
    fn rayleigh_indicator(&self, set: &HashSet<usize>) -> f64;
}

// ──────────────────────────────────────────────────────────────────────────────
// Implementation for GraphLaplacian
// ──────────────────────────────────────────────────────────────────────────────

impl Motives for GraphLaplacian {
    fn try_spot_motives_featurespace(
        &self,
        cfg: &MotiveConfig,
    ) -> Result<Vec<Vec<usize>>, crate::error::ArrowSpaceError> {
        info!(
            "Spotting motifs: top_l={}, min_tri={}, min_clust={:.2}, max_size={}",
            cfg.top_l, cfg.min_triangles, cfg.min_clust, cfg.max_motif_size
        );

        // Nodes of this Laplacian as built: on pipeline EigenMaps graphs the
        // F×F bootstrap enumerates feature dimensions (issue #165 contract).
        let n = self.init_data.shape().0;

        // Shared deterministic detector (invariant #4): identical seeding,
        // expansion and dedup as the item-space tracks. This replaces the
        // legacy body whose HashSet frontier + unstable seed sort made two
        // calls on the same Laplacian diverge on ties (0.27.3 behaviour was
        // not reproducible; see CHANGELOG).
        let results = self.motif_node_sets(cfg, n);

        info!("Motifs found: {}", results.len());

        let mut out: Vec<Vec<usize>> = results
            .into_iter()
            .map(|res| {
                let mut v: Vec<usize> = res.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        out.shrink_to_fit();
        Ok(out)
    }

    #[allow(deprecated)]
    fn spot_motives_eigen(&self, cfg: &MotiveConfig) -> Vec<Vec<usize>> {
        self.try_spot_motives_featurespace(cfg)
            .expect("feature-space spotting operates on the Laplacian as given and never fails")
    }

    fn try_spot_motives_eigen(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &MotiveConfig,
    ) -> Result<Vec<Vec<usize>>, crate::error::ArrowSpaceError> {
        use crate::error::ArrowSpaceError;

        // Item-space contract for the eigen track (issue #165), mirroring
        // try_spot_motives_energy: detection runs on the X-node centroid
        // graph and centroid sets expand to ITEM indices through the
        // item→cluster bookkeeping. Serving the F×F bootstrap node ids as
        // item indices is the #161 failure family, so missing structure is
        // refused instead of degrading.
        if self.energy {
            return Err(ArrowSpaceError::EigenModeRequired {
                missing: "eigen build (use ArrowSpaceBuilder::build); \
                          EnergyMaps graphs must use try_spot_motives_energy",
            });
        }
        if aspace.n_clusters < 2 {
            return Err(ArrowSpaceError::EigenModeRequired {
                missing: "at least 2 clusters (n_clusters)",
            });
        }
        // Full assignment validation BEFORE any accumulation: the item-space
        // contract is that every raw-data item participates in the projection.
        // A None (outlier) or an out-of-range cluster id would silently drop
        // that item from `sums` and `c_to_items`, yielding motifs that are
        // not unions of whole clusters — refuse instead of degrading.
        if aspace.cluster_assignments.len() != aspace.nitems {
            return Err(ArrowSpaceError::EigenModeRequired {
                missing: "cluster_assignments on the ArrowSpace index",
            });
        }
        let n_c = aspace.n_clusters;
        for assign in aspace.cluster_assignments.iter() {
            match assign {
                None => {
                    return Err(ArrowSpaceError::EigenModeRequired {
                        missing: "a cluster assignment for every item \
                                  (item without cluster; outliers cannot be \
                                  projected in item space)",
                    });
                }
                Some(c) if *c >= n_c => {
                    return Err(ArrowSpaceError::EigenModeRequired {
                        missing: "cluster_assignments values within \
                                  0..n_clusters",
                    });
                }
                Some(_) => {}
            }
        }

        // Centroid coordinates: recompute the X×F centroid matrix from the
        // index's own rows (`aspace.data`) and the pipeline's item→cluster
        // assignment. This mirrors the energy track, which rebuilds its
        // subcentroid graph from stored coordinates (see #161): detection
        // runs on a Laplacian over the same clustered structure the pipeline
        // produced — no parallel data path is introduced, and the
        // reconstruction is deterministic.
        //
        // (gl.init_data would be the natural source — it stores the F×X
        // bootstrap input — but its columns cannot currently be read back as
        // centroid coordinates: run_incremental_clustering_with_sampling
        // feeds a row-major flat buffer to DenseMatrix::from_iterator with
        // axis=1, which reinterprets it as column-major (issue #167). Until
        // that layout fix lands in its own breaking release, init_data is
        // not a usable coordinate source.)
        let (data_rows, n_feats) = aspace.data.shape();
        if data_rows != aspace.nitems || n_feats < 2 {
            return Err(ArrowSpaceError::EigenModeRequired {
                missing: "the raw data matrix (aspace.data) matching nitems \
                          with a feature axis of at least 2",
            });
        }
        let mut sums = vec![vec![0.0f64; n_feats]; n_c];
        let mut counts = vec![0usize; n_c];
        for (it, assign) in aspace.cluster_assignments.iter().enumerate() {
            if let Some(c) = assign
                && *c < n_c
            {
                counts[*c] += 1;
                for (k, x) in aspace.data.get_row(it).iterator(0).enumerate() {
                    sums[*c][k] += x;
                }
            }
        }
        for c in 0..n_c {
            if counts[c] == 0 {
                return Err(ArrowSpaceError::EigenModeRequired {
                    missing: "a non-empty cluster for every centroid \
                              (cluster_assignments values in 0..n_clusters)",
                });
            }
            let inv = 1.0 / counts[c] as f64;
            for x in sums[c].iter_mut() {
                *x *= inv;
            }
        }
        // Assemble X×F row-major (centroid-major flat, axis=0) — the buffer
        // order DenseMatrix::from_iterator needs to interpret as rows.
        let mut flat: Vec<f64> = Vec::with_capacity(n_c * n_feats);
        for row in &sums {
            flat.extend_from_slice(row);
        }
        let centroids = smartcore::linalg::basic::matrix::DenseMatrix::<f64>::from_iterator(
            flat.iter().copied(),
            n_c,
            n_feats,
            0,
        );

        // Rebuild the X×X Laplacian over centroids with the pipeline's own
        // graph parameters (clamped to the smaller node count, as in the
        // energy path). nnodes is pinned to the centroid count so the
        // rebuilt struct's node space is unambiguous.
        let mut params = self.graph_params.clone();
        params.k = params.k.min(n_c - 1);
        params.topk = params.topk.min(4).min(n_c - 1);
        let centroid_graph =
            crate::laplacian::build_laplacian_matrix(centroids, &params, Some(n_c), false);

        info!(
            "Spotting eigen motifs (item space): top_l={}, min_tri={}, min_clust={:.2}, max_size={}, n_clusters={}",
            cfg.top_l, cfg.min_triangles, cfg.min_clust, cfg.max_motif_size, n_c
        );

        let c_sets = centroid_graph.motif_node_sets(cfg, n_c);

        info!("Eigen motifs: {} centroid motifs found", c_sets.len());

        // cluster → items, built sequentially in ascending item order so the
        // projection is deterministic by construction (invariant #4).
        let mut c_to_items: Vec<Vec<usize>> = vec![Vec::new(); n_c];
        for (it, assign) in aspace.cluster_assignments.iter().enumerate() {
            if let Some(c) = assign
                && *c < n_c
            {
                c_to_items[*c].push(it);
            }
        }

        let out = project_node_sets_to_items(c_sets, &c_to_items, cfg);

        info!(
            "Eigen motifs: {} item-level motifs after mapping",
            out.len()
        );
        Ok(out)
    }

    fn try_spot_motives_energy(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &MotiveConfig,
    ) -> Result<Vec<Vec<usize>>, crate::error::ArrowSpaceError> {
        use crate::error::ArrowSpaceError;

        // Enforce the documented requirements (issue #161). Degrading silently
        // here ran motif detection over the F×F bootstrap Laplacian — whose
        // nodes enumerate FEATURES — and returned the ids as item indices.
        if !self.energy {
            return Err(ArrowSpaceError::EnergyModeRequired {
                missing: "energy build (use EnergyMapsBuilder::build_energy)",
            });
        }
        if aspace.sub_centroids.is_none() {
            return Err(ArrowSpaceError::EnergyModeRequired {
                missing: "sub_centroids on the ArrowSpace index",
            });
        }
        let cmap = match &aspace.centroid_map {
            Some(m) => m,
            None => {
                return Err(ArrowSpaceError::EnergyModeRequired {
                    missing: "centroid_map on the ArrowSpace index",
                });
            }
        };

        // Operate strictly on the energy Laplacian over subcentroids
        let (rows, cols) = self.matrix.shape();
        if rows == 0 || rows != cols {
            return Ok(Vec::new());
        }
        let n_sc = rows;

        // The energy pipeline hands us the F×F bootstrap Laplacian used for
        // taumode λ read-outs; its nodes enumerate FEATURES, while
        // aspace.centroid_map enumerates SUBCENTROIDS. Motif detection must run
        // in subcentroid space so that projection onto items joins matching
        // namespaces: rebuild the X×X subcentroid Laplacian from stored
        // coordinates whenever dimensions disagree.
        let rebuilt: Option<GraphLaplacian> = match &aspace.sub_centroids {
            Some(sc) if sc.shape().0 >= 2 && sc.shape().0 != rows => {
                let (x, _) = sc.shape();
                let mut params = self.graph_params.clone();
                params.k = params.k.min(x - 1);
                params.topk = params.topk.min(4).min(x - 1);
                Some(crate::laplacian::build_laplacian_matrix(
                    sc.clone(),
                    &params,
                    Some(x),
                    true,
                ))
            }
            _ => None,
        };

        let (neigh_source, n_sc): (&GraphLaplacian, usize) = match &rebuilt {
            Some(g) => (g, g.matrix.shape().0),
            None => (self, n_sc),
        };

        info!(
            "Spotting energy motifs: top_l={}, min_tri={}, min_clust={:.2}, max_size={}, n_sc={}",
            cfg.top_l, cfg.min_triangles, cfg.min_clust, cfg.max_motif_size, n_sc
        );

        // Steps 1–5 (triangle seeding + deterministic greedy expansion) and
        // steps 6–7 (projection onto items + canonical dedup) are shared with
        // the eigen item-space track (issue #165); see motif_node_sets and
        // project_node_sets_to_items.
        let sc_results = neigh_source.motif_node_sets(cfg, n_sc);

        info!(
            "Energy motifs: {} subcentroid motifs found",
            sc_results.len()
        );

        // 6) Map to item indices via centroid_map — built sequentially in
        // ascending item order, so each subcentroid's item list is sorted and
        // the grouping is deterministic by construction (invariant #4).
        // The map is guaranteed present by the requirement checks above.
        let mut sc_to_items: Vec<Vec<usize>> = vec![Vec::new(); n_sc];
        for (it, &sc) in cmap.iter().enumerate() {
            if sc < n_sc {
                sc_to_items[sc].push(it);
            }
        }

        let mut out: Vec<Vec<usize>> = project_node_sets_to_items(sc_results, &sc_to_items, cfg);
        out.shrink_to_fit();

        info!(
            "Energy motifs: {} item-level motifs after mapping",
            out.len()
        );

        // Output vecs are already sorted from canonicalisation above.
        Ok(out)
    }

    fn spot_motives_energy(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &MotiveConfig,
    ) -> Vec<Vec<usize>> {
        self.try_spot_motives_energy(aspace, cfg).expect(
            "spot_motives_energy requires an energy build with sub_centroids \
             and centroid_map; use try_spot_motives_energy for a typed error",
        )
    }

    fn is_clique(&self, set: &HashSet<usize>) -> bool {
        let sz = set.len();
        if sz < 2 {
            return false;
        }
        // Parallel short-circuit check

        set.par_iter().all(|&u| {
            let nbrs: HashSet<usize> = self.neighbors_of(u).iter().map(|(j, _)| *j).collect();
            let need = sz - 1;
            let have = nbrs.intersection(set).count();
            have == need
        })
    }

    /// unused: potential improvements using rayleigh energy boundaries
    fn rayleigh_indicator(&self, set: &HashSet<usize>) -> f64 {
        // Active computation space derived from the Laplacian itself
        let (rows, cols) = self.matrix.shape();
        if rows == 0 || rows != cols || set.is_empty() {
            return f64::INFINITY;
        }
        let n = rows;
        if set.iter().any(|&u| u >= n) {
            return f64::INFINITY;
        }
        let mut x = vec![0.0f64; n];
        for &i in set {
            x[i] = 1.0;
        }
        self.rayleigh_quotient(&x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared machinery for the item-space tracks (#161, #165)
// ──────────────────────────────────────────────────────────────────────────────

impl GraphLaplacian {
    /// Steps 1–5 of the item-space motif pipeline: triangle-based seeding and
    /// greedy expansion over the first `n` nodes of this Laplacian, followed
    /// by Jaccard dedup. Returns motif sets in this graph's node space
    /// (centroids / subcentroids); projection onto items is
    /// [`project_node_sets_to_items`].
    ///
    /// Fully deterministic per design invariant #4 — see the notes on the
    /// seed sort and the expansion loop.
    fn motif_node_sets(&self, cfg: &MotiveConfig, n: usize) -> Vec<HashSet<usize>> {
        // 1) Neighbors with clamped indices (parallel)
        let neigh_idx: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut ids: Vec<usize> = self
                    .neighbors_of(i)
                    .into_iter()
                    .filter_map(|(j, w)| {
                        if j < n && j != i && w > 0.0 {
                            Some(j)
                        } else {
                            None
                        }
                    })
                    .collect();
                ids.sort_unstable();
                if ids.len() > cfg.top_l {
                    ids.truncate(cfg.top_l);
                }
                ids
            })
            .collect();

        // 2) Triangle stats (parallel)
        let (tri_count, clust) = triangle_stats_sorted(&neigh_idx, n);

        debug!(
            "Triangle stats: max_tri={}, max_clust={:.3}",
            tri_count.iter().copied().max().unwrap_or(0),
            clust.iter().cloned().fold(0.0f64, f64::max)
        );

        // 3) Seeds (parallel filter + deterministic sort)
        //
        // par_filter output order is scheduler-dependent; par_sort_unstable_by_key
        // can leave equal-scored seeds in arbitrary relative order because it is
        // unstable. We finish with a sequential sort_by_key so that seeds with
        // identical (tri_count, clust) scores always appear in ascending node-index
        // order, giving the greedy expansion in step 4 a fully reproducible input.
        let mut seeds: Vec<usize> = (0..n)
            .into_par_iter()
            .filter(|&i| tri_count[i] >= cfg.min_triangles && clust[i] >= cfg.min_clust)
            .collect();

        // Primary key: triangle count descending + clustering coefficient descending.
        // Tie-breaker: node index ascending — deterministic across all runs.
        seeds.sort_by_key(|&i| {
            (
                std::cmp::Reverse(tri_count[i]),
                std::cmp::Reverse((clust[i] * 1e6) as i64),
                i, // ascending node index breaks all remaining ties
            )
        });

        debug!("Motif seeds (graph nodes): {:?}", seeds);

        // 4) Parallel greedy expansions per seed in node space.
        //
        // Each expansion is independent, so par_iter is safe here.
        // The two sources of non-determinism fixed below are:
        //
        //   a) Candidate frontier built from HashSet iteration:
        //      `for &u in &seeds_hashset` and `for &v in &neigh_idx[u]` both iterated
        //      a HashSet, whose order is undefined. Two runs could produce different
        //      `cand` sets with different insertion orders, affecting which u wins
        //      when multiple candidates share the same best_gain.
        //      Fix: collect cand into a sorted Vec before the gain loop.
        //
        //   b) Tie-breaking in the gain loop:
        //      When two candidates share best_gain, `best_u` was last-write-wins
        //      over HashSet iteration order. After fix (a) the loop is ordered, so
        //      the first maximum found is always the lowest node index — stable.
        let expansions: Vec<Option<HashSet<usize>>> = seeds
            .par_iter()
            .map(|&s| {
                // Use a sorted Vec as the working set so `for &u in &seeds_vec`
                // always iterates in ascending node-index order.
                let mut seeds_vec: Vec<usize> = vec![s];

                loop {
                    if seeds_vec.len() >= cfg.max_motif_size {
                        break;
                    }

                    // Build frontier: neighbours of current set not yet in set.
                    // Collect into a HashSet first to deduplicate, then sort for
                    // deterministic iteration order in the gain loop below.
                    let seeds_set: HashSet<usize> = seeds_vec.iter().copied().collect();
                    let cand: Vec<usize> = {
                        let mut c = HashSet::new();
                        for &u in &seeds_vec {
                            for &v in &neigh_idx[u] {
                                if !seeds_set.contains(&v) {
                                    c.insert(v);
                                }
                            }
                        }
                        let mut v: Vec<usize> = c.into_iter().collect();
                        v.sort_unstable(); // deterministic candidate order
                        v
                    };

                    if cand.is_empty() {
                        break;
                    }

                    // Select candidate with highest triangle-gain.
                    // Iterating a sorted Vec means the first maximum encountered is
                    // always the lowest node index — fully deterministic tie-breaking.
                    let mut best_u: Option<usize> = None;
                    let mut best_gain: i64 = -1;

                    for &u in &cand {
                        let mut s_nbrs: Vec<usize> = neigh_idx[u]
                            .iter()
                            .copied()
                            .filter(|v| seeds_set.contains(v))
                            .collect();
                        s_nbrs.sort_unstable();
                        let mut edges = 0i64;
                        for i in 0..s_nbrs.len() {
                            let ui = s_nbrs[i];
                            edges += count_edges_among(&neigh_idx[ui], &s_nbrs, i + 1) as i64;
                        }
                        if edges > best_gain {
                            best_gain = edges;
                            best_u = Some(u);
                        }
                    }

                    match best_u {
                        Some(u) => {
                            seeds_vec.push(u);
                            seeds_vec.sort_unstable(); // keep working set sorted
                        }
                        None => break,
                    }
                }

                if seeds_vec.len() >= 3 {
                    Some(seeds_vec.iter().copied().collect::<HashSet<usize>>())
                } else {
                    None
                }
            })
            .collect();

        // 5) Global dedup in node space
        let mut results: Vec<HashSet<usize>> = Vec::new();
        for opt in expansions.into_iter().flatten() {
            let mut keep = true;
            for res in &results {
                if jaccard(&opt, res) >= cfg.jaccard_dedup {
                    keep = false;
                    break;
                }
            }
            if keep {
                results.push(opt);
                if results.len() >= cfg.max_sets {
                    break;
                }
            }
        }

        results
    }
}

/// Steps 6–7 of the item-space motif pipeline: expand node sets (centroids /
/// subcentroids) to item indices through `node_to_items` (whose buckets are
/// sorted ascending by item index), drop sets with fewer than 3 items, and
/// deduplicate. Returns item-index sets, each sorted ascending.
///
/// Determinism note (inherited from #161): `item_sets` is produced by a
/// parallel map, whose delivery order is scheduler-dependent, while the
/// sequential dedup pass is order-sensitive — different input orderings evict
/// different sets on Jaccard ties. Canonicalising every set into a sorted Vec
/// and sorting the whole slice before dedup gives the loop a stable,
/// reproducible input regardless of how Rayon delivered the results.
fn project_node_sets_to_items(
    node_sets: Vec<HashSet<usize>>,
    node_to_items: &[Vec<usize>],
    cfg: &MotiveConfig,
) -> Vec<Vec<usize>> {
    // Project each node motif to items (parallel)
    let item_sets: Vec<HashSet<usize>> = node_sets
        .par_iter()
        .map(|s_nodes| {
            let mut s_items = HashSet::new();
            for &nd in s_nodes {
                for &it in &node_to_items[nd] {
                    s_items.insert(it);
                }
            }
            s_items
        })
        .filter(|s_items| s_items.len() >= 3)
        .collect();

    // Canonicalise: HashSet → sorted Vec so both content and ordering are stable.
    let mut item_sets_sorted: Vec<Vec<usize>> = item_sets
        .into_iter()
        .map(|set| {
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable(); // canonical form for each set
            v
        })
        .collect();

    // Sort the collection itself so dedup always sees the same input order.
    // Lexicographic order on sorted Vecs is deterministic and cheap here
    // since item_sets_sorted is bounded by cfg.max_sets (typically ≤ 60).
    item_sets_sorted.sort_unstable();

    let mut deduped_items: Vec<Vec<usize>> = Vec::new();
    for item in item_sets_sorted {
        let item_set: HashSet<usize> = item.iter().copied().collect();
        let mut keep = true;
        for cmp in &deduped_items {
            let cmp_set: HashSet<usize> = cmp.iter().copied().collect();
            if jaccard(&item_set, &cmp_set) >= cfg.jaccard_dedup {
                keep = false;
                break;
            }
        }
        if keep {
            deduped_items.push(item); // already sorted, no re-sort needed
            if deduped_items.len() >= cfg.max_sets {
                break;
            }
        }
    }

    deduped_items
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers (parallel-friendly)
// ──────────────────────────────────────────────────────────────────────────────

fn triangle_stats_sorted(neigh_idx: &[Vec<usize>], n: usize) -> (Vec<usize>, Vec<f64>) {
    // Count triangles per node by intersecting neighbor lists
    let tri_count: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let nbrs_i = &neigh_idx[i];
            if nbrs_i.len() < 2 {
                return 0usize;
            }
            let mut t = 0usize;
            for &j in nbrs_i {
                if j <= i {
                    continue;
                }
                let nbrs_j = &neigh_idx[j];
                t += count_intersection(nbrs_i, nbrs_j, i, j);
            }
            t
        })
        .collect();

    // Local clustering per node in parallel
    let clust: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let k = neigh_idx[i].len();
            if k >= 2 {
                (2.0 * tri_count[i] as f64) / ((k * (k - 1)) as f64)
            } else {
                0.0
            }
        })
        .collect();

    (tri_count, clust)
}

// Count common neighbors excluding i and j using two-pointer scan on sorted lists
#[inline]
fn count_intersection(a: &[usize], b: &[usize], i: usize, j: usize) -> usize {
    let mut x = 0usize;
    let (mut p, mut q) = (0usize, 0usize);
    while p < a.len() && q < b.len() {
        let va = a[p];
        let vb = b[q];
        if va == vb {
            if va != i && va != j {
                x += 1;
            }
            p += 1;
            q += 1;
        } else if va < vb {
            p += 1;
        } else {
            q += 1;
        }
    }
    x
}

// Count edges among s_nbrs after position start by intersecting with neigh(u)
#[inline]
fn count_edges_among(neigh_u: &[usize], s_nbrs: &[usize], start: usize) -> usize {
    let mut x = 0usize;
    let mut p = 0usize;
    let mut q = start;
    while p < neigh_u.len() && q < s_nbrs.len() {
        let va = neigh_u[p];
        let vb = s_nbrs[q];
        if va == vb {
            x += 1;
            p += 1;
            q += 1;
        } else if va < vb {
            p += 1;
        } else {
            q += 1;
        }
    }
    x
}

pub fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    let inter = a.intersection(b).count() as f64;
    let union = (a.len() + b.len()) as f64 - inter;
    if union == 0.0 { 0.0 } else { inter / union }
}
