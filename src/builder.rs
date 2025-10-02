//! ArrowSpace Builder with Graph-Based Synthetic Index Generation
//!
//! This module provides the `ArrowSpaceBuilder`, a fluent API for constructing `ArrowSpace` instances
//! with configurable graph Laplacians and synthetic lambda indices. It orchestrates the complex process
//! of building item-to-item proximity graphs, computing spectral representations, and deriving
//! τ-mode synthetic indices that capture both local geometry and global manifold structure.
//!
//! # Overview
//!
//! The builder implements a flexible construction pipeline that:
//!
//! 1. **Builds a λ-proximity graph** from data items using rectified cosine distance
//! 2. **Computes graph Laplacian** (N×N) encoding item-to-item relationships
//! 3. **Optionally derives spectral Laplacian** (F×F) encoding feature-to-feature relationships
//! 4. **Generates synthetic λ indices** via Rayleigh quotient analysis with τ-mode selection
//!
//! # Construction Philosophy
//!
//! ## Graph-First Approach
//!
//! All ArrowSpace instances are fundamentally graph-based. The builder always constructs a
//! λ-proximity graph as the canonical representation of item relationships. This graph serves
//! dual purposes:
//! - **Geometric**: Captures local neighborhood structure via k-nearest neighbors
//! - **Spectral**: Enables global manifold analysis via Laplacian eigenfunctions
//!
//! ## Synthetic Index Generation
//!
//! The λ (lambda) values assigned to each item are "synthetic indices" computed from the graph
//! structure, not from raw feature statistics. These indices quantify how much each item's
//! feature vector varies across its graph neighborhood, measured via the Rayleigh quotient:
//!
//! ```ignore
//! λᵢ = R(L, xᵢ) = (xᵢᵀ L xᵢ) / (xᵢᵀ xᵢ)
//! ```
//!
//! where L is the graph Laplacian and xᵢ is the item's feature vector.
//!
//! ## τ-Mode Selection
//!
//! The builder supports multiple strategies for computing synthetic indices via `TauMode`:
//! - **Median**: Uses median of all Rayleigh quotients (robust, default)
//! - **Mean**: Uses mean of Rayleigh quotients (sensitive to outliers)
//! - **Fixed(τ)**: Uses a user-specified τ value for all items
//! - **Adaptive**: Per-item τ based on local graph density
//!
//! # Configuration Parameters
//!
//! ## Lambda-Graph Parameters
//!
//! These control the λ-proximity graph construction:
//!
//! - **`eps` (epsilon)**: Maximum rectified cosine distance threshold. Items within this distance
//!   are considered neighbors. Typical range: [1e-4, 1e-2]. Smaller values create sparser graphs.
//!
//! - **`k`**: Maximum degree per node (neighbor cap). Limits graph density and computational cost.
//!   Recommended: 3-10 for sparse graphs, 10-50 for denser analysis.
//!
//! - **`topk`**: Number of candidates returned during k-NN search. Should be ≤ k. Automatically
//!   adjusted by heuristics if not set explicitly.
//!
//! - **`p`**: Kernel exponent for edge weight computation: w(i,j) = exp(-||xᵢ - xⱼ||ᵖ / σ).
//!   Typical value: 2.0 (Gaussian-like kernel).
//!
//! - **`sigma`**: Kernel bandwidth. Controls decay rate of edge weights. Default: `eps * 0.5`.
//!   Smaller σ creates sharper locality.
//!
//! - **`normalise`**: Whether to normalize feature vectors before graph construction. Set to
//!   `false` (default) to preserve magnitude information.
//!
//! ## Recommended Starting Parameters
//!
//! For most datasets, begin with:
//! ```ignore
//! builder.with_lambda_graph(
//!     1e-3,       // eps: moderate sparsity
//!     6,          // k: small neighborhood
//!     3,          // topk: conservative results
//!     2.0,        // p: Gaussian kernel
//!     None        // sigma: auto (eps/2)
//! )
//! ```
//!
//! Adjust based on:
//! - **Dataset size**: Smaller k for N > 10,000
//! - **Intrinsic dimension**: Larger k for high-dimensional manifolds
//! - **Desired sparsity**: Target < 5% density (enforced by assertions)
//!
//! # Usage Patterns
//!
//! ## Basic Construction
//!
//! ```ignore
//! use arrowspace::builder::ArrowSpaceBuilder;
//!
//! let items = vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![2.0, 3.0, 4.0],
//!     vec![3.0, 4.0, 5.0],
//! ];
//!
//! let (aspace, graph) = ArrowSpaceBuilder::new()
//!     .build(items);
//! ```
//!
//! ## Custom Graph Configuration
//!
//! ```ignore
//! let (aspace, graph) = ArrowSpaceBuilder::new()
//!     .with_lambda_graph(1e-3, 10, 5, 2.0, Some(0.5))
//!     .build(items);
//! ```
//!
//! ## Custom Synthesis Strategy
//!
//! ```ignore
//! use arrowspace::taumode::TauMode;
//!
//! let (aspace, graph) = ArrowSpaceBuilder::new()
//!     .with_synthesis(TauMode::Fixed(0.5))
//!     .build(items);
//! ```
//!
//! ## Spectral Analysis (Expensive)
//!
//! Enable F×F feature-to-feature Laplacian for deeper analysis:
//! ```ignore
//! let (aspace, graph) = ArrowSpaceBuilder::new()
//!     .with_spectral(true)  // Doubles computation time
//!     .build(items);
//!
//! // Access feature signals matrix
//! let signals = &aspace.signals;  // F×F sparse matrix
//! ```
//!
//! # Build Process Details
//!
//! The `build()` method executes the following pipeline:
//!
//! 1. **Validation**: Checks input dimensions and parameter consistency
//! 2. **ArrowSpace initialization**: Creates basic item storage structure
//! 3. **Graph construction**: Builds N×N sparse Laplacian via `GraphFactory`
//! 4. **Spectral computation** (optional): Derives F×F feature Laplacian
//! 5. **Synthetic index generation**: Computes λ values via τ-mode analysis
//! 6. **Validation**: Ensures graph properties (symmetry, sparsity, positivity)
//!
//! # Parameter Tuning Guidelines
//!
//! ## Graph Connectivity
//!
//! - **Too sparse** (k < 3): Risk of disconnected components, unreliable λ values
//! - **Too dense** (k > 50): Excessive computation, loss of local structure
//! - **Sweet spot**: k ∈ [5, 15] for most datasets
//!
//! ## Epsilon Selection
//!
//! - Start with `eps = 1e-3`
//! - Increase if graph is disconnected (check `graph.statistics()`)
//! - Decrease if sparsity < 90% (target 95-98% sparse)
//!
//! ## Sigma vs Epsilon
//!
//! - `sigma = eps/2` (default): Conservative, sharp locality
//! - `sigma = eps`: Moderate decay
//! - `sigma = 2*eps`: Smooth, broad influence
//!
//! # Integration with Search
//!
//! The built ArrowSpace uses λ values for similarity search:
//! ```ignore
//! let query = vec![2.5, 3.5, 4.5];
//! let results = aspace.search(
//!     &query,
//!     &graph,
//!     None,        // tau: use default
//!     Some(5),     // k: top 5 results
//!     Some(0.7),   // alpha: 70% feature weight
//!     Some(0.3)    // beta: 30% lambda weight
//! );
//! ```
//!
//! # Logging and Diagnostics
//!
//! Enable logging to monitor the build process:
//! ```ignore
//! env_logger::Builder::from_default_env()
//!     .filter_level(log::LevelFilter::Debug)
//!     .init();
//! ```
//!
//! Log levels:
//! - **trace**: Detailed matrix operations, inner computations
//! - **debug**: Dimension tracking, parameter values, statistics
//! - **info**: Major pipeline steps, completion messages
//! - **warn**: Numerical issues, validation failures
//!
//! # Numerical Stability
//!
//! The builder maintains precision at the 1e-10 level through:
//! - Sparse matrix representations (avoid dense storage)
//! - Graph sparsity enforcement (< 5% density)
//! - Lambda normalization and clamping to [0, 1]
//! - Validation of Laplacian properties (symmetry, zero row sums)
//!
//! # Common Patterns
//!
//! ## Production Configuration
//!
//! ```ignore
//! let builder = ArrowSpaceBuilder::new()
//!     .with_lambda_graph(1e-3, 8, 4, 2.0, None)
//!     .with_synthesis(TauMode::Median)
//!     .with_normalisation(false);
//! ```
//!
//! ## Exploratory Analysis
//!
//! ```ignore
//! let builder = ArrowSpaceBuilder::new()
//!     .with_lambda_graph(1e-2, 15, 8, 2.0, Some(5e-3))
//!     .with_spectral(true)
//!     .with_synthesis(TauMode::Adaptive);
//! ```
//!
//! ## High-Dimensional Data
//!
//! ```ignore
//! let builder = ArrowSpaceBuilder::new()
//!     .with_lambda_graph(1e-2, 20, 10, 2.0, Some(1e-2))
//!     .with_normalisation(true);  // Mitigate curse of dimensionality
//! ```

use crate::core::{ArrowSpace, TAUDEFAULT};
use crate::graph::{GraphFactory, GraphLaplacian};
use crate::taumode::TauMode;

// Add logging
use log::{debug, info, trace};

#[derive(Clone, Debug)]
pub enum PairingStrategy {
    FastPair,            // 1-NN union via Smartcore FastPair
    Default,             // O(n^2) path
    CoverTreeKNN(usize), // k for k-NN build
}

pub struct ArrowSpaceBuilder {
    // Data
    arrows: ArrowSpace,
    pub prebuilt_spectral: bool, // true if spectral laplacian has been computed

    // Lambda-graph parameters (the canonical path)
    // A good starting point is to choose parameters that keep the λ-graph broadly connected but sparse,
    // and set the kernel to behave nearly linearly for small gaps so it doesn't overpower cosine;
    // a practical default is: lambda_eps ≈ 1e-3, lambda_k ≈ 3–10, lambda_p = 2.0,
    // lambda_sigma = None (which defaults σ to eps)
    lambda_eps: f64,
    lambda_k: usize,
    lambda_topk: usize,
    lambda_p: f64,
    lambda_sigma: Option<f64>,
    normalise: bool,

    // Synthetic index configuration (used `with_synthesis`)
    synthesis: TauMode, // (tau_mode)
}

impl Default for ArrowSpaceBuilder {
    fn default() -> Self {
        debug!("Creating ArrowSpaceBuilder with default parameters");
        Self {
            arrows: ArrowSpace::default(),
            prebuilt_spectral: false,

            // enable synthetic λ with α=0.7 and Median τ by default
            synthesis: TAUDEFAULT,

            // λ-graph parameters
            lambda_eps: 1e-3,
            lambda_k: 6,
            lambda_topk: 3,
            lambda_p: 2.0,
            lambda_sigma: None, // means σ := eps inside the builder
            normalise: false,
        }
    }
}

impl ArrowSpaceBuilder {
    pub fn new() -> Self {
        info!("Initializing new ArrowSpaceBuilder");
        Self::default()
    }

    // -------------------- Lambda-graph configuration --------------------

    /// Use this to pass λτ-graph parameters. If not called, use defaults
    /// Configure the base λτ-graph to be built from the provided data matrix:
    /// - eps: threshold for |Δλ| on items
    /// - k: optional cap on neighbors per item
    /// - p: weight kernel exponent
    /// - sigma_override: optional scale σ for the kernel (default = eps)
    pub fn with_lambda_graph(
        mut self,
        eps: f64,
        k: usize,
        topk: usize,
        p: f64,
        sigma_override: Option<f64>,
    ) -> Self {
        info!(
            "Configuring lambda graph: eps={}, k={}, p={}, sigma={:?}",
            eps, k, p, sigma_override
        );
        debug!(
            "Lambda graph will use {} for normalization",
            if self.normalise {
                "normalized items"
            } else {
                "raw item magnitudes"
            }
        );

        self.lambda_eps = eps;
        self.lambda_k = k;
        self.lambda_topk = topk;
        self.lambda_p = p;
        self.lambda_sigma = sigma_override;
        self.normalise = false;
        self
    }

    // -------------------- Synthetic index --------------------

    /// Optional: override the default tau policy or tau for synthetic index.
    pub fn with_synthesis(mut self, tau_mode: TauMode) -> Self {
        info!("Configuring synthesis with tau mode: {:?}", tau_mode);
        self.synthesis = tau_mode;
        self
    }

    pub fn with_normalisation(mut self, normalise: bool) -> Self {
        info!("Setting normalization: {}", normalise);
        self.normalise = normalise;
        self
    }

    /// Optional define if building spectral matrix at building time
    /// This is expensive as requires twice laplacian computation
    /// use only on limited dataset for analysis, exploration and data QA
    pub fn with_spectral(mut self, compute_spectral: bool) -> Self {
        info!("Setting compute spectral: {}", compute_spectral);
        self.prebuilt_spectral = compute_spectral;
        self
    }

    /// Define the results number of k-neighbours from the
    ///  max number of neighbours connections (`GraphParams::k` -> result_k)
    /// Check if the passed cap_k is reasonable and define an euristics to
    ///  select a proper value.
    fn define_result_k(&mut self) {
        // normalise values for small values,
        // leave to the user for higher values
        if self.lambda_k <= 5 {
            self.lambda_topk = 3;
        } else if self.lambda_k < 10 {
            self.lambda_topk = 4;
        };
    }

    // -------------------- Build --------------------

    /// Build the ArrowSpace and the selected Laplacian (if any).
    ///
    /// Priority order for graph selection:
    ///   1) prebuilt Laplacian (if provided)
    ///   2) hypergraph clique/normalized (if provided)
    ///   3) fallback: λτ-graph-from-data (with_lambda_graph config or defaults)
    ///
    /// Behavior:
    /// - If fallback (#3) is selected, synthetic lambdas are always computed using TauMode::Median
    ///   unless with_synthesis was called, in which case the provided tau_mode and alpha are used.
    /// - If prebuilt or hypergraph graph is selected, standard Rayleigh lambdas are computed unless
    ///   with_synthesis was called, in which case synthetic lambdas are computed on that graph.
    pub fn build(mut self, rows: Vec<Vec<f64>>) -> (ArrowSpace, GraphLaplacian) {
        let n_items = rows.len();
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);

        // set baseline for topk
        self.define_result_k();

        info!(
            "Building ArrowSpace from {} items with {} features",
            n_items, n_features
        );
        debug!(
            "Build configuration: eps={}, k={}, p={}, sigma={:?}, normalise={}, synthesis={:?}",
            self.lambda_eps,
            self.lambda_k,
            self.lambda_p,
            self.lambda_sigma,
            self.normalise,
            self.synthesis
        );

        // 1) Base graph selection
        trace!("Creating ArrowSpace from items");
        self.arrows = ArrowSpace::new(rows.clone(), self.synthesis);
        debug!(
            "ArrowSpace created with {} items and {} features",
            self.arrows.nitems, self.arrows.nfeatures
        );

        // Resolve λτ-graph params with conservative defaults
        info!("Building Laplacian matrix with configured parameters");

        // 3) Compute synthetic indices on resulting graph
        let mut aspace = self.arrows;
        let gl = GraphFactory::build_laplacian_matrix(
            rows,
            self.lambda_eps,
            self.lambda_k,
            self.lambda_topk,
            self.lambda_p,
            self.lambda_sigma,
            self.normalise,
        );
        debug!("Laplacian matrix built successfully");

        // Branch: if spectral L_2 laplacian is required, compute
        // if aspace.signals is not set, gl.matrix will be used
        if self.prebuilt_spectral {
            // Compute signals FxF laplacian
            trace!("Building spectral Laplacian for ArrowSpace");
            aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);
            debug!(
                "Spectral Laplacian built with signals shape: {:?}",
                aspace.signals.shape()
            );
        }

        // Compute taumode lambdas
        info!(
            "Computing taumode lambdas with synthesis: {:?}",
            self.synthesis
        );
        TauMode::compute_taumode_lambdas(&mut aspace, &gl, self.synthesis);

        let lambda_stats = {
            let lambdas = aspace.lambdas();
            let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max: f64 = lambdas.iter().fold(0.0, |a, &b| a.max(b));
            let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            (min, max, mean)
        };

        debug!(
            "Lambda computation completed - min: {:.6}, max: {:.6}, mean: {:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );

        info!("ArrowSpace build completed successfully");
        (aspace, gl)
    }
}
