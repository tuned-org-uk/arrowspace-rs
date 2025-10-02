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
