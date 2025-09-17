use crate::core::ArrowSpace;
use crate::graph_factory::{GraphFactory, GraphLaplacian};
use crate::taumode::{TauMode, compute_synthetic_lambdas};

pub struct ArrowSpaceBuilder {
    // Data
    arrows: ArrowSpace,
    pub prebuilt_gl: Option<GraphLaplacian>, // use as-is if already built

    // Synthetic index configuration (used `with_synthesis`)
    synthesis: Option<(f64, TauMode)>, // (alpha, tau_mode)

    // Lambda-graph parameters (the canonical path)
    // A good starting point is to choose parameters that keep the λ-graph broadly connected but sparse,
    // and set the kernel to behave nearly linearly for small gaps so it doesn’t overpower cosine;
    // a practical default is: lambda_eps ≈ 1e-3, lambda_k ≈ 3–10, lambda_p = 2.0,
    // lambda_sigma = None (which defaults σ to eps)
    lambda_eps: f64,
    lambda_k: usize,
    lambda_p: f64,
    lambda_sigma: Option<f64>,
}

impl Default for ArrowSpaceBuilder {
    fn default() -> Self {
        Self {
            arrows: ArrowSpace::default(),
            prebuilt_gl: None,

            // enable synthetic λ with α=0.7 and Median τ by default
            synthesis: Some((0.7, TauMode::Median)),

            // λ-graph parameters
            lambda_eps: 1e-3,
            lambda_k: 6,
            lambda_p: 2.0,
            lambda_sigma: None, // means σ := eps inside the builder
        }
    }
}

impl ArrowSpaceBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------- Data --------------------

    /// Mandatory.
    /// Always provide the row-major 2D Vec<Vec<..>> (as per a database, each row is an item).
    /// The 2D array is automatically transposed to compute Rayleigh.
    pub fn with_rows(mut self, rows: Vec<Vec<f64>>) -> Self {
        self.arrows = ArrowSpace::from_items(rows);
        self
    }

    // -------------------- Lambda-graph configuration --------------------

    /// Use this to pass λτ-graph parameters.
    /// Configure the base λτ-graph to be built from the provided data matrix:
    /// - eps: threshold for |Δλ| on items
    /// - k: optional cap on neighbors per item
    /// - p: weight kernel exponent
    /// - sigma_override: optional scale σ for the kernel (default = eps)
    pub fn with_lambda_graph(
        mut self,
        eps: f64,
        k: usize,
        p: f64,
        sigma_override: Option<f64>,
    ) -> Self {
        self.lambda_eps = eps;
        self.lambda_k = k;
        self.lambda_p = p;
        self.lambda_sigma = sigma_override;
        self
    }

    // -------------------- Synthetic index --------------------

    /// Optional: override the default tau policy or alpha for synthetic index.
    /// Note: when the fallback λτ-graph (priority #3) is chosen, synthesis is always ON.
    /// If this method is not called, the default is alpha=0.7, TauMode::Median.
    pub fn with_synthesis(mut self, tau: f64, tau_mode: TauMode) -> Self {
        assert!((0.0..=1.0).contains(&tau), "alpha must be in [0,1]");
        self.synthesis = Some((tau, tau_mode));
        self
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
    pub fn build(mut self) -> (ArrowSpace, GraphLaplacian) {
        assert!(self.arrows.shape() != (0, 0));
        // 1) Base graph selection
        assert!(
            self.prebuilt_gl.is_none(),
            "GraphLaplacian already built, should be None at this point"
        );

        // Recreate matrix from ArrowSpace column-major to wor-major
        let (nrows, _) = self.arrows.shape();
        let mut data_matrix: Vec<Vec<f64>> = Vec::with_capacity(nrows);
        for r in 0..self.arrows.ncols {
            // loop cols number as ArrowSpace is column-major
            data_matrix.push(self.arrows.get_item(r).item.to_vec());
        }
        // Resolve λτ-graph params with conservative defaults
        self.prebuilt_gl = Some(GraphFactory::build_lambda_graph(
            &data_matrix,
            self.lambda_eps,
            self.lambda_k,
            self.lambda_p,
            self.lambda_sigma,
        ));

        // 3) Compute synthetic indices on resulting graph
        let mut aspace = self.arrows;
        let gl = self.prebuilt_gl.unwrap();
        let synth = self.synthesis.unwrap();
        compute_synthetic_lambdas(&mut aspace, &gl, synth.0, synth.1);

        // aspace.recompute_lambdas(&self.prebuilt_gl.clone().unwrap());

        (aspace, gl)
    }
}

/// Transpose N×F → F×N.
/// the transpose makes each feature become a “graph signal” defined on the item nodes,
///  which is exactly the shape needed to compute a Rayleigh quotient λ per feature against
///  a Laplacian whose nodes are the items; without that F×N layout (features over items),
/// λ can’t be evaluated correctly for each feature row on the item graph
pub fn transpose_nf_to_fn(nxf: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = nxf.len();
    let f = nxf.first().map(|row| row.len()).unwrap_or(0);
    let mut fxf = vec![Vec::with_capacity(n); f];
    for row in nxf {
        assert_eq!(row.len(), f, "All rows must have same feature length");
        for (j, &v) in row.iter().enumerate() {
            fxf[j].push(v);
        }
    }
    fxf
}
