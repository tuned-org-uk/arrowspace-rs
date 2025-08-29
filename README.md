# ArrowSpace

`ArrowSpace` is a data structure library that encapsulate use of `λτ` indexing; a novel scoring method that mixes Rayleigh and Laplacian scoring (see [`RESEARCH.md`](./RESEARCH.md)) for building vector-search-friendly lookup tables with built-in spectral-awareness. This allows better managing of datasets where spectral characteristics are most relevant. It pairs dense, row‑major arrays with per‑row spectral scores (`λτ`) derived from a Rayleigh-Laplacian score built over items, enabling lambda‑aware similarity, range queries, and composable operations like superposition and element‑wise multiplication over rows. It has been designed to work on datasets where spectral characteristics can be leveraged to find matches that are usually ranked lower by commonly used distance metrics.

Run `cargo run --example proteins_lookup` for an example about how it compares with cosine similarity.

### Requirements

- Rust 1.78+ (edition 2024)

### Run example
```
$ cargo run --example hypergraph_showcase
// run a lookup on AlphaFold vectors
$ cargo run --example proteins_lookup
```

### Run Bench
```
$ cargo bench
```

### Minimal usage

Construct an `ArrowSpace` from rows and compute a synthetic index `λτ` used in similarities search (spectral search):

- Build λτ‑graph from data (preferred path):
    - Use `ArrowSpaceBuilder::with_rows(...).build()` to get an `ArrowSpace` and its Laplacian+Tau mode; the builder will compute per‑row synthetic indices immediately.
    - Use `ArrowSpaceBuilder::with_rows(...).with_lambda_graph(...).build()` to get an `ArrowSpace` and its Laplacian+Tau mode by specifying the parameters for the graph where the Laplacian is computed.
    - Use `ArrowSpaceBuilder::with_rows(...).with_lambdas(...).build()` to get an `ArrowSpace` and its Laplacian+Tau indices by specifying which lambdas values to use.
    - Other bulding options to use hypergraph cliques extensions and boost, ensembles, ...


## Main Features (spectral graph construction and search)

- Data structure for vector search:
    - Lambda+Tau graph from data (default): builds a Laplacian over items from the row matrix, then computes per‑row synthetic λτ using laplacian + TauMode (see paper) with Median policy by default; override via `with_synthesis(alpha, mode)` to change α or τ policy.
    - Direct lambda ε‑graph (lower‑level): constructs a Laplacian from a vector of λ values with ε thresholding and k‑capping, union‑symmetrized CSR; use when supplying external λ instead of synthetic.
    - (optional) Hypergraph overlays: build Laplacians from hyperedges (clique expansion, normalized variant) and overlay “boosts” to strengthen pairs; for prebuilt/hypergraph paths, synthetic λ is opt‑in via with_synthesis.
    - (optional) Ensembles: parameterized variants (k adjust, radius/ε expand, hypergraph transforms) for graph experimentation while reusing the same data matrix; synthetic λ is computed per chosen base when enabled.
- Examples:
    - End‑to‑end examples: protein‑like lookup with λ‑band range query using a ZSET‑style index; showcases for hypergraph, λ‑graph, and synthetic laplacian + TauMode flows.
    - Extensive tests spanning `ArrowSpace` algebra, Rayleigh properties, lambda scale‑invariance, superposition bounds, λ‑graph symmetry and k‑capping semantics, hypergraph correctness, diffusion/random‑walk simulations, fractal integrations, and synthetic λ via Median/Mean/Percentile τ policies.

## Key concepts

- Rows vs. items
    - `ArrowSpace` stores a matrix with rows = feature signals and columns = items; lambdas are per‑row by default and recomputed against a Laplacian whose nodes are items (columns).
- Lambda (Rayleigh quotient)
    - Given Laplacian L and row x, lambda = (xᵀ L x) / (xᵀ x), non‑negative with diagonal degree and negative off‑diagonals; constant vectors approach zero on connected graphs, alternating/high‑frequency rows increase lambda.
    - λ defaults to a synthetic transform of the Rayleigh energy via TauMode (Median), with alternatives selectable through with_synthesis. This makes the lineage explicit: Rayleigh → bounded transform (E′) → optional dispersion blend (G) → final λ used by search.
- Lambda proximity graph
    - The λ‑graph connects items whose aggregated per‑item λ values are within ε; edges are k‑capped by smallest |Δλ| and weighted via a monotone kernel w = 1 / (1 + (|Δλ|/σ)^p), then union‑symmetrized and emitted as CSR Laplacian .

## What Rayleigh does

- Rayleigh is the per‑row smoothness/roughness energy on the item Laplacian: \$ \lambda_{Rayleigh}(x) = \frac{x^\top L x}{x^\top x} \$, non‑negative, scale‑invariant, near‑zero for constant signals, and larger for high‑frequency rows on connected graphs. It is the canonical spectral score tying rows to graph geometry and eigenstructure.


## Why synthesize on top

- In small, smooth, or degenerate regimes, raw Rayleigh can collapse toward 0 or be hard to compare across datasets; “laplacian + TauMode” maps $E_r$ to $E'_r = \frac{E_r}{E_r+\tau}$ and blends it with an edge‑dispersion summary $G_r$ to capture whether roughness is concentrated or diffuse. This produces a bounded, comparable λ that is more informative for search and ranking while preserving the spectral meaning that Rayleigh provides.

## Key concepts

- Rows vs. items
    - ArrowSpace stores a matrix with rows = feature signals and columns = items; lambdas are per‑row and are computed against a Laplacian whose nodes are items (columns). On the fallback graph path, lambdas are the synthetic index by default via laplacian + TauMode (Median), providing a bounded, comparable score per row.
- Lambda (Rayleigh quotient) and synthetic index
    - Given Laplacian L and row x, the Rayleigh energy is \$ \lambda_{Rayleigh} = \frac{x^\top L x}{x^\top x} \$, non‑negative with diagonal degree and negative off‑diagonals; constant vectors approach zero on connected graphs, while alternating/high‑frequency rows increase this value. The synthetic index used by laplacian + TauMode maps Rayleigh energy to $E'_r = \frac{E_r}{E_r + \tau}$ and blends it with an edgewise dispersion term $G_r$ to distinguish diffuse versus concentrated roughness. The default τ policy is TauMode::Median; alternative policies (Fixed, Mean, Percentile) can be selected with with_synthesis.
- Lambda proximity graph
    - The λ‑graph connects items whose aggregated per‑item λ values are within ε; edges are k‑capped by smallest $|\Delta \lambda|$ and weighted via a monotone kernel \$ w = \frac{1}{1 + (|\Delta \lambda|/\sigma)^p} \$, then union‑symmetrized and emitted as a CSR Laplacian. When using the fallback path, per‑row λ values feeding this process are the synthetic index from laplacian + TauMode (Median) by default; custom TauMode or α can be enabled via with_synthesis.

## Usage patterns

- Per‑row vs per‑item λ
    - `ArrowSpace` stores λ per row; the λ‑graph over items is constructed from data rows; tests also show how to aggregate per‑item proxies or compute domain‑level scores for ranking/grouping.
- Superposition and multiplicative combinations
    - add_rows and mul_rows perform in‑place algebra on rows and immediately refresh the row’s λ with the current Laplacian; tests validate positivity, boundedness, and common smoothness behaviors.
- Ensembles and overlays
    - Use GraphFactory::ensemble_from_variants to explore k and ε, or overlay hyperedge boosts to inject prior relationships before recomputing lambdas.


## Notes and caveats

- Temporary dense item similarity is O(N²)
    - For large N columns/items, replace build_item_similarity_laplacian_dense with a sparse or approximate strategy suitable for scale; CSR interfaces remain compatible.
- Determinism and NaN
    - OrderedFloat is used in the example ZSET index; do not store NaN scores to avoid ordering panics; `ArrowSpace` lambdas are recomputed via Rayleigh and guarded in tests for basic invariants.
- API stability
    - The crate uses edition 2024 and wildcard dependencies in places; for production pin versions and consider exposing only stable builder paths (λ‑graph/hypergraph).