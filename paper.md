---
title: 'ArrowSpace: Spectral Indexing of Embeddings using taumode (λτ)'
tags:
  - embeddings
  - vector database
  - RAG
  - numerical
  - scientific computing
  - Rust
authors:
  - name: Lorenzo Moriondo
    orcid: 0000-0002-8804-2963
    affiliation: "1"
affiliations:
 - name: Independent Researcher - tuned.org.uk
   index: 1
date: 28 August 2025
bibliography: paper.bib
---

# Summary

`ArrowSpace` [@ArrowSpace:2025] is a Rust library that implements a novel spectral indexing approach for vector similarity search, combining traditional semantic similarity with graph-based spectral properties [@Mahadevan:2006;@Spielman:2007]. The library introduces taumode (`λτ`, lambda-tau) indexing, which blends Rayleigh quotient smoothness energy from graph Laplacians [@Bai:2007;@Bai:2010] with edge-wise dispersion statistics to create bounded, comparable spectral scores. This enables similarity search that considers both semantic content and spectral characteristics of high-dimensional vector datasets.

# Statement of Need

Traditional vector similarity search relies primarily on geometric measures, like cosine similarity or Euclidean distance which capture semantic relationships but ignore the spectral structure inherent in many datasets. For example in domains such as protein analysis, signal processing and molecular dynamics, the "roughness" or "smoothness" of feature signals across data relationships can provide valuable discriminative information that complements semantic similarity.

Existing vector databases and similarity search systems lack integrated spectral-aware indexing capabilities. While spectral methods exist in graph theory and signal processing (for spectral clustering see [@VonLuxburg:2007]), they are typically computationally expensive and they are not considered for database applications. With the increasing demand for vector searching though (in particular, at current state, for the components called "retrievers" in RAG applications[@Lewis:2020]), the research on spectral indexing gains traction for database applications.
`ArrowSpace` addresses this gap by providing:

1. **Spectral-aware similarity search** that combines semantic and spectral properties
2. **Bounded synthetic indexing** that produces comparable scores across datasets
3. **Memory-efficient representation** that avoids storing graph structures at query time
4. **High-performance Rust implementation** with potentially zero-copy operations and cache-friendly data layouts

# Data Model and Algorithm

`ArrowSpace` provides an API to use taumode (`λτ`) that is a single, bounded, synthetic score per signal that blends the Rayleigh smoothness energy on a graph with an edgewise dispersion summary; enabling spectra-aware search and range filtering. Operationally, `ArrowSpace` stores dense features (inspired by CSR [@Kelly:2020] and `smartcore` [@smartcore:2021]) as rows over item nodes, computes a Laplacian on items, derives per-row Rayleigh energies, compresses them via a bounded map $E/(E+\tau)$, mixes in a dispersion term and uses the resulting `λτ` both for similarity and to build a λ-proximity item graph used across the API. This way the `λτ` (taumode) score can rely on a synthesis of the characteristics proper of diffusion models and geometric/topological representation of graphs. 

## Motivation
From an engineering perspective, there is increasing demand for vector database indices that can spot vector similarities beyond the current available methods (L2 distance, cosine distance, or more complex algorithms like HNSW that requires multiple graphs, or typical caching mechanism requiring hashing). New methods to search vector spaces can lead to more accurate and fine-tunable procedures to adapt the search to the specific needs of the domain the embeddings belong to.

## Foundation
The starting score is Rayleigh as described in [@Chen:2020]. Chen emphasises that the Rayleigh quotient provides a variational characterization of eigenvalues, it offers a way to find eigenvalues through optimization rather than solving the characteristic polynomial. This perspective is fundamental in numerical linear algebra and spectral analysis.
The treatment is particularly valuable for understanding how spectral properties of matrices emerge naturally from optimization problems, which connects to applications in data analysis, graph theory, and machine learning.

Basic points:
- Definition: for a feature row $x$ and item-Laplacian $L$, the smoothness is $E = \frac{x^\top L x}{x^\top x}$, which is non‑negative, scale‑invariant in $x$, near‑zero for constants on connected graphs, and larger for high‑frequency signals; the Rayleigh quotient is the normalized Dirichlet Energy, it is the discrete Dirichlet energy normalized by signal power.
- Physical Interpretation: Dirichlet energy measure the "potential energy" or "stiffness" of a configuration while the Rayleigh quotient normalises this by the total "mass" or "signal power". the result is a scale-invariant measure of how much energy is required per unit mass (in our case the items-nodes).
- The numerator equals the sum of weighted edge differences $\sum_{(i,j)} w_{ij}(x_i-x_j)^2$, directly capturing roughness over the graph, a classical link between Laplacians and Dirichlet energy used throughout spectral methods.

Some implementation starting points:
- Rayleigh energy $x^\top L x / x^\top x$ measures how "wiggly" a feature signal is over an item graph; constants yield near-zero on connected graphs, while alternating patterns are larger, making it a principled spectral smoothness score for search and structure discovery.
- Pure Rayleigh can collapse near zero or be hard to compare across datasets; mapping energy to a bounded score and blending with a dispersion statistic produces a stable, comparable score that preserves spectral meaning while improving robustness for ranking and filtering.


### Graph and data model
Rayleigh energy score is complemented for spectral indexing by computing the graph Laplacian [@Spielman:2007] of the dataset:
- Items and features: `ArrowSpace` stores a matrix with rows = feature signals and columns = items; the item graph nodes are the columns, and Rayleigh is evaluated per feature row against that item-Laplacian, aligning spectral scores with dataset geometry.
- Item Laplacian: a Laplacian matrix is constructed over the graph of the items using a `λ`‑proximity policy (`ε` threshold on per‑item `λ`, union-symmetrized, k‑capped, kernel-weighted); diagonals store degrees and off‑diagonals are $−weights$, satisfying standard Laplacian invariants used by the Rayleigh quotient.

Example:
```rust
// 1. Build item graph based on lambda proximity
let items = vec![
    vec![1.0, 2.0],  // Item 0: λ_0 = 0.3
    vec![1.1, 2.1],  // Item 1: λ_1 = 0.35  
    vec![3.0, 1.0],  // Item 2: λ_2 = 0.8
];
let aspace = ArrowSpace::from_items(...)

// 2. Connect items with |λ_i - λ_j| ≤ ε
// Items 0,1 connected (|0.3-0.35| = 0.05 ≤ ε)
// Items 0,2 not connected (|0.3-0.8| = 0.5 > ε)
// Items 1,2 not connected (|0.35-0.8| = 0.45 > ε)

// 3. Resulting Laplacian (simplified):
//     [  w   -w    0 ]
// L = [ -w    w    0 ]  where w = kernel weight
//     [  0    0    0 ]
```

### Role of Laplacian
What the graph Laplacian contributes to Rayleigh energy:
1. Spectral Smoothness: Captures how features vary across item relationships
2. Graph Structure: Encodes similarity topology beyond simple pairwise distances
3. Efficient Computation: Sparse matrix enables fast spectral calculations
4. Theoretical Foundation: Connects to harmonic analysis and diffusion processes


## taumode and bounded energy
The main idea for this design is to *build a score that synthesises the energy features and geometric features of the dataset* and apply it to vector searching.

Rayleigh and Laplacian as bounded energy transformation score become a bounded map: raw energy $E$ is compressed to $E'=\frac{E}{E+\tau}\in$ using a strictly positive scale $\tau$, stabilizing tails and making scores comparable across rows and datasets while preserving order within moderate ranges.

Additional τ selection: taumode supports `Fixed`, `Mean`, `Median`, and `Percentile`; non‑finite inputs are filtered and a small floor ensures positivity; the default `Median` policy provides robust scaling across heterogeneously distributed energies.

Rayleigh, Laplacian and τ selection enable the taumode score, so to use this score as an indexing score for dataset indexing.

### Purpose of τ in the Bounded Transform

The τ parameter is crucial for the bounded energy transformation: **E' = E/(E+τ)**. This maps raw Rayleigh energies from [0,∞) to [0,1), making scores:

- **Comparable across datasets** with different energy scales
- **Numerically stable** by preventing division issues with very small energies
- **Bounded** for consistent similarity computations


### taumode Options and Their Use Cases

#### 1. `taumode::Fixed(value)`

```rust
taumode::Fixed(0.1)  // Use exactly τ = 0.1
```

**When to use:**

- You have **domain knowledge** about the appropriate energy scale
- **Consistency** across multiple datasets is critical
- **Reproducibility** is paramount (no dependence on data distribution)

**Example:** If you know protein dynamics typically have Rayleigh energies around 0.05-0.2, you might fix τ = 0.1.

#### 2. `taumode::Median` (Default)

```rust
taumode::Median  // Use median of all computed energies
```

**When to use:**

- **Robust scaling** - less sensitive to outliers than mean
- **Heterogeneous energy distributions** with potential skewness
- **General-purpose** applications where you want automatic adaptation

**Why it's default:** The median provides a stable central tendency that works well across diverse datasets without being thrown off by extreme values.

#### 3. `taumode::Mean`

```rust
taumode::Mean  // Use arithmetic mean of energies
```

**When to use:**

- **Normally distributed** energy values
- You want the transform to **preserve relative distances** around the center
- **Mathematical simplicity** is preferred

**Caution:** Sensitive to outliers - a few very high-energy features can skew the entire transformation.

#### 4. `taumode::Percentile(p)`

```rust
taumode::Percentile(0.25)  // Use 25th percentile
taumode::Percentile(0.75)  // Use 75th percentile
```

**When to use:**

- **Fine-tuned control** over the energy threshold
- **Emphasizing different regimes:**
    - Low percentiles (0.1-0.3): Emphasize discrimination among low-energy (smooth) features
    - High percentiles (0.7-0.9): Emphasize discrimination among high-energy (rough) features


## Practical Impact on Search

The choice of taumode affects how the bounded energies E' distribute in [0,1):

```rust
// Low-energy feature with different τ values
let energy = 0.01;
let tau_small = 0.001;  // E' = 0.01/0.011 ≈ 0.91 (high sensitivity)
let tau_large = 0.1;    // E' = 0.01/0.11 ≈ 0.09 (low sensitivity)
```


#### Effect on Lambda-Aware Similarity

In the lambda-aware similarity score: **s = α·cosine + β·(1/(1+|λ_q-λ_i|))**

- **Smaller τ** → More compressed E' values → **Less discrimination** between different energy levels
- **Larger τ** → More spread E' values → **Greater emphasis** on spectral differences


### Implementation Robustness

The code includes several safeguards:

```rust
pub const TAU_FLOOR: f64 = 1e-9;

// Filters out NaN/Inf values and enforces minimum
let filtered_energies: Vec<f64> = energies
    .iter()
    .copied()
    .filter(|x| x.is_finite())
    .collect();
    
if result <= 0.0 { TAU_FLOOR } else { result }
```


#### Recommendation Strategy

1. **Start with `taumode::Median`** (default) - works well generally
2. **Use `taumode::Fixed`** when you need reproducibility across runs/datasets
3. **Try `taumode::Percentile(0.25)`** if you want to emphasize smooth features
4. **Try `taumode::Percentile(0.75)`** if rough/high-frequency features are most important
5. **Avoid `taumode::Mean`** unless you're confident about normal distribution

The choice fundamentally determines **how much the spectral component (λ) influences similarity** relative to semantic cosine similarity, making it a key hyperparameter for tuning search behavior in your specific domain.


# Summary and Conclusion

## Lambda-Tau (λτ) Indexing

The core innovation of `ArrowSpace` is the $λτ$ synthetic index, which combines:

- **Rayleigh Energy**: For each feature signal x over an item graph with Laplacian L, computes the smoothness energy $E = (x^T L x)/(x^T x)$
- **Bounded Transform**: Maps raw energy $E$ to $E' = E/(E+τ)$ using a robust $τ$ selection policy (Median, Mean, Percentile, or Fixed)
- **Dispersion Term**: Captures edge-wise concentration of spectral energy using Gini-like statistics
- **Synthetic Score**: Blends $E'$ and dispersion via $λ = α·E' + (1-α)·G$, producing bounded  scores


## Graph Construction

`ArrowSpace` builds similarity graphs from vector data using lambda-proximity connections:

- **Item Graphs**: Connects items whose aggregated λ values differ by at most ε
- **K-Capping**: Limits neighbors per node while maintaining graph connectivity
- **Union Symmetrization**: Ensures undirected Laplacian properties
- **Kernel Weighting**: Uses monotone kernels $w = 1/(1 + (|Δλ|/σ)^p)$ for edge weights


## Memory-Efficient Design

The library consider by-design several optimizations for performance:

- **Column-Major Storage**: Dense arrays with features as rows, items as columns
- **Potentially Zero-Copy Operations**: Slice-based access without unnecessary allocations as already present in [@smartcore:2021]
- **Single-Pass Computation**: $λτ$ indices computed once, graph can be discarded
- **Cache-Friendly Layout**: Contiguous memory access patterns for potential SIMD optimization


# Implementation

`ArrowSpace` is implemented in Rust (edition 2024) with the following architecture:

## Core Components

- **``ArrowSpace``**: Dense matrix container with per-item $λτ$ scores
- **`ArrowItem`**: Individual vector with spectral metadata and similarity operations
- **`GraphFactory`**: Constructs various graph types from vector data
- **`ArrowSpaceBuilder`**: Fluent API for configuration and construction


## Key Algorithms

For insertion and `λτ` computing at insertion time:
```rust
/// Aggregate from feature signals to per-item scores.
/// For each item i, combine the feature-level spectral information weighted by the item’s feature magnitudes
/// (or another explicit weighting) to get a single synthetic index S_i.
///
/// Use the same bounded energy map and dispersion blend idea, but perform it per item:
/// - For each feature row f, compute Rayleigh energy E_f and dispersion G_f once (as today).
/// - For each item i, compute a weight w_fi = |x_f[i]| (magnitude of feature f at item i) and use it to
/// Aggregate the feature contributions to item level:
/// - E_i_raw = (sum_f w_fi * E_f) / (sum_f w_fi) with 0-guard.
/// - G_i_raw = (sum_f w_fi * G_f) / (sum_f w_fi) with 0-guard.
/// - Select τ from the per-item energy population {E_i_raw} using the same taumode.
/// - Map E_i = E_i_raw / (E_i_raw + τ), clamp G_i to , then S_i = α * E_i + (1−α) * G_i.
/// - Finally, update `ArrowSpace`.lambdas with the n_items synthetic vector S (length equals gl.nnodes).
pub fn compute_synthetic_lambdas(
    aspace: &mut `ArrowSpace`,
    gl: &GraphLaplacian,
    alpha: f64,
    tau_mode: taumode,
) { ... }
```

For search and lookup (alpha=1.0 and beta=0.0 is equivalent to cosine similarity):
```rust
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
```


## Usage Example

```rust
use `ArrowSpace`::builder::`ArrowSpace`Builder;
use `ArrowSpace`::core::ArrowItem;

// Build `ArrowSpace` from item vectors
let items = vec![
    vec![1.0, 2.0, 3.0],  // Item 1
    vec![2.0, 3.0, 1.0],  // Item 2
    vec![3.0, 1.0, 2.0],  // Item 3
];

let (aspace, _graph) = `ArrowSpace`Builder::new()
    .with_rows(items)
    .with_lambda_graph(1e-3, 6, 2.0, None)
    .build();

// Query with lambda-aware similarity
let query = ArrowItem::new(vec![1.5, 2.5, 2.0], 0.0);
let results = aspace.search_lambda_aware(&query, 5, 0.8, 0.2);
```

# Performance Characteristics

## Computational Complexity

- **Index Construction**: $O(N²)$ for similarity graph (already identified a solution to make this into $O(N log N)$); $O(F·nnz(L))$ for $λτ$ computation.
- **Query Time**: O(N) for linear scan, O(1) for $λτ$ lookup
- **Memory Usage**: O(F·N) for dense storage, O(N) for $λτ$ indices


## Benchmarks

The library includes comprehensive benchmarks comparing `ArrowSpace` with baseline cosine similarity:

- **Single Query**: ~15% overhead for $λτ$-aware search vs pure cosine
- **Batch Queries**: Scales linearly with batch size, maintains constant per-query overhead
- **Memory Footprint**: 4-8 bytes per $λτ$ index vs graph storage


# Scientific Applications

`ArrowSpace` has been designed with several scientific domains in mind:

## Protein Structure Analysis

The examples demonstrate protein-like vector databases with molecular dynamics features (inspired by [@Nelson:2015]).

1. Build items from features for a protein domain:
```rust
// Trajectory features for spectral analysis
fn trajectory_features(domain: &ProteinDomain) -> Vec<f64> {
    let mut features = Vec::new();
    for frame in &domain.trajectory {
        features.push(frame.rmsd);
        features.push(frame.energy / 1000.0);
        features.push(frame.temperature / 300.0);
        // ... additional biophysical features
    }
    features
}

let items: Vec<Vec<f64>> = domains
    .into_iter()
    .map(extract_features)
    .collect();
```
2. Pass the arrays of items and features to the index:
```rust
let (aspace, _gl) = ArrowSpaceBuilder::new()
    .with_rows(items) // N×F -> auto-transposed to F×N
    .build();
```
Lookup the index based on a range of lambdas built on an interval around a reference feature.


## Fractal and Signal Analysis

Integration with fractal dimension estimation for complex signal analysis. Methods are provided in the library for dimensionality operations but they are extra features relative to the core vector search feature:

```rust
// Combine fractal properties with spectral indexing
let fractal_dimension = DimensionalOps::box_count_dimension(&trajectory_2d, &scales);
let spectral_features = extract_spectral_features(&signal);
```

# Results

`ArrowSpace` has substantial potential for raw improvements plus all the advantages provided to downstream more complex operations like matching, comparison and search due to the $\lambda$ spectrum. Capabilities are demonstrated in the other tests present in the code. Please check the `proteins_lookup` example that demonstrates the functionality in a small dataset. The time complexity for a range-based lookup is the same as a sorted set $O(log(N)+M)$. As demonstrated in the `proteins_lookup` example, starting from a collection of $\lambda$s with a standard deviation of $0.06$, it is possible to sort out the single nearest neighbour with a range query on an query interval of $\lambda \pm 10^{-7}$.

Present libraries provide the building blocks (graph Laplacians, Rayleigh quotient evaluators, HKS/heat-kernel descriptors) but not the exact, reusable container/API that mirrors this paper's `ArrowSpace` concept.
The arrow space packages a general-purpose programming/data-structure pattern that keeps vector signals and their operator-derived scale/distribution together and available for algorithmic scheduling and result novel as a unified abstraction.

This basic reference implementation can be improved in multiple dimensions to reach state-of-the-art capabilities for well-targeted applications like pattern matching for embeddings. Generally, any kind of embedding sensitive to spectral characteristics; every application designed for given target embeddings can be fine-tuned appropriately in different dimensions and potentially improve performances in these areas:


## Testing and Validation

The library includes extensive test coverage:

- **Unit Tests**: Core algorithms, edge cases, mathematical properties
- **Integration Tests**: End-to-end workflows, builder patterns
- **Property Tests**: Scale invariance, non-negativity, boundedness
- **Domain Tests**: Molecular dynamics simulations, fractal analysis
- **Performance Tests**: Benchmarks against baseline implementations


## Availability and Installation

`ArrowSpace` is available as an open-source Rust crate:

```toml
[dependencies]
`ArrowSpace` = "0.1.0"
```

The source code, documentation, and examples are available at the project repository, with comprehensive API documentation generated via rustdoc.

## Theoretical properties and tests

- Invariants: tests enforce non‑negativity of Rayleigh, near‑zero for constant vectors on connected graphs, scale‑invariance $λ(cx)=λ(x)$, and conservative upper bounds via diagonal degrees, aligning with standard spectral graph theory expectations. [@Chen:2020]
- Laplacian structure: CSR symmetry, negative off‑diagonals, non‑negative diagonals, degree–diagonal equality, and deterministic ordering are validated to ensure stable Rayleigh evaluation and reproducible λτ synthesis across builds.[@Grindrod:2020]


## Practical guidance

- Defaults: a practical starting point is ε≈1e‑3, k in , p=2.0, σ=ε, and taumode::Median with $\alpha≈0.7$; this keeps the λ‑graph connected but sparse and yields bounded λτ values that mix energy and dispersion robustly for search [@Wikipedia:DirichletEnergy;@Chua:2025].
- Usage patterns: build ArrowSpace from item rows (auto‑transposed internally), let the builder construct the λ‑graph and compute synthetic λτ, then use lambda‑aware similarity for ranking or ε‑band zsets for range‑by‑score retrieval; in‑place algebra over items supports superposition experiments while preserving spectral semantics through recompute.[@Chen:2020;@Mahadevan:2006;Grindrod:2020]


###Relation to classical theory

- Link to Dirichlet energy: the Rayleigh quotient over graph Laplacians is the discrete analogue of Dirichlet energy normalized by signal power, with eigenvalues/eigenvectors characterizing smoothness classes; λτ leverages this foundation but adds a bounded transform and dispersion term for practical search and indexing.[^8][^4][^1][^2]

## Summary of contributions

- A single‑scalar `λτ` per feature-row that is bounded, comparable, and spectrally meaningful, derived from Rayleigh energy and an edgewise dispersion statistic with robust taumode scaling.
- A dense `ArrowSpace` that stores features over items, recomputes `λτ` after algebraic item operations, and exposes lambda‑aware search primitives and λ‑proximity graphs with strong Laplacian guarantees.
- A builder that unifies λ‑graph construction and synthetic index computation, yielding reproducible, spectrally aware vector search behavior validated by extensive tests and examples.


## Conclusion

`ArrowSpace` provides a novel approach to vector similarity search by integrating spectral graph properties with traditional semantic similarity measures. The $λτ$ indexing system offers a memory-efficient way to capture spectral characteristics of vector datasets while maintaining practical query performance. The library's design emphasizes both mathematical rigor and computational efficiency, making it suitable for scientific applications requiring spectral-aware similarity search.

The combination of Rust's performance characteristics with innovative spectral indexing algorithms positions `ArrowSpace` as a valuable tool for researchers and practitioners working with high-dimensional vector data where both semantic content and structural properties matter.

- Lambda‑aware similarity: for query and item ArrowItems, the score combines semantic cosine and λ proximity via $s=\alpha\,\cos(q,i)+\beta\,(1/(1+|\lambda_q-\lambda_i|))$, making search sensitive to both content and spectral smoothness class; setting $\alpha=1,\beta=0$ recovers plain cosine.
- Range and top‑k: `ArrowSpace` exposes lambda‑aware top‑k, radius queries, and pairwise cosine matrices; examples validate that λ‑aware rankings agree with cosine when $\beta=0$ and diverge meaningfully when blending in λ proximity, with tests covering Jaccard overlap and commutativity of algebraic operations.

The definition of a core library to be used to develop a database solution based on spectral indexing is left to another paper that will include further improvements in terms of algorithms and idioms to make this approach to indexing feasible and efficient in modern cloud installations.
