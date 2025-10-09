Full pre-pring paper at [TechArxiv page](https://www.techrxiv.org/users/685780/articles/1329993-arrowspace-spectral-indexing-of-embeddings-using-taumode-%CE%BB%CF%84). 


`ArrowSpace` provides an API to use taumode (`λτ`) that is a single, bounded synthetic score per signal that blends the Rayleigh smoothness energy on a graph with an edgewise dispersion summary; enabling spectra-aware search, range filters, and composable algebra over vector datasets. Operationally, `ArrowSpace` stores dense features as rows over item nodes, computes a Laplacian on items, derives per-row Rayleigh energies, compresses them via a bounded map $E/(E+\tau)$, mixes in a dispersion term, and uses the resulting `λτ` both for similarity and to build a λ-proximity item graph used across the API.[^1][^2][^3][^4]

## Motivation
From an engineering perspective, there is increasing demand for vector database indices that can spot vector similarities beyond the current available methods (L2 distance, cosine distance, etc.). New methods to search vector spaces can lead to more accurate and fine-tunable procedures to adapt the search to the specific needs of the domain the dataset belongs to. 

## Foundation
The starting score is Rayleigh:
- Rayleigh energy $x^\top L x / x^\top x$ measures how “wiggly” a feature signal is over an item graph; constants yield near-zero on connected graphs, while alternating patterns are larger, making it a principled spectral smoothness score for search and structure discovery.[^2][^3][^4][^1]
- Pure Rayleigh can collapse near zero or be hard to compare across datasets; mapping energy to a bounded score and blending with a dispersion statistic produces a stable, comparable score that preserves spectral meaning while improving robustness for ranking and filtering.[^3][^4][^1][^2]

### Graph and data model
Rayleigh energy is flanked by graph Laplacian of the graph built on the items:
- Items and features: `ArrowSpace` stores a matrix with rows = feature signals and columns = items; the item graph nodes are the columns, and Rayleigh is evaluated per feature row against that item-Laplacian, aligning spectral scores with dataset geometry.[^4][^1][^2][^3]
- Item Laplacian: a Laplacian matrix is constructed over the graph of the items using a `λ`‑proximity policy (`ε` threshold on per‑item `λ`, union-symmetrized, k‑capped, kernel-weighted); diagonals store degrees and off‑diagonals are `−`weights, satisfying standard Laplacian invariants used by the Rayleigh quotient.[^1][^2][^3][^4]

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
1. Spectral Smoothness: Captures how features vary across item relationships
2. Graph Structure: Encodes similarity topology beyond simple pairwise distances
3. Efficient Computation: Sparse matrix enables fast spectral calculations
4. Theoretical Foundation: Connects to harmonic analysis and diffusion processes

### Rayleigh foundation

- Definition: for a feature row $x$ and item-Laplacian $L$, the smoothness is $E = \frac{x^\top L x}{x^\top x}$, which is non‑negative, scale‑invariant in $x$, near‑zero for constants on connected graphs, and larger for high‑frequency signals; this is the discrete Dirichlet energy normalized by signal power.[^2][^3][^4][^1]
- Interpretation: the numerator equals the sum of weighted edge differences $\sum_{(i,j)} w_{ij}(x_i-x_j)^2$, directly capturing roughness over the graph, a classical link between Laplacians and Dirichlet energy used throughout spectral methods.[^3][^4][^1][^2]


### taumode and bounded energy
The idea is to build a score that synthesises the energy features and geometric features of the dataset.

Rayleigh and Laplacian as bounded energy transformation score:
- Bounded map: raw energy $E$ is compressed to $E'=\frac{E}{E+\tau}\in$ using a strictly positive scale $\tau$, stabilizing tails and making scores comparable across rows and datasets while preserving order within moderate ranges.[^5][^4][^1][^3]
- τ selection: taumode supports `Fixed`, `Mean`, `Median`, and `Percentile`; non‑finite inputs are filtered and a small floor ensures positivity; the default `Median` policy provides robust scaling across heterogeneously distributed energies.[^5][^4][^1][^3]

### Purpose of τ in the Bounded Transform

The τ parameter is crucial for the bounded energy transformation: **E' = E/(E+τ)**. This maps raw Rayleigh energies from [0,∞) to [0,1), making scores:

- **Comparable across datasets** with different energy scales
- **Numerically stable** by preventing division issues with very small energies
- **Bounded** for consistent similarity computations


### TauMode Options and Their Use Cases

#### 1. `TauMode::Fixed(value)`

```rust
TauMode::Fixed(0.1)  // Use exactly τ = 0.1
```

**When to use:**

- You have **domain knowledge** about the appropriate energy scale
- **Consistency** across multiple datasets is critical
- **Reproducibility** is paramount (no dependence on data distribution)

**Example:** If you know protein dynamics typically have Rayleigh energies around 0.05-0.2, you might fix τ = 0.1.

#### 2. `TauMode::Median` (Default)

```rust
TauMode::Median  // Use median of all computed energies
```

**When to use:**

- **Robust scaling** - less sensitive to outliers than mean
- **Heterogeneous energy distributions** with potential skewness
- **General-purpose** applications where you want automatic adaptation

**Why it's default:** The median provides a stable central tendency that works well across diverse datasets without being thrown off by extreme values.

#### 3. `TauMode::Mean`

```rust
TauMode::Mean  // Use arithmetic mean of energies
```

**When to use:**

- **Normally distributed** energy values
- You want the transform to **preserve relative distances** around the center
- **Mathematical simplicity** is preferred

**Caution:** Sensitive to outliers - a few very high-energy features can skew the entire transformation.

#### 4. `TauMode::Percentile(p)`

```rust
TauMode::Percentile(0.25)  // Use 25th percentile
TauMode::Percentile(0.75)  // Use 75th percentile
```

**When to use:**

- **Fine-tuned control** over the energy threshold
- **Emphasizing different regimes:**
    - Low percentiles (0.1-0.3): Emphasize discrimination among low-energy (smooth) features
    - High percentiles (0.7-0.9): Emphasize discrimination among high-energy (rough) features


## Practical Impact on Search

The choice of TauMode affects how the bounded energies E' distribute in [0,1):

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


## Implementation Robustness

The code includes several safeguards:

```rust
pub const TAU_FLOOR: f64 = 1e-11;

// Filters out NaN/Inf values and enforces minimum
let filtered_energies: Vec<f64> = energies
    .iter()
    .copied()
    .filter(|x| x.is_finite())
    .collect();
    
if result <= 0.0 { TAU_FLOOR } else { result }
```


#### Recommendation Strategy

1. **Start with `TauMode::Median`** (default) - works well generally
2. **Use `TauMode::Fixed`** when you need reproducibility across runs/datasets
3. **Try `TauMode::Percentile(0.25)`** if you want to emphasize smooth features
4. **Try `TauMode::Percentile(0.75)`** if rough/high-frequency features are most important
5. **Avoid `TauMode::Mean`** unless you're confident about normal distribution

The choice fundamentally determines **how much the spectral component (λ) influences similarity** relative to semantic cosine similarity, making it a key hyperparameter for tuning search behavior in your specific domain.


## ArrowSpace API

### From items to a λ‑graph

- Per‑item λ: to build a λ‑proximity graph over items, ArrowSpace aggregates feature‑level spectral information into per‑item synthetic values by weighting each row’s λτ by the item’s feature magnitude, yielding one scalar per item for proximity thresholding.[^1][^2][^3][^4]
- ε‑graph and k‑cap: items whose λ differ by at most ε are connected; per‑row neighbor selection is k‑capped and union‑symmetrized to ensure an undirected CSR Laplacian with negative off‑diagonals, non‑negative diagonal degrees, and deterministic ordering.[^2][^3][^4][^1]


### ArrowSpace structure and operations

- Column‑major layout: ArrowSpace stores features as rows and items as columns in a dense, cache‑friendly array, with per‑row λτ alongside; zero‑copy views support fast iteration and in‑place row arithmetic (add/mul/scale) while keeping spectral semantics consistent after recomputation.[^3][^4][^1][^2]
- Spectral recompute: after in‑place item operations, Rayleigh energies are recomputed against the current Laplacian and mapped back through taumode to refresh λτ, preserving invariants like non‑negativity and scale invariance in the input vectors.[^4][^1][^2][^3]


### Similarity and search

- Lambda‑aware similarity: for query and item ArrowItems, the score combines semantic cosine and λ proximity via $s=\alpha\,\cos(q,i)+\beta\,(1/(1+|\lambda_q-\lambda_i|))$, making search sensitive to both content and spectral smoothness class; setting $\alpha=1,\beta=0$ recovers plain cosine [^1][^2][^4][^3].
- Range and top‑k: ArrowSpace exposes lambda‑aware top‑k, radius queries, and pairwise cosine matrices; examples validate that λ‑aware rankings agree with cosine when $\beta=0$ and diverge meaningfully when blending in λ proximity, with tests covering Jaccard overlap and commutativity of algebraic operations.[^1][^2][^3][^4]


## Theoretical properties and tests

- Invariants: tests enforce non‑negativity of Rayleigh, near‑zero for constant vectors on connected graphs, scale‑invariance $λ(cx)=λ(x)$, and conservative upper bounds via diagonal degrees, aligning with standard spectral graph theory expectations.[^4][^1][^2][^3]
- Laplacian structure: CSR symmetry, negative off‑diagonals, non‑negative diagonals, degree–diagonal equality, and deterministic ordering are validated to ensure stable Rayleigh evaluation and reproducible λτ synthesis across builds.[^1][^2][^3][^4]


### Practical guidance

- Defaults: a practical starting point is ε≈1e‑3, k in , p=2.0, σ=ε, and taumode::Median with $\alpha≈0.7$; this keeps the λ‑graph connected but sparse and yields bounded λτ values that mix energy and dispersion robustly for search.[^6][^7][^2][^3][^4][^1]
- Usage patterns: build ArrowSpace from item rows (auto‑transposed internally), let the builder construct the λ‑graph and compute synthetic λτ, then use lambda‑aware similarity for ranking or ε‑band zsets for range‑by‑score retrieval; in‑place algebra over items supports superposition experiments while preserving spectral semantics through recompute.[^2][^3][^4][^1]


### Relation to classical theory

- Link to Dirichlet energy: the Rayleigh quotient over graph Laplacians is the discrete analogue of Dirichlet energy normalized by signal power, with eigenvalues/eigenvectors characterizing smoothness classes; λτ leverages this foundation but adds a bounded transform and dispersion term for practical search and indexing.[^8][^4][^1][^2]


## Summary of contributions

- A single‑scalar `λτ` per feature-row that is bounded, comparable, and spectrally meaningful, derived from Rayleigh energy and an edgewise dispersion statistic with robust taumode scaling.[^3][^4][^1][^2]
- A dense `ArrowSpace` that stores features over items, recomputes `λτ` after algebraic item operations, and exposes lambda‑aware search primitives and λ‑proximity graphs with strong Laplacian guarantees.[^4][^1][^2][^3]
- A builder that unifies λ‑graph construction and synthetic index computation, yielding reproducible, spectrally aware vector search behavior validated by extensive tests and examples.[^1][^2][^3][^4]

[^1]: arrowspace-rs codebase

[^2]: https://people.cs.umass.edu/~mahadeva/cs791bb/lectures-s2006/lec4.pdf

[^3]: https://people.maths.bris.ac.uk/~maajg/teaching/complexnets/laplacians.pdf

[^4]: https://ocw.mit.edu/courses/18-409-topics-in-theoretical-computer-science-an-algorithmists-toolkit-fall-2009/535add3f6457cc13e51d9774f16bf48f_MIT18_409F09_scribe3.pdf

[^5]: https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec4RayleighQuotient.pdf

[^6]: https://mathweb.ucsd.edu/~fan/teach/262/notes/paul/10_3_notes.pdf

[^7]: https://cvgmt.sns.it/media/doc/paper/1580/giamodmuc07.pdf

[^8]: https://en.wikipedia.org/wiki/Dirichlet_energy

[^9]: https://core.ac.uk/download/pdf/39216607.pdf

[^10]: https://www.cs.yale.edu/homes/spielman/462/2007/lect7-07.pdf

[^11]: https://en.wikipedia.org/wiki/Hilbert_transform