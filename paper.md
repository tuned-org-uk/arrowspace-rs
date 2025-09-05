---
title: 'ArrowSpace: A primer on Spectral Indexing for vector search'
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
 - name: Independent Researcher (London, UK / Tokyo, Japan) - tuned.org.uk
   index: 1
date: 28 August 2025
bibliography: paper.bib
---

# Summary

`ArrowSpace` [@ArrowSpace:2025] is a library that implements a novel spectral indexing approach for vector similarity search, combining traditional semantic similarity with graph-based spectral properties [@Mahadevan:2006;@Spielman:2007]. The library introduces taumode (`λτ`, lambda-tau) indexing, which blends Rayleigh quotient smoothness energy from graph Laplacians [@Bai:2007;@Bai:2010] with edge-wise dispersion statistics to create bounded, comparable spectral scores. This enables similarity search that considers both semantic content and spectral characteristics of high-dimensional vector datasets.

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
The starting score is Rayleigh as described in [@Chen:2020]. Chen emphasises that the Rayleigh quotient provides a variational characterisation of eigenvalues, it offers a way to find eigenvalues through optimisation rather than solving the characteristic polynomial. This perspective is fundamental in numerical linear algebra and spectral analysis.
The treatment is particularly valuable for understanding how spectral properties of matrices emerge naturally from optimisation problems, which connects to applications in data analysis, graph theory, and machine learning.

Basic points:
- Definition: for a feature row $x$ and item-Laplacian $L$, the smoothness is $E = \frac{x^\top L x}{x^\top x}$, which is non‑negative, scale‑invariant in $x$, near‑zero for constants on connected graphs, and larger for high‑frequency signals; the Rayleigh quotient is the normalised Dirichlet Energy, it is the discrete Dirichlet energy normalised by signal power.
- Physical Interpretation: Dirichlet energy measure the "potential energy" or "stiffness" of a configuration while the Rayleigh quotient normalises this by the total "mass" or "signal power". the result is a scale-invariant measure of how much energy is required per unit mass (in our case the items-nodes).

## taumode and bounded energy
The main idea for this design is to *build a score that synthesises the energy features and geometric features of the dataset* and apply it to vector searching.

Rayleigh and Laplacian as bounded energy transformation score become a bounded map: raw energy $E$ is compressed to $E'=\frac{E}{E+\tau}\in$ using a strictly positive scale $\tau$, stabilising tails and making scores comparable across rows and datasets while preserving order within moderate ranges.

Additional τ selection: taumode supports `Fixed`, `Mean`, `Median`, and `Percentile`; non‑finite inputs are filtered and a small floor ensures positivity; the default `Median` policy provides robust scaling across heterogeneously distributed energies.

Rayleigh, Laplacian and τ selection enable the taumode score, so to use this score as an indexing score for dataset indexing.

### Purpose of τ in the Bounded Transform

The τ parameter is crucial for the bounded energy transformation: **E' = E/(E+τ)**. This maps raw Rayleigh energies from [0,∞) to [0,1), making scores:

- **Comparable across datasets** with different energy scales
- **Numerically stable** by preventing division issues with very small energies
- **Bounded** for consistent similarity computations

## Usage Example
	
```rust
use ArrowSpace::builder::ArrowSpaceBuilder;
use ArrowSpace::core::ArrowItem;

// Build ArrowSpace from item vectors
let items = vec![
vec![1.0, 2.0, 3.0],  // Item 1
vec![2.0, 3.0, 1.0],  // Item 2
vec![3.0, 1.0, 2.0],  // Item 3
];

let (aspace, _graph) = ArrowSpaceBuilder::new()
.with_rows(items)
.with_lambda_graph(1e-3, 6, 2.0, None)
.build();

// Query with lambda-aware similarity
let query = ArrowItem::new(vec![1.5, 2.5, 2.0], 0.0);
// with alpha=1.0 and beta=0.0, same results as cosine similarity
let results = aspace.search_lambda_aware(&query, 5, 1.0, 0.0);
``````
	

### Practical Impact on Search

The choice of taumode affects how the bounded energies $E'$ distribute in $[0,1)$:

```rust
// Low-energy feature with different τ values
let energy = 0.01;
let tau_small = 0.001;  // E' = 0.01/0.011 ≈ 0.91 (high sensitivity)
let tau_large = 0.1;    // E' = 0.01/0.11 ≈ 0.09 (low sensitivity)
```

# Summary and Conclusion

## Performance Characteristics

### Computational Complexity

- **Index Construction**: $O(N²)$ for similarity graph (already identified a solution to make this into $O(N log N)$); $O(F·nnz(L))$ for $λτ$ computation.
- **Query Time**: O(N) for linear scan, O(1) for $λτ$ lookup
- **Memory Usage**: O(F·N) for dense storage, O(N) for $λτ$ indices

### Benchmarks

The library includes comprehensive benchmarks comparing `ArrowSpace` with baseline cosine similarity:

- **Single Query**: ~15% overhead for $λτ$-aware search vs pure cosine
- **Batch Queries**: Scales linearly with batch size, maintains constant per-query overhead
- **Memory Footprint**: 4-8 bytes per $λτ$ index vs graph storage

## Results

`ArrowSpace` has substantial potential for raw improvements plus all the advantages provided to downstream more complex operations like matching, comparison and search due to the $\lambda$ spectrum. Capabilities are demonstrated in the other tests present in the code. Please check the `proteins_lookup` example that demonstrates the functionality in a small dataset. The time complexity for a range-based lookup is the same as a sorted set $O(log(N)+M)$. As demonstrated in the `proteins_lookup` example, starting from a collection of $\lambda$s with a standard deviation of $0.06$, it is possible to sort out the single nearest neighbour with a range query on an query interval of $\lambda \pm 10^{-7}$.

## Conclusion

`ArrowSpace` provides a novel approach to vector similarity search by integrating spectral graph properties with traditional semantic similarity measures. The $λτ$ indexing system offers a memory-efficient way to capture spectral characteristics of vector datasets while maintaining practical query performance. The library's design emphasises both mathematical rigor and computational efficiency, making it suitable for scientific applications requiring spectral-aware similarity search.

The combination of Rust's performance characteristics with innovative spectral indexing algorithms positions `ArrowSpace` as a valuable tool for researchers and practitioners working with high-dimensional vector data where both semantic content and structural properties matter.

The definition of a core library to be used to develop a database solution based on spectral indexing is left to another paper that will include further improvements in terms of algorithms and idioms to make this approach to indexing feasible and efficient in modern cloud installations.

# Disclaimer

For this research LLMs have been used extensively in the ideation and development phase.

# References