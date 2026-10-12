# Research notes

## Rayleigh filter-bank coordinates

ArrowSpace computes spectral information from the graph Laplacian. It does not
compute eigenvectors. This keeps the method discrete, sparse, and deterministic.

The λτ scalar compresses one item vector into one bounded spectral value. The
filter bank added in `0.29.0` behind the default-off `filter_banks` feature
keeps more structure. It returns a short vector of Rayleigh band energies for
each item.

For one item vector $$x$$ and Laplacian $$L = D - A$$, the total Rayleigh energy
is:

$$
R(x) = \frac{x^T L x}{x^T x}
$$

For a sparse graph Laplacian this is also the sum of edge energies:

$$
R(x) = \frac{\sum_{i < j} w_{ij}(x_i - x_j)^2}{\sum_i x_i^2}
$$

The filter bank partitions these sparse edge-energy contributions into
normalised bands. The bands are non-overlapping and cover the interval
$$[0, 1]$$. Therefore the band energies add back to the Rayleigh energy:

$$
\sum_m e_m(x) = R(x)
$$

This is an invariant-compatible substitute for dense spectral projectors. It
keeps the useful property of a filter bank, but it avoids eigensolvers and dense
spectral embeddings.

Use it when a downstream head needs features such as:

$$
[\cos(q, x), e_0(x), e_1(x), e_2(x)]
$$

The query path can also return query-side band energies with
`try_prepare_query_item_with_bands`.
