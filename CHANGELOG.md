# Changelog

All notable changes to `arrowspace` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [0.28.0] — breaking

This release fixes the `DenseMatrix` flat-buffer layout defect (#167):
every call site that assembled a matrix from a **row-major** buffer passed
`axis=1` to `DenseMatrix::from_iterator`, which smartcore maps to
`column_major = true` — adopting the buffer as-is. Every `get((r, c))` then
reads a transposed mixture of the intended rows. The matrices were
self-consistent (all consumers read through the same scrambled layout), so
nothing crashed — but the values were not the quantities the code claims to
assemble, and downstream spectral content was calibrated against the
scramble.

**No signatures change and no API is removed.** The break is value-level:
graph content and every stored λ derived from it change for EigenMaps and
EnergyMaps builds. Pin 0.28.0 and expect re-tuned indexes; persisted
indexes built with ≤ 0.27.4 embed the old graph content (rebuild, or treat
loaded λs as legacy — see "Note for the Python bindings").

### Fixed — `DenseMatrix` flat-buffer layout (#167)

`axis=1 → axis=0` at every row-major call site:

- `run_incremental_clustering_with_sampling` (clustering/mod.rs) — the
  `clustered_dm` / `GraphLaplacian::init_data` centroid matrix. `init_data`
  columns are now the true cluster means (probe evidence in #167: pre-fix
  columns had norms 0.53–1.36 and cos ≈ 0 against means of norm 1.000).
- `sparse_to_dense` (graph.rs) — values-identical for the symmetric
  Laplacians it feeds, correctness-true for any input.
- `kmeans_lloyd` input assembly (clustering/mod.rs) — the clustering
  heuristic (`compute_optimal_k`) previously scored k candidates on
  scrambled data; `k_opt`/`radius` selections may differ.
- `project_matrix` (reduction.rs) — JL-projected centroid matrices.
- optical compression + `diffuse_and_split_subcentroids` (maps/energymaps.rs)
  — `sub_centroids` rows are now true subcentroid coordinates.
- `extract_columns` (analysis/subgraphs/sg_from_motives.rs) — subgraph
  Laplacian coordinates.

`try_prepare_query_item`'s energy branch no longer compares a raw F-dim
query against reduced-dim subcentroid rows through a silently truncating
`zip`: unprojected queries are projected exactly like the indexing path,
already-reduced queries are accepted as-is, and anything else returns
`DimensionMismatch` instead of mis-mapping (this latent defect is what
broke energy self-retrieval once subcentroid rows held true coordinates).

Parquet round-trip is unaffected: `save_dense_matrix` streams logical
columns, so the loaded buffer is genuinely column-major and `axis=1` is
correct there.

### Changed — builder defaults calibrated for true signal graphs

The pre-#167 defaults (`lambda_eps=1e-3, lambda_k=6, lambda_topk=3`) were
only survivable because scrambled `clustered_dm` buffers made adjacent
"feature signals" overlapping windows of one centroid (artificially
correlated at ~0 distance). On true signals they collapse the bootstrap
graph to zero edges and every λ to 0. New defaults:
`lambda_eps=0.5, lambda_k=12, lambda_topk=6` — inside the documented
0.5–4.0 eps regime and matching one-off `arrowspace_tuner` optima on
unit-norm corpora (eps ≈ 0.45–0.52, k ≈ 25–34 at 10³-scale).

### Test-contract changes (post-fix behaviour, by design)

- K-Means may converge to unbalanced local minima — explorative behaviour
  is by design; only empty clusters are gated now.
- EigenMaps and EnergyMaps motif tracks may return different results: the
  energy graph is the subcentroid Laplacian with λ-proximity item mapping,
  the eigen track detects on the centroid graph. Cross-track agreement is
  reported, never gated; each track is gated on its own contract
  (item-space validity, planted-structure recovery for the primary eigen
  track).
- `test_builder_unit_norm_diagonal_similarity` no longer asserts equal
  cluster counts for raw vs unit-norm inputs: the (squared-Euclidean)
  incremental clusterer is scale-sensitive, and the old equality was an
  artifact of the scrambled heuristic.
- A λ golden test pins min/max/mean on a fixed fixture so future graph or
  read-out changes are explicit.

### Note for the Python bindings

No signature or variant changes — `map_arrow_error` is untouched. pyarrowspace
pinned at `arrowspace = "0.27"` keeps compiling and keeps its current
(semantically scrambled-era) λ values; bump to 0.28 to pick up corrected
graph content, and expect stored-index rebuilds.

## [0.27.4]

This release closes the second half of the motif-namespace failure family:
`spot_motives_eigen` returned **feature-space** node ids of the F×F bootstrap
Laplacian while `GraphLaplacian::nnodes` advertised the item count (#165).
The eigen track now has the same item-space contract the energy track got in
0.27.3 (#161), and the historical feature-space behaviour keeps a properly
named, fallible entry point. **No signature changes or removals**; one
additive `ArrowSpaceError` variant — the Python bindings must add one
`map_arrow_error` arm when they bump to 0.27.4 (they currently pin 0.27.3
and are unaffected until then).

### Added — item-space motifs for the EigenMaps track (#165)

`Motives::try_spot_motives_eigen(&gl, &aspace, &cfg) ->
Result<Vec<Vec<usize>>, ArrowSpaceError>` mirrors
`try_spot_motives_energy` for EigenMaps builds:

1. Rebuilds the X×X Laplacian over cluster centroids from the index's own
   rows (`aspace.data`) and the pipeline's `cluster_assignments` — the same
   clustered structure the bootstrap graph was assembled from, no raw-data
   bypass of L (invariant #1).
2. Runs triangle-based motif detection on the centroid graph.
3. Expands each centroid set to **item indices** and deduplicates, like the
   energy path. Returned ids live in `0..aspace.nitems`; every motif is a
   union of whole clusters.

```rust
// before (0.27.3): feature ids, easy to misread as items (nnodes says 1000)
let motifs = gl.spot_motives_eigen(&cfg);          // ids in 0..F-1

// after: item-space motifs on the eigen track
let motifs = gl.try_spot_motives_eigen(&aspace, &cfg)?;   // ids in 0..nitems-1
```

Requirements are enforced (no silent degradation, per the #161 lesson):
the Laplacian must be an eigen build (`energy == false`), `n_clusters >= 2`,
`cluster_assignments` must cover **every** item with an in-range cluster id
(`Some(c)`, `c < n_clusters` — outliers or out-of-range ids are refused
rather than silently dropped from the projection), and every centroid must
be non-empty. Violations return the new
`ArrowSpaceError::EigenModeRequired { missing }`.

Also added: `Motives::try_spot_motives_featurespace(&gl, &cfg) -> Result<...>`
— detection over the Laplacian's own nodes (feature ids on pipeline builds)
under an explicit name, as a fallible call. Use it for feature-space /
dimension-ensemble analysis.

### Deprecated

- `Motives::spot_motives_eigen` — the name hid its node space: the ids are
  feature indices on pipeline builds while `nnodes` reports items (#165).
  It now delegates to `try_spot_motives_featurespace`. Callers keep
  compiling with a warning; the bindings are unaffected because they do not
  enable `deny(warnings)`.

### Fixed — feature-space motif detection is now deterministic

`spot_motives_eigen` / `try_spot_motives_featurespace` used an unstable
seed sort and walked the expansion frontier through `HashSet` iteration
order, so two calls on the same Laplacian could return different motif sets
when candidates tied on triangle gain (observed: 14 vs 22 motifs on one
fixture). Both now run the same deterministic detector as the item-space
tracks (sorted frontier, lowest-index tie-break, invariant #4). This is a
determinism bugfix, not a behaviour redefinition: on tie-free graphs the
output is unchanged, and 0.27.3's output was not reproducible to begin with.
The long-standing `test_motives_eigen_deterministic` guard is now robust
instead of passing by luck.

### Fixed — documented node spaces

- `GraphLaplacian::nnodes` is now documented as the **original item count,
  not the matrix's node space**: on pipeline EigenMaps builds `matrix` is
  the F×F feature bootstrap (`matrix.shape() == (F, F)`) while `nnodes`
  still reports `n_items` (#165).
- The `Motives` trait docs state the node space of every entry point
  (feature ids vs item ids).

### Note for the Python bindings

`map_arrow_error` in pyarrowspace matches `ArrowSpaceError` exhaustively;
when bumping to 0.27.4 it needs one arm:

```rust
ArrowSpaceError::EigenModeRequired { .. } => PyValueError::new_err(format!("{}", e)),
```

(and, optionally, a `spot_motives_eigen_items` method wrapping
`try_spot_motives_eigen`, mirroring the existing `spot_motives_energy`
wrapper).

## [0.27.0] — breaking

This release deprecates the panic-as-API query surface and replaces the last
stringly-typed seam in the build pipeline with a typed enum. **One signature
changes** — `build_for_persistence` — everything else is additive or
deprecation-only. The Python bindings ([pyarrowspace]) are **unaffected**:
they do not call `build_for_persistence` and already use the `try_*` query
path exclusively.

[pyarrowspace]: https://github.com/tuned-org-uk/pyarrowspace

### Breaking — `build_for_persistence` takes `PipelineKind` (#154)

`ArrowSpaceBuilder::build_for_persistence` no longer dispatches on magic
strings. A mis-typed pipeline name (`"engery"`) is now a **compile-time
error** instead of a runtime panic, and the energy parameters travel with the
variant instead of an `Option<EnergyParams>` that was only meaningful for one
of the two string values.

```rust
// before (0.26.14)
let (aspace, gl) = builder.build_for_persistence(data, "eigen", None);
let (aspace, gl) = builder.build_for_persistence(data, "energy", Some(params));

// after
use arrowspace::builder::PipelineKind;
let (aspace, gl) = builder.build_for_persistence(data, PipelineKind::Eigen);
let (aspace, gl) = builder.build_for_persistence(data, PipelineKind::Energy(params));
```

**Migration for stringly contexts** (CLI flags, config files, serde): parse
the string with `FromStr`. Accepted names are `eigen`, `energy`, and the
legacy alias `default` (previously dispatched to the energy arm):

```rust
let pipeline: PipelineKind = "energy".parse()?;   // → Energy(EnergyParams::default())
let pipeline: PipelineKind = "eigen".parse()?;    // → Eigen
"engery".parse::<PipelineKind>()                  // → Err(InvalidPipelineError)
```

The parse error type is `arrowspace::builder::InvalidPipelineError` (implements
`Display + std::error::Error`, and echoes the offending string plus the valid
names) — replacing the previous `panic!("Invalid pipeline value: …")`.

Behavioral mapping of the old strings:

| 0.26.14 call | 0.27.0 replacement |
|---|---|
| `build_for_persistence(data, "eigen", _)` | `build_for_persistence(data, PipelineKind::Eigen)` |
| `build_for_persistence(data, "energy", Some(p))` | `build_for_persistence(data, PipelineKind::Energy(p))` |
| `build_for_persistence(data, "energy", None)` | panicked before; now `PipelineKind::Energy(EnergyParams::default())` is available — pick params explicitly |
| `build_for_persistence(data, "default", Some(p))` | `build_for_persistence(data, PipelineKind::Energy(p))` |
| `build_for_persistence(data, <typo>, _)` | panicked before; now does not compile — parse errors surface as `InvalidPipelineError` |

### Deprecated — panicking query wrappers (#153)

`ArrowSpace::prepare_query_item` and `ArrowSpace::search_lambda_aware` are
now `#[deprecated]`. They remain fully functional (same signatures, same
panic behaviour, same `expect` sites) so existing callers keep compiling with
warnings. Deprecation, not removal, is the policy going forward.

Both panic on recoverable input conditions — non-finite query, dimension
mismatch, degenerate λ (~0) — which is exactly the failure class the `try_*`
twins (added in 0.26.7, #122) return as typed `ArrowSpaceError` values.

```rust
// before (panics on degenerate λ / wrong dims / NaN)
let lambda = aspace.prepare_query_item(&q, &gl);
let hits = aspace.search_lambda_aware(&item, k, alpha);

// after (typed, catchable across FFI)
let lambda = aspace.try_prepare_query_item(&q, &gl)?;          // f64
let hits  = aspace.try_search_lambda_aware(&item, k, alpha)?;  // Vec<(usize, f64)>
```

Not deprecated: `ArrowSpace::search_lambda_aware_hybrid` (no `try_*` twin
exists; downstream bindings call it directly).

Rustdoc reordering: the `try_*` variants are now the primary documented entry
points with the authoritative examples; the panicking wrappers carry a
deprecation notice pointing at their fallible twins.

### Added

- `builder::PipelineKind` enum (`Eigen`, `Energy(EnergyParams)`) with
  `FromStr` (`Err = InvalidPipelineError`) — #154.
- `builder::InvalidPipelineError` typed parse error — #154.
- `PartialEq` derive on `EnergyParams` (needed to compare parsed pipelines;
  harmless for all other uses).

### Deprecated

- `builder::Pipeline` enum — superseded by `PipelineKind`. Still parses
  `"eigen"` / `"energy"` / `"default"` for code that holds a `Pipeline`
  value, but new code should not reference it.
- `ArrowSpace::prepare_query_item` — use `try_prepare_query_item` — #153.
- `ArrowSpace::search_lambda_aware` — use `try_search_lambda_aware` — #153.

### Fixed

- Internal query-path callers (`search_linear_sorted`, `range_search`,
  `EigenMaps::search`, `search_energy`, energy build arms) migrated to the
  `try_*` twins, so the crate itself no longer routes through the deprecated
  surface.

### Deferred

- #155 (`try_build_energy` / `EnergyParams::validate` typed validation for
  the energy builder): postponed — energymaps is lower priority this cycle;
  `build_energy` keeps its current `assert!`-based validation.
