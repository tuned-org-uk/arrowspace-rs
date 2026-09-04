# Changelog

All notable changes to `arrowspace` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

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
`cluster_assignments` populated for every item, and every centroid
non-empty. Violations return the new
`ArrowSpaceError::EigenModeRequired { missing }`.

Also added: `Motives::try_spot_motives_featurespace(&gl, &cfg) -> Result<...>`
— the exact historical `spot_motives_eigen` behaviour (detection over the
Laplacian's own nodes, i.e. feature ids on pipeline builds), under an
explicit name, as a fallible call. Use it for feature-space/dimension
ensemble analysis.

### Deprecated

- `Motives::spot_motives_eigen` — unchanged behaviour (it now delegates to
  `try_spot_motives_featurespace`), but the name hid its node space and the
  ids are feature indices on pipeline builds while `nnodes` reports items
  (#165). Callers keep compiling with a warning; the bindings are unaffected
  because they do not enable `deny(warnings)`.

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
