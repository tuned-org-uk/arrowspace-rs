# Changelog

All notable changes to `arrowspace` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

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
