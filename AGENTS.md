# AGENTS.md — Design Invariants

Normative constraints for any modification to this crate. Workflow guidance and
license acceptance live in [CONTRIBUTING.md](CONTRIBUTING.md). A change that
violates an invariant here must be rejected regardless of its local merits.

Design background: *The ArrowSpace Algorithm: From Graph Wiring to τ-Mode
Spectral Search* — L. Moriondo, DOI
[10.5281/zenodo.21679021](https://zenodo.org/records/21679021).

## 0. Check the Python bindings for downstream usage

Before changing any public API — signatures, enum variants, panic behaviour,
removals — ALWAYS inspect the Python bindings at
[pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace) (`src/lib.rs`)
for actual downstream call sites. The Rust test suite does not cover the FFI
surface: a signature that is "unused" inside this crate may still be
load-bearing across the PyO3 boundary. Prefer deprecation over breakage
(e.g. the `try_*` twins from #122/#153); never assume a method is safe to
repurpose without grepping the bindings first.

## 1. The graph Laplacian is the centerpiece

Every capability — search, clustering, partitioning, sequencing, analysis,
compression — MUST derive from the feature-space graph Laplacian built by the
pipeline (`GraphLaplacian.matrix`: L = D − A, its weights, and per-node λτ
read-outs). Do not introduce parallel data paths that bypass it.

- Lives in: `src/graph.rs` (`GraphLaplacian`), `src/laplacian.rs` (construction)
- Violation example: computing similarity structures from raw vectors when the
  adjacency recoverable from L already encodes them.

## 2. Discrete spaces only

Arrowspace computes in discrete spaces. NEVER use eigensolvers, continuous
optimisation or dense spectral embeddings. Spectral quantities are per-node
Rayleigh quotients (λ) computed by graph–vector products; structural ordering
problems are solved combinatorially on the graph.

- Lives in: `src/search/taumode.rs` (λτ read-out)
- Violation example: a Fiedler-vector ordering via Lanczos/power iteration,
  or any call into an eigen-decomposition routine.

## 3. Sparse-first, flat-first

Keep data sparse (`sprs::CsMat`) and memory flat (contiguous `Vec` /
row-major `DenseMatrix`). Never densify N×N or N×F structures beyond bounded
sizes; never route hot paths through hash maps where sorted arrays suffice.
Dense buffers must be row-major so rows are addressable as contiguous slices.

- Lives in: `src/laplacian.rs` (CSR assembly), `src/core.rs`
  (`ArrowSpace::new` single-flatten); see issues #107 and #31 for the
  DashMap→sort-merge and clone-removal precedents.
- Violation example: an O(N·E) map scan, or cloning the dataset between
  pipeline stages that could share one owned buffer.

## 4. Determinism by construction

Identical inputs MUST produce identical outputs. Every randomised stage takes
an explicit seed; tie-breaking in sorts, heaps and merges is explicit;
parallel reductions merge deterministically (e.g. max-weight).

- Lives in: `src/clustering/mod.rs` (seeded StdRng), `src/laplacian.rs`
  (deterministic symmetrisation).
- Violation example: relying on HashMap iteration order or last-write-wins
  concurrent updates to resolve conflicting values.

## 5. Pipeline separation

The build pipeline is clustering → sampling → Laplacian construction → score
read-out. New capabilities plug into the final read-out slot and MUST NOT
reach backwards into earlier stages or mutate their inputs.

- Lives in: `src/builder.rs` (stage structure), `src/maps/eigenmaps.rs`
  (read-out traits).
- Violation example: a search feature that re-runs kNN inside the query path.
