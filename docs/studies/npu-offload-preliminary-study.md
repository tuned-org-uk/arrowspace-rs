# Preliminary study: NPU offload for ArrowSpace (GPU-free acceleration)

Status: preliminary / proposal. Nothing in this document is implemented. No benchmarks have been run.

## 1. Goal

Evaluate how NPU hardware could accelerate parts of an ArrowSpace workflow without requiring a GPU, while keeping the lambda-tau index definition unchanged.

## 2. Summary

The most practical GPU-free design is a hybrid CPU-NPU pipeline:

- NPU: embedding-model inference and, possibly, dense fixed-shape transforms.
- CPU: graph construction, sparse Rayleigh quotient, edge dispersion, indexing, filtering, top-k and final exact reranking.

NPU runtimes deploy compiled tensor graphs with device-specific operator, precision and shape constraints. Dense MatMul-style work maps well; irregular sparse graph traversal does not automatically.

## 3. Proposed responsibility split

| Stage | Device | Rationale |
|---|---|---|
| Embedding model inference | NPU | Natural inference workload; subject to operator/model compatibility |
| Dense projection / learned feature transform | NPU candidate | Expressible as fixed-shape matrix multiplication |
| Graph construction, kNN, CSR assembly | CPU | Irregular logic |
| Exact sparse Rayleigh quotient (`compute_rayleigh_quotient_from_matrix`) | CPU | Keeps sparse representation and numerical reference |
| Edge dispersion (`compute_item_dispersion`) | CPU | Preserves traversal and normalisation semantics |
| Large batched dense candidate scoring | NPU experiment | Only if batching amortises dispatch and transfer cost |
| Index lookups, filtering, top-k, rerank | CPU | Keep first implementation exact |

## 4. Vendor runtime routes to evaluate

| Hardware family | Route |
|---|---|
| Intel NPU | OpenVINO, explicit `NPU` device |
| AMD Ryzen AI / XDNA | Ryzen AI software with ONNX Runtime execution provider |
| Qualcomm Hexagon | ONNX Runtime QNN execution provider / Qualcomm AI stack |

Compatibility must be verified for the exact processor, OS, driver, model and runtime version. For a strict no-GPU requirement: select the NPU explicitly (not an automatic multi-device policy), allow CPU fallback only where intended, and inspect operator placement to confirm what actually ran on the NPU.

## 5. Can Rayleigh scoring run on the NPU?

For a batch X (B x F) and a fixed feature-space Laplacian L (F x F):

    Y = X L
    E_b = sum_j(X_bj * Y_bj) / sum_j(X_bj^2)

This is MatMul + elementwise + reduction. Caveat: if L is sparse, densifying it can remove the advantage (sparse CPU cost scales with nnz; dense cost with F^2) and may require recompiling the model whenever the graph changes. Recommendation: keep exact sparse Rayleigh on the CPU unless benchmarks show a fixed, reasonably small graph and large batches make the dense route beneficial.

A dense projection Z = X P is a cleaner first dense target. Note that materialising P changes memory use, and adding a new reduction step changes the representation being scored; it is not automatically equivalent to full-dimensional lambda-tau.

## 6. Dispersion

The current implementation uses contributions c_ij = max(-L_ij, 0) * (x_i - x_j)^2 over stored directed entries and sums squared normalised shares. A possible NPU formulation would use fixed edge-endpoint arrays: gather source/target values, subtract, square, weight, reduce, normalise. Gather/reduction support, precision and static-shape requirements must be checked per backend. Rules: preserve the directed-edge counting convention; do not substitute normalised-incidence energy shares for the implementation's contributions. Keep on CPU for the first release.

## 7. Numerical validation

NPUs may compute at lower internal precision (e.g. FP16). Keep the CPU implementation as the reference and measure:

- absolute/relative Rayleigh error
- dispersion error
- absolute lambda-tau error
- candidate recall of any approximate filter
- final top-k overlap / recall
- self-retrieval and repeatability
- near-zero-energy behaviour

If approximate NPU scores are used for candidate generation, recompute exact scores on the CPU; exact reranking cannot recover candidates already discarded, so candidate recall must be tested explicitly.

## 8. Implementation plan

### Phase 1: inference boundary

```rust
trait EmbeddingBackend {
    fn embed_batch(&self, inputs: &[Input]) -> Result<Vec<Vec<f32>>, BackendError>;
}
```

Implement a CPU backend and a vendor NPU backend feeding the existing CPU ArrowSpace engine. Validate model conversion and device placement in Python first; integrate the chosen runtime into Rust afterwards.

### Phase 2: GPU-free baseline

Compare A) CPU embeddings + CPU ArrowSpace vs B) NPU embeddings + CPU ArrowSpace. Measure model load/compile time separately from warm execution, embedding throughput, end-to-end p50/p95 query latency, index-build time, memory, energy per document/query if measurable, and retrieval-quality change. Use fixed batch/sequence-length buckets where the backend requires static shapes.

### Phase 3: one dense ArrowSpace kernel

After profiling, test exactly one of: batched dense projection; fixed small-graph batched Rayleigh; batched dense scoring over a preselected candidate set. Compare against an optimised multithreaded CPU baseline including dispatch, packing and readback.

### Phase 4 (research): surrogate lambda-tau prediction

A small NPU model predicting lambda-tau would be a surrogate, not the exact formula. It needs graph-specific training data, recalibration when the graph changes, and candidate-recall validation. Treat as a separate experiment.

## 9. Open questions

- Target NPU platform and OS (Intel, AMD Ryzen AI, Qualcomm, other)?
- Which embedding model(s) and precisions are in scope?
- Typical feature dimension F and graph sparsity for target workloads?
- Acceptable score drift versus the CPU reference?

## 10. Caveats

- Operator, precision and dynamic-shape support differs by vendor, driver and runtime release and changes quickly; re-verify against current documentation before implementation.
- This study is derived from reading `src/search/taumode.rs` and `src/maps/energymaps.rs` and from vendor documentation; no code was changed or benchmarked.
