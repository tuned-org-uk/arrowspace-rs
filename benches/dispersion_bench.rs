//! #200 — item dispersion cost: sparse directed-edge traversal vs dense
//! per-pair lookup.
//!
//! A/B within one binary: the pre-#200 implementation survives verbatim as
//! the CSC/non-canonical fallback of `compute_item_dispersion`, and
//! `compute_synthetic_lambda` is public — so the same matrix fed as CSR
//! takes the sparse fast path while its CSC conversion takes the dense
//! lookup path (identical results; parity proven in
//! test_dispersion_sparse.rs). The end-to-end read-out bench covers the
//! per-item pipeline around it.

use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::ArrowSpace;
use arrowspace::graph::GraphLaplacian;
use arrowspace::search::taumode::{TauMode, TauMode as TM};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

const SEED: u64 = 3407;
const N: usize = 300;
const F: usize = 384;

fn fixture() -> (ArrowSpace, GraphLaplacian) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let rows: Vec<Vec<f64>> = (0..N)
        .map(|_| (0..F).map(|_| rng.random_range(0.0..1.0)).collect())
        .collect();
    // eps=1.5 keeps the clustered-graph shape healthy (eps trap); topk=15
    // mirrors the CVE graph parameters (nnz per feature row ~16).
    ArrowSpaceBuilder::new()
        .with_lambda_graph(1.5, 25, 15, 2.0, None)
        .with_seed(SEED)
        .build(rows)
}

fn bench_dispersion(c: &mut Criterion) {
    let (_aspace, gl) = fixture();
    let item: Vec<f64> = {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED + 2);
        (0..F).map(|_| rng.random_range(0.0..1.0)).collect()
    };
    let csr = &gl.matrix;
    let csc = csr.to_csc();

    // Sanity: both storage formats must produce the identical lambda.
    let a = TM::compute_synthetic_lambda(&item, None, csr, 0.5);
    let b = TM::compute_synthetic_lambda(&item, None, &csc, 0.5);
    assert_eq!(a.to_bits(), b.to_bits(), "CSR and CSC lambda bits diverge");

    let mut group = c.benchmark_group("dispersion");
    group.bench_with_input(
        BenchmarkId::new("compute_synthetic_lambda", "sparse_csr_path"),
        csr,
        |b, graph| {
            b.iter(|| {
                black_box(TM::compute_synthetic_lambda(
                    black_box(&item),
                    None,
                    black_box(graph),
                    0.5,
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("compute_synthetic_lambda", "dense_lookup_path"),
        &csc,
        |b, graph| {
            b.iter(|| {
                black_box(TM::compute_synthetic_lambda(
                    black_box(&item),
                    None,
                    black_box(graph),
                    0.5,
                ))
            })
        },
    );
    group.finish();

    let (mut aspace2, gl2) = fixture();
    c.bench_function("taumode_readout/300x384_topk15", |b| {
        b.iter(|| {
            TM::compute_taumode_lambdas_parallel(
                black_box(&mut aspace2),
                black_box(&gl2),
                black_box(TauMode::Median),
            );
            black_box(&aspace2.lambdas());
        })
    });
}

criterion_group!(benches, bench_dispersion);
criterion_main!(benches);
