//! #201 follow-up — samplers are seeded, not OS-entropy driven.
//!
//! Invariant 4 (determinism by construction): every randomised stage takes
//! an explicit seed. Both sampler types seeded their internal StdRng from
//! `SysRng`, so every default build (which uses `Simple(0.6)`) consumed OS
//! entropy: repeat builds differed in which rows were rejected, and
//! sampling-enabled tests flickered.
//!
//! The seed is an explicit constructor argument now. The enum variant
//! shapes (`Simple(rate)`, `DensityAdaptive(rate)`) are unchanged — the
//! bindings and stored configs only ever carry the rate.

use crate::sampling::{InlineSampler, SamplerType};

const SEED: u64 = 3407;

fn drive(sampler: &mut dyn InlineSampler, n: usize) -> Vec<bool> {
    (0..n)
        .map(|_| sampler.should_keep(&[1.0], 1.0, 1, 10))
        .collect()
}

#[test]
fn simple_same_seed_same_stream() {
    let mut a = SamplerType::new_simple(0.5, SEED);
    let mut b = SamplerType::new_simple(0.5, SEED);
    let sa = drive(&mut a, 512);
    let sb = drive(&mut b, 512);
    assert_eq!(sa, sb, "same-seed samplers diverged");
}

#[test]
fn simple_different_seed_different_stream() {
    let mut a = SamplerType::new_simple(0.5, 3407);
    let mut b = SamplerType::new_simple(0.5, 3408);
    let sa = drive(&mut a, 512);
    let sb = drive(&mut b, 512);
    assert_ne!(sa, sb, "different-seed samplers produced identical streams");
}

#[test]
fn density_adaptive_same_seed_same_stream() {
    let mut a = SamplerType::new_density_adaptive(0.5, SEED);
    let mut b = SamplerType::new_density_adaptive(0.5, SEED);
    // Varied inputs so the adaptive branch structure is exercised.
    let sa: Vec<bool> = (0..256)
        .map(|i| a.should_keep(&[i as f64], (i % 7) as f64 * 0.1, i % 40, 40))
        .collect();
    let sb: Vec<bool> = (0..256)
        .map(|i| b.should_keep(&[i as f64], (i % 7) as f64 * 0.1, i % 40, 40))
        .collect();
    assert_eq!(sa, sb, "same-seed adaptive samplers diverged");
}

#[test]
fn builder_passes_its_seed_to_the_sampler() {
    // Two builds with the same clustering seed must produce bit-identical
    // indexes even though the default policy samples (Simple(0.6)).
    let rows: Vec<Vec<f64>> = {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        (0..1200usize)
            .map(|i| {
                let base = (i % 6) as f64;
                (0..8).map(|_| base + rng.random_range(0.0..1.0)).collect()
            })
            .collect()
    };

    let build = || ArrowSpaceBuilder::new().with_seed(SEED).build(rows.clone());
    let (a, _) = build();
    let (b, _) = build();

    assert_eq!(a.lambdas().len(), b.lambdas().len());
    for (la, lb) in a.lambdas().iter().zip(b.lambdas().iter()) {
        assert_eq!(la.to_bits(), lb.to_bits(), "seeded builds diverged");
    }
    for i in 0..a.lambdas().len() {
        assert_eq!(a.cluster_of(i), b.cluster_of(i));
    }
}

use crate::builder::ArrowSpaceBuilder;
use rand::{RngExt, SeedableRng};
