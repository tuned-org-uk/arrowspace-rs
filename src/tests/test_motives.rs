#![allow(deprecated)] // guards for the spot_motives_eigen compatibility surface (#165)

use crate::analysis::motives::{MotiveConfig, Motives};
use crate::analysis::subgraphs::SubgraphsMotive;
use crate::builder::ArrowSpaceBuilder;
use crate::maps::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::tests::test_data::make_gaussian_cliques;

use log::{debug, info};

#[test]
fn test_motives_basic() {
    crate::tests::init();

    // 3 near-cliques + outliers
    let rows = make_gaussian_cliques(12, 0.05, 15, 10, 42);

    // Build a denser, normalized graph to preserve triangle closures
    // (seeded: an unseeded builder draws its clustering seed from
    // rand::rng(), making the motif count a per-run coin flip).
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_seed(42)
        .with_lambda_graph(0.4, 14, 8, 2.0, None) // k=14, topk=8
        .with_normalisation(true)
        .with_sparsity_check(false)
        .build(rows);

    // Keep at least as many as topk; relax thresholds; disable Rayleigh for the first run
    let cfg = MotiveConfig {
        top_l: 16, // ≥ topk (if no motives is spotted, increase top_l)
        min_triangles: 2,
        min_clust: 0.4,
        max_motif_size: 24,
        max_sets: 100,
        jaccard_dedup: 0.8,
    };

    let motifs = gl.spot_motives_eigen(&cfg);
    info!("Found {} motifs:", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        debug!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
}

#[test]
fn test_motives_basic_2() {
    crate::tests::init();

    // 3 near-cliques of 24 points each + 27 outliers = 99 total, 10D
    let rows = make_gaussian_cliques(24, 0.05, 27, 10, 42);

    // Build a denser, normalized graph to preserve triangle closures at N=99
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.3, 18, 12, 2.0, None) // slightly denser intra-group
        .with_sparsity_check(false)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None)
        .with_seed(42)
        .build(rows);

    // Keep at least as many neighbors as topk; thresholds tuned for N=99 near-cliques
    let cfg = MotiveConfig {
        top_l: 16, // ≥ topk to avoid double pruning
        min_triangles: 3,
        min_clust: 0.45,
        max_motif_size: 32,
        max_sets: 100,
        jaccard_dedup: 0.7,
    };

    let motifs = gl.spot_motives_eigen(&cfg);
    info!("Found {} motifs:", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        debug!("  Motif {}: {:?}", i, m);
    }

    assert!(!motifs.is_empty(), "Expected motifs at N=99");
}

#[test]
fn test_motives_energy_basic() {
    crate::tests::init();

    // 3 near-cliques + outliers
    let rows = make_gaussian_cliques(12, 0.05, 15, 10, 42);

    let p = EnergyParams::default();
    // Mild diffusion and balanced weights tend to give usable local density
    // p.steps, p.neighbork, etc. can be tuned in your codebase if exposed

    // Build Energy-only ArrowSpace and GraphLaplacian
    // Note: build_energy requires dimensionality reduction enabled in this codebase.
    let (aspace, gl_energy) = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_lambda_graph(0.4, 14, 8, 2.0, None) // k=14, topk=8
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build_energy(rows, p);

    // Keep at least as many neighbors as the energy graph retains; avoid double-pruning
    let cfg = MotiveConfig {
        top_l: 16,        // keep neighbors available from energy Laplacian
        min_triangles: 2, // permissive seeding
        min_clust: 0.4,   // moderate clustering threshold
        max_motif_size: 24,
        max_sets: 100,
        jaccard_dedup: 0.8,
    };

    let motifs = gl_energy
        .try_spot_motives_energy(&aspace, &cfg)
        .expect("energy build must satisfy energy-mode requirements");

    info!("Found {} motifs (energy):", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        debug!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
}

#[test]
fn test_motives_eigen_vs_energy_consistency() {
    // Deterministic logs and RNG
    crate::tests::init();

    // Synthetic data: 3 planted cliques, no outliers — the eigen item-space
    // contract requires every item to carry a cluster assignment (#166
    // review), so the shared fixture must be fully clusterable.
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
    // Positive quadrant: smartcore's cosine distance is 1 - cos (unrectified,
    // up to 2.0), so negative-cosine centroid pairs would exceed any eps < 2
    // and isolate nodes in the reconstructed centroid graph.
    let centers: [Vec<f64>; 3] = [vec![10.0, 0.0], vec![0.0, 10.0], vec![7.0, 7.0]];
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(36);
    for c in &centers {
        for _ in 0..12 {
            let mut v = vec![0.0f64; 10];
            for (k, m) in c.iter().enumerate() {
                v[k] = m + Normal::new(0.0, 0.04).unwrap().sample(&mut rng);
            }
            rows.push(v);
        }
    }

    // Common motif config
    let cfg = MotiveConfig {
        top_l: 16,
        min_triangles: 2,
        min_clust: 0.35,
        max_motif_size: 24,
        max_sets: 64,
        jaccard_dedup: 0.8,
    };

    // -----------------------
    // EigenMaps pipeline
    // -----------------------
    let (aspace_eig, gl_eig) = ArrowSpaceBuilder::new()
        .with_seed(42)
        .with_lambda_graph(1.2, 14, 8, 2.0, None) // eps>=1 joins the orthogonal clique centroids
        .with_cluster_max_clusters(3)
        .with_cluster_radius(10.0)
        .with_sparsity_check(false)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build(rows.clone());

    // Item-space motifs on the EigenMaps track (#165): the ids are item
    // indices and must recover the planted cliques.
    let motifs_eig = gl_eig
        .try_spot_motives_eigen(&aspace_eig, &cfg)
        .expect("pipeline EigenMaps build must satisfy the item-space requirements");
    debug!("Eigen motifs ({}): {:?}", motifs_eig.len(), motifs_eig);
    assert!(
        !motifs_eig.is_empty(),
        "EigenMaps returned 0 item-space motifs; expected planted clusters"
    );

    // -----------------------
    // EnergyMaps pipeline
    // -----------------------
    let p = EnergyParams::default();

    let (aspace_eng, gl_eng) = ArrowSpaceBuilder::new()
        .with_seed(42)
        .with_lambda_graph(0.35, 18, 10, 2.0, None) // denser k/topk
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build_energy(rows, p);

    // Energy-aware motifs: discovered on subcentroid graph, mapped to items.
    // The two tracks may legitimately return different results — the energy
    // graph is the subcentroid Laplacian with its own λ-proximity item
    // mapping, so agreement is REPORTED, never gated. What is gated is the
    // contract: Ok, and every returned motif is valid item space.
    let motifs_eng = gl_eng
        .try_spot_motives_energy(&aspace_eng, &cfg)
        .expect("energy build must satisfy energy-mode requirements");
    debug!("Energy motifs ({}): {:?}", motifs_eng.len(), motifs_eng);
    for m in &motifs_eng {
        assert!(m.len() >= 3, "energy motif below minimum size: {m:?}");
        assert!(
            m.iter().all(|&i| i < aspace_eng.nitems),
            "energy motif {m:?} leaves item space 0..{}",
            aspace_eng.nitems
        );
    }

    // -----------------------
    // Agreement (reported, not gated)
    // -----------------------

    fn dedup(mut sets: Vec<Vec<usize>>, thr: f64) -> Vec<Vec<usize>> {
        use std::collections::HashSet;
        let mut out: Vec<HashSet<usize>> = Vec::new();
        for v in sets.drain(..) {
            let s: HashSet<usize> = v.into_iter().collect();
            let mut keep = true;
            for t in &out {
                let inter = s.intersection(t).count() as f64;
                let uni = (s.len() + t.len()) as f64 - inter;
                let j = if uni == 0.0 { 0.0 } else { inter / uni };
                if j >= thr {
                    keep = false;
                    break;
                }
            }
            if keep {
                out.push(s);
            }
        }
        let mut vs: Vec<Vec<usize>> = out
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        vs.sort_by_key(|v| std::cmp::Reverse(v.len()));
        vs
    }

    fn jaccard(a: &[usize], b: &[usize]) -> f64 {
        use std::collections::HashSet;
        let sa: HashSet<usize> = a.iter().copied().collect();
        let sb: HashSet<usize> = b.iter().copied().collect();
        let inter = sa.intersection(&sb).count() as f64;
        let uni = (sa.len() + sb.len()) as f64 - inter;
        if uni == 0.0 { 0.0 } else { inter / uni }
    }

    let eig_d = dedup(motifs_eig.clone(), 0.8);
    let eng_d = dedup(motifs_eng.clone(), 0.8);

    // EigenMaps is the primary track: its top item-space motif must recover
    // planted structure.
    let planted: Vec<Vec<usize>> = vec![(0..12).collect(), (12..24).collect(), (24..36).collect()];
    if let Some(top_eig) = eig_d.first() {
        let eig_ground = planted
            .iter()
            .map(|p| jaccard(top_eig, p))
            .fold(0.0_f64, f64::max);
        assert!(
            eig_ground >= 0.3,
            "EigenMaps top motif misses planted cliques (best J={eig_ground:.3})"
        );
    }

    // Coverage report: how many eigen motifs have a matching energy motif.
    // Informational only — the tracks explore different node spaces by design.
    if !eng_d.is_empty() {
        let mut matched = 0usize;
        for e in &eig_d {
            let best = eng_d.iter().map(|x| jaccard(e, x)).fold(0.0_f64, f64::max);
            if best >= 0.5 {
                matched += 1;
            }
        }
        debug!(
            "Eigen motifs matched by energy at J>=0.5: {}/{} (|eig|={}, |eng|={})",
            matched,
            eig_d.len(),
            eig_d.len(),
            eng_d.len()
        );
    } else {
        debug!(
            "EnergyMaps returned 0 item-level motifs on this fixture; the              tracks may return different results by design"
        );
    }
}

#[test]
fn test_motives_energy_stable() {
    crate::tests::init();

    // EnergyMaps contract: motifs are λ-proximity communities over
    // subcentroids mapped to items via centroid_map. Oscillation across
    // runs is expected on this track, so gate validity only: motifs exist
    // and reference valid item indices on every independent build.
    let rows = make_gaussian_cliques(12, 0.04, 12, 10, 3407);

    let build = || {
        let p = EnergyParams::default();
        let (aspace_eng, gl_eng) = ArrowSpaceBuilder::new()
            .with_seed(3407)
            .with_lambda_graph(0.35, 18, 10, 2.0, None)
            .with_dims_reduction(true, Some(0.3))
            .with_inline_sampling(None)
            .build_energy(rows.clone(), p);

        let cfg = MotiveConfig {
            top_l: 16,
            min_triangles: 2,
            min_clust: 0.35,
            max_motif_size: 24,
            max_sets: 64,
            jaccard_dedup: 0.8,
        };
        gl_eng
            .try_spot_motives_energy(&aspace_eng, &cfg)
            .expect("energy build must satisfy energy-mode requirements")
    };

    for motifs in [build(), build()] {
        assert!(
            !motifs.is_empty(),
            "energy pipeline returned no item-level motifs"
        );
        for m in &motifs {
            assert!(m.len() >= 3, "energy motif below minimum size: {:?}", m);
            assert!(
                m.iter().all(|&i| i < 51),
                "energy motif references out-of-range item index: {:?}",
                m
            );
        }
    }
}

#[test]
fn test_spot_subg_motives_eigenmaps_must_not_return_feature_indices() {
    use crate::error::ArrowSpaceError;

    crate::tests::init();

    // Issue #161 regression guard: on an EigenMaps build (no sub_centroids,
    // no centroid_map) spot_subg_motives used to run motif detection over the
    // F×F bootstrap Laplacian — whose nodes enumerate FEATURES — and label
    // the results as item_indices. With N=48 items and F=120 features the
    // namespaces cannot be confused: the pre-fix failure returned ids in
    // 9..=117 on this exact fixture. The fix rejects the call instead.
    let rows = crate::tests::test_data::make_gaussian_cliques_multi(48, 0.2, 4, 120, 3407);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 12, 8, 2.0, None)
        .with_seed(3407)
        .build(rows);

    let cfg = crate::analysis::subgraphs::SubgraphConfig {
        min_size: 3,
        ..Default::default()
    };
    let err = gl
        .try_spot_subg_motives(&aspace, &cfg)
        .expect_err("EigenMaps builds must be rejected, not served feature-space indices");
    assert!(
        matches!(err, ArrowSpaceError::EnergyModeRequired { .. }),
        "expected EnergyModeRequired, got: {err}"
    );
}

#[test]
fn test_try_spot_motives_energy_rejects_eigenmaps_build() {
    use crate::error::ArrowSpaceError;

    crate::tests::init();

    // Issue #161: the energy path must refuse EigenMaps builds instead of
    // silently degrading to feature-space motif detection.
    let rows = make_gaussian_cliques(12, 0.04, 12, 10, 3407);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_seed(3407)
        .with_lambda_graph(0.4, 14, 8, 2.0, None)
        .with_inline_sampling(None)
        .build(rows);

    let cfg = MotiveConfig::default();
    let err = gl
        .try_spot_motives_energy(&aspace, &cfg)
        .expect_err("EigenMaps builds must be rejected");
    assert!(
        matches!(err, ArrowSpaceError::EnergyModeRequired { .. }),
        "expected EnergyModeRequired, got: {err}"
    );
}

#[test]
fn test_try_spot_motives_energy_returns_item_space_indices() {
    crate::tests::init();

    // On a genuine energy build the fallible path succeeds and every motif
    // index lives in item space (0..nitems), never in subcentroid space.
    let rows = make_gaussian_cliques(12, 0.04, 12, 10, 3407);
    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(3407)
        .with_lambda_graph(0.35, 18, 10, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let p = crate::maps::energymaps::EnergyParams::new(&builder);
    let (aspace, gl) = builder.build_energy(rows, p);

    let cfg = MotiveConfig {
        top_l: 16,
        min_triangles: 2,
        min_clust: 0.35,
        max_motif_size: 24,
        max_sets: 64,
        jaccard_dedup: 0.8,
    };
    let motifs = gl
        .try_spot_motives_energy(&aspace, &cfg)
        .expect("energy build must satisfy the energy-mode requirements");

    assert!(!motifs.is_empty(), "energy pipeline returned no motifs");
    for m in &motifs {
        assert!(
            m.iter().all(|&i| i < aspace.nitems),
            "motif {m:?} leaves item space 0..{}",
            aspace.nitems
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Issue #165: item-space motifs on the EigenMaps track
// ──────────────────────────────────────────────────────────────────────────────

/// EigenMaps fixture with namespaces that cannot be confused:
/// 9 clusters of ~22 unit-norm items each, F=256 features, dims reduction
/// disabled (mirrors the #165 reproduction layout: unit-norm embeddings,
/// F > N). Feature-space ids reach 255, item-space ids stop at 198.
///
/// Geometry: 9 cluster directions grouped into 3 angular neighbourhoods of
/// 3 directions each. Intra-neighbourhood angle 0.5 rad (cos distance
/// ≈ 0.12 < eps 0.5), inter-neighbourhood angle ≈ 1.4 rad (distance ≈ 0.83
/// > eps), so the reconstructed centroid graph forms exactly 3 disjoint
/// triangles — one item-space motif per neighbourhood, each a union of
/// whole clusters.
///
/// Hyperparameter regime from a one-off arrowspace_tuner (Optuna) study on
/// unit-norm corpora of this shape: best eps ∈ [0.45, 0.52], k ∈ [25, 34]
/// (k≈25 on 200×256, k≈34 on the 1000×128 #165 repro corpus), topk = k/2.
fn eigen_fixture_165() -> (crate::core::ArrowSpace, crate::graph::GraphLaplacian) {
    // Directions: 3 neighbourhoods × 3 directions; within a neighbourhood
    // directions fan out by 0.4 rad around the neighbourhood's own axis
    // (cos distance ≈ 0.08 < eps 0.5), neighbourhoods sit 2.5 rad apart
    // (distance ≥ 1.24 > eps).
    let mut dirs: Vec<Vec<f64>> = Vec::with_capacity(9);
    for g in 0..3usize {
        let base = g as f64 * 2.5;
        for j in 0..3usize {
            let theta = base + j as f64 * 0.4;
            let mut d = vec![0.0; 256];
            d[0] = theta.cos();
            d[1] = theta.sin();
            dirs.push(d);
        }
    }
    // Points: direction + per-dim noise small enough that cluster members
    // stay within the (squared-L2) cluster radius: 2·256·σ² ≈ 0.002.
    let mut rng = rand::rngs::StdRng::seed_from_u64(3407);
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(200);
    for d in &dirs {
        for _ in 0..22 {
            let mut v: Vec<f64> = Vec::with_capacity(256);
            for k in 0..256 {
                let noise = Normal::new(0.0, 0.002).unwrap().sample(&mut rng);
                v.push(d[k] + noise);
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            rows.push(v.into_iter().map(|x| x / norm).collect());
        }
    }

    ArrowSpaceBuilder::new()
        .with_seed(3407)
        .with_lambda_graph(0.5, 25, 12, 2.0, None)
        .with_cluster_max_clusters(9)
        .with_cluster_radius(0.01)
        .with_dims_reduction(false, None)
        .with_sparsity_check(false)
        .with_inline_sampling(None)
        .build(rows)
}

fn cfg_165() -> MotiveConfig {
    MotiveConfig {
        top_l: 24,
        min_triangles: 1,
        min_clust: 0.1,
        max_motif_size: 60,
        max_sets: 64,
        jaccard_dedup: 0.8,
    }
}

#[test]
fn test_try_spot_motives_eigen_returns_item_space_indices() {
    crate::tests::init();

    // Issue #165: the eigen track must offer the same item-space contract as
    // try_spot_motives_energy — motif ids are item indices, never feature ids.
    let (aspace, gl) = eigen_fixture_165();
    let motifs = gl
        .try_spot_motives_eigen(&aspace, &cfg_165())
        .expect("pipeline EigenMaps build must satisfy the item-space requirements");

    assert!(
        !motifs.is_empty(),
        "EigenMaps item-space path returned no motifs"
    );
    for m in &motifs {
        assert!(m.len() >= 3, "item motif below minimum size: {:?}", m);
        assert!(
            m.iter().all(|&i| i < aspace.nitems),
            "motif {m:?} leaves item space 0..{}",
            aspace.nitems
        );
    }
}

#[test]
fn test_try_spot_motives_eigen_motifs_are_cluster_unions() {
    crate::tests::init();

    // The item-space eigen path detects on the centroid graph and expands via
    // the item→cluster map, so every returned motif must be a union of whole
    // clusters: if item i is in a motif, every item assigned to cluster_of(i)
    // must be in the same motif. Feature-space ids cannot satisfy this.
    let (aspace, gl) = eigen_fixture_165();
    let motifs = gl
        .try_spot_motives_eigen(&aspace, &cfg_165())
        .expect("pipeline EigenMaps build must satisfy the item-space requirements");
    assert!(!motifs.is_empty());

    for m in &motifs {
        let members: std::collections::HashSet<usize> = m.iter().copied().collect();
        for &i in m {
            let ci = aspace
                .cluster_of(i)
                .expect("motif item must carry a cluster assignment");
            for j in 0..aspace.nitems {
                if aspace.cluster_of(j) == Some(ci) {
                    assert!(
                        members.contains(&j),
                        "motif {m:?} splits cluster {ci} (missing item {j})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_try_spot_motives_featurespace_replicates_current_behaviour() {
    crate::tests::init();

    // The featurespace variant is the one true implementation behind the
    // deprecated spot_motives_eigen twin (bit-for-bit: the twin delegates),
    // and its ids live in the node space of this Laplacian — the F×F
    // bootstrap over feature dimensions, NOT items (issue #165).
    let (aspace, gl) = eigen_fixture_165();
    let cfg = cfg_165();

    let now = gl
        .try_spot_motives_featurespace(&cfg)
        .expect("featurespace spotting operates on the Laplacian as given");
    #[allow(deprecated)]
    let legacy = gl.spot_motives_eigen(&cfg);

    let (rows, _) = gl.shape();
    assert!(rows > aspace.nitems, "fixture must keep F > N");

    for motifs in [&now, &legacy] {
        assert!(!motifs.is_empty(), "featurespace path found no motifs");
        for m in motifs {
            assert!(m.len() >= 3, "motif below minimum size: {m:?}");
            assert!(
                m.windows(2).all(|w| w[0] < w[1]),
                "motif not sorted-unique: {m:?}"
            );
            for &i in m {
                assert!(
                    i < rows,
                    "feature-space id {i} leaves matrix node space ({rows}) in {m:?}"
                );
            }
        }
    }

    // Feature-space and item-space namespaces genuinely differ on this
    // fixture: some featurespace id must be outside item space, proving the
    // #165 confusion is possible here and that the two methods are distinct.
    let max_feature_id = now.iter().flatten().copied().max().unwrap_or(0);
    assert!(
        max_feature_id >= aspace.nitems,
        "featurespace motifs unexpectedly fit in item space (max id {max_feature_id})"
    );
}

#[test]
fn test_try_spot_motives_eigen_rejects_energy_build() {
    use crate::error::ArrowSpaceError;

    crate::tests::init();

    // Mirror of the #161 enforcement in the other direction: the eigen
    // item-space entry point refuses EnergyMaps builds (which already have
    // try_spot_motives_energy with finer subcentroid resolution).
    let rows = make_gaussian_cliques(12, 0.04, 12, 10, 3407);
    let p = crate::maps::energymaps::EnergyParams::new(&ArrowSpaceBuilder::new());
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_seed(3407)
        .with_lambda_graph(0.35, 18, 10, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build_energy(rows, p);

    let err = gl
        .try_spot_motives_eigen(&aspace, &MotiveConfig::default())
        .expect_err("energy builds must be routed to try_spot_motives_energy");
    assert!(
        matches!(err, ArrowSpaceError::EigenModeRequired { .. }),
        "expected EigenModeRequired, got: {err}"
    );
}

#[test]
fn test_try_spot_motives_eigen_requires_cluster_bookkeeping() {
    use crate::error::ArrowSpaceError;

    crate::tests::init();

    // Without the item→cluster bookkeeping there is no safe projection to
    // item space; the call must refuse instead of serving feature-space ids
    // (#161 lesson applied to the eigen track).
    let (mut aspace, gl) = eigen_fixture_165();
    aspace.cluster_assignments = Vec::new();

    let err = gl
        .try_spot_motives_eigen(&aspace, &cfg_165())
        .expect_err("missing cluster_assignments must be rejected, not degraded");
    assert!(
        matches!(err, ArrowSpaceError::EigenModeRequired { .. }),
        "expected EigenModeRequired, got: {err}"
    );
}

#[test]
fn test_try_spot_motives_eigen_rejects_unassigned_items() {
    use crate::error::ArrowSpaceError;

    crate::tests::init();

    // A None assignment (outlier) would silently drop that item from the
    // centroid sums and from the cluster→items projection, yielding motifs
    // that are not unions of whole clusters. The call must refuse (review
    // on PR #166), not produce a partial item-space projection.
    let (mut aspace, gl) = eigen_fixture_165();
    aspace.cluster_assignments[0] = None;

    let err = gl
        .try_spot_motives_eigen(&aspace, &cfg_165())
        .expect_err("unassigned items must be rejected, not silently dropped");
    assert!(
        matches!(err, ArrowSpaceError::EigenModeRequired { .. }),
        "expected EigenModeRequired, got: {err}"
    );
}

#[test]
fn test_try_spot_motives_eigen_rejects_out_of_range_assignments() {
    use crate::error::ArrowSpaceError;

    crate::tests::init();

    // An assignment beyond n_clusters has no centroid to accumulate into and
    // no bucket in the cluster→items map; the projection would be partial.
    // The call must refuse instead of degrading (review on PR #166).
    let (mut aspace, gl) = eigen_fixture_165();
    aspace.cluster_assignments[0] = Some(aspace.n_clusters + 5);

    let err = gl
        .try_spot_motives_eigen(&aspace, &cfg_165())
        .expect_err("out-of-range cluster ids must be rejected, not silently dropped");
    assert!(
        matches!(err, ArrowSpaceError::EigenModeRequired { .. }),
        "expected EigenModeRequired, got: {err}"
    );
}

#[test]
fn test_try_spot_motives_eigen_deterministic() {
    crate::tests::init();

    // Design invariant #4: identical inputs → identical outputs.
    let run = || {
        let (aspace, gl) = eigen_fixture_165();
        gl.try_spot_motives_eigen(&aspace, &cfg_165())
            .expect("pipeline EigenMaps build must satisfy the item-space requirements")
    };

    let first = run();
    let second = run();
    assert!(!first.is_empty());
    assert_eq!(
        first, second,
        "item-space eigen motifs differ across identical runs"
    );
}

#[test]
fn test_motives_eigen_deterministic() {
    crate::tests::init();

    // EigenMaps contract: identical inputs must yield identical motifs
    // (design invariant #4).
    let rows = make_gaussian_cliques(12, 0.04, 12, 10, 3407);

    let build = || {
        let (_, gl_eig) = ArrowSpaceBuilder::new()
            .with_seed(3407)
            .with_lambda_graph(0.4, 14, 8, 2.0, None)
            .with_sparsity_check(false)
            .with_dims_reduction(true, Some(0.3))
            .with_inline_sampling(None)
            .build(rows.clone());

        let cfg = MotiveConfig {
            top_l: 16,
            min_triangles: 2,
            min_clust: 0.35,
            max_motif_size: 24,
            max_sets: 64,
            jaccard_dedup: 0.8,
        };
        gl_eig.spot_motives_eigen(&cfg)
    };

    let first = build();
    assert!(!first.is_empty(), "eigen pipeline returned no motifs");

    let second = build();
    assert_eq!(first.len(), second.len(), "motif count differs across runs");
    for (a, b) in first.iter().zip(second.iter()) {
        let mut sa = a.clone();
        let mut sb = b.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        assert_eq!(sa, sb, "motif contents differ across identical runs");
    }
}
