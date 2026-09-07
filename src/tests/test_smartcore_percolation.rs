//! Pins the behaviour arrowspace inherits from the pinned smartcore
//! CosinePair implementation (crates.io 0.6.11, commit 6e285b3) and the
//! eps / topk semantics of the lambda-graph builder.
//!
//! Contract notes:
//! - `build_laplacian_matrix` takes a TRANSPOSED matrix (F x N): graph
//!   nodes are FEATURES, columns are observations. All edge assertions
//!   below therefore speak of feature profiles, not items.
//! - smartcore 0.6.11 made `CosinePair::query_row_top_k` exact; distances
//!   are unrectified `1 - cos` in [0, 2]. No cap at 1.0 exists.

use crate::builder::ArrowSpaceBuilder;
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::tests::init;
use smartcore::algorithm::neighbour::cosinepair::CosinePair;
use smartcore::linalg::basic::matrix::DenseMatrix;

fn params(eps: f64, k: usize, topk: usize) -> GraphParams {
    GraphParams {
        eps,
        k,
        topk,
        p: 2.0,
        sigma: Some(0.5),
        normalise: false,
        sparsity_check: false,
    }
}

/// Build a Laplacian directly from feature profiles (rows = features).
fn laplacian_from_features(profiles: &Vec<Vec<f64>>, p: GraphParams) -> GraphLaplacian {
    let n_obs = profiles.first().map(|r| r.len()).unwrap_or(0);
    let m = DenseMatrix::<f64>::from_2d_vec(profiles).unwrap();
    build_laplacian_matrix(m, &p, Some(n_obs), false)
}

/// True when an off-diagonal edge exists between distinct nodes i and j.
/// L = D - A, so an edge shows up as a negative off-diagonal entry.
fn has_edge(gl: &GraphLaplacian, i: usize, j: usize) -> bool {
    assert_ne!(i, j);
    gl.get(i, j) < 0.0
}

fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return f64::MAX;
    }
    1.0 - dot / (na * nb)
}

// ---------------------------------------------------------------------------
// 1. CosinePair exactness (the 0.6.11 fix that percolates into graph builds)
// ---------------------------------------------------------------------------

/// Twelve feature profiles in four near-duplicate consecutive trios. Any
/// fixed-stride candidate sampler (the pre-0.6.11 query strategy) can reach
/// at most one sibling of a trio besides the query row itself, so an exact
/// result must contain BOTH siblings at near-zero distance.
#[test]
fn cosinepair_knn_finds_both_trio_siblings_exactly() {
    init();

    let dirs: [Vec<f64>; 4] = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(12);
    for d in &dirs {
        for m in 0..3usize {
            let mut v = d.clone();
            v[m % 3] += 0.01 * m as f64; // tiny intra-trio spread, no ties
            rows.push(v);
        }
    }

    let dm = DenseMatrix::<f64>::from_2d_vec(&rows).unwrap();
    let fastpair = CosinePair::with_top_k(&dm, 4).unwrap();

    let k = 4; // mirrors arrowspace: query with params.topk + 1
    for i in 0..12usize {
        let trio = i / 3 * 3;
        let off = i - trio;
        let sib_a = trio + (off + 1) % 3;
        let sib_b = trio + (off + 2) % 3;

        let hits = fastpair.query_row_top_k(i, k).unwrap();
        let neighbours: Vec<(usize, f64)> = hits
            .iter()
            .filter_map(|(d, j)| if *j != i { Some((*j, *d)) } else { None })
            .collect();

        let found_sibs = neighbours
            .iter()
            .filter(|(j, _)| *j == sib_a || *j == sib_b)
            .count();
        assert_eq!(
            found_sibs,
            2,
            "row {}: exact kNN must contain both siblings {:?}, got {:?}",
            i,
            (sib_a, sib_b),
            neighbours
        );
        for (j, d) in &neighbours {
            if *j == sib_a || *j == sib_b {
                let brute = cosine_distance(&rows[i], &rows[*j]);
                assert!(
                    (d - brute).abs() < 1e-9,
                    "distance mismatch row {} -> {}: {} vs {}",
                    i,
                    j,
                    d,
                    brute
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. eps semantics over feature nodes: uncapped 1 - cos, NOT capped at 1.0
// ---------------------------------------------------------------------------

/// Feature profiles observed over four items. With eps = 1.2 (> 1.0):
/// f1 is anti-parallel to f0 (distance 2.0) and must stay disconnected.
/// Under a rectified/capped distance it would read 1.0 <= eps and connect.
#[test]
fn eps_above_one_does_not_collapse_antiparallel_pair_into_range() {
    init();

    //           item0  item1  item2  item3
    let f0 = vec![1.0, 1.0, 1.0, 1.0]; // baseline profile
    let f1 = vec![-1.0, -1.0, -1.0, -1.0]; // cos(f0,f1) = -1, dist = 2.0
    let f2 = vec![1.0, 1.0, 1.0, 0.0]; // close to f0

    // sanity on geometry before building
    assert!((cosine_distance(&f0, &f1) - 2.0).abs() < 1e-9);
    assert!(cosine_distance(&f0, &f2) < 0.2);

    let gl = laplacian_from_features(&vec![f0, f1, f2], params(1.2, 3, 3));

    assert!(
        !has_edge(&gl, 0, 1),
        "anti-parallel pair connected at eps=1.2: distance was capped at 1.0"
    );
    assert!(has_edge(&gl, 0, 2), "positive control edge f0-f2 missing");
}

/// Same anti-parallel pair, eps past 2.0: the unrectified range reaches 2.0,
/// so the edge appears. This contradicts the stale doc claim in this file's
/// neighbourhood that negative-cosine items are 'effectively disconnected'.
#[test]
fn antiparallel_pair_connects_when_eps_exceeds_two() {
    init();

    let f0 = vec![1.0, 1.0, 1.0];
    let f1 = vec![-1.0, -1.0, -1.0];

    let gl = laplacian_from_features(&vec![f0, f1], params(2.5, 2, 2));

    assert!(
        has_edge(&gl, 0, 1),
        "anti-parallel pair must connect at eps=2.5 (dist 2.0); dead zone above 1.0 would forbid it"
    );
}

// ---------------------------------------------------------------------------
// 3. explicit topk vs define_result_k
// ---------------------------------------------------------------------------

/// The user configures topk=3 with k=7. The old define_result_k silently
/// rewrote it to 4 because 6 <= k < 10. Explicit configuration must win.
#[test]
fn explicit_topk_survives_build_with_small_k() {
    init();

    let rows: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![(i % 5) as f64, (i / 5) as f64, 1.0])
        .collect();

    let (_, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.9, 7, 3, 2.0, None)
        .build(rows);

    assert_eq!(
        gl.topk(),
        3,
        "explicit topk=3 was overridden by define_result_k"
    );
}
