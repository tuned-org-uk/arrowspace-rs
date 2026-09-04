//! Layout contract for `DenseMatrix` construction from flat buffers (issue
//! #167): `DenseMatrix::from_iterator(iter, nrows, ncols, axis)` maps
//! `axis != 0` to `column_major = true` and adopts the buffer as-is, so
//! **row-major** buffers must be built with `axis = 0`. Feeding a row-major
//! buffer with `axis = 1` scrambles every `get((r, c))` into a
//! transposed-mixture read.
//!
//! These tests pin the contract at the crate's call sites: the sparse→dense
//! bridge, the clustered-centroid matrix handed to the eigen bootstrap, and
//! the K-Means heuristic input.

use crate::graph::sparse_to_dense;
use smartcore::linalg::basic::arrays::{Array, Array2};
use sprs::CsMat;

#[test]
fn test_sparse_to_dense_preserves_element_positions() {
    // Non-symmetric 3×3 on purpose: the pre-fix implementation reinterpreted
    // the row-major buffer as column-major, which is a correct read only for
    // symmetric inputs — exactly why this bug survived the suite.
    let mut tm = sprs::TriMat::new((3, 3));
    tm.add_triplet(0, 1, 2.5);
    tm.add_triplet(1, 0, 7.0);
    tm.add_triplet(1, 2, -4.0);
    tm.add_triplet(2, 2, 1.0);
    let sparse: CsMat<f64> = tm.to_csr();

    let dense = sparse_to_dense(&sparse);

    let expected = [[0.0, 2.5, 0.0], [7.0, 0.0, -4.0], [0.0, 0.0, 1.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert_eq!(
                dense.get((i, j)),
                v,
                "sparse_to_dense scrambled ({i},{j}): expected {v}"
            );
        }
    }
}

#[test]
fn test_kmeans_lloyd_assigns_by_true_proximity() {
    use crate::clustering::kmeans_lloyd;

    // Two far-apart tight groups in 4 dims: every partition that respects
    // true distances puts {0,1,2} together and {3,4,5} together.
    let rows = vec![
        vec![10.0, 0.0, 0.0, 0.0],
        vec![10.1, 0.1, 0.0, 0.0],
        vec![9.9, -0.1, 0.1, 0.0],
        vec![0.0, 10.0, 0.0, 0.0],
        vec![0.1, 10.1, 0.0, 0.0],
        vec![-0.1, 9.9, 0.1, 0.0],
    ];
    let assignments = kmeans_lloyd(&rows, 2, 20, 42);

    let a0 = assignments[0];
    for i in 0..3 {
        assert_eq!(
            assignments[i], a0,
            "row {i} split from its own group (assignments: {assignments:?})"
        );
    }
    for i in 3..6 {
        assert_ne!(
            assignments[i], a0,
            "row {i} merged into the other group (assignments: {assignments:?})"
        );
    }
}

#[test]
fn test_clustered_dm_holds_true_centroid_means() {
    use crate::builder::ArrowSpaceBuilder;

    crate::tests::init();

    // Three tight direction clusters (unit-norm, noise 0.002 per dim, well
    // separated). With manual k=3 / radius 0.01 each cluster is exactly one
    // group of points, so gl.init_data columns must equal the true cluster
    // means of aspace.data.
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(3407);
    let dirs: [Vec<f64>; 3] = [vec![1.0], vec![0.0, 1.0], vec![-1.0, -1.0]];
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(3 * 20);
    for d in &dirs {
        for _ in 0..20 {
            let mut v = vec![0.0f64; 32];
            v[0] = 10.0 * d[0];
            if d.len() > 1 {
                v[1] = 10.0 * d[1];
            }
            for x in v.iter_mut() {
                *x += Normal::new(0.0, 0.002).unwrap().sample(&mut rng);
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            rows.push(v.into_iter().map(|x| x / norm).collect());
        }
    }

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_seed(3407)
        .with_lambda_graph(0.5, 12, 6, 2.0, None)
        .with_cluster_max_clusters(3)
        .with_cluster_radius(0.01)
        .with_dims_reduction(false, None)
        .with_sparsity_check(false)
        .with_inline_sampling(None)
        .build(rows);

    assert_eq!(aspace.n_clusters, 3, "fixture must yield 3 clusters");
    assert!(
        aspace.cluster_assignments.iter().all(|a| a.is_some()),
        "fixture must assign every item"
    );

    let (d_rows, d_cols) = gl.init_data.shape();
    assert_eq!((d_cols, d_rows), (3, aspace.nfeatures));

    // True means from the index's own rows + assignments
    let mut means = vec![vec![0.0f64; aspace.nfeatures]; 3];
    let mut counts = vec![0usize; 3];
    for (it, a) in aspace.cluster_assignments.iter().enumerate() {
        let c = a.expect("checked above");
        counts[c] += 1;
        for (k, x) in aspace.data.get_row(it).iterator(0).enumerate() {
            means[c][k] += x;
        }
    }
    for (c, m) in means.iter_mut().enumerate() {
        for x in m.iter_mut() {
            *x /= counts[c] as f64;
        }
    }

    for c in 0..3 {
        // init_data is stored F×X (rows = features, cols = centroids)
        let (mut dot, mut ni, mut nj) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..aspace.nfeatures {
            let a = *gl.init_data.get((k, c));
            let b = means[c][k];
            dot += a * b;
            ni += a * a;
            nj += b * b;
        }
        let cos = dot / (ni.sqrt() * nj.sqrt());
        assert!(
            cos > 0.99,
            "init_data column {c} is not centroid {c}'s mean (cos={cos:.4}) — \
             clustered_dm layout is scrambled (issue #167)"
        );
    }
}

#[test]
fn test_lambda_golden_after_167_layout_fix() {
    use crate::builder::ArrowSpaceBuilder;

    crate::tests::init();

    // Post-#167 golden: with the clustered_dm layout corrected, the eigen
    // bootstrap graph is built from true centroid coordinates and the λ
    // read-out lands on these exact values for this fixture. Any future
    // change to graph construction or the λ read-out must consciously
    // update this pin (see issue #167 for the before/after policy).
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(3407);
    let dirs: [Vec<f64>; 3] = [vec![1.0], vec![0.0, 1.0], vec![-1.0, -1.0]];
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(60);
    for d in &dirs {
        for _ in 0..20 {
            let mut v = vec![0.0f64; 32];
            v[0] = 10.0 * d[0];
            if d.len() > 1 {
                v[1] = 10.0 * d[1];
            }
            for x in v.iter_mut() {
                *x += Normal::new(0.0, 0.002).unwrap().sample(&mut rng);
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            rows.push(v.into_iter().map(|x| x / norm).collect());
        }
    }
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_seed(3407)
        .with_lambda_graph(0.5, 12, 6, 2.0, None)
        .with_cluster_max_clusters(3)
        .with_cluster_radius(0.01)
        .with_dims_reduction(false, None)
        .with_sparsity_check(false)
        .with_inline_sampling(None)
        .build(rows);

    let lmin = aspace.lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
    let lmax = aspace
        .lambdas
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let lmean = aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64;

    let tol = 1e-12;
    assert!((lmin - 0.0).abs() < tol, "golden λ min drifted: {lmin}");
    assert!((lmax - 1.0).abs() < tol, "golden λ max drifted: {lmax}");
    assert!(
        (lmean - 0.44232788702618059).abs() < tol,
        "golden λ mean drifted: {lmean:.17}"
    );
}
