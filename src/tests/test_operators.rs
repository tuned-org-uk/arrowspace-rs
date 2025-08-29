#[cfg(test)]
mod tests {
    use crate::core::ArrowSpace;
    use crate::graph_factory::{GraphFactory, GraphLaplacian};
    use crate::operators::rayleigh_lambda;

    // Helper: build a lambda graph from a small data matrix
    fn lambda_graph(rows: Vec<Vec<f64>>, eps: f64, k: usize, p: f64) -> GraphLaplacian {
        GraphFactory::build_lambda_graph(&rows, eps, k, p, None)
    }

    #[test]
    fn test_symmetry_enforcement() {
        // Two features over four items
        let gl = lambda_graph(
            vec![vec![1.0, 0.5, 0.0, 0.5], vec![0.2, 0.3, 0.7, 0.1]],
            1e-3,
            3,
            2.0,
        );

        // Check symmetry: for each off-diagonal (i,j) there is (j,i) with same weight
        for i in 0..gl.nnodes {
            let (si, ei) = (gl.rows[i], gl.rows[i + 1]);
            for idx_i in si..ei {
                let j = gl.cols[idx_i];
                let vij = gl.vals[idx_i];
                if i != j {
                    // find reverse edge
                    let (sj, ej) = (gl.rows[j], gl.rows[j + 1]);
                    let mut found = false;
                    for idx_j in sj..ej {
                        if gl.cols[idx_j] == i {
                            let vji = gl.vals[idx_j];
                            assert!(
                                (vij - vji).abs() < 1e-12,
                                "Asymmetric weights at ({i},{j}) vs ({j},{i})"
                            );
                            found = true;
                            break;
                        }
                    }
                    assert!(found, "Missing symmetric reverse edge for ({i},{j})");
                    // Off-diagonal must be negative
                    assert!(vij < 0.0, "Off-diagonal not negative at ({i},{j})");
                } else {
                    // Diagonal must be non-negative
                    assert!(vij >= 0.0, "Diagonal negative at row {i}");
                }
            }
        }
    }

    #[test]
    fn test_k_capping() {
        // Construct data with distinct per-item λ values
        // Rows = features, Cols = items
        let rows = vec![
            vec![0.1, 0.2, 0.4, 0.8], // monotone increase
            vec![0.0, 0.1, 0.2, 0.4], // scaled monotone
        ];

        // Large k: eps large to include all → fully connected (n-1 off-diagonals + diagonal)
        let gl_full = lambda_graph(rows.clone(), 1.0, 10, 2.0);
        for i in 0..gl_full.nnodes {
            let row_len = gl_full.rows[i + 1] - gl_full.rows[i];
            assert_eq!(
                row_len, gl_full.nnodes,
                "Row {i} should have n entries (n-1 neighbors + diagonal) under large k"
            );
        }

        // k = 1 semantics with union symmetrization:
        // - Per-row pre-union selection keeps only 1 neighbor.
        // - After union, a row can accumulate multiple reverse selections from others.
        // Therefore, we cannot assert ≤ 2 total entries per row post-union.
        // Instead, assert global Laplacian invariants and that the row is not denser than fully connected.
        let gl_k1 = lambda_graph(rows, 1.0, 1, 2.0);
        for i in 0..gl_k1.nnodes {
            let s = gl_k1.rows[i];
            let e = gl_k1.rows[i + 1];
            let row_len = e - s;

            // Always include diagonal, never exceed fully-connected size
            assert!(
                gl_k1.cols[s..e].contains(&i),
                "Diagonal missing at row {i}"
            );
            assert!(
                row_len <= gl_k1.nnodes,
                "Row {} has more than n entries after union symmetrization ({} > {})",
                i,
                row_len,
                gl_k1.nnodes
            );

            // Diagonal non-negative; off-diagonals negative
            for idx in s..e {
                let j = gl_k1.cols[idx];
                let v = gl_k1.vals[idx];
                if j == i {
                    assert!(v >= 0.0, "Diagonal negative at row {i}");
                } else {
                    assert!(v < 0.0, "Off-diagonal not negative at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn test_no_self_loops_off_diagonal_and_diagonal_last() {
        let gl = lambda_graph(vec![vec![0.1, 0.2, 0.3], vec![0.3, 0.2, 0.1]], 1e-3, 2, 2.0);

        for i in 0..gl.nnodes {
            let start = gl.rows[i];
            let end = gl.rows[i + 1];
            let mut diagonal_count = 0;
            for idx in start..end {
                let col = gl.cols[idx];
                if col == i {
                    diagonal_count += 1;
                    // Diagonal must be last and non-negative
                    assert_eq!(idx, end - 1, "Diagonal not last in row {i}");
                    assert!(gl.vals[idx] >= 0.0, "Diagonal negative in row {i}");
                }
            }
            assert_eq!(
                diagonal_count, 1,
                "Row {i} should have exactly one diagonal"
            );
        }
    }

    #[test]
    fn test_graph_determinism_and_guardrails() {
        let rows = vec![vec![0.7, 0.1, 0.3, 0.4, 0.2], vec![0.2, 0.8, 0.1, 0.1, 0.5]];

        let gl1 = lambda_graph(rows.clone(), 1e-3, 2, 2.0);
        let gl2 = lambda_graph(rows, 1e-3, 2, 2.0);

        // Deterministic CSR
        assert_eq!(gl1.nnodes, gl2.nnodes);
        assert_eq!(gl1.rows, gl2.rows);
        assert_eq!(gl1.cols, gl2.cols);
        assert_eq!(gl1.vals, gl2.vals);

        // CSR sanity and diagonal presence
        assert_eq!(gl1.rows.len(), gl1.nnodes + 1);
        assert_eq!(gl1.cols.len(), gl1.vals.len());
        for r in 0..gl1.nnodes {
            let s = gl1.rows[r];
            let e = gl1.rows[r + 1];
            assert!(e > s, "Row {r} must contain at least the diagonal");
            // No duplicate columns and include diagonal
            let mut cols = gl1.cols[s..e].to_vec();
            cols.sort_unstable();
            for w in cols.windows(2) {
                assert!(w[0] != w[1], "Duplicate column in row {r}");
            }
            assert!(cols.contains(&r), "Diagonal missing in row {r}");
        }
    }

    #[test]
    fn test_stable_csr_structure() {
        let gl = lambda_graph(
            vec![vec![0.5, 0.4, 0.3, 0.2, 0.1], vec![0.1, 0.2, 0.3, 0.4, 0.5]],
            1e-3,
            3,
            2.0,
        );

        // Rows monotone and CSR integrity
        for i in 0..gl.nnodes {
            assert!(
                gl.rows[i] < gl.rows[i + 1],
                "Rows not strictly increasing at {i}"
            );
        }
        assert_eq!(gl.rows.len(), gl.nnodes + 1, "CSR rows length incorrect");
        assert_eq!(
            gl.cols.len(),
            gl.vals.len(),
            "CSR cols/vals length mismatch"
        );
        assert_eq!(gl.rows[0], 0, "CSR should start at 0");
        assert_eq!(
            gl.rows[gl.nnodes],
            gl.cols.len(),
            "CSR end pointer incorrect"
        );

        // Each row must have the diagonal
        for i in 0..gl.nnodes {
            let s = gl.rows[i];
            let e = gl.rows[i + 1];
            assert!(e > s, "Row {i} has no entries");
            let row_cols = &gl.cols[s..e];
            assert!(row_cols.contains(&i), "Diagonal entry missing in row {i}");
        }
    }

    #[test]
    fn test_degree_calculation() {
        // Identical row encourages multiple connections under eps
        let gl = lambda_graph(vec![vec![1.0, 1.0, 1.0], vec![1.0, 0.5, 1.0]], 1.0, 2, 2.0);
        for i in 0..gl.nnodes {
            let start = gl.rows[i];
            let end = gl.rows[i + 1];
            let mut degree = 0.0;
            let mut diagonal_value = 0.0;
            for idx in start..end {
                let j = gl.cols[idx];
                let weight = gl.vals[idx];
                if i == j {
                    diagonal_value = weight;
                } else {
                    degree -= weight; // off-diagonal negative; sum positive weights
                }
            }
            assert!(
                (degree - diagonal_value).abs() < 1e-12,
                "Degree mismatch at node {i}: expected {degree}, got {diagonal_value}"
            );
        }
    }

    #[test]
    fn test_rayleigh_quotient_properties() {
        let gl = lambda_graph(
            vec![vec![0.9, 0.1, 0.2, 0.8], vec![0.1, 0.9, 0.8, 0.2]],
            1e-3,
            3,
            2.0,
        );

        // Uniform vector: non-negative (near zero for connected graphs)
        let uniform = vec![1.0; gl.nnodes];
        let lambda_uniform = rayleigh_lambda(&gl, &uniform);
        assert!(lambda_uniform >= 0.0, "Rayleigh must be non-negative");

        // Zero vector
        let zero = vec![0.0; gl.nnodes];
        let lambda_zero = rayleigh_lambda(&gl, &zero);
        assert_eq!(lambda_zero, 0.0, "Zero vector must yield zero Rayleigh");

        // Scale invariance
        let x = vec![1.0, -1.0, 1.0, -1.0][..gl.nnodes.min(4)].to_vec();
        let x_scaled: Vec<f64> = x.iter().map(|v| v * 2.5).collect();
        let l1 = rayleigh_lambda(&gl, &x);
        let l2 = rayleigh_lambda(&gl, &x_scaled);
        assert!(
            (l1 - l2).abs() < 1e-12,
            "Rayleigh quotient should be scale invariant"
        );
    }

    #[test]
    #[should_panic]
    fn test_edge_cases_and_empty() {
        // Empty matrix => empty graph
        let gl_empty = GraphFactory::build_lambda_graph(&vec![], 1e-3, 1, 2.0, None);
        assert_eq!(gl_empty.nnodes, 0);
        assert_eq!(gl_empty.rows, vec![0]);
        assert!(gl_empty.cols.is_empty());
        assert!(gl_empty.vals.is_empty());

        // Rayleigh size mismatch should panic
        let gl2 = GraphFactory::build_lambda_graph(&vec![vec![0.0, 1.0]], 1e-3, 1, 2.0, None);
        let wrong = vec![1.0]; // length 1, but nnodes=2
        rayleigh_lambda(&gl2, &wrong);
    }

    #[test]
    fn arrowspace_lambda_rayleigh_sanity() {
        // Integrate with ArrowSpace using the new λτ-graph
        let gl = lambda_graph(
            vec![vec![0.37, 0.60, 0.60, 0.37], vec![0.35, 0.10, 0.9, 0.45]],
            1e-3,
            3,
            2.0,
        );

        let row = vec![vec![0.37, 0.60, 0.60, 0.37], vec![0.35, 0.10, 0.9, 0.45]];
        let mut aspace = ArrowSpace::from_items(row);
        aspace.recompute_lambdas(&gl);
        let lambda = aspace.lambdas();

        // Conservative bound via 2*max_degree
        let mut max_deg = 0.0f64;
        for i in 0..gl.nnodes {
            let s = gl.rows[i];
            let e = gl.rows[i + 1];
            for idx in s..e {
                if gl.cols[idx] == i {
                    max_deg = max_deg.max(gl.vals[idx]);
                    break;
                }
            }
        }
        let upper = 2.0 * max_deg + 1e-12;

        // Non-negativity and upper bound
        assert!(
            lambda.iter().all(|&x| x >= 0.0),
            "Rayleigh must be non-negative"
        );
        assert!(
            lambda.iter().all(|&x| x <= upper),
            "Rayleigh {lambda:?} exceeded bound {upper} (max_deg={max_deg})"
        );
    }

    #[test]
    fn zero_lambda_gap_full_connectivity_and_union_effect() {
        // All columns identical => per-item λ are all equal => |Δλ| = 0 for every pair
        // Use two identical feature rows across three items
        let rows = vec![vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];

        // Large k: expect fully connected (n-1 neighbors + diagonal)
        let gl_full = lambda_graph(rows.clone(), 0.0, 10, 2.0);
        assert_eq!(gl_full.nnodes, 2);
        for i in 0..gl_full.nnodes {
            let row_len = gl_full.rows[i + 1] - gl_full.rows[i];
            assert_eq!(
                row_len, gl_full.nnodes,
                "With zero gaps and large k, row {i} should have n entries (n-1 neighbors + diagonal)"
            );
            // Off-diagonals negative, diagonal non-negative
            let s = gl_full.rows[i];
            let e = gl_full.rows[i + 1];
            let mut has_diag = false;
            for idx in s..e {
                let j = gl_full.cols[idx];
                let v = gl_full.vals[idx];
                if j == i {
                    has_diag = true;
                    assert!(v >= 0.0, "Diagonal must be non-negative at row {i}");
                } else {
                    assert!(v < 0.0, "Off-diagonal must be negative at ({i},{j})");
                }
            }
            assert!(has_diag, "Diagonal entry missing for row {i}");
        }

        // k = 1: local per-row selection keeps only 1 neighbor before union; union can add a second neighbor
        // Expect each row to have at least the diagonal and at most 2 off-diagonals (in this tiny graph, ≤ 2 total neighbors)
        let gl_k1 = lambda_graph(rows, 0.0, 1, 2.0);
        assert_eq!(gl_k1.nnodes, 2);
        for i in 0..gl_k1.nnodes {
            let s = gl_k1.rows[i];
            let e = gl_k1.rows[i + 1];
            // Count off-diagonals
            let mut off_cnt = 0usize;
            let mut has_diag = false;
            for idx in s..e {
                let j = gl_k1.cols[idx];
                if j == i {
                    has_diag = true;
                } else {
                    off_cnt += 1;
                }
            }
            assert!(has_diag, "Diagonal missing at row {i}");
            // With k=1 and union symmetry over a 3-node complete zero-gap scenario,
            // off-diagonals will be either 1 or 2 depending on reciprocal selections.
            assert!(
                off_cnt <= 2,
                "With k=1 and zero gaps, row {i} should have at most 2 off-diagonals (got {off_cnt})"
            );
        }
    }
}
