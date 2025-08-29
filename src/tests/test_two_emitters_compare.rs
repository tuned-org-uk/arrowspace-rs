mod tests {
    use crate::core::ArrowSpace;
    use crate::dimensional::{ArrowDimensionalOps, DimensionalOps};
    use crate::operators::{build_knn_graph, rayleigh_lambda};

    /// Compare results of running the same superposition of signal with
    /// new `ArrowSpace`. They shall return the same results
    /// when computing Raighley.
    #[test]
    fn test_compare_value() {
        let length = 3usize.pow(5);
        let c1 = DimensionalOps::make_cantor_1d(4, 1.0 / 3.0, length);
        let height = 48usize;
        let support = DimensionalOps::make_product_support(&c1, height);
        let n = support.len();
        println!("Support size: {}", n);

        let gl = build_knn_graph(&support, 8, 1.0, None);

        let src_a = (length as f64 * 0.3, height as f64 * 0.5);
        let src_b = (length as f64 * 0.7, height as f64 * 0.5);
        let alpha = 0.05;
        let eps = 1e-6;

        let mk = |src: (f64, f64)| -> Vec<f64> {
            support
                .iter()
                .map(|&(r, c)| {
                    let d = {
                        let dr = src.0 - r as f64;
                        let dc = src.1 - c as f64;
                        (dr * dr + dc * dc).sqrt()
                    };
                    (-alpha * d).exp() / (d + eps)
                })
                .collect()
        };

        // Arrow workflow
        let row_a = mk(src_a);
        let row_b = mk(src_b);
        let mut aspace = ArrowSpace::from_items(vec![row_a.clone(), row_b.clone()], vec![0.0, 0.0]);
        aspace.recompute_lambdas(&gl);

        // Superposition compare
        let mut row_ab = row_a.clone();
        for i in 0..n {
            row_ab[i] += row_b[i];
        }
        let lam_dense_ab = rayleigh_lambda(&gl, &row_ab);

        aspace. add_features(0, 1, &gl);
        let lam_arrow_ab = aspace.lambdas()[0];
        println!(
            "Arrow lambda(A+B)={:.6} vs Dense lambda(A+B)={:.6}",
            lam_arrow_ab, lam_dense_ab
        );
        assert!(lam_arrow_ab.to_bits() == lam_dense_ab.to_bits());
    }

    /// Compare results of running the same superposition of signal with
    /// new `ArrowSpace` and `DenseMatrix`.
    #[test]
    fn test_compare_perf() {
        use std::time::Instant;

        // instantiate a arrow space
        let length = 3usize.pow(5);
        let c1 = DimensionalOps::make_cantor_1d(4, 1.0 / 3.0, length);
        let height = 48usize;
        let support = DimensionalOps::make_product_support(&c1, height);
        let n = support.len();
        println!("Support size: {}", n);

        let gl = build_knn_graph(&support, 8, 1.0, None);

        let src_a = (length as f64 * 0.3, height as f64 * 0.5);
        let src_b = (length as f64 * 0.7, height as f64 * 0.5);
        let alpha = 0.05;
        let eps = 1e-6;

        let mk = |src: (f64, f64)| -> Vec<f64> {
            support
                .iter()
                .map(|&(r, c)| {
                    let d = {
                        let dr = src.0 - r as f64;
                        let dc = src.1 - c as f64;
                        (dr * dr + dc * dc).sqrt()
                    };
                    (-alpha * d).exp() / (d + eps)
                })
                .collect()
        };

        let row_a = mk(src_a);
        let row_b = mk(src_b);

        // Arrow workflow
        let mut aspace = ArrowSpace::from_items(vec![row_a.clone(), row_b.clone()], vec![0.0, 0.0]);
        aspace.recompute_lambdas(&gl);
        let now = Instant::now();
        {
            aspace. add_features(0, 1, &gl);
        }
        let elapsed = now.elapsed();
        println!("Elapsed Arrow: {:.2?}", elapsed);

    }
}