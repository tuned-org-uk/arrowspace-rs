use crate::graph_factory::{GraphFactory};
use crate::dimensional::{DimensionalOps, ArrowDimensionalOps};
use crate::core::ArrowSpace;


// pressure kernel proxy on integer grid support: K(d) = exp(-alpha * d)/(d+eps)
#[cfg(test)]
fn emitter_row(edges: &[(usize, usize)], src: (f64, f64), alpha: f64) -> Vec<f64> {
    let eps = 1e-6;
    edges
        .iter()
        .map(|&(r, c)| {
            let dr = src.0 - r as f64;
            let dc = src.1 - c as f64;
            let d = (dr * dr + dc * dc).sqrt();
            (-alpha * d).exp() / (d + eps)
        })
        .collect()
}

#[test]
fn two_emitters_superposition_lambda() {
    // Build the discrete support (Cantor x height)
    let length = 3usize.pow(5); // 243
    let c1 = DimensionalOps::make_cantor_1d(4, 1.0 / 3.0, length);
    let height = 24usize;
    let support = DimensionalOps::make_product_support(&c1, height);

    // Two emitter sources on that support
    let src_a = (length as f64 * 0.3, height as f64 * 0.5);
    let src_b = (length as f64 * 0.7, height as f64 * 0.5);

    // Build two signal rows over the same set of items (support)
    let row_a = emitter_row(&support, src_a, 0.06);
    let row_b = emitter_row(&support, src_b, 0.06);

    // ArrowSpace from rows (rows are features over items)
    let mut aspace_sum = ArrowSpace::from_items(vec![row_a.clone(), row_b.clone()]);
    let mut aspace_mul = ArrowSpace::from_items(vec![row_a, row_b]);

    // Build λτ-graph directly from the data matrix (rows are features over items)
    // Reconstruct a matrix view in row-major (feature-major) shape F×N for the factory API
    let (_, nitems) = aspace_sum.shape();
    let mut data_matrix: Vec<Vec<f64>> = Vec::with_capacity(nitems);
    for r in 0..nitems {
        // loop cols number as ArrowSpace is column-major
        data_matrix.push(aspace_sum.get_item(r).item.to_vec());
    }
    assert_eq!(data_matrix.len(), 2);
    assert_eq!(data_matrix[0].len(), 1152);

    let eps = 1e-3; // λ proximity threshold
    let k = 8usize; // cap neighbors per item
    let p = 2.0; // kernel exponent
    let sigma = None; // default σ=eps
    let gl = GraphFactory::build_lambda_graph(&data_matrix, eps, k, p, sigma);

    println!("{gl:?}");

    // Compute lambdas for both spaces
    aspace_sum.recompute_lambdas(&gl);
    aspace_mul.recompute_lambdas(&gl);
    assert!(aspace_sum.lambdas().iter().all(|&l| l >= 0.0));

    println!("aspace_sum {:?}", &aspace_sum.lambdas());

    // Superpose b into a (item-wise add) and recompute λ
    aspace_sum.add_items(0, 1, &gl);

    println!("added {:?}", &aspace_sum.lambdas());

    // Sanity bound: superposed lambda remains finite and non-negative
    assert!(aspace_sum.lambdas().iter().all(|&l| l.is_finite()));

    // Multiply a times b (item-wise mul) and recompute λ
    aspace_mul.mul_items(0, 1, &gl);

    println!("multiplied {:?}", &aspace_mul.lambdas());

    assert!(aspace_mul.lambdas().iter().all(|&l| l.is_finite()));
    assert!(aspace_mul.lambdas().iter().all(|&l| l >= 0.0));
}
