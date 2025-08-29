use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::ArrowItem;

fn main() {
    // =====================
    // 1. Setup data (3 items, 24 features)
    // =====================
    // generate more varied signals:
    let a: Vec<f64> = (0..24).map(|i| (i as f64 * 0.3).sin()).collect();
    let b: Vec<f64> = (0..24).map(|i| (i as f64 * 0.3).cos()).collect();
    let c: Vec<f64> = (0..24)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    // Build ArrowSpace (col-major internally) and lambda-graph
    let (mut aspace, gl) = ArrowSpaceBuilder::new()
        .with_rows(vec![a.clone(), b.clone(), c.clone()])
        .with_lambda_graph(1e-3, 3, 2.0, None)
        .build();

    println!("Initial lambdas: {:?}", aspace.lambdas());

    // =====================
    // 2. Zero-copy feature view
    // =====================
    {
        let f0 = aspace.iter_feature(0);
        println!(
            "feature 0 len={} (first values={:?})",
            f0.len(),
            &f0[0..3.min(f0.len())]
        );

        // Mutate feature row 1
        let mut f1_vals = aspace.get_feature_mut(1).data;
        for v in f1_vals.iter_mut() {
            *v *= 1.1;
        }
        aspace.recompute_lambdas(&gl);
        println!("After scaling row1, lambdas: {:?}", aspace.lambdas());
    }

    // =====================
    // 3. Owned ArrowItems
    // =====================
    let ar_a = aspace.get_item(0);
    let ar_b = aspace.get_item(1);
    let ar_c = aspace.get_item(2);

    // Arithmetic in-place
    let mut ar_sum = ar_a.clone();
    ar_sum.add_inplace(&ar_b);
    let mut ar_prod = ar_a.clone();
    ar_prod.mul_inplace(&ar_c);

    println!(
        "cos(sum,a)={:.4}, cos(prod,a)={:.4}",
        ar_sum.cosine_similarity(&ar_a),
        ar_prod.cosine_similarity(&ar_a)
    );

    println!(
        "λτ-aware: sum↔a={:.4}, prod↔a={:.4}",
        ar_sum.lambda_similarity(&ar_a, 0.8, 0.2),
        ar_prod.lambda_similarity(&ar_a, 0.8, 0.2)
    );

    // =====================
    // 4. In-place ops on ArrowSpace
    // =====================
    {
        let before = aspace.lambdas().to_vec();
        aspace.add_items(0, 1, &gl);
        aspace.mul_items(2, 0, &gl);
        let after = aspace.lambdas().to_vec();
        println!("lambdas before: {before:?}");
        println!(
            "lambdas after add(item0,item1) & mul(item2,item0): {after:?}"
        );
    }

    // =====================
    // 5. Query + lambda-aware top-2 similarity
    // =====================
    let q = ArrowItem::new(
        vec![
            0.89, 0.09, 0.46, 0.25, 0.55, 0.30, 0.50, 0.40, 0.20, 0.60, 0.10, 0.30, 0.55, 0.20,
            0.40, 0.30, 0.50, 0.20, 0.60, 0.40, 0.25, 0.48, 0.35, 0.28,
        ],
        0.0,
    );

    let mut scores: Vec<(usize, f64)> = (0..3)
        .map(|i| {
            let item = aspace.get_item(i); // owned ArrowItem
            let q_like = ArrowItem::new(q.item.clone(), item.lambda);
            (i, q_like.lambda_similarity(&item, 0.8, 0.2))
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("lambda-aware top-2 items: {:?}", &scores[..2]);

    // =====================
    // 6. Range search
    // =====================
    let q2 = ArrowItem::new(
        vec![
            0.49, 0.41, 0.14, 0.20, 0.30, 0.50, 0.40, 0.60, 0.30, 0.45, 0.15, 0.35, 0.50, 0.25,
            0.40, 0.20, 0.48, 0.18, 0.55, 0.33, 0.28, 0.42, 0.31, 0.30,
        ],
        0.0,
    );
    let hits = aspace.range_search(&q2, 0.6);
    println!("range hits within 0.6: {hits:?}");

    // =====================
    // 7. Pairwise cosine matrix
    // =====================
    let sims = aspace.pairwise_similarities(&[0, 1, 2]);
    println!("pairwise cosine 3x3: {sims:?}");
}
