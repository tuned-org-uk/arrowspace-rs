use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::filter_bank::{BandEdges, FilterBankConfig};

fn main() {
    let items = vec![
        vec![1.0, 0.2, 0.1, 0.0, 0.4, 0.3],
        vec![0.9, 0.3, 0.1, 0.1, 0.5, 0.2],
        vec![0.1, 1.0, 0.8, 0.2, 0.0, 0.3],
        vec![0.2, 0.9, 0.7, 0.3, 0.1, 0.4],
        vec![0.4, 0.2, 0.3, 1.0, 0.9, 0.8],
        vec![0.5, 0.1, 0.4, 0.9, 1.0, 0.7],
    ];

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 4, 3, 2.0, None)
        .with_filter_bank(FilterBankConfig::new(3, BandEdges::Uniform))
        .with_normalisation(true)
        .with_inline_sampling(None)
        .with_dims_reduction(false, None)
        .with_seed(3407)
        .build(items.clone());

    println!("filter bands: {}", aspace.filter_bank_bands());
    for i in 0..aspace.nitems {
        println!(
            "item {i}: lambda={:.6}, bands={:?}",
            aspace.lambdas()[i],
            aspace.band_energies_for_item(i)
        );
    }

    let prepared = aspace
        .try_prepare_query_item_with_bands(&items[0], &gl)
        .expect("query preparation should succeed");

    println!(
        "query: lambda={:.6}, bands={:?}",
        prepared.lambda, prepared.band_energies
    );
}
