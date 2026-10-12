use approx::assert_relative_eq;

use crate::builder::ArrowSpaceBuilder;
use crate::filter_bank::{BandEdges, FilterBankConfig};
use crate::search::taumode::TauMode;
use crate::tests::init;
use crate::tests::test_data::make_moons_hd;

#[test]
fn test_filter_bank_default_off_is_empty() {
    init();

    let items = make_moons_hd(32, 0.08, 0.2, 8, 3407);
    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 5, 3, 2.0, None)
        .with_normalisation(true)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None)
        .with_seed(3407)
        .build(items);

    assert_eq!(aspace.filter_bank_bands(), 0);
    assert!(aspace.band_energies().is_empty());
    assert!(aspace.band_energies_for_item(0).is_empty());
}

#[test]
fn test_uniform_filter_bank_sums_to_rayleigh_energy() {
    init();

    let items = make_moons_hd(36, 0.08, 0.25, 8, 3407);
    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 5, 3, 2.0, None)
        .with_filter_bank(FilterBankConfig::new(3, BandEdges::Uniform))
        .with_normalisation(true)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None)
        .with_seed(3407)
        .build(items);

    assert_eq!(aspace.filter_bank_bands(), 3);
    assert_eq!(aspace.band_energies().len(), aspace.nitems * 3);

    for item_idx in 0..aspace.nitems {
        let item = aspace.get_item(item_idx);
        let bands = aspace.band_energies_for_item(item_idx);
        assert_eq!(bands.len(), 3);

        let band_sum: f64 = bands.iter().sum();
        let rayleigh = TauMode::compute_rayleigh_quotient_from_matrix(&gl.matrix, &item.item);
        assert_relative_eq!(band_sum, rayleigh, epsilon = 1e-8);
    }
}

#[test]
fn test_filter_bank_scale_invariance_and_query_preparation() {
    init();

    let items = make_moons_hd(36, 0.08, 0.25, 8, 3407);
    let query = items[3].clone();
    let scaled_query: Vec<f64> = query.iter().map(|v| v * 7.0).collect();

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 5, 3, 2.0, None)
        .with_filter_bank(FilterBankConfig::new(4, BandEdges::Uniform))
        .with_normalisation(true)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None)
        .with_seed(3407)
        .build(items);

    let lambda_only = aspace.try_prepare_query_item(&query, &gl).unwrap();
    let prepared = aspace
        .try_prepare_query_item_with_bands(&query, &gl)
        .unwrap();
    let prepared_scaled = aspace
        .try_prepare_query_item_with_bands(&scaled_query, &gl)
        .unwrap();

    assert_relative_eq!(prepared.lambda, lambda_only, epsilon = 1e-12);
    assert_eq!(prepared.band_energies.len(), 4);
    assert_eq!(prepared_scaled.band_energies.len(), 4);

    for (a, b) in prepared
        .band_energies
        .iter()
        .zip(prepared_scaled.band_energies.iter())
    {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }
}

#[test]
fn test_custom_filter_bank_edges_are_accepted() {
    init();

    let items = make_moons_hd(32, 0.08, 0.2, 8, 3407);
    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 5, 3, 2.0, None)
        .with_filter_bank(FilterBankConfig::new(
            3,
            BandEdges::Custom(vec![0.0, 0.2, 0.7, 1.0]),
        ))
        .with_normalisation(true)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None)
        .with_seed(3407)
        .build(items);

    assert_eq!(aspace.filter_bank_bands(), 3);
    assert_eq!(aspace.band_energies_for_item(0).len(), 3);
}

#[test]
#[should_panic(expected = "custom filter-bank edges must be finite and strictly increasing")]
fn test_custom_filter_bank_edges_must_not_overlap() {
    let _ = FilterBankConfig::new(3, BandEdges::Custom(vec![0.0, 0.5, 0.4, 1.0]));
}
