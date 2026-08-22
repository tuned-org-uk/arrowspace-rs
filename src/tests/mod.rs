mod test_arrow;
mod test_builder;
pub mod test_data;
mod test_error;
mod test_helpers;
// mod test_dimensional;
mod test_clustering;
mod test_eigenmaps;
mod test_energy_builder;
mod test_energy_search;
mod test_graph_factory;
mod test_laplacian;
mod test_laplacian_unnormalised;
mod test_motives;
mod test_querying_proj;
mod test_reduction;
mod test_sequencing;
mod test_taumode;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env)
            .is_test(true) // nicer formatting for tests
            .try_init();
    });
}

use crate::tests::test_data::generate_test_data;
use crate::tests::test_data::make_gaussian_hd;
use once_cell::sync::Lazy;

#[cfg(test)]
static GAUSSIAN_DATA: Lazy<Vec<Vec<f64>>> = Lazy::new(|| {
    // generate synthetic data
    make_gaussian_hd(99, 0.6)
});

#[cfg(test)]
static CLUSTERING_TEST_DATA: Lazy<Vec<Vec<f64>>> = Lazy::new(|| {
    // generate synthetic data
    generate_test_data(99, 500, 12345)
});
