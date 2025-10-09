mod test_arrow;
mod test_builder;
mod test_data;
mod test_helpers;
// mod test_dimensional;
mod test_clustering;
mod test_graph_factory;
mod test_laplacian;
mod test_laplacian_unnormalised;
mod test_querying_proj;
mod test_reduction;
mod test_taumode;

use crate::graph::GraphParams;
use crate::taumode::TauMode;

pub const GRAPH_PARAMS: GraphParams = GraphParams {
    eps: 0.1,
    k: 3,
    topk: 3,
    p: 2.0,
    sigma: Some(0.1 * 0.75),
    normalise: true,
    sparsity_check: true,
};

pub const TAU_PARAMS: TauMode = TauMode::Median;
