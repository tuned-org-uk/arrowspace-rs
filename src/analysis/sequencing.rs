//! Sequencing: total orderings over the nodes of the feature-space graph Laplacian.
//!
//! Sequencing assigns each node of `GraphLaplacian.matrix` (L = D − A) a discrete
//! position along a path through the graph, producing a permutation usable for
//! data curricula, de-duplication sweeps or coherent walk generation.
//!
//! Following the design invariants (`AGENTS.md`), sequencing is computed with
//! purely combinatorial graph algorithms — no eigensolvers, no continuous
//! embeddings. Two strategies are provided:
//!
//! * [`sequence_by_lambda`] orders nodes by their per-node λ score: a "spectral
//!   curriculum" from low to high Rayleigh energy (or the reverse).
//! * [`sequence_by_graph`] performs MST-chain seriation: a minimum spanning
//!   forest is grown over the weighted adjacency recovered from L (A = D − L),
//!   then each tree is walked in depth-first preorder from an approximate
//!   diameter endpoint found by double-sweep BFS. This is the discrete
//!   counterpart of Fiedler-vector seriation: it approximates a minimum linear
//!   arrangement without computing any vector.
//!
//! Reference: *The ArrowSpace Algorithm: From Graph Wiring to τ-Mode Spectral
//! Search*, DOI [10.5281/zenodo.21679021](https://zenodo.org/records/21679021).
//!
//! # Examples
//!
//! ```
//! use arrowspace::analysis::sequencing::{sequence_by_graph, sequence_by_lambda};
//!
//! // λ-based curriculum ordering over per-node scores.
//! let seq = sequence_by_lambda(&[0.9, 0.1, 0.4], false);
//! assert_eq!(seq.order, vec![1, 2, 0]);
//!
//! // Graph seriation over a Laplacian built from items.
//! use arrowspace::graph::GraphParams;
//! use arrowspace::laplacian::build_laplacian_matrix;
//! use smartcore::linalg::basic::arrays::Array2;
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//!
//! let items = vec![
//!     vec![1.0, 0.0, 0.0],
//!     vec![0.9, 0.1, 0.0],
//!     vec![0.8, 0.6, 0.0],
//!     vec![0.0, 1.0, 0.0],
//!     vec![0.1, 0.9, 0.0],
//! ];
//! let params = GraphParams {
//!     eps: 0.5,
//!     k: 3,
//!     topk: 3,
//!     p: 2.0,
//!     sigma: Some(0.1),
//!     normalise: true,
//!     sparsity_check: false,
//! };
//! let gl = build_laplacian_matrix(
//!     DenseMatrix::<f64>::from_2d_vec(&items).unwrap().transpose(),
//!     &params,
//!     None,
//!     false,
//! );
//! let seq = sequence_by_graph(&gl);
//! assert_eq!(seq.order.len(), gl.matrix.shape().0);
//! ```

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use log::{debug, info};
use ordered_float::OrderedFloat;

use crate::graph::GraphLaplacian;
use sprs::CsMat;

/// A total order over the nodes of a graph Laplacian.
#[derive(Clone, Debug)]
pub struct Sequence {
    /// Node indices in sequence order: a permutation of `0..n_nodes`.
    pub order: Vec<usize>,
    /// Discrete coordinate per sequence step, aligned with `order`:
    /// the λ score for [`sequence_by_lambda`], or the DFS discovery depth
    /// within its component for [`sequence_by_graph`].
    pub positions: Vec<f64>,
    /// Number of connected components traversed (`1` for λ orderings).
    pub components: usize,
}

/// Orders items by their per-node λ scores in ascending or descending order.
///
/// This is the simplest sequencing strategy: a spectral curriculum that walks
/// items from low Rayleigh energy to high (ascending), or the reverse. Ties
/// break on ascending node index; the result is fully deterministic.
///
/// # Panics
///
/// Panics if `lambdas` contains fewer than two items.
pub fn sequence_by_lambda(lambdas: &[f64], descending: bool) -> Sequence {
    assert!(
        lambdas.len() >= 2,
        "sequencing requires at least two items, got {}",
        lambdas.len()
    );

    let mut order: Vec<usize> = (0..lambdas.len()).collect();
    if descending {
        order.sort_unstable_by(|&a, &b| {
            lambdas[b]
                .partial_cmp(&lambdas[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
    } else {
        order.sort_unstable_by(|&a, &b| {
            lambdas[a]
                .partial_cmp(&lambdas[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
    }

    debug!(
        "sequence_by_lambda ordered {} items ({})",
        order.len(),
        if descending {
            "descending"
        } else {
            "ascending"
        }
    );

    let positions = order.iter().map(|&i| lambdas[i]).collect();
    Sequence {
        order,
        positions,
        components: 1,
    }
}

/// Serialises the nodes of a graph Laplacian by walking its minimum spanning
/// forest in depth-first preorder from approximate diameter endpoints.
///
/// The adjacency is recovered exactly from the Laplacian (`A[i,j] = -L[i,j]`
/// off-diagonal); no similarity recomputation happens here. Each connected
/// component yields one contiguous block of the sequence; blocks are ordered
/// by descending size, ties by ascending smallest member index. Within a
/// block, the walk starts at a double-sweep BFS endpoint and always visits
/// smaller-indexed children first, so the output is deterministic.
///
/// Positions report the DFS discovery depth of each node within its component.
///
/// Complexity: `O(E log V + V)` time, `O(V + E)` space.
///
/// # Panics
///
/// Panics if the Laplacian has fewer than two nodes or is not square.
pub fn sequence_by_graph(gl: &GraphLaplacian) -> Sequence {
    let n = gl.matrix.shape().0;
    assert!(n >= 2, "sequencing requires at least two nodes, got {}", n);
    assert_eq!(
        gl.matrix.shape().1,
        n,
        "the laplacian must be a square matrix"
    );

    info!("Sequencing {} laplacian nodes via MST-chain seriation", n);

    let adjacency = _adjacency_from_laplacian(&gl.matrix);
    debug!(
        "Recovered undirected adjacency with {} nodes",
        adjacency.len()
    );

    let forest = _minimum_spanning_forest(&adjacency, n);
    let sequence = _serialise_forest(forest);

    info!(
        "Sequenced {} nodes across {} component(s)",
        sequence.order.len(),
        sequence.components
    );
    sequence
}

/// Extracts the undirected weighted adjacency implied by L = D − A as
/// flat per-node neighbour lists (sorted by neighbour index).
///
/// Assumes L is symmetric: each undirected edge is read once from the upper
/// triangle (`j > i`). The pipeline guarantees this via deterministic
/// symmetrisation; hand-built Laplacians must be symmetrised first or
/// edges present only below the diagonal will be missed.
fn _adjacency_from_laplacian(l: &CsMat<f64>) -> Vec<Vec<(usize, OrderedFloat<f64>)>> {
    let n = l.shape().0;

    // Off-diagonal entries of L are -w; take each edge once from its upper
    // triangle, then emit both directions for flat sort-merge grouping.
    let mut directed: Vec<(usize, usize, OrderedFloat<f64>)> = Vec::new();
    for (i, row) in l.outer_iterator().enumerate() {
        for (j, &v) in row.iter() {
            if j > i && v < 0.0 {
                let w = OrderedFloat(-v);
                directed.push((i, j, w));
                directed.push((j, i, w));
            }
        }
    }
    directed.sort_unstable_by_key(|&(i, j, _)| (i, j));

    let mut adjacency: Vec<Vec<(usize, OrderedFloat<f64>)>> = vec![Vec::new(); n];
    let mut last = (usize::MAX, usize::MAX);
    for &(i, j, w) in &directed {
        if (i, j) != last {
            adjacency[i].push((j, w));
            last = (i, j);
        }
    }
    adjacency
}

/// Prim's algorithm grown from every unvisited node in index order, producing
/// a spanning *forest* whose trees span the graph's connected components.
/// Heap entries carry `(weight, candidate, parent)` so pops resolve weight
/// ties deterministically by node index.
fn _minimum_spanning_forest(
    adjacency: &[Vec<(usize, OrderedFloat<f64>)>],
    n: usize,
) -> Vec<Vec<usize>> {
    let mut tree: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_tree = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize, usize)>> = BinaryHeap::new();

    for start in 0..n {
        if in_tree[start] {
            continue;
        }
        in_tree[start] = true;
        for &(j, w) in &adjacency[start] {
            heap.push(Reverse((w, j, start)));
        }
        while let Some(Reverse((_, j, parent))) = heap.pop() {
            if in_tree[j] {
                continue;
            }
            in_tree[j] = true;
            tree[parent].push(j);
            tree[j].push(parent);
            for &(k, wk) in &adjacency[j] {
                if !in_tree[k] {
                    heap.push(Reverse((wk, k, j)));
                }
            }
        }
    }

    // Ascending child lists make every downstream walk deterministic.
    for neighbours in &mut tree {
        neighbours.sort_unstable();
    }
    tree
}

/// Labels the forest's trees, orders them by (size desc, min index asc),
/// roots each at a double-sweep BFS endpoint and concatenates DFS preorders.
fn _serialise_forest(tree: Vec<Vec<usize>>) -> Sequence {
    let n = tree.len();

    // Component labelling by stack DFS; members kept sorted ascending so
    // downstream scans and tie-breaks are index-deterministic.
    let mut comp_of = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for s in 0..n {
        if comp_of[s] != usize::MAX {
            continue;
        }
        let id = components.len();
        let mut members = Vec::new();
        let mut stack = vec![s];
        comp_of[s] = id;
        while let Some(u) = stack.pop() {
            members.push(u);
            for &v in &tree[u] {
                if comp_of[v] == usize::MAX {
                    comp_of[v] = id;
                    stack.push(v);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }

    // Larger components first; ties broken by smallest member index.
    components.sort_unstable_by_key(|members| (std::cmp::Reverse(members.len()), members[0]));

    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut depth_of = vec![0f64; n];
    let mut visited = vec![false; n];
    // Scratch buffers reused by both sweeps of every component; entries are
    // reset per call so cost stays proportional to the component, not n.
    let mut dist = vec![usize::MAX; n];
    let mut seen = vec![false; n];

    for members in &components {
        let seed = members[0];

        // Double sweep: BFS farthest twice to land near a tree diameter end.
        let far = _bfs_farthest(seed, &tree, members, &mut dist, &mut seen);
        let root = _bfs_farthest(far, &tree, members, &mut dist, &mut seen);

        // Iterative DFS preorder; children pushed in reverse so the
        // smallest index is visited first.
        let mut stack = vec![(root, 0usize)];
        visited[root] = true;
        while let Some((u, depth)) = stack.pop() {
            order.push(u);
            depth_of[u] = depth as f64;
            for &v in tree[u].iter().rev() {
                if !visited[v] {
                    visited[v] = true;
                    stack.push((v, depth + 1));
                }
            }
        }
    }

    Sequence {
        positions: order.iter().map(|&u| depth_of[u]).collect(),
        order,
        components: components.len(),
    }
}

/// BFS over a component's tree returning the farthest member from `src`;
/// distance ties break on the smaller node index (`members` is ascending).
/// `dist` and `seen` are caller-owned scratch buffers reset for `members`
/// only, so each call costs O(|members| + edges within the component) and
/// the total across all components stays O(V + E).
fn _bfs_farthest(
    src: usize,
    tree: &[Vec<usize>],
    members: &[usize],
    dist: &mut [usize],
    seen: &mut [bool],
) -> usize {
    for &m in members {
        dist[m] = usize::MAX;
        seen[m] = false;
    }
    let mut queue = VecDeque::with_capacity(members.len());
    dist[src] = 0;
    seen[src] = true;
    queue.push_back(src);

    while let Some(u) = queue.pop_front() {
        for &v in &tree[u] {
            if !seen[v] {
                seen[v] = true;
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }

    // Every member is reachable within its own tree, so no MAX guard is
    // needed; strict > keeps the first (smallest) index on ties.
    let mut best = src;
    let mut best_dist = 0;
    for &u in members {
        if dist[u] > best_dist {
            best = u;
            best_dist = dist[u];
        }
    }
    best
}
