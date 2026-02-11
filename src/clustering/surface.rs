use kalman_centroids::KalmanClusterer;
use smartcore::linalg::basic::matrix::DenseMatrix;

// Helper to bundle the output of the new clustering stage
pub struct KalmanClusteredOutput {
    pub centroids: DenseMatrix<f64>,
    pub variances: Vec<Vec<f64>>, // We need variances for thickness
    pub assignments: Vec<Option<usize>>,
    pub sizes: Vec<usize>,
}

// The new replacement for `run_incremental_clustering_with_sampling`
pub(crate) fn run_kalman_clustering(rows: &[Vec<f64>], max_k: usize) -> KalmanClusteredOutput {
    let n_items = rows.len();

    // 1. Run Kalman Clustering
    let mut clusterer = KalmanClusterer::new(max_k, n_items);
    clusterer.fit(rows); // Assumes fit takes &[Vec<f64>] or similar

    // 2. Export Data
    let centroids_dm = clusterer.export_centroids(); // DenseMatrix

    // Extract variances for thickness computation later
    let variances: Vec<Vec<f64>> = clusterer
        .centroids
        .iter()
        .map(|c| c.variance.clone())
        .collect();

    let sizes: Vec<usize> = clusterer.centroids.iter().map(|c| c.count).collect();

    KalmanClusteredOutput {
        centroids: centroids_dm,
        variances,
        assignments: clusterer.assignments,
        sizes,
    }
}

// The Surface Optimization Ordering Logic
pub(crate) fn compute_surface_order(
    centroids: &DenseMatrix<f64>,
    variances: &[Vec<f64>],
    // These defaults can be hardcoded or passed from builder
    k_order: usize,
    eps: f64,
) -> Vec<usize> {
    let n_centroids = centroids.shape().0;
    if n_centroids <= 1 {
        return vec![0];
    }

    // 1. Compute Thickness (Confidence)
    // w_i = 1 / sqrt(mean_variance + epsilon)
    let thickness: Vec<f64> = variances
        .iter()
        .map(|vars| {
            let mean_var = vars.iter().sum::<f64>() / vars.len() as f64;
            1.0 / (mean_var + 1e-9).sqrt()
        })
        .collect();

    // 2. Build Sparse Centroid Graph (Rectified Cosine)
    // We reuse the existing CosinePair infrastructure
    use smartcore::algorithm::neighbour::cosine_pair::CosinePair;
    let topk = (k_order * 2).max(16);
    let index = CosinePair::with_top_k(centroids, topk).unwrap();

    // 3. Compute MST with Surface Costs
    // Cost = d_ij * (r_i + r_j) / 2
    // We use a simple Prim's algorithm here for the skeleton
    let mut visited = vec![false; n_centroids];
    // (cost, node_index)
    let mut pq = std::collections::BinaryHeap::new();

    // Find root: thickest node (highest confidence)
    let (root_idx, _) = thickness
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // MST Adjacency list for traversal
    let mut mst_adj: Vec<Vec<usize>> = vec![vec![]; n_centroids];

    visited[root_idx] = true;

    // Push neighbors of root
    // Note: CosinePair queries return distances.
    // We strictly filter by `eps` to match Laplacian logic.
    let root_neighbors = index.query_row_top_k(root_idx, topk).unwrap();
    for (neigh_idx, dist_d) in root_neighbors {
        if neigh_idx != root_idx && dist_d <= eps {
            let surface_cost = dist_d * (thickness[root_idx] + thickness[neigh_idx]) / 2.0;
            // FloatOrd wrapper needed for BinaryHeap
            pq.push(std::cmp::Reverse(OrderedFloat(
                surface_cost,
                root_idx,
                neigh_idx,
            )));
        }
    }

    // Standard MST loop
    while let Some(std::cmp::Reverse(OrderedFloat(cost, parent, child))) = pq.pop() {
        if visited[child] {
            continue;
        }

        visited[child] = true;
        mst_adj[parent].push(child);
        mst_adj[child].push(parent); // Undirected skeleton

        // Add child's neighbors
        let neighbors = index.query_row_top_k(child, topk).unwrap();
        for (next_idx, dist_d) in neighbors {
            if !visited[next_idx] && next_idx != child && dist_d <= eps {
                let surface_cost = dist_d * (thickness[child] + thickness[next_idx]) / 2.0;
                pq.push(std::cmp::Reverse(OrderedFloat(
                    surface_cost,
                    child,
                    next_idx,
                )));
            }
        }
    }

    // 4. Linearize via Traversal (Trunk-First)
    // DFS/BFS starting from root.
    // When visiting children, sort them by Descending Thickness.
    let mut order = Vec::with_capacity(n_centroids);
    let mut traversal_stack = vec![root_idx];
    let mut ordered_visited = vec![false; n_centroids];
    ordered_visited[root_idx] = true;

    while let Some(curr) = traversal_stack.pop() {
        order.push(curr);

        // Get unvisited children from MST
        let mut children: Vec<usize> = mst_adj[curr]
            .iter()
            .cloned()
            .filter(|&n| !ordered_visited[n])
            .collect();

        // Mark as visited immediately to avoid cycles in queue
        for &c in &children {
            ordered_visited[c] = true;
        }

        // Sort children: Thicker (higher confidence) -> Thinner
        // If tied, use distance from current parent (closer first)
        children.sort_by(|&a, &b| {
            thickness[b]
                .partial_cmp(&thickness[a])
                .unwrap() // Descending thickness
                .then_with(|| {
                    // Break tie with raw distance from current node
                    let dist_a = index.query_dist(curr, a);
                    let dist_b = index.query_dist(curr, b);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
        });

        // Push to stack in reverse so highest priority is popped first
        for c in children.into_iter().rev() {
            traversal_stack.push(c);
        }
    }

    // Handle disconnected components (if eps was too small) by just appending them
    for i in 0..n_centroids {
        if !ordered_visited[i] {
            order.push(i);
        }
    }

    order
}

// Helper struct for Heap
struct OrderedFloat(f64, usize, usize); // cost, from, to
impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
