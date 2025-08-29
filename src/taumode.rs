// Encapsulates tau selection policies for the synthetic index transform.
//
// API:
// - TauMode: policy enum
// - select_tau(&[f64], TauMode) -> f64: returns a strictly positive tau (> 0)
//
// Notes:
// - Energies may contain NaN/inf; this utility filters non-finite values out.
// - If the filtered set is empty or produces a non-positive tau, a small floor is used.
//
// TauMode policy and selection utility for synthetic lambda transform.
// Default is Median, use `with_synthesis` to change the default to one of the other

// The computation should run once and the graph should not be stored, only the syntethic index should be stored

// A single-pass, stored synthetic index can be designed as a per-row scalar that blends the global Rayleigh energy with a localized Laplacian-weighted roughness summary, so downstream search can use “semantic” scores together with “spectral roughness” without retaining the graph. Concretely, compute once per row r a scalar S_r that merges the Rayleigh quotient $E_r = \frac{x_r^\top L x_r}{x_r^\top x_r}$ with a Laplacian-locality moment derived from the same per-edge Dirichlet energy, normalize it, and store only S_r alongside the row; discard L afterward. This preserves the smoothness signal that Rayleigh encodes while incorporating a stable, intensity-aware normalization and local variability component, enabling alpha–beta blending at query time with no graph in memory.[^1][^2]
// ## Key idea
// - Use the Rayleigh energy $E_r$ as the primary smoothness measure; it is non‑negative and scale‑invariant in x, reflecting how “wiggly” a row is over the item graph built once for the dataset.[^2][^1]
// - Augment $E_r$ with a local Dirichlet statistic that captures how energy distributes across edges, e.g., an energy-weighted Gini or variance of per-edge contributions, to distinguish uniformly rough rows from rows whose roughness is concentrated; summarize this as $V_r$ and combine with $E_r$ into a single scalar index $S_r$ via a bounded transform that is comparable across rows.[^3][^1]
// ## Per-row synthetic index
// For each row vector x in R^N and Laplacian L computed once:
// - Rayleigh energy: $E_r = \frac{x^\top L x}{x^\top x}$ with the convention $E_r = 0$ if $x^\top x = 0$.[^1][^2]
// - Edgewise energy distribution: expand the Dirichlet energy as a sum over edges using the standard identity $x^\top L x = \sum_{(i,j)\in E} w_{ij} (x_i - x_j)^2$; define per-edge share $e_{ij} = \frac{w_{ij} (x_i - x_j)^2}{\sum w_{uv} (x_u - x_v)^2}$ if the denominator is nonzero, else 0.[^3][^1]
// - Locality moment: compute a dispersion summary $V_r$ over {e_{ij}} such as:
//     - Gini-like concentration: $G_r = \sum e_{ij}^2$ (higher means energy concentrated on fewer edges), or
//     - Variance: $\operatorname{Var}_r = \sum (e_{ij} - \frac{1}{|E_x|})^2$ over edges with nonzero contribution [^3][^1].
// - Normalization: map $E_r$ to a bounded score $E_r' = \frac{E_r}{E_r + \tau}$ with a small positive scale $\tau$ (e.g., median E across rows) to stabilize tails and keep 0–1 range; keep $G_r$ already in  by construction.`ArrowSpace`[^2][^1]
// - Synthetic index: $S_r = \alpha \, E_r' + (1-\alpha) \, G_r$, with \$\alpha \in \$ (e.g., 0.7). Store S_r per row and discard L and all edge stats.`ArrowSpace`[^2][^1]
// ## Why this works
// - $E_r$ is the classical Dirichlet (Rayleigh) energy; lower means smoother over the item graph, higher means high-frequency variation; it is standard in spectral methods and tightly coupled to Laplacian eigen-analysis.[^2][^1]
// - $G_r$ distinguishes equal-energy rows by how the energy is distributed across edges: a row with many small, diffuse irregularities differs from one with a few sharp discontinuities; a single scalar keeps this nuance without storing the graph.[^1][^3]
// ## One-time computation plan
// - Build the item graph Laplacian L once (any construction policy, e.g., your λτ-graph or KNN), compute S_r for all rows, then drop L.[^3][^1]
// - Complexity: computing $x^\top L x$ is O(nnz(L)) per row using CSR; computing e_{ij} shares during the same pass has the same complexity; no extra matrices are needed; storage is O(nrows) for S_r.[^2][^3]
// ## Practical details
// - Zero/constant rows: if $x^\top x = 0$, set $E_r=0$ and $G_r=0$, hence $S_r=0$, which is consistent with smoothness on connected graphs.[^1][^2]
// - Scale selection: pick $\tau$ as median or a small quantile of {E_r} to get a stable logistic-like compression; this makes S_r robust and comparable across datasets.[^2][^1]
// - Optional calibration: z-score or rank-normalize E_r across rows before the bounded map if heavy tails; keep G_r as-is in.`ArrowSpace`[^3][^1]
// ## Using the index
// - At search time, blend semantic similarity with S_r proximity using the same alpha–beta logic already used for λ, replacing lambda proximity with an S_r proximity term, e.g., $1/(1+|S_q - S_i|)$; this preserves a spectral bias without reconstructing or storing any graph [^2][^1].
// - The single scalar S_r per row can also be used for band filters, clustering seeds, or diversity penalties biased by spectral roughness, again without graphs.[^3][^1]
// <span style="display:none">[^10][^11][^5][^6][^7][^8][^9]</span>
// <div style="text-align: center">⁂</div>
// [^1]: https://www.cs.yale.edu/homes/spielman/462/2007/lect7-07.pdf
// [^2]: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/07/Krishnan-SG13.pdf
// [^3]: https://www.math.uni-potsdam.de/fileadmin/user_upload/Prof-GraphTh/Keller/KellerLenzWojciechowski_GraphsAndDiscreteDirichletSpaces_personal.pdf
// [^5]: https://arxiv.org/html/2501.11024v1
// [^6]: https://arxiv.org/html/2508.06123v1
// [^7]: https://d-nb.info/1162140003/34
// [^8]: https://www.ins.uni-bonn.de/media/public/publication-media/MA_Schwartz_2016.pdf?pk=703
// [^9]: https://proceedings.neurips.cc/paper_files/paper/2024/file/f57de20ab7bb1540bcac55266ebb5401-Paper-Conference.pdf
// [^10]: https://arxiv.org/pdf/2309.11251.pdf
// [^11]: https://academic.oup.com/mnras/article-pdf/370/4/1713/3388125/mnras0370-1713.pdf
use crate::core::ArrowSpace;
use crate::graph_factory::GraphLaplacian;

#[derive(Clone, Copy, Debug)]
pub enum TauMode {
    Fixed(f64),
    Median,
    Mean,
    Percentile(f64),
}

pub const TAU_FLOOR: f64 = 1e-9;

pub fn select_tau(energies: &[f64], mode: TauMode) -> f64 {
    match mode {
        TauMode::Fixed(t) => {
            if t.is_finite() && t > 0.0 {
                t
            } else {
                TAU_FLOOR
            }
        }
        TauMode::Mean => {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for &e in energies {
                if e.is_finite() {
                    sum += e;
                    cnt += 1;
                }
            }
            let m = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
            m.max(TAU_FLOOR)
        }
        TauMode::Median | TauMode::Percentile(_) => {
            let mut v: Vec<f64> = energies.iter().copied().filter(|x| x.is_finite()).collect();
            if v.is_empty() {
                return TAU_FLOOR;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let TauMode::Percentile(p) = mode {
                let pp = p.clamp(0.0, 1.0);
                let idx = ((v.len() - 1) as f64 * pp).round() as usize;
                return v[idx].max(TAU_FLOOR);
            }
            if v.len() % 2 == 1 {
                v[v.len() / 2].max(TAU_FLOOR)
            } else {
                let mid = 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]);
                mid.max(TAU_FLOOR)
            }
        }
    }
}

/// Aggregate from feature signals to per-item scores.
/// For each item i, combine the feature-level spectral information weighted by the item’s feature magnitudes
/// (or another explicit weighting) to get a single synthetic index S_i.
///
/// Use the same bounded energy map and dispersion blend idea, but perform it per item:
/// - For each feature row f, compute Rayleigh energy E_f and dispersion G_f once (as today).
/// - For each item i, compute a weight w_fi = |x_f[i]| (magnitude of feature f at item i) and use it to
/// Aggregate the feature contributions to item level:
/// - E_i_raw = (sum_f w_fi * E_f) / (sum_f w_fi) with 0-guard.
/// - G_i_raw = (sum_f w_fi * G_f) / (sum_f w_fi) with 0-guard.
/// - Select τ from the per-item energy population {E_i_raw} using the same TauMode.
/// - Map E_i = E_i_raw / (E_i_raw + τ), clamp G_i to , then S_i = α * E_i + (1−α) * G_i.
/// - Finally, update ArrowSpace.lambdas with the n_items synthetic vector S (length equals gl.nnodes).
pub fn compute_synthetic_lambdas(
    aspace: &mut ArrowSpace,
    gl: &GraphLaplacian,
    alpha: f64,
    tau_mode: TauMode,
) {
    let (n_features, n_items) = aspace.shape();
    assert_eq!(
        gl.nnodes, n_items,
        "GraphLaplacian nodes must match ArrowSpace items"
    );

    // 1) Compute per-feature Rayleigh energy E_f and dispersion G_f (as before)
    let mut energies_f: Vec<f64> = Vec::with_capacity(n_features);
    let mut dispersions_f: Vec<f64> = Vec::with_capacity(n_features);

    for f in 0..n_features {
        let x = aspace.iter_feature(f); // length n_items
        let den: f64 = x.iter().map(|&v| v * v).sum();
        if den <= 0.0 {
            energies_f.push(0.0);
            dispersions_f.push(0.0);
            continue;
        }

        // Rayleigh numerator
        let mut num = 0.0;
        // For dispersion G_f we need total edge energy to normalize shares
        let mut edge_energy_sum = 0.0;
        for i in 0..n_items {
            let xi = x[i];
            let (s, e) = (gl.rows[i], gl.rows[i + 1]);
            // (Lx)_i
            let mut lx_i = 0.0;
            for idx in s..e {
                let j = gl.cols[idx];
                let lij = gl.vals[idx];
                lx_i += lij * x[j];
            }
            num += xi * lx_i;

            // accumulate off-diagonal edge energy
            for idx in s..e {
                let j = gl.cols[idx];
                if j == i {
                    continue;
                }
                let w = (-gl.vals[idx]).max(0.0);
                if w > 0.0 {
                    let d = xi - x[j];
                    edge_energy_sum += w * d * d;
                }
            }
        }
        let e_f = num / den;
        energies_f.push(e_f.max(0.0));

        // G_f: sum of squared normalized edge shares
        let mut g_sq_sum = 0.0;
        if edge_energy_sum > 0.0 {
            for i in 0..n_items {
                let xi = x[i];
                let (s, e) = (gl.rows[i], gl.rows[i + 1]);
                for idx in s..e {
                    let j = gl.cols[idx];
                    if j == i {
                        continue;
                    }
                    let w = (-gl.vals[idx]).max(0.0);
                    if w > 0.0 {
                        let d = xi - x[j];
                        let contrib = w * d * d;
                        let share = contrib / edge_energy_sum;
                        g_sq_sum += share * share;
                    }
                }
            }
        }
        let g_f = g_sq_sum.clamp(0.0, 1.0);
        dispersions_f.push(g_f);
    }

    // 2) Aggregate feature contributions into per-item raw energies and dispersions
    let mut e_item_raw = vec![0.0f64; n_items];
    let mut g_item_raw = vec![0.0f64; n_items];
    let mut wsum_item = vec![0.0f64; n_items];

    for f in 0..n_features {
        let x = aspace.iter_feature(f);
        let e_f = energies_f[f];
        let g_f = dispersions_f[f];
        for i in 0..n_items {
            let w = x[i].abs();
            if w > 0.0 {
                e_item_raw[i] += w * e_f;
                g_item_raw[i] += w * g_f;
                wsum_item[i] += w;
            }
        }
    }

    for i in 0..n_items {
        if wsum_item[i] > 0.0 {
            e_item_raw[i] /= wsum_item[i];
            g_item_raw[i] /= wsum_item[i];
        } else {
            e_item_raw[i] = 0.0;
            g_item_raw[i] = 0.0;
        }
    }

    // 3) Select tau over the per-item energies and map to bounded scores
    let tau = select_tau(&e_item_raw, tau_mode);
    let mut synthetic_items = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let e_bounded = {
            let e = e_item_raw[i].max(0.0);
            e / (e + tau)
        };
        let g_clamped = g_item_raw[i].clamp(0.0, 1.0);
        let s = alpha * e_bounded + (1.0 - alpha) * g_clamped;
        synthetic_items.push(s);
    }

    // 4) Store per-item lambdas
    aspace.update_lambdas(synthetic_items);
}
