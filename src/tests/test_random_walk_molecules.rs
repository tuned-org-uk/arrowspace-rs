//! Test module: free random walk molecular diffusion simulation
//! Reference: Physical Models of Living Systems, Chapter 6
//! Simulates Brownian motion of a molecule and computes Arrow lambda on the trajectory
//!
//! Arrow Space Integration
//! * Uses your existing build_knn_graph and ArrowSpace implementations
//! * Computes spectral lambda on different physical signals over trajectory graphs
//! * Tests superposition behavior (molecular interaction effects)
//!
//! * Validates lambda positivity and bounds for physical consistency
//! Biological/Physical Insights
//! * Lambda as smoothness measure: Lower lambda indicates more homogeneous concentration
//! * Superposition effects: Combined molecular fields can have non-additive spectral properties
//! * Time evolution: Tracks how diffusion spreads and its spectral signature changes

#[cfg(test)]
mod free_random_walk_tests {
    use crate::core::ArrowSpace;
    use crate::operators::{build_knn_graph};
    use rand::prelude::*;

    /// Simulate 2D free random walk of one particle on integer lattice
    /// Returns sequence of visited coordinates
    fn simulate_free_random_walk(steps: usize, start: (usize, usize)) -> Vec<(usize, usize)> {
        let mut rng = rand::rng();
        let mut pos = start;
        let mut trajectory = Vec::with_capacity(steps);
        trajectory.push(pos);

        for _ in 1..steps {
            // Random step in 4 directions: up, down, left, right
            let direction = rng.random_range(0..4);
            pos = match direction {
                0 => (pos.0.wrapping_add(1), pos.1), // right
                1 => (pos.0.wrapping_sub(1), pos.1), // left
                2 => (pos.0, pos.1.wrapping_add(1)), // up
                _ => (pos.0, pos.1.wrapping_sub(1)), // down
            };
            trajectory.push(pos);
        }
        trajectory
    }

    /// Generate concentration field from random walk trajectory
    /// Models local concentration as function of visit frequency
    fn concentration_field(trajectory: &[(usize, usize)]) -> Vec<f64> {
        use std::collections::HashMap;

        // Count visits to each position
        let mut visit_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for &pos in trajectory {
            *visit_counts.entry(pos).or_insert(0) += 1;
        }

        // Create concentration signal: log(1 + visit_count)
        let mut signal = Vec::with_capacity(trajectory.len());
        for &pos in trajectory {
            let count = visit_counts[&pos] as f64;
            signal.push((1.0 + count).ln());
        }
        signal
    }

    /// Generate diffusion potential field (distance from starting point)
    fn diffusion_potential(trajectory: &[(usize, usize)], start: (usize, usize)) -> Vec<f64> {
        trajectory
            .iter()
            .map(|&(x, y)| {
                let dx = (x as f64) - (start.0 as f64);
                let dy = (y as f64) - (start.1 as f64);
                (dx * dx + dy * dy).sqrt()
            })
            .collect()
    }

    #[test]
    fn test_molecular_diffusion_single_particle() {
        let steps = 1000;
        let start_pos = (50, 50);

        // Simulate random walk
        let trajectory = simulate_free_random_walk(steps, start_pos);

        // Build spatial graph from trajectory points
        let graph = build_knn_graph(&trajectory, 6, 2.0, None);

        // Test 1: Uniform concentration (all positions equally likely)
        let uniform_signal = vec![1.0f64; trajectory.len()];
        let mut aspace_uniform = ArrowSpace::from_items(vec![uniform_signal], vec![0.0]);
        aspace_uniform.recompute_lambdas(&graph);
        let lambda_uniform = aspace_uniform.lambdas()[0];

        // Test 2: Visit-frequency concentration field
        let concentration = concentration_field(&trajectory);
        let mut aspace_conc = ArrowSpace::from_items(vec![concentration], vec![0.0]);
        aspace_conc.recompute_lambdas(&graph);
        let lambda_concentration = aspace_conc.lambdas()[0];

        // Test 3: Diffusion potential (distance from origin)
        let potential = diffusion_potential(&trajectory, start_pos);
        let mut aspace_pot = ArrowSpace::from_items(vec![potential], vec![0.0]);
        aspace_pot.recompute_lambdas(&graph);
        let lambda_potential = aspace_pot.lambdas()[0];

        println!("Random walk diffusion simulation results:");
        println!("  Steps: {}", steps);
        println!("  Lambda (uniform): {:.6}", lambda_uniform);
        println!("  Lambda (concentration): {:.6}", lambda_concentration);
        println!("  Lambda (potential): {:.6}", lambda_potential);

        // Physical expectations:
        // - All lambdas should be non-negative (energy condition)
        assert!(lambda_uniform >= 0.0, "Uniform lambda negative");
        assert!(lambda_concentration >= 0.0, "Concentration lambda negative");
        assert!(lambda_potential >= 0.0, "Potential lambda negative");

        // - Concentration field should be smoother (lower lambda) than potential
        // because visited regions create local smoothness
        println!(
            "  Smoothness ratio (conc/pot): {:.3}",
            lambda_concentration / lambda_potential
        );

        // - Reasonable bounds for graph Laplacian eigenvalues
        assert!(
            lambda_uniform < 20.0,
            "Uniform lambda too large: {}",
            lambda_uniform
        );
        assert!(
            lambda_concentration < 30.0,
            "Concentration lambda too large: {}",
            lambda_concentration
        );
        assert!(
            lambda_potential < 50.0,
            "Potential lambda too large: {}",
            lambda_potential
        );
    }

    #[test]
    fn test_diffusion_superposition() {
        let steps = 500;

        // Two particles starting from different positions
        let traj_a = simulate_free_random_walk(steps, (30, 50));
        let traj_b = simulate_free_random_walk(steps, (70, 50));

        // Combined trajectory (particle interaction simulation)
        let mut combined_traj = traj_a.clone();
        combined_traj.extend_from_slice(&traj_b);

        let graph = build_knn_graph(&combined_traj, 4, 2.0, None);

        // Individual concentration fields
        let conc_a = concentration_field(&traj_a);
        let conc_b = concentration_field(&traj_b);

        // Pad to combined length
        let mut signal_a = conc_a;
        signal_a.resize(combined_traj.len(), 0.0);
        let mut signal_b = vec![0.0; traj_a.len()];
        signal_b.extend_from_slice(&concentration_field(&traj_b));

        let mut aspace = ArrowSpace::from_items(vec![signal_a, signal_b], vec![0.0, 0.0]);
        aspace.recompute_lambdas(&graph);

        let lambda_a = aspace.lambdas()[0];
        let lambda_b = aspace.lambdas()[1];

        println!("Two-particle diffusion:");
        println!("  Lambda A: {:.6}", lambda_a);
        println!("  Lambda B: {:.6}", lambda_b);

        // Superpose concentrations (molecular interaction)
        aspace. add_features(0, 1, &graph);
        let lambda_combined = aspace.lambdas()[0];

        println!("  Lambda combined: {:.6}", lambda_combined);
        println!(
            "  Interaction effect: {:.3}",
            lambda_combined / lambda_a.max(lambda_b)
        );

        // Verify lambda behavior
        assert!(lambda_a >= 0.0 && lambda_b >= 0.0);
        assert!(lambda_combined >= 0.0);

        // Combined field can have different smoothness due to interference
        assert!(
            lambda_combined < 2.0 * lambda_a.max(lambda_b),
            "Combined lambda unexpectedly large"
        );
    }

    #[test]
    fn test_diffusion_time_evolution() {
        let start_pos = (50, 50);
        let time_points = vec![100, 250, 500, 1000];

        println!("Diffusion time evolution:");

        for &t in &time_points {
            let trajectory = simulate_free_random_walk(t, start_pos);
            let graph = build_knn_graph(&trajectory, 4, 2.0, None);

            // Measure spread: standard deviation of positions
            let mean_x: f64 = trajectory.iter().map(|&(x, _)| x as f64).sum::<f64>() / t as f64;
            let mean_y: f64 = trajectory.iter().map(|&(_, y)| y as f64).sum::<f64>() / t as f64;
            let spread = trajectory
                .iter()
                .map(|&(x, y)| {
                    let dx = (x as f64) - mean_x;
                    let dy = (y as f64) - mean_y;
                    dx * dx + dy * dy
                })
                .sum::<f64>()
                / t as f64;
            let rms_displacement = spread.sqrt();

            // Arrow lambda for position field
            let position_signal: Vec<f64> = trajectory
                .iter()
                .map(|&(x, y)| {
                    ((x as f64 - start_pos.0 as f64).powi(2)
                        + (y as f64 - start_pos.1 as f64).powi(2))
                    .sqrt()
                })
                .collect();

            let mut aspace = ArrowSpace::from_items(vec![position_signal], vec![0.0]);
            aspace.recompute_lambdas(&graph);
            let lambda = aspace.lambdas()[0];

            println!(
                "  t={:4}: RMS displacement={:.2}, Lambda={:.4}",
                t, rms_displacement, lambda
            );

            assert!(lambda >= 0.0);
            assert!(rms_displacement >= 0.0);
        }
    }
}
