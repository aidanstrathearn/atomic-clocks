use super::operator::{BlochVec, Hamiltonian, TimeDependentHamiltonian, Unitary, commutator_norm};

#[derive(Clone)]
pub struct Solver {
    hamiltonians: Vec<Hamiltonian>,
    times: Vec<f64>,
}

impl Solver {
    pub fn new(hamiltonians: Vec<Hamiltonian>, times: Vec<f64>) -> Self {
        Self::validate_times(&times);
        assert_eq!(
            times.len() - 1,
            hamiltonians.len(),
            "one Hamiltonian is required per interval"
        );
        Self {
            hamiltonians,
            times,
        }
    }

    pub fn from(system: &impl TimeDependentHamiltonian, times: Vec<f64>) -> Self {
        Self::validate_times(&times);
        Self {
            hamiltonians: times
                .windows(2)
                .map(|interval| system.h(interval[0]))
                .collect(),
            times,
        }
    }

    pub fn hamiltonians(&self) -> &[Hamiltonian] {
        &self.hamiltonians
    }

    /// Interval boundaries, including the initial and final times.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    fn validate_times(times: &[f64]) {
        assert!(!times.is_empty(), "at least one time boundary is required");
        assert!(
            times.iter().all(|t| t.is_finite()),
            "time boundaries must be finite"
        );
        assert!(
            times.windows(2).all(|pair| pair[1] > pair[0]),
            "time boundaries must be strictly increasing"
        );
        assert!(
            (times[times.len() - 1] - times[0]).is_finite(),
            "evolution duration must be finite"
        );
    }

    fn steps(&self) -> impl Iterator<Item = (Hamiltonian, f64)> + '_ {
        self.hamiltonians
            .iter()
            .zip(self.times.windows(2))
            .map(|(&h, interval)| (h, interval[1] - interval[0]))
    }

    pub fn linear_response(&self, initial: BlochVec, perturbation: Hamiltonian) -> Vec<f64> {
        let n = self.hamiltonians.len();
        let mut trajectory = Vec::with_capacity(n);
        let mut forward = initial;

        for (h, dt) in self.steps() {
            let step = Unitary::from_hamiltonian(h, dt);
            forward = step.apply_to(forward);
            trajectory.push((step, forward));
        }

        let mut responses = vec![0.0; n];
        // A pure state's Bloch vector also specifies its measurement projector.
        let mut measurement = BlochVec::ground();

        for i in (0..n).rev() {
            let (step, forward_state) = trajectory[i];
            responses[i] = 0.5 * measurement.r.dot(perturbation.r.cross(forward_state.r));
            measurement = step.inverse().apply_to(measurement);
        }

        responses
    }

    #[must_use]
    pub fn trotter_reduce(&self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= 0.0,
            "reduction tolerance must be finite and nonnegative"
        );
        if self.hamiltonians.len() < 2 || tolerance == 0.0 {
            return self.clone();
        }
        let mut hamiltonians = Vec::new();
        let mut times = vec![self.times[0]];
        let mut current = self.hamiltonians[0];
        let mut start = self.times[0];
        let mut end = self.times[1];
        let tol = 4.0 * (tolerance / self.hamiltonians.len() as f64);
        for (i, (h, dt)) in self.steps().enumerate().skip(1) {
            let duration = end - start;
            let next_end = self.times[i + 1];
            let current_action = Hamiltonian {
                r: current.r * duration,
            };
            let next_action = Hamiltonian { r: h.r * dt };
            if commutator_norm(current_action, next_action) < tol {
                current = Hamiltonian {
                    r: (current_action.r + next_action.r) * (1.0 / (next_end - start)),
                };
            } else {
                hamiltonians.push(current);
                times.push(end);
                current = h;
                start = end;
            }
            end = next_end;
        }
        hamiltonians.push(current);
        times.push(end);
        Self::new(hamiltonians, times)
    }

    /// Returns the state after each step, excluding `initial`. These states
    /// correspond to `times()[1..]`, including on a reduced grid.
    pub fn propagate(&self, initial: BlochVec) -> Vec<BlochVec> {
        let mut state = initial;
        self.steps()
            .map(|(h, dt)| {
                state = Unitary::from_hamiltonian(h, dt).apply_to(state);
                state
            })
            .collect()
    }

    pub fn propagate_to_final(&self, initial: BlochVec) -> BlochVec {
        let mut state = initial;
        for (h, dt) in self.steps() {
            state = Unitary::from_hamiltonian(h, dt).apply_to(state);
        }
        state
    }

    /// Composes all step unitaries before applying the net rotation to `initial`.
    pub fn propagate_to_final_composed(&self, initial: BlochVec) -> BlochVec {
        let mut total = Unitary::identity();
        for (h, dt) in self.steps() {
            let step = Unitary::from_hamiltonian(h, dt);
            total = step.compose(total);
        }
        total.normalised().apply_to(initial)
    }
}
