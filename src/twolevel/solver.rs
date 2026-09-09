use super::operator::{BlochVec, Hamiltonian, TimeDependentHamiltonian, Unitary, commutator_norm};
use crate::maths::Linspace;

pub struct Solver {
    pub h_t: Vec<Hamiltonian>,
    times: Linspace,
}

impl Solver {
    /// Samples the Hamiltonian at each interval's left endpoint. The final
    /// boundary ends the last step and is not an additional Hamiltonian sample.
    pub fn from(system: &impl TimeDependentHamiltonian, times: Linspace) -> Self {
        Self {
            h_t: times
                .array
                .windows(2)
                .map(|interval| system.h(interval[0]))
                .collect(),
            times,
        }
    }

    /// Returns the derivative of final ground probability with respect to a kick
    /// `exp(-i * epsilon * perturbation.r.sigma / 2)` after each stored step.
    /// No time-step factor is included. After reduction, entries correspond to
    /// the reduced steps rather than the original Hamiltonian samples.
    /// Before reduction, the kick times are `times.array[1..]`, ending at `stop`.
    pub fn linear_response(&self, initial: BlochVec, perturbation: Hamiltonian) -> Vec<f64> {
        let n = self.h_t.len();
        let mut trajectory = Vec::with_capacity(n);
        let mut forward = initial;

        for &h in &self.h_t {
            let step = Unitary::from_hamiltonian(h, self.times.step);
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

    /// Reduces the original step sequence and returns the end time of each
    /// reduced step, matching the states and kicks returned by this solver.
    pub fn trotter_reduce(&mut self, tolerance: f64) -> Vec<f64> {
        let mut new_h_t: Vec<Hamiltonian> = Vec::new();
        let mut new_times: Vec<f64> = Vec::new();
        let mut current = self.h_t[0];
        let mut current_t = self.times.array[1];
        let tol = tolerance * 4.0 / (self.times.array.len() - 1) as f64;
        for (i, &h) in self.h_t[1..].iter().enumerate() {
            if self.times.step.powi(2) * commutator_norm(h, current) < tol {
                current = Hamiltonian::new(
                    current.r.x + h.r.x,
                    current.r.y + h.r.y,
                    current.r.z + h.r.z,
                );
            } else {
                new_h_t.push(current);
                new_times.push(current_t);
                current = h;
            }
            current_t = self.times.array[i + 2];
        }
        new_h_t.push(current);
        new_times.push(current_t);
        (*self).h_t = new_h_t;
        new_times
    }

    /// Returns the state after each step, excluding `initial`. Before reduction,
    /// these states correspond to `times.array[1..]`, ending at `stop`.
    pub fn propagate(&self, initial: BlochVec) -> Vec<BlochVec> {
        let mut state = initial;
        self.h_t
            .iter()
            .map(|&h| {
                state = Unitary::from_hamiltonian(h, self.times.step).apply_to(state);
                state
            })
            .collect()
    }

    pub fn propagate_to_final(&self, initial: BlochVec) -> BlochVec {
        let mut state = initial;
        for &h in &self.h_t {
            state = Unitary::from_hamiltonian(h, self.times.step).apply_to(state);
        }
        state
    }

    /// Composes all step unitaries before applying the net rotation to `initial`.
    pub fn propagate_to_final_composed(&self, initial: BlochVec) -> BlochVec {
        let mut total = Unitary::identity();
        for &h in &self.h_t {
            let step = Unitary::from_hamiltonian(h, self.times.step);
            total = step.compose(total);
        }
        total.normalised().apply_to(initial)
    }
}
