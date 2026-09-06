use num_complex::Complex64;

use super::operator::{commutator_norm, Hamiltonian, Operator, TimeDependentHamiltonian, Unitary};
use crate::maths::Linspace;
use crate::twolevel::Vec3;

#[derive(Copy, Clone)]
pub struct BlochVec {
    pub r: Vec3,
}

impl BlochVec {
    pub fn ground() -> Self {
        Self {
            r: Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
        }
    }
    pub fn excited_probability(&self) -> f64 {
        // The excited state lies at r.z = +1.
        0.5 * (1.0 + self.r.z)
    }

    pub fn ground_probability(&self) -> f64 {
        // The excited state lies at r.z = +1.
        0.5 * (1.0 - self.r.z)
    }

}

#[derive(Copy, Clone)]
pub struct QubitState {
    pub ground: Complex64,
    pub excited: Complex64,
}

impl QubitState {
    pub fn ground_probability(&self) -> f64 {
        self.ground.norm_sqr()
    }

    pub fn ground() -> Self {
        Self {
            ground: Complex64::new(1.0, 0.0),
            excited: Complex64::new(0.0, 0.0),
        }
    }

    pub fn linear_response(
        &self,
        hamiltonian: &impl TimeDependentHamiltonian,
        times: &Linspace,
        perturbation: Operator,
    ) -> Vec<f64> {
        let n = times.array.len();
        let mut trajectory = Vec::with_capacity(n);
        let mut forward = *self;

        for &t in &times.array {
            let step = Operator::ti_propagator(hamiltonian.h(t), times.step);
            forward = step.apply_to(forward);
            trajectory.push((step, forward));
        }

        let final_ground = forward.ground;
        let mut responses = vec![0.0; n];
        let mut future = Operator::identity();

        for i in (0..n).rev() {
            let (step, forward_state) = trajectory[i];
            let inserted = perturbation.apply_to(forward_state);
            let response_amplitude = future.gg * inserted.ground + future.ge * inserted.excited;

            responses[i] = 2.0 * (final_ground.conj() * response_amplitude).im;
            future = future.multiply(step);
        }

        responses
    }
}

pub struct Solver {
    pub h_t: Vec<Hamiltonian>,
    times: Linspace,
}

impl Solver {
    pub fn from(system: &impl TimeDependentHamiltonian, times: Linspace) -> Self {
        Self {
            h_t: times.array.iter().map(|&t| system.h(t)).collect(),
            times,
        }
    }

    pub fn trotter_reduce(&mut self, tolerance: f64) -> Vec<f64> {
        let mut new_h_t: Vec<Hamiltonian> = Vec::new();
        let mut new_times: Vec<f64> = Vec::new();
        let mut current = self.h_t[0];
        let mut current_t = self.times.array[0];
        let tol = tolerance * 4.0 / self.times.array.len() as f64;
        for &h in &self.h_t[1..] {
            if self.times.step.powi(2) * commutator_norm(h, current) < tol {
                current = Hamiltonian::new(
                    current.r.x + h.r.x,
                    current.r.y + h.r.y,
                    current.r.z + h.r.z,
                );
                current_t += self.times.step;
            } else {
                new_h_t.push(current);
                new_times.push(current_t);
                current = h;
                current_t += self.times.step;
            }
        }
        new_h_t.push(current);
        new_times.push(current_t);
        (*self).h_t = new_h_t;
        new_times
    }

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
}
