use num_complex::{Complex, Complex64};

use super::operator::{Hamiltonian, Operator, TimeDependentHamiltonian};
use crate::maths::Linspace;

#[derive(Copy, Clone)]
pub struct QubitState {
    pub ground: Complex64,
    pub excited: Complex64,
}

impl QubitState {
    pub fn ground_probability(&self) -> f64 {
        self.ground.norm_sqr()
    }

    pub fn excited_probability(&self) -> f64 {
        self.excited.norm_sqr()
    }

    pub fn apply_pauli_x(&self) -> Self {
        Self {
            ground: self.excited,
            excited: self.ground,
        }
    }

    pub fn apply_pauli_y(&self) -> Self {
        Self {
            ground: self.excited * Complex::I,
            excited: -self.ground * Complex::I,
        }
    }

    pub fn apply_pauli_z(&self) -> Self {
        Self {
            ground: -self.ground,
            excited: self.excited,
        }
    }

    pub fn scale_real(&self, scale: f64) -> Self {
        Self {
            ground: self.ground * scale,
            excited: self.excited * scale,
        }
    }

    pub fn scale_complex(&self, scale: Complex64) -> Self {
        Self {
            ground: self.ground * scale,
            excited: self.excited * scale,
        }
    }

    pub fn add(&self, other: Self) -> Self {
        Self {
            ground: self.ground + other.ground,
            excited: self.excited + other.excited,
        }
    }

    pub fn ground() -> Self {
        Self {
            ground: Complex64::new(1.0, 0.0),
            excited: Complex64::new(0.0, 0.0),
        }
    }

    pub fn apply_operator(&self, op: Operator) -> Self {
        Self {
            ground: op.gg * self.ground + op.ge * self.excited,
            excited: op.eg * self.ground + op.ee * self.excited,
        }
    }

    pub fn ti_propagate(&self, hamiltonian: Hamiltonian, dt: f64) -> Self {
        let prop = Operator::ti_propagator(hamiltonian, dt);
        self.apply_operator(prop)
    }

    pub fn propagate(
        &self,
        hamiltonian: &impl TimeDependentHamiltonian,
        times: &Linspace,
    ) -> Vec<Self> {
        let mut state = *self;
        times
            .array
            .iter()
            .map(|&t| {
                state = state.ti_propagate(hamiltonian.h(t), times.step);
                state
            })
            .collect()
    }

    pub fn propagate_to_final(
        &self,
        hamiltonian: &impl TimeDependentHamiltonian,
        times: &Linspace,
    ) -> Self {
        let mut state = *self;
        for &t in &times.array {
            state = state.ti_propagate(hamiltonian.h(t), times.step);
        }
        state
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
            forward = forward.apply_operator(step);
            trajectory.push((step, forward));
        }

        let final_ground = forward.ground;
        let mut responses = vec![0.0; n];
        let mut future = Operator::identity();

        for i in (0..n).rev() {
            let (step, forward_state) = trajectory[i];
            let inserted = forward_state.apply_operator(perturbation);
            let response_amplitude = future.gg * inserted.ground + future.ge * inserted.excited;

            responses[i] = 2.0 * (final_ground.conj() * response_amplitude).im;
            future = future.multiply(step);
        }

        responses
    }
}
