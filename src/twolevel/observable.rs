//! Real Pauli coefficients of Hermitian qubit observables.

use crate::maths::vec3::Vec3;

use super::state::BlochVec;

/// A Hermitian qubit operator `O = scalar I + vector.sigma`.
///
/// For a state `rho = (I + r.sigma) / 2`, its expectation is
/// `scalar + vector.dot(r)`. The same operator can specify a Hamiltonian
/// perturbation in a response calculation.
#[derive(Clone, Copy)]
pub struct Observable {
    scalar: f64,
    vector: Vec3,
}

impl Observable {
    pub fn new(scalar: f64, vector: Vec3) -> Self {
        Self { scalar, vector }
    }

    /// Ground-state projector `(I - sigma_z) / 2`.
    pub fn ground_projector() -> Self {
        Self::new(
            0.5,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: -0.5,
            },
        )
    }

    pub fn scalar(&self) -> f64 {
        self.scalar
    }

    pub fn vector(&self) -> Vec3 {
        self.vector
    }

    /// `Tr(O rho)` for a normalized state.
    pub fn expectation(&self, state: BlochVec) -> f64 {
        self.scalar + self.vector.dot(state.r)
    }

    /// `Tr(O X)` for the traceless operator `X = variation.sigma / 2`.
    /// The observable's identity coefficient does not contribute.
    pub fn pair_traceless(&self, variation: Vec3) -> f64 {
        self.vector.dot(variation)
    }
}
