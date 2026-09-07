use num_complex::Complex64;

use super::state::{BlochVec, QubitState};
use crate::vec3::Vec3;

#[derive(Copy, Clone)]
pub struct Unitary {
    pub r: Vec3,
}

impl Unitary {
    pub fn from_hamiltonian(hamiltonian: Hamiltonian, dt: f64) -> Self {
        Self {
            r: hamiltonian.r * dt,
        }
    }

    pub fn inverse(&self) -> Self {
        Self { r: self.r * -1.0 }
    }

    pub fn apply_to(&self, bloch: BlochVec) -> BlochVec {
        let angle = self.r.norm();
        if angle == 0.0 {
            return bloch;
        }
        let axis = self.r * (1.0 / angle);
        BlochVec {
            r: bloch.r.rotate(axis, angle),
        }
    }
}



#[derive(Copy, Clone)]
pub struct Hamiltonian {
    pub(crate) r: Vec3,
}

impl Hamiltonian {
    pub fn new(hx: f64, hy: f64, hz: f64) -> Self {
        Self {
            r: Vec3 {
                x: hx,
                y: hy,
                z: hz,
            },
        }
    }

    pub fn norm(&self) -> f64 {
        self.r.norm()
    }
}

pub fn commutator_norm(h1: Hamiltonian, h2: Hamiltonian) -> f64 {
    let h1h2 = h1.norm() * h2.norm();
    let cos_theta = (h1.r.x * h2.r.x + h1.r.y * h2.r.y + h1.r.z * h2.r.z) / h1h2;
    h1h2 * (1.0 - cos_theta * cos_theta).sqrt()
}

pub trait TimeDependentHamiltonian {
    fn h(&self, t: f64) -> Hamiltonian;
}

#[derive(Copy, Clone)]
pub struct Operator {
    pub(crate) gg: Complex64,
    pub(crate) ge: Complex64,
    pub(crate) eg: Complex64,
    pub(crate) ee: Complex64,
}

impl Operator {
    pub fn apply_to(&self, state: QubitState) -> QubitState {
        QubitState {
            ground: self.gg * state.ground + self.ge * state.excited,
            excited: self.eg * state.ground + self.ee * state.excited,
        }
    }

    pub fn identity() -> Self {
        Self {
            gg: Complex64::new(1.0, 0.0),
            eg: Complex64::new(0.0, 0.0),
            ge: Complex64::new(0.0, 0.0),
            ee: Complex64::new(1.0, 0.0),
        }
    }

    pub fn ti_propagator(hamiltonian: Hamiltonian, dt: f64) -> Self {
        let norm = hamiltonian.norm();
        if norm == 0.0 {
            return Self::identity();
        }
        let theta = norm * dt * 0.5;
        let (sin_theta, cos_theta) = theta.sin_cos();

        let k = sin_theta / norm;

        let sx = k * hamiltonian.r.x;
        let sy = k * hamiltonian.r.y;
        let sz = k * hamiltonian.r.z;

        Self {
            gg: Complex64::new(cos_theta, sz),
            eg: Complex64::new(-sy, -sx),
            ge: Complex64::new(sy, -sx),
            ee: Complex64::new(cos_theta, -sz),
        }
    }

    pub fn multiply(&self, other: Self) -> Self {
        Self {
            gg: self.gg * other.gg + self.ge * other.eg,
            ge: self.gg * other.ge + self.ge * other.ee,
            eg: self.eg * other.gg + self.ee * other.eg,
            ee: self.eg * other.ge + self.ee * other.ee,
        }
    }
}
