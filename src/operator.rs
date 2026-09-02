use num_complex::Complex64;

use crate::maths::normalised_gaussian;

#[derive(Copy, Clone)]
pub struct Hamiltonian {
    hx: f64,
    hy: f64,
    hz: f64,
}

impl Hamiltonian {
    pub fn new(hx: f64, hy: f64, hz: f64) -> Self {
        Self { hx, hy, hz }
    }

    pub fn norm(&self) -> f64 {
        (self.hx * self.hx + self.hy * self.hy + self.hz * self.hz).sqrt()
    }
}

pub trait TimeDependentHamiltonian {
    fn h(&self, t: f64) -> Hamiltonian;
}

pub struct GaussianPulse {
    pub(crate) pulse_area: f64,
    pub(crate) detuning: f64,
    pub(crate) width: f64,
    pub(crate) center: f64,
}

impl TimeDependentHamiltonian for GaussianPulse {
    fn h(&self, t: f64) -> Hamiltonian {
        Hamiltonian::new(
            self.pulse_area * normalised_gaussian(t, self.center, self.width),
            0.0,
            self.detuning,
        )
    }
}

#[derive(Copy, Clone)]
pub struct Operator {
    pub(crate) gg: Complex64,
    pub(crate) ge: Complex64,
    pub(crate) eg: Complex64,
    pub(crate) ee: Complex64,
}

impl Operator {
    pub fn identity() -> Self {
        Self {
            gg: Complex64::new(1.0, 0.0),
            eg: Complex64::new(0.0, 0.0),
            ge: Complex64::new(0.0, 0.0),
            ee: Complex64::new(1.0, 0.0),
        }
    }

    pub fn pauli_x() -> Self {
        Self {
            gg: Complex64::new(0.0, 0.0),
            eg: Complex64::new(1.0, 0.0),
            ge: Complex64::new(1.0, 0.0),
            ee: Complex64::new(0.0, 0.0),
        }
    }

    pub fn pauli_y() -> Self {
        Self {
            gg: Complex64::new(0.0, 0.0),
            eg: Complex64::new(0.0, -1.0),
            ge: Complex64::new(0.0, 1.0),
            ee: Complex64::new(0.0, 0.0),
        }
    }

    pub fn pauli_z() -> Self {
        Self {
            gg: Complex64::new(-1.0, 0.0),
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

        let sx = k * hamiltonian.hx;
        let sy = k * hamiltonian.hy;
        let sz = k * hamiltonian.hz;

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
