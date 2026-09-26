//! Normalized qubit states in real Bloch coordinates.

use crate::maths::vec3::Vec3;

/// A normalized qubit state `rho = (I + r.sigma) / 2`.
/// Physical states have `r.norm() <= 1`; the trace-one coefficient is implicit.
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

    pub fn excited() -> Self {
        Self {
            r: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
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
