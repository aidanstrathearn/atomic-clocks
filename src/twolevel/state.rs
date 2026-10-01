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

    /// Rotates this state about a unit Bloch-sphere axis using the right-hand rule.
    #[must_use]
    #[inline]
    pub fn rotate(self, axis: Vec3, angle: f64) -> Self {
        Self {
            r: self.r.rotate(axis, angle),
        }
    }

    /// Rotates this state about +z using the right-hand rule.
    #[must_use]
    #[inline]
    pub fn rotate_about_z(mut self, angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = self.r.x;
        let y = self.r.y;
        self.r.x = cos * x - sin * y;
        self.r.y = sin * x + cos * y;
        self
    }

    /// Scales transverse coherence while preserving the populations.
    ///
    /// `coherence` is the remaining transverse-coherence fraction: `1` leaves
    /// the state unchanged and `0` completely dephases it in the
    /// ground/excited basis.
    #[must_use]
    #[inline]
    pub fn dephase(mut self, coherence: f64) -> Self {
        self.r.x *= coherence;
        self.r.y *= coherence;
        self
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
