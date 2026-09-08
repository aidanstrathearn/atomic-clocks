use super::state::BlochVec;
use crate::maths::vec3::Vec3;

#[derive(Copy, Clone)]
pub struct Unitary {
    // U = c I - i u.sigma; c and u together form a unit quaternion.
    c: f64,
    u: Vec3,
}

impl Unitary {
    pub fn identity() -> Self {
        Self {
            c: 1.0,
            u: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    pub fn from_hamiltonian(hamiltonian: Hamiltonian, dt: f64) -> Self {
        let v = hamiltonian.r * dt;
        let angle = v.norm();
        if angle == 0.0 {
            return Self::identity();
        }
        let (sin_theta, cos_theta) = (0.5 * angle).sin_cos();
        Self {
            c: cos_theta,
            u: v * (sin_theta / angle),
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            c: self.c,
            u: self.u * -1.0,
        }
    }

    /// Returns `self * earlier`: apply `earlier` first, then `self`.
    pub fn compose(&self, earlier: Self) -> Self {
        Self {
            c: self.c * earlier.c - self.u.dot(earlier.u),
            u: earlier.u * self.c + self.u * earlier.c + self.u.cross(earlier.u),
        }
    }

    /// Removes floating-point norm drift after accumulating rotations.
    pub(crate) fn normalised(&self) -> Self {
        let scale = 1.0 / (self.c * self.c + self.u.dot(self.u)).sqrt();
        Self {
            c: self.c * scale,
            u: self.u * scale,
        }
    }

    pub fn apply_to(&self, bloch: BlochVec) -> BlochVec {
        let t = self.u.cross(bloch.r) * 2.0;
        BlochVec {
            r: bloch.r + t * self.c + self.u.cross(t),
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
