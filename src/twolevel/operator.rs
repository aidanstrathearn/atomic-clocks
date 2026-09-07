use super::state::BlochVec;
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
