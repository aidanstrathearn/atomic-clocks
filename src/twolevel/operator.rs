use num_complex::Complex64;

#[derive(Copy, Clone)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

#[derive(Copy, Clone)]
pub struct Hamiltonian {
    pub(crate) r: Vec3,
}

impl Hamiltonian {
    pub fn new(hx: f64, hy: f64, hz: f64) -> Self {
        Self {
            r: Vec3 { x: hx, y: hy, z: hz },
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
