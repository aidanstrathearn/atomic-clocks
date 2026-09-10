use std::ops::Range;
use crate::maths::vec3::Vec3;

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
pub struct TrotterConfig {
    pub start: f64,
    pub stop: f64,
    pub nsteps: usize,
    pub tolerance: f64,
}

impl TrotterConfig {
    fn validate(self) {
        assert!(self.nsteps > 0, "at least one integration step is required");
        assert!(
            self.start.is_finite() && self.stop.is_finite() && self.stop >= self.start,
            "time boundaries must be finite with stop >= start"
        );
        assert!(
            self.tolerance.is_finite() && self.tolerance >= 0.0,
            "reduction tolerance must be finite and nonnegative"
        );
    }

    fn dt(self) -> f64 {
        (self.stop - self.start) / self.nsteps as f64
    }
}

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

    pub fn from_system(system: &impl TimeDependentHamiltonian, config: TrotterConfig) -> Self {
        config.validate();
        if config.start == config.stop {
            return Self::identity();
        }
        let total = if config.tolerance > 0.0 {
            Self::reduced_compose(system, config)
        } else {
            Self::direct_compose(system, config)
        };
        total.normalised()
    }

    fn direct_compose(system: &impl TimeDependentHamiltonian, config: TrotterConfig) -> Self {
        let dt = config.dt();
        let mut total = Self::identity();
        for i in 0..config.nsteps {
            let t = config.start + i as f64 * dt;
            total = Self::from_hamiltonian(system.h(t), dt).compose(total);
        }
        total
    }

    fn reduced_compose(system: &impl TimeDependentHamiltonian, config: TrotterConfig) -> Self {
        let dt = config.dt();
        let threshold = 4.0 * (config.tolerance / config.nsteps as f64);
        let mut total = Self::identity();
        let mut pending = Hamiltonian::default();
        for i in 0..config.nsteps {
            let t = config.start + i as f64 * dt;
            let action = Hamiltonian {
                r: system.h(t).r * dt,
            };
            if commutator_norm(pending, action) < threshold {
                pending.r = pending.r + action.r;
            } else {
                total = Self::from_hamiltonian(pending, 1.0).compose(total);
                pending = action;
            }
        }
        Self::from_hamiltonian(pending, 1.0).compose(total)
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

impl Default for Hamiltonian {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
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

/// Hamiltonian and excitation, decay, and pure-dephasing rates for a two-level system.
#[derive(Copy, Clone)]
pub struct Liouvillian {
    pub hamiltonian: Hamiltonian,
    pub gamma_up: f64,
    pub gamma_down: f64,
    pub gamma_phi: f64,
}

pub fn commutator_norm(h1: Hamiltonian, h2: Hamiltonian) -> f64 {
    h1.r.cross(h2.r).norm()
}

pub trait TimeDependentHamiltonian {
    fn h(&self, t: f64) -> Hamiltonian;
}
