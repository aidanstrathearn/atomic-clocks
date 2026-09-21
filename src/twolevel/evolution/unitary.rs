//! Hamiltonian evolution and unitary channel construction.

use crate::maths::Linspace;
use crate::maths::mat3::Mat3;
use crate::maths::vec3::Vec3;

use crate::twolevel::{AffineChannel, BlochVec, Channel, ComposableChannel, Observable, Process};

#[derive(Copy, Clone, Debug)]
pub struct TrotterConfig {
    pub start: f64,
    pub stop: f64,
    pub nsteps: usize,
    pub tolerance: f64,
}

impl TrotterConfig {
    pub(crate) fn validate(self) {
        assert!(self.nsteps > 0, "at least one integration step is required");
        assert!(
            self.start.is_finite() && self.stop.is_finite() && self.stop > self.start,
            "time boundaries must be finite with stop > start"
        );
        assert!(
            (self.stop - self.start).is_finite(),
            "evolution duration must be finite"
        );
        assert!(
            self.tolerance.is_finite() && self.tolerance >= 0.0,
            "reduction tolerance must be finite and nonnegative"
        );
    }

    /// Returns the `nsteps + 1` boundaries of the configured time grid.
    pub fn time_grid(self) -> Linspace {
        self.validate();
        Linspace::new(self.start, self.stop, self.nsteps)
    }

    fn intervals(self) -> impl ExactSizeIterator<Item = (f64, f64)> {
        let dt = (self.stop - self.start) / self.nsteps as f64;
        (0..self.nsteps).map(move |i| (self.start + i as f64 * dt, dt))
    }
}

#[derive(Copy, Clone)]
pub struct Unitary {
    // U = c I - i u.sigma; c and u together form a unit quaternion.
    c: f64,
    u: Vec3,
}

impl Unitary {
    /// Equivalent to `hamiltonian.for_duration(dt)`.
    pub fn from_hamiltonian(hamiltonian: Hamiltonian, dt: f64) -> Self {
        hamiltonian.for_duration(dt)
    }

    pub fn from_system(system: &impl TimeDependentHamiltonian, config: TrotterConfig) -> Self {
        config.validate();
        let total = if config.tolerance > 0.0 {
            Self::reduced_compose(system, config)
        } else {
            Self::direct_compose(system, config)
        };
        total.normalised()
    }

    fn direct_compose(system: &impl TimeDependentHamiltonian, config: TrotterConfig) -> Self {
        Process::new(
            config
                .intervals()
                .map(|(t, dt)| system.h(t).for_duration(dt)),
        )
        .compose()
    }

    fn reduced_compose(system: &impl TimeDependentHamiltonian, config: TrotterConfig) -> Self {
        let threshold = 4.0 * (config.tolerance / config.nsteps as f64);
        let mut total = Self::identity();
        let mut pending = Hamiltonian::default();
        for (t, dt) in config.intervals() {
            let action = Hamiltonian {
                r: system.h(t).r * dt,
            };
            if commutator_norm(pending, action) < threshold {
                pending.r = pending.r + action.r;
            } else {
                total = pending.for_duration(1.0).compose(&total);
                pending = action;
            }
        }
        pending.for_duration(1.0).compose(&total)
    }

    pub fn inverse(&self) -> Self {
        Self {
            c: self.c,
            u: self.u * -1.0,
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
}

impl Channel for Unitary {
    fn apply_to(&self, bloch: BlochVec) -> BlochVec {
        BlochVec {
            r: self.apply_traceless(bloch.r),
        }
    }

    fn apply_traceless(&self, r: Vec3) -> Vec3 {
        let t = self.u.cross(r) * 2.0;
        r + t * self.c + self.u.cross(t)
    }

    fn pull_back_traceless_observable(&self, o: Vec3) -> Vec3 {
        self.inverse().apply_traceless(o)
    }

    fn to_affine(&self) -> AffineChannel {
        let columns = [
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        ]
        .map(|r| {
            let rotated = self.apply_to(BlochVec { r }).r;
            [rotated.x, rotated.y, rotated.z]
        });
        AffineChannel::new(
            Mat3::from_columns(columns),
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
    }
}

impl ComposableChannel for Unitary {
    fn identity() -> Self {
        Self {
            c: 1.0,
            u: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    /// Apply `earlier` first, then `self`, removing quaternion norm drift.
    fn compose(&self, earlier: &Self) -> Self {
        Self {
            c: self.c * earlier.c - self.u.dot(earlier.u),
            u: earlier.u * self.c + self.u * earlier.c + self.u.cross(earlier.u),
        }
        .normalised()
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

    /// Evolution under this constant Hamiltonian for `dt`.
    /// Negative durations produce the inverse rotation.
    pub fn for_duration(&self, dt: f64) -> Unitary {
        let v = self.r * dt;
        let angle = v.norm();
        if angle == 0.0 {
            return Unitary::identity();
        }
        let (sin_theta, cos_theta) = (0.5 * angle).sin_cos();
        Unitary {
            c: cos_theta,
            u: v * (sin_theta / angle),
        }
    }
}

impl From<Hamiltonian> for Observable {
    /// Preserve `H = h.sigma / 2` in the observable convention `a I + o.sigma`.
    fn from(hamiltonian: Hamiltonian) -> Self {
        Self::new(0.0, hamiltonian.r * 0.5)
    }
}

pub fn commutator_norm(h1: Hamiltonian, h2: Hamiltonian) -> f64 {
    h1.r.cross(h2.r).norm()
}

pub trait TimeDependentHamiltonian {
    fn h(&self, t: f64) -> Hamiltonian;
}
