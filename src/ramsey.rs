use std::f64::consts::PI;

use crate::maths::{Linspace, normalised_gaussian};
use crate::operator::{Hamiltonian, TimeDependentHamiltonian};
use crate::state::QubitState;

pub struct Ramsey {
    pub(crate) pulse_area: f64,
    pub(crate) detuning: f64,
    pub(crate) pulse_width: f64,
    pub(crate) pulse_separation: f64,
    pub(crate) phase_diff: f64,
}

impl Ramsey {
    pub fn propagate_to_final(&self, initial: &QubitState, times: &Linspace) -> QubitState {
        initial.propagate_to_final(self, times)
    }
}

impl TimeDependentHamiltonian for Ramsey {
    fn h(&self, t: f64) -> Hamiltonian {
        let (sin_phi, cos_phi) = self.phase_diff.sin_cos();
        let pulse1 = normalised_gaussian(t, 0.0, self.pulse_width);
        let pulse2 = normalised_gaussian(t, self.pulse_separation, self.pulse_width);
        Hamiltonian::new(
            self.pulse_area * (pulse1 + pulse2 * cos_phi),
            self.pulse_area * pulse2 * sin_phi,
            self.detuning,
        )
    }
}

pub struct ModulatedRamsey {
    pub(crate) pulse_area: f64,
    pub(crate) detuning: f64,
    pub(crate) mod_freq: f64,
    pub(crate) mod_depth: f64,
    pub(crate) start_time: f64,
    pub(crate) pulse_width: f64,
    pub(crate) pulse_separation: f64,
}

impl TimeDependentHamiltonian for ModulatedRamsey {
    fn h(&self, t: f64) -> Hamiltonian {
        let hx = self.pulse_area
            * (normalised_gaussian(t, self.start_time, self.pulse_width)
                + normalised_gaussian(
                    t,
                    self.pulse_separation + self.start_time,
                    self.pulse_width,
                ));
        Hamiltonian::new(
            hx,
            0.0,
            self.detuning + self.mod_depth * (2.0 * PI * self.mod_freq * t).sin(),
        )
    }
}
