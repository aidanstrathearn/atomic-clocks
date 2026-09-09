use crate::maths::normalised_gaussian;
use crate::twolevel::{Hamiltonian, TimeDependentHamiltonian};

pub struct Ramsey {
    pub pulse_area: f64,
    pub detuning: f64,
    pub pulse_width: f64,
    pub pulse_separation: f64,
    pub phase_diff: f64,
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
    pub pulse_area: f64,
    pub detuning: f64,
    pub mod_freq: f64,
    pub mod_depth: f64,
    pub start_time: f64,
    pub pulse_width: f64,
    pub pulse_separation: f64,
}

impl TimeDependentHamiltonian for ModulatedRamsey {
    fn h(&self, t: f64) -> Hamiltonian {
        let hx = self.pulse_area
            * (normalised_gaussian(t, 0.0, self.pulse_width)
                + normalised_gaussian(t, self.pulse_separation, self.pulse_width));
        Hamiltonian::new(
            hx,
            0.0,
            self.detuning + self.mod_depth * (self.mod_freq * (t + self.start_time)).sin(),
        )
    }
}
