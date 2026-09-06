use crate::maths::normalised_gaussian;
use crate::twolevel::{Hamiltonian, TimeDependentHamiltonian};

trait Pulse: TimeDependentHamiltonian {

}
pub struct GaussianPulse {
    pub pulse_area: f64,
    pub detuning: f64,
    pub width: f64,
    pub center: f64,
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