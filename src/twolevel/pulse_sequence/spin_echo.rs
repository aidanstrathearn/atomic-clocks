use crate::twolevel::BlochVec;

use super::resolved::apply_dark_time;
use super::{PulseParameters, ResolvedPulseSequence, prepare_pulse};

#[derive(Clone, Copy)]
pub struct SpinEcho {
    pi_over_two_duration_s: f64,
    pi_duration_s: f64,
    dark_before_s: f64,
    dark_after_s: f64,
    final_pulse_phase_rad: f64,
}

impl SpinEcho {
    pub fn new(
        pi_over_two_duration_s: f64,
        pi_duration_s: f64,
        dark_before_s: f64,
        dark_after_s: f64,
        final_pulse_phase_rad: f64,
    ) -> Self {
        Self {
            pi_over_two_duration_s,
            pi_duration_s,
            dark_before_s,
            dark_after_s,
            final_pulse_phase_rad,
        }
    }
}

impl ResolvedPulseSequence for SpinEcho {
    fn apply_to(&self, parameters: PulseParameters, initial: BlochVec) -> BlochVec {
        let pi_over_two_pulse = prepare_pulse(parameters, self.pi_over_two_duration_s);
        let pi_pulse = prepare_pulse(parameters, self.pi_duration_s);
        let after_first_pulse = pi_over_two_pulse.apply_to(initial);
        let before_pi_pulse = apply_dark_time(
            after_first_pulse,
            parameters.detuning_angular_frequency * self.dark_before_s,
            (-parameters.gamma_phi * self.dark_before_s).exp(),
        );
        let after_pi_pulse = pi_pulse.apply_to(before_pi_pulse);
        let before_final_pulse = apply_dark_time(
            after_pi_pulse,
            parameters.detuning_angular_frequency * self.dark_after_s,
            (-parameters.gamma_phi * self.dark_after_s).exp(),
        );
        pi_over_two_pulse.apply_to_with_phase(before_final_pulse, self.final_pulse_phase_rad)
    }
}
