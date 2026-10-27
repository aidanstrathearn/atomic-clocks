use crate::twolevel::BlochVec;

use super::resolved::apply_dark_time;
use super::{PulseParameters, ResolvedPulseSequence, prepare_pulse};

#[derive(Clone, Copy)]
/// Two equal square pulses separated by a dark interval.
pub struct RamseyPair {
    pulse_duration_s: f64,
    dark_duration_s: f64,
}

impl RamseyPair {
    /// `pair_duration_s` is the combined duration of both pulses.
    /// `dark_duration_s` excludes the pulses.
    pub fn new(pair_duration_s: f64, dark_duration_s: f64) -> Self {
        Self {
            pulse_duration_s: 0.5 * pair_duration_s,
            dark_duration_s,
        }
    }
}

impl ResolvedPulseSequence for RamseyPair {
    fn apply_to(&self, parameters: PulseParameters, initial: BlochVec) -> BlochVec {
        if self.dark_duration_s == 0.0 {
            return prepare_pulse(parameters, 2.0 * self.pulse_duration_s).apply_to(initial);
        }

        let pulse = prepare_pulse(parameters, self.pulse_duration_s);
        let after_first = pulse.apply_to(initial);
        let after_dark_time = apply_dark_time(
            after_first,
            parameters.detuning_angular_frequency * self.dark_duration_s,
            (-parameters.gamma_phi * self.dark_duration_s).exp(),
        );
        pulse.apply_to(after_dark_time)
    }
}
