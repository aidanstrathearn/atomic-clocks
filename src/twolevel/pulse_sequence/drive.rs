use super::{PulseParameters, ResolvedPulseSequence};

/// Fixed drive and dephasing parameters for a resolved pulse sequence.
#[derive(Clone, Copy)]
pub struct PulseDrive {
    rabi_angular_frequency: f64,
    gamma_x: f64,
    gamma_phi: f64,
}

impl PulseDrive {
    pub fn new(rabi_angular_frequency: f64, gamma_x: f64, gamma_phi: f64) -> Self {
        Self {
            rabi_angular_frequency,
            gamma_x,
            gamma_phi,
        }
    }

    pub fn at_detuning_hz(self, detuning_hz: f64) -> PulseParameters {
        PulseParameters {
            rabi_angular_frequency: self.rabi_angular_frequency,
            detuning_angular_frequency: std::f64::consts::TAU * detuning_hz,
            gamma_x: self.gamma_x,
            gamma_phi: self.gamma_phi,
        }
    }
}

/// A resolved sequence together with its fixed drive and noise parameters.
#[derive(Clone, Copy)]
pub struct DrivenPulseSequence<S> {
    sequence: S,
    drive: PulseDrive,
}

impl<S> DrivenPulseSequence<S> {
    pub fn new(sequence: S, drive: PulseDrive) -> Self {
        Self { sequence, drive }
    }
}

impl<S> DrivenPulseSequence<S>
where
    S: ResolvedPulseSequence,
{
    pub fn z_response(&self, detuning_hz: f64) -> f64 {
        self.sequence
            .z_response(self.drive.at_detuning_hz(detuning_hz))
    }

    pub fn ground_probability(&self, detuning_hz: f64) -> f64 {
        self.sequence
            .ground_probability(self.drive.at_detuning_hz(detuning_hz))
    }
}
