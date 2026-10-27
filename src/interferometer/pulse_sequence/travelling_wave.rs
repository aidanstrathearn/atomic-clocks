use crate::twolevel::pulse_sequence::{DrivenPulseSequence, ResolvedPulseSequence};

use super::{BeamDirection, MomentumBranchDetunings, PulseSequence};

/// A resolved two-level sequence driven by one travelling wave.
#[derive(Clone, Copy)]
pub struct TravellingWavePulseSequence<S> {
    pulse: DrivenPulseSequence<S>,
    direction: BeamDirection,
}

impl<S> TravellingWavePulseSequence<S> {
    pub fn new(pulse: DrivenPulseSequence<S>, direction: BeamDirection) -> Self {
        Self { pulse, direction }
    }
}

impl<S> PulseSequence for TravellingWavePulseSequence<S>
where
    S: ResolvedPulseSequence + Sync,
{
    fn ground_probability(&self, detunings: MomentumBranchDetunings) -> f64 {
        let detuning_hz = match self.direction {
            BeamDirection::Forward => detunings.forward_hz,
            BeamDirection::Backward => detunings.backward_lower_hz,
        };
        self.pulse.ground_probability(detuning_hz)
    }
}
