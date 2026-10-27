use crate::twolevel::BlochVec;

use super::{PulseParameters, ResolvedPulseSequence, prepare_pulse};

#[derive(Clone, Copy)]
pub struct SinglePulse {
    duration_s: f64,
}

impl SinglePulse {
    pub fn new(duration_s: f64) -> Self {
        Self { duration_s }
    }
}

impl ResolvedPulseSequence for SinglePulse {
    fn apply_to(&self, parameters: PulseParameters, initial: BlochVec) -> BlochVec {
        prepare_pulse(parameters, self.duration_s).apply_to(initial)
    }
}
