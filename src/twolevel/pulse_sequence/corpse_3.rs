use std::f64::consts::PI;

use crate::twolevel::BlochVec;

use super::{PulseParameters, ResolvedPulseSequence, prepare_pulse};

/// Canonical three-pulse CORPSE sequence with winding numbers (1, 1, 0).
#[derive(Clone, Copy)]
pub struct Corpse3 {
    segment_durations_s: [f64; 3],
}

impl Corpse3 {
    pub const SEGMENT_PHASES_RAD: [f64; 3] = [0.0, PI, 0.0];
    pub const DURATION_FACTORS: [f64; 3] = [7.0 / 3.0, 5.0 / 3.0, 1.0 / 3.0];

    /// Scales the canonical CORPSE pi-pulse template from its primitive pulse duration.
    pub fn new(target_duration_s: f64) -> Self {
        assert!(target_duration_s.is_finite() && target_duration_s >= 0.0);

        Self {
            segment_durations_s: Self::DURATION_FACTORS.map(|factor| factor * target_duration_s),
        }
    }

    pub fn segment_durations_s(&self) -> [f64; 3] {
        self.segment_durations_s
    }

    pub fn total_duration_s(&self) -> f64 {
        self.segment_durations_s.iter().sum()
    }
}

impl ResolvedPulseSequence for Corpse3 {
    fn apply_to(&self, parameters: PulseParameters, initial: BlochVec) -> BlochVec {
        self.segment_durations_s
            .iter()
            .zip(Self::SEGMENT_PHASES_RAD)
            .fold(initial, |state, (&duration_s, phase_rad)| {
                prepare_pulse(parameters, duration_s).apply_to_with_phase(state, phase_rad)
            })
    }
}
