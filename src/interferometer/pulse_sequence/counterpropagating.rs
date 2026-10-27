use crate::twolevel::pulse_sequence::{DrivenPulseSequence, ResolvedPulseSequence};

use super::{MomentumBranchDetunings, PulseSequence};

/// Forward and backward unital sequences coupling three momentum branches.
///
/// Combines populations between the directional sequences; momentum-branch
/// coherence between them is discarded. Coherence within each sequence is
/// retained. The population formula requires unital evolution.
#[derive(Clone, Copy)]
pub struct CounterpropagatingPulseSequence<F, B> {
    forward: DrivenPulseSequence<F>,
    backward: DrivenPulseSequence<B>,
}

impl<F, B> CounterpropagatingPulseSequence<F, B> {
    pub fn new(forward: DrivenPulseSequence<F>, backward: DrivenPulseSequence<B>) -> Self {
        Self { forward, backward }
    }
}

impl<F, B> PulseSequence for CounterpropagatingPulseSequence<F, B>
where
    F: ResolvedPulseSequence + Sync,
    B: ResolvedPulseSequence + Sync,
{
    fn ground_probability(&self, detunings: MomentumBranchDetunings) -> f64 {
        let forward_z_response = self.forward.z_response(detunings.forward_hz);
        let backward_lower_z_response = self.backward.z_response(detunings.backward_lower_hz);
        let backward_upper_z_response = self.backward.z_response(detunings.backward_upper_hz);

        // For a unital channel, P(g|g) = (1 + M_zz) / 2 and
        // P(g|e) = (1 - M_zz) / 2.
        0.25 * ((1.0 + forward_z_response) * (1.0 + backward_lower_z_response)
            + (1.0 - forward_z_response) * (1.0 - backward_upper_z_response))
    }
}
