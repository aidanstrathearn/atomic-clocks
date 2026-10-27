//! Travelling-wave sequences with Doppler- and recoil-resolved momentum branches.
mod counterpropagating;
mod detunings;
mod travelling_wave;

pub use counterpropagating::CounterpropagatingPulseSequence;
pub use detunings::{BeamDirection, MomentumBranchDetunings};
pub use travelling_wave::TravellingWavePulseSequence;

/// A complete sequence evaluated for one atom's resolved momentum branches.
/// All detunings are ordinary frequencies in hertz.
pub trait PulseSequence: Sync {
    fn ground_probability(&self, detunings: MomentumBranchDetunings) -> f64;
}
