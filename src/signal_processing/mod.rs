//! Scalar LTI transfer functions and one-sided power spectral densities.

mod error;
mod feedback;
mod frequency;
mod psd;
mod transfer;

pub use error::SpectrumError;
pub use frequency::AngularFrequencyGrid;
pub use psd::{FunctionalPsd, Psd, PsdSamples, WhiteRw};
pub use transfer::{
    Delay, FunctionalFilter, Gain, Integrator, LtiFilter, Series, TransferFunctionSamples,
};
