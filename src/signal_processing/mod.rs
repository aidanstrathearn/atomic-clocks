//! Scalar LTI transfer functions, one-sided PSDs, and stochastic processes.

mod adev;
mod error;
mod feedback;
mod frequency;
mod psd;
pub mod stochastic;
mod transfer;

pub use adev::AdevSamples;
pub use error::SpectrumError;
pub use feedback::{FeedbackPsdSamples, LtiFeedback};
pub use frequency::AngularFrequencyGrid;
pub use psd::{FunctionalPsd, Psd, PsdSamples, WhiteRw};
pub use transfer::{
    Delay, FunctionalFilter, Gain, Integrator, LtiFilter, Series, TransferFunctionSamples,
};
