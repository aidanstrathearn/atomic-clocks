use crate::maths::FourierSpectrum;
use rustfft::num_complex::Complex64;

use super::error::SpectrumError;
use super::frequency::{AngularFrequencyGrid, interpolation_location, validate_lengths};

/// Complex samples of an LTI transfer function on an angular-frequency grid.
#[derive(Clone, Debug)]
pub struct TransferFunctionSamples {
    grid: AngularFrequencyGrid,
    response: Vec<Complex64>,
}

impl TransferFunctionSamples {
    pub fn new(
        grid: AngularFrequencyGrid,
        response: Vec<Complex64>,
    ) -> Result<Self, SpectrumError> {
        validate_lengths(&grid, response.len())?;
        if let Some(index) = response
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(SpectrumError::NonFiniteTransferValue { index });
        }
        Ok(Self { grid, response })
    }

    pub fn grid(&self) -> &AngularFrequencyGrid {
        &self.grid
    }

    pub fn response_values(&self) -> &[Complex64] {
        &self.response
    }

    pub fn response_at(&self, omega: f64) -> Result<Complex64, SpectrumError> {
        let (lower, upper, fraction) = interpolation_location(&self.grid, omega)?;
        if lower == upper {
            return Ok(self.response[lower]);
        }
        Ok(self.response[lower] * (1.0 - fraction) + self.response[upper] * fraction)
    }
}

impl TryFrom<FourierSpectrum> for TransferFunctionSamples {
    type Error = SpectrumError;

    fn try_from(spectrum: FourierSpectrum) -> Result<Self, Self::Error> {
        Self::new(
            AngularFrequencyGrid::new(spectrum.angular_frequencies)?,
            spectrum.amplitudes,
        )
    }
}

pub trait LtiFilter {
    fn frequency_response(&self, omega: f64) -> Complex64;

    fn sample(
        &self,
        grid: &AngularFrequencyGrid,
    ) -> Result<TransferFunctionSamples, SpectrumError> {
        TransferFunctionSamples::new(
            grid.clone(),
            grid.values()
                .iter()
                .map(|&omega| self.frequency_response(omega))
                .collect(),
        )
    }

    fn then<F: LtiFilter>(self, next: F) -> Series<Self, F>
    where
        Self: Sized,
    {
        Series::new(self, next)
    }
}

impl<F: LtiFilter + ?Sized> LtiFilter for &F {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        (**self).frequency_response(omega)
    }
}

/// A real, frequency-independent gain. Signed gains are supported.
#[derive(Clone, Copy, Debug)]
pub struct Gain {
    gain: f64,
}

impl Gain {
    pub fn new(gain: f64) -> Self {
        assert!(gain.is_finite(), "Gain must be finite");
        Self { gain }
    }
}

impl LtiFilter for Gain {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        assert!(omega.is_finite());
        Complex64::new(self.gain, 0.0)
    }
}

/// A pure time delay with response `exp(-i omega duration)`.
#[derive(Clone, Copy, Debug)]
pub struct Delay {
    duration: f64,
}

impl Delay {
    pub fn new(duration: f64) -> Self {
        assert!(
            duration.is_finite() && duration >= 0.0,
            "Delay duration must be finite and nonnegative"
        );
        Self { duration }
    }
}

impl LtiFilter for Delay {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        assert!(omega.is_finite());
        let (sin_phase, cos_phase) = (-omega * self.duration).sin_cos();
        Complex64::new(cos_phase, sin_phase)
    }
}

/// An integrator with response `gain / (i omega)`.
#[derive(Clone, Copy, Debug)]
pub struct Integrator {
    gain: f64,
}

impl Integrator {
    pub fn new(gain: f64) -> Self {
        assert!(gain.is_finite(), "Integrator gain must be finite");
        Self { gain }
    }
}

impl Default for Integrator {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LtiFilter for Integrator {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        assert!(omega.is_finite());
        assert_ne!(
            omega, 0.0,
            "Integrator response is singular at zero frequency"
        );
        Complex64::new(0.0, -self.gain / omega)
    }
}

/// Two scalar LTI filters applied in sequence.
#[derive(Clone, Copy, Debug)]
pub struct Series<A, B> {
    first: A,
    second: B,
}

impl<A: LtiFilter, B: LtiFilter> Series<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: LtiFilter, B: LtiFilter> LtiFilter for Series<A, B> {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        assert!(omega.is_finite());
        self.second.frequency_response(omega) * self.first.frequency_response(omega)
    }
}

pub struct FunctionalFilter<F: Fn(f64) -> Complex64> {
    pub response: F,
}

impl<F: Fn(f64) -> Complex64> LtiFilter for FunctionalFilter<F> {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        assert!(omega.is_finite());
        (self.response)(omega)
    }
}

impl LtiFilter for TransferFunctionSamples {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        self.response_at(omega)
            .unwrap_or_else(|error| panic!("cannot evaluate sampled transfer function: {error}"))
    }
}
