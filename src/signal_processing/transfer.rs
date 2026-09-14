use crate::maths::FourierSpectrum;
use rustfft::num_complex::Complex64;

use super::error::SpectrumError;
use super::frequency::{interpolation_location, validate_lengths, AngularFrequencyGrid};

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
}

impl<F: LtiFilter + ?Sized> LtiFilter for &F {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        (**self).frequency_response(omega)
    }
}

pub struct Integrator;

impl LtiFilter for Integrator {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        assert!(omega.is_finite());
        assert_ne!(
            omega, 0.0,
            "Integrator response is singular at zero frequency"
        );
        Complex64::new(0.0, -1.0 / omega)
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
