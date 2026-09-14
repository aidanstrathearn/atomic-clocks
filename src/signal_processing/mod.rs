//! Scalar LTI transfer functions and one-sided power spectral densities.

use crate::maths::FourierSpectrum;
use rustfft::num_complex::Complex64;
use std::{error::Error, fmt};

#[derive(Clone, Debug, PartialEq)]
pub enum SpectrumError {
    EmptyFrequencyGrid,
    LengthMismatch {
        frequencies: usize,
        values: usize,
    },
    NonFiniteFrequency {
        index: usize,
    },
    FrequenciesNotStrictlyIncreasing {
        lower_index: usize,
    },
    NegativePsdFrequency {
        index: usize,
    },
    NonFiniteTransferValue {
        index: usize,
    },
    InvalidPsdValue {
        index: usize,
    },
    NonFiniteQuery,
    FrequencyOutsideRange {
        frequency: f64,
        minimum: f64,
        maximum: f64,
    },
}

impl fmt::Display for SpectrumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrequencyGrid => write!(formatter, "frequency grid must not be empty"),
            Self::LengthMismatch {
                frequencies,
                values,
            } => write!(
                formatter,
                "frequency grid has {frequencies} points but spectrum has {values} values"
            ),
            Self::NonFiniteFrequency { index } => {
                write!(formatter, "frequency at index {index} is not finite")
            }
            Self::FrequenciesNotStrictlyIncreasing { lower_index } => write!(
                formatter,
                "frequencies at indices {lower_index} and {} are not strictly increasing",
                lower_index + 1
            ),
            Self::NegativePsdFrequency { index } => {
                write!(formatter, "PSD frequency at index {index} is negative")
            }
            Self::NonFiniteTransferValue { index } => {
                write!(
                    formatter,
                    "transfer response at index {index} is not finite"
                )
            }
            Self::InvalidPsdValue { index } => write!(
                formatter,
                "PSD value at index {index} must be finite and nonnegative"
            ),
            Self::NonFiniteQuery => write!(formatter, "query frequency must be finite"),
            Self::FrequencyOutsideRange {
                frequency,
                minimum,
                maximum,
            } => write!(
                formatter,
                "frequency {frequency} is outside the sampled range [{minimum}, {maximum}]"
            ),
        }
    }
}

impl Error for SpectrumError {}

/// A validated grid of angular frequencies in radians per model time unit.
#[derive(Clone, Debug, PartialEq)]
pub struct AngularFrequencyGrid {
    values: Vec<f64>,
}

impl AngularFrequencyGrid {
    pub fn new(values: Vec<f64>) -> Result<Self, SpectrumError> {
        if values.is_empty() {
            return Err(SpectrumError::EmptyFrequencyGrid);
        }
        if let Some(index) = values.iter().position(|frequency| !frequency.is_finite()) {
            return Err(SpectrumError::NonFiniteFrequency { index });
        }
        if let Some(lower_index) = values.windows(2).position(|pair| pair[0] >= pair[1]) {
            return Err(SpectrumError::FrequenciesNotStrictlyIncreasing { lower_index });
        }
        Ok(Self { values })
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn require_nonnegative(&self) -> Result<(), SpectrumError> {
        if let Some(index) = self.values.iter().position(|frequency| *frequency < 0.0) {
            return Err(SpectrumError::NegativePsdFrequency { index });
        }
        Ok(())
    }
}

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

/// Real samples of a one-sided PSD on a nonnegative angular-frequency grid.
#[derive(Clone, Debug)]
pub struct PsdSamples {
    grid: AngularFrequencyGrid,
    density: Vec<f64>,
}

impl PsdSamples {
    pub fn new(grid: AngularFrequencyGrid, density: Vec<f64>) -> Result<Self, SpectrumError> {
        grid.require_nonnegative()?;
        validate_lengths(&grid, density.len())?;
        if let Some(index) = density
            .iter()
            .position(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(SpectrumError::InvalidPsdValue { index });
        }
        Ok(Self { grid, density })
    }

    pub fn grid(&self) -> &AngularFrequencyGrid {
        &self.grid
    }

    pub fn density_values(&self) -> &[f64] {
        &self.density
    }

    pub fn density_at(&self, omega: f64) -> Result<f64, SpectrumError> {
        let (lower, upper, fraction) = interpolation_location(&self.grid, omega)?;
        if lower == upper {
            return Ok(self.density[lower]);
        }
        Ok(self.density[lower] * (1.0 - fraction) + self.density[upper] * fraction)
    }
}

pub trait Psd {
    fn spectrum(&self, omega: f64) -> f64;

    fn sample(&self, grid: &AngularFrequencyGrid) -> Result<PsdSamples, SpectrumError> {
        grid.require_nonnegative()?;
        PsdSamples::new(
            grid.clone(),
            grid.values()
                .iter()
                .map(|&omega| self.spectrum(omega))
                .collect(),
        )
    }

    fn apply_filter(self, filter: impl LtiFilter) -> impl Psd
    where
        Self: Sized,
    {
        FunctionalPsd {
            spectrum: move |omega| {
                self.spectrum(omega) * filter.frequency_response(omega).norm_sqr()
            },
        }
    }

    fn add(self, other: impl Psd) -> impl Psd
    where
        Self: Sized,
    {
        FunctionalPsd {
            spectrum: move |omega| self.spectrum(omega) + other.spectrum(omega),
        }
    }
}

impl<P: Psd + ?Sized> Psd for &P {
    fn spectrum(&self, omega: f64) -> f64 {
        (**self).spectrum(omega)
    }
}

pub struct WhiteRw {
    pub white_noise: f64,
    pub random_walk: f64,
}

impl Psd for WhiteRw {
    fn spectrum(&self, omega: f64) -> f64 {
        assert!(omega.is_finite() && omega >= 0.0);
        assert!(self.white_noise.is_finite() && self.white_noise >= 0.0);
        assert!(self.random_walk.is_finite() && self.random_walk >= 0.0);
        if self.random_walk == 0.0 {
            return self.white_noise;
        }
        assert!(omega > 0.0, "Random-walk PSD is singular at zero frequency");
        self.white_noise + self.random_walk / omega / omega
    }
}

pub struct FunctionalPsd<F> {
    pub spectrum: F,
}

impl<F: Fn(f64) -> f64> Psd for FunctionalPsd<F> {
    fn spectrum(&self, omega: f64) -> f64 {
        assert!(omega.is_finite() && omega >= 0.0);
        (self.spectrum)(omega)
    }
}

impl Psd for PsdSamples {
    fn spectrum(&self, omega: f64) -> f64 {
        self.density_at(omega)
            .unwrap_or_else(|error| panic!("cannot evaluate sampled PSD: {error}"))
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

struct LtiFeedback<S, N, G, C>
where
    S: Psd,
    N: Psd,
    G: LtiFilter,
    C: LtiFilter,
{
    signal: S,
    noise: N,
    measurement: G,
    control: C,
}

impl<S: Psd, N: Psd, M: LtiFilter, C: LtiFilter> LtiFeedback<S, N, M, C> {
    pub fn loop_response(&self) -> FunctionalFilter<impl Fn(f64) -> Complex64> {
        FunctionalFilter {
            response: |omega| {
                Complex64::new(1.0, 0.0)
                    / (Complex64::new(1.0, 0.0)
                        + self.control.frequency_response(omega)
                            * self.measurement.frequency_response(omega))
            },
        }
    }

    pub fn output_psd(&self) -> impl Psd {
        let tmp_psd = (&self.signal).add((&self.noise).apply_filter(&self.control));

        tmp_psd.apply_filter(self.loop_response())
    }
}

fn validate_lengths(grid: &AngularFrequencyGrid, values: usize) -> Result<(), SpectrumError> {
    if grid.len() != values {
        return Err(SpectrumError::LengthMismatch {
            frequencies: grid.len(),
            values,
        });
    }
    Ok(())
}

fn interpolation_location(
    grid: &AngularFrequencyGrid,
    omega: f64,
) -> Result<(usize, usize, f64), SpectrumError> {
    if !omega.is_finite() {
        return Err(SpectrumError::NonFiniteQuery);
    }
    let frequencies = grid.values();
    let minimum = frequencies[0];
    let maximum = frequencies[frequencies.len() - 1];
    if omega < minimum || omega > maximum {
        return Err(SpectrumError::FrequencyOutsideRange {
            frequency: omega,
            minimum,
            maximum,
        });
    }

    let upper = frequencies.partition_point(|&frequency| frequency < omega);
    if frequencies[upper] == omega {
        return Ok((upper, upper, 0.0));
    }
    let lower = upper - 1;
    let fraction = (omega - frequencies[lower]) / (frequencies[upper] - frequencies[lower]);
    Ok((lower, upper, fraction))
}
