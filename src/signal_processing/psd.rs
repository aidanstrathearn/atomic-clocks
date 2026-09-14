use super::error::SpectrumError;
use super::frequency::{interpolation_location, validate_lengths, AngularFrequencyGrid};
use super::transfer::LtiFilter;

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
