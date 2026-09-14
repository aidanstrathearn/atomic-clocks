use super::error::SpectrumError;

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

    pub(super) fn require_nonnegative(&self) -> Result<(), SpectrumError> {
        if let Some(index) = self.values.iter().position(|frequency| *frequency < 0.0) {
            return Err(SpectrumError::NegativePsdFrequency { index });
        }
        Ok(())
    }
}

pub(super) fn validate_lengths(
    grid: &AngularFrequencyGrid,
    values: usize,
) -> Result<(), SpectrumError> {
    if grid.values.len() != values {
        return Err(SpectrumError::LengthMismatch {
            frequencies: grid.values.len(),
            values,
        });
    }
    Ok(())
}

pub(super) fn interpolation_location(
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
