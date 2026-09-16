use super::error::SpectrumError;
use super::psd::PsdSamples;

/// Allan-deviation samples on a strictly increasing grid of averaging times.
#[derive(Clone, Debug)]
pub struct AdevSamples {
    averaging_times: Vec<f64>,
    deviations: Vec<f64>,
}

impl AdevSamples {
    pub fn new(averaging_times: Vec<f64>, deviations: Vec<f64>) -> Result<Self, SpectrumError> {
        validate_averaging_times(&averaging_times)?;
        if averaging_times.len() != deviations.len() {
            return Err(SpectrumError::LengthMismatch {
                frequencies: averaging_times.len(),
                values: deviations.len(),
            });
        }
        if let Some(index) = deviations
            .iter()
            .position(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(SpectrumError::InvalidAdevValue { index });
        }
        Ok(Self {
            averaging_times,
            deviations,
        })
    }

    pub fn averaging_times(&self) -> &[f64] {
        &self.averaging_times
    }

    pub fn deviation_values(&self) -> &[f64] {
        &self.deviations
    }
}

fn validate_averaging_times(averaging_times: &[f64]) -> Result<(), SpectrumError> {
    if averaging_times.is_empty() {
        return Err(SpectrumError::EmptyAveragingTimeGrid);
    }
    if let Some(index) = averaging_times.iter().position(|time| !time.is_finite()) {
        return Err(SpectrumError::NonFiniteAveragingTime { index });
    }
    if let Some(index) = averaging_times.iter().position(|time| *time <= 0.0) {
        return Err(SpectrumError::NonPositiveAveragingTime { index });
    }
    if let Some(lower_index) = averaging_times
        .windows(2)
        .position(|pair| pair[0] >= pair[1])
    {
        return Err(SpectrumError::AveragingTimesNotStrictlyIncreasing { lower_index });
    }
    Ok(())
}

fn allan_variance_weight(omega: f64, averaging_time: f64) -> f64 {
    let phase = 0.5 * omega * averaging_time;
    if phase == 0.0 {
        return 0.0;
    }
    if phase.abs() < 1.0 {
        let sinc = phase.sin() / phase;
        2.0 * phase * phase * sinc.powi(4)
    } else {
        2.0 * phase.sin().powi(4) / phase.powi(2)
    }
}

impl PsdSamples {
    /// Computes the band-limited Allan deviation represented by these samples.
    ///
    /// For one-sided angular-frequency density `S(omega)`, this evaluates
    /// `sigma^2(tau) = 2 integral S(omega) sin^4(omega tau / 2)
    /// / (omega tau / 2)^2 d omega`.
    ///
    /// The one-sided angular-frequency PSD is integrated only over the sampled
    /// frequency interval. Adjacent samples are joined using trapezoidal
    /// quadrature; no low- or high-frequency extrapolation is performed.
    pub fn to_adev(&self, averaging_times: &[f64]) -> Result<AdevSamples, SpectrumError> {
        validate_averaging_times(averaging_times)?;

        let angular_frequencies = self.grid().values();
        let deviations = averaging_times
            .iter()
            .map(|&averaging_time| {
                let variance = angular_frequencies
                    .windows(2)
                    .zip(self.density_values().windows(2))
                    .map(|(frequencies, densities)| {
                        let lower_integrand =
                            densities[0] * allan_variance_weight(frequencies[0], averaging_time);
                        let upper_integrand =
                            densities[1] * allan_variance_weight(frequencies[1], averaging_time);
                        0.5 * (lower_integrand + upper_integrand)
                            * (frequencies[1] - frequencies[0])
                    })
                    .sum::<f64>();
                variance.sqrt()
            })
            .collect();

        AdevSamples::new(averaging_times.to_vec(), deviations)
    }
}
