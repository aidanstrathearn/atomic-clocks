use rustfft::num_complex::Complex64;

use super::error::SpectrumError;
use super::frequency::AngularFrequencyGrid;
use super::psd::{FunctionalPsd, Psd, PsdSamples};
use super::transfer::{FunctionalFilter, LtiFilter};

/// Sampled decomposition of the disturbances in a scalar feedback loop.
#[derive(Clone, Debug)]
pub struct FeedbackPsdSamples {
    free_running: PsdSamples,
    residual_signal: PsdSamples,
    injected_measurement_noise: PsdSamples,
    output: PsdSamples,
}

impl FeedbackPsdSamples {
    pub fn free_running(&self) -> &PsdSamples {
        &self.free_running
    }

    pub fn residual_signal(&self) -> &PsdSamples {
        &self.residual_signal
    }

    pub fn injected_measurement_noise(&self) -> &PsdSamples {
        &self.injected_measurement_noise
    }

    pub fn output(&self) -> &PsdSamples {
        &self.output
    }
}

/// Scalar negative feedback with independent input and measurement noise.
///
/// The input disturbance is added directly at the output. Measurement noise is
/// added after `measurement` and passes through `control`. The closed-loop
/// denominator is `1 + control * measurement`.
pub struct LtiFeedback<S, N, G, C>
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
    pub fn new(signal: S, noise: N, measurement: M, control: C) -> Self {
        Self {
            signal,
            noise,
            measurement,
            control,
        }
    }

    fn open_loop_at(&self, omega: f64) -> Complex64 {
        self.control.frequency_response(omega) * self.measurement.frequency_response(omega)
    }

    fn sensitivity_at(&self, omega: f64) -> Complex64 {
        Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + self.open_loop_at(omega))
    }

    fn measurement_noise_response_at(&self, omega: f64) -> Complex64 {
        -self.control.frequency_response(omega) * self.sensitivity_at(omega)
    }

    /// Open-loop response `control * measurement` used by the feedback denominator.
    pub fn open_loop(&self) -> FunctionalFilter<impl Fn(f64) -> Complex64> {
        FunctionalFilter {
            response: |omega| self.open_loop_at(omega),
        }
    }

    /// Closed-loop response from the free-running input disturbance to output.
    pub fn sensitivity(&self) -> FunctionalFilter<impl Fn(f64) -> Complex64> {
        FunctionalFilter {
            response: |omega| self.sensitivity_at(omega),
        }
    }

    /// Closed-loop response from additive measurement noise to output.
    pub fn measurement_noise_response(&self) -> FunctionalFilter<impl Fn(f64) -> Complex64> {
        FunctionalFilter {
            response: |omega| self.measurement_noise_response_at(omega),
        }
    }

    pub fn free_running_psd(&self) -> FunctionalPsd<impl Fn(f64) -> f64> {
        FunctionalPsd {
            spectrum: |omega| self.signal.spectrum(omega),
        }
    }

    pub fn residual_signal_psd(&self) -> FunctionalPsd<impl Fn(f64) -> f64> {
        FunctionalPsd {
            spectrum: |omega| self.signal.spectrum(omega) * self.sensitivity_at(omega).norm_sqr(),
        }
    }

    pub fn injected_measurement_noise_psd(&self) -> FunctionalPsd<impl Fn(f64) -> f64> {
        FunctionalPsd {
            spectrum: |omega| {
                self.noise.spectrum(omega) * self.measurement_noise_response_at(omega).norm_sqr()
            },
        }
    }

    pub fn output_psd(&self) -> FunctionalPsd<impl Fn(f64) -> f64> {
        FunctionalPsd {
            spectrum: |omega| {
                self.signal.spectrum(omega) * self.sensitivity_at(omega).norm_sqr()
                    + self.noise.spectrum(omega)
                        * self.measurement_noise_response_at(omega).norm_sqr()
            },
        }
    }

    /// Samples every component of the feedback PSD decomposition on one grid.
    /// Each input spectrum and transfer function is evaluated once per frequency.
    pub fn sample_psds(
        &self,
        grid: &AngularFrequencyGrid,
    ) -> Result<FeedbackPsdSamples, SpectrumError> {
        grid.require_nonnegative()?;
        let mut free_running = Vec::with_capacity(grid.values().len());
        let mut residual_signal = Vec::with_capacity(grid.values().len());
        let mut injected_measurement_noise = Vec::with_capacity(grid.values().len());
        let mut output = Vec::with_capacity(grid.values().len());

        for &omega in grid.values() {
            let signal = self.signal.spectrum(omega);
            let noise = self.noise.spectrum(omega);
            let control = self.control.frequency_response(omega);
            let measurement = self.measurement.frequency_response(omega);
            let sensitivity =
                Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + control * measurement);
            let measurement_noise_response = -control * sensitivity;
            let residual = signal * sensitivity.norm_sqr();
            let injected = noise * measurement_noise_response.norm_sqr();

            free_running.push(signal);
            residual_signal.push(residual);
            injected_measurement_noise.push(injected);
            output.push(residual + injected);
        }

        Ok(FeedbackPsdSamples {
            free_running: PsdSamples::new(grid.clone(), free_running)?,
            residual_signal: PsdSamples::new(grid.clone(), residual_signal)?,
            injected_measurement_noise: PsdSamples::new(grid.clone(), injected_measurement_noise)?,
            output: PsdSamples::new(grid.clone(), output)?,
        })
    }
}
