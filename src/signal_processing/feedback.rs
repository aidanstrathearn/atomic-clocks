use rustfft::num_complex::Complex64;

use super::psd::FunctionalPsd;
use super::psd::Psd;
use super::transfer::{FunctionalFilter, LtiFilter};

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

    fn sensitivity_at(&self, omega: f64) -> Complex64 {
        Complex64::new(1.0, 0.0)
            / (Complex64::new(1.0, 0.0)
                + self.control.frequency_response(omega)
                    * self.measurement.frequency_response(omega))
    }

    fn measurement_noise_response_at(&self, omega: f64) -> Complex64 {
        -self.control.frequency_response(omega) * self.sensitivity_at(omega)
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
                self.noise.spectrum(omega) //* self.measurement_noise_response_at(omega).norm_sqr()
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
}
