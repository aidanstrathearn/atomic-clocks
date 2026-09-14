use rustfft::num_complex::Complex64;

use super::psd::Psd;
use super::transfer::{FunctionalFilter, LtiFilter};

pub(super) struct LtiFeedback<S, N, G, C>
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
