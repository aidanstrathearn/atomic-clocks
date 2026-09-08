use crate::maths::FourierSpectrum;
use rustfft::num_complex::Complex64;

pub trait Psd {
    fn spectrum(&self, omega: f64) -> f64;

    fn sample(&self, angular_frequencies: &[f64]) -> FourierSpectrum {
        FourierSpectrum {
            angular_frequencies: angular_frequencies.to_vec(),
            amplitudes: angular_frequencies
                .iter()
                .map(|&omega| Complex64::new(self.spectrum(omega), 0.0))
                .collect(),
        }
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
            spectrum: move |omega| {
                self.spectrum(omega) + other.spectrum(omega)
            },
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

pub struct NumericalPsd {
    spectrum: FourierSpectrum,
}

impl NumericalPsd {
    pub fn new(spectrum: FourierSpectrum) -> Self {
        validate_spectrum(&spectrum);
        assert!(spectrum.angular_frequencies[0] >= 0.0);
        assert!(
            spectrum
                .amplitudes
                .iter()
                .all(|value| value.im == 0.0 && value.re >= 0.0)
        );
        Self { spectrum }
    }
}

impl Psd for NumericalPsd {
    fn spectrum(&self, omega: f64) -> f64 {
        interpolate(&self.spectrum, omega).re
    }
}

pub trait LtiFilter {
    fn frequency_response(&self, omega: f64) -> Complex64;

    fn sample(&self, angular_frequencies: &[f64]) -> FourierSpectrum {
        FourierSpectrum {
            angular_frequencies: angular_frequencies.to_vec(),
            amplitudes: angular_frequencies
                .iter()
                .map(|&omega| self.frequency_response(omega))
                .collect(),
        }
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

pub struct NumericalFilter {
    spectrum: FourierSpectrum,
}

impl NumericalFilter {
    pub fn new(spectrum: FourierSpectrum) -> Self {
        validate_spectrum(&spectrum);
        Self { spectrum }
    }
}

impl LtiFilter for NumericalFilter {
    fn frequency_response(&self, omega: f64) -> Complex64 {
        interpolate(&self.spectrum, omega)
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
                Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0)
                    + self.control.frequency_response(omega)
                        * self.measurement.frequency_response(omega))
            },
        }
    }

    pub fn output_psd(&self) -> impl Psd {
        let tmp_psd = (&self.signal)
            .add((&self.noise).apply_filter(&self.control));

        tmp_psd.apply_filter(self.loop_response())
    }
}

fn validate_spectrum(spectrum: &FourierSpectrum) {
    let frequencies = &spectrum.angular_frequencies;
    assert!(
        !frequencies.is_empty(),
        "Numerical spectrum must not be empty"
    );
    assert_eq!(frequencies.len(), spectrum.amplitudes.len());
    assert!(frequencies.iter().all(|omega| omega.is_finite()));
    assert!(
        frequencies.windows(2).all(|pair| pair[0] < pair[1]),
        "Frequencies must be strictly increasing"
    );
    assert!(
        spectrum
            .amplitudes
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

fn interpolate(spectrum: &FourierSpectrum, omega: f64) -> Complex64 {
    let frequencies = &spectrum.angular_frequencies;
    assert!(omega.is_finite());
    assert!(
        omega >= frequencies[0] && omega <= frequencies[frequencies.len() - 1],
        "Frequency is outside the numerical spectrum range"
    );
    let upper = frequencies.partition_point(|&frequency| frequency < omega);
    if frequencies[upper] == omega {
        return spectrum.amplitudes[upper];
    }
    let lower = upper - 1;
    let fraction = (omega - frequencies[lower]) / (frequencies[upper] - frequencies[lower]);
    spectrum.amplitudes[lower] * (1.0 - fraction) + spectrum.amplitudes[upper] * fraction
}
