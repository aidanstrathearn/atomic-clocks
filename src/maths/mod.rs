pub(crate) mod complex;
pub mod demodulation;
pub mod mat3;
pub mod vec3;

pub use mat3::Mat3;

use std::f64::consts::PI;

use rustfft::{FftPlanner, num_complex::Complex64};

#[derive(Clone)]
pub struct FourierSpectrum {
    /// Angular frequencies in radians per unit time.
    pub angular_frequencies: Vec<f64>,
    pub amplitudes: Vec<Complex64>,
}

/// Transforms real samples at `start_time + n * step` using the convention
/// `F(omega) = step * sum_n samples[n] * exp(-i * omega * (start_time + n * step))`.
/// This is a rectangular-quadrature approximation to the continuous transform.
///
/// `df` is the maximum angular-frequency spacing, in radians per unit time.
/// Pads with zeros to `M = max(N, ceil(2*pi / (step*df)))` samples, returning bins
/// `omega[k] = 2*pi*k / (M*step)` for `k = 0..=M/2`. The actual spacing can be
/// smaller than `df`; input samples are never truncated. Includes Nyquist when M
/// is even. Negative frequencies follow by complex conjugation.
/// Amplitudes are not doubled, windowed, or normalized by N. Empty input returns
/// an empty spectrum. `step` and `df` must be finite and positive; `start_time`
/// finite. Padding leaves the time-step scaling and time origin unchanged.
pub fn fourier_transform(samples: &[f64], step: f64, start_time: f64, df: f64) -> FourierSpectrum {
    assert!(
        step.is_finite() && step > 0.0,
        "Fourier sample step must be finite and positive"
    );
    assert!(start_time.is_finite(), "Fourier start time must be finite");
    assert!(
        df.is_finite() && df > 0.0,
        "Fourier frequency spacing must be finite and positive"
    );
    if samples.is_empty() {
        return FourierSpectrum {
            angular_frequencies: Vec::new(),
            amplitudes: Vec::new(),
        };
    }

    let required_len = (2.0 * PI / step / df).ceil();
    assert!(
        required_len.is_finite()
            && required_len < (isize::MAX as usize / size_of::<Complex64>()) as f64,
        "Fourier frequency spacing requires an unsupported transform length"
    );
    let n = samples.len().max(required_len as usize);
    let mut amplitudes: Vec<_> = samples
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    amplitudes.resize(n, Complex64::new(0.0, 0.0));
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(n).process(&mut amplitudes);
    amplitudes.truncate(n / 2 + 1);

    let frequency_step = 2.0 * PI / (n as f64 * step);
    let angular_frequencies: Vec<_> = (0..amplitudes.len())
        .map(|k| k as f64 * frequency_step)
        .collect();
    for (amplitude, &omega) in amplitudes.iter_mut().zip(&angular_frequencies) {
        let (sin_phase, cos_phase) = (-omega * start_time).sin_cos();
        *amplitude *= Complex64::new(cos_phase, sin_phase) * step;
    }

    FourierSpectrum {
        angular_frequencies,
        amplitudes,
    }
}

pub fn normalised_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    f64::exp(-0.5 * ((x - mu) / sigma).powi(2)) / (PI * 2.0).sqrt() / sigma
}

#[derive(Clone)]
pub struct Linspace {
    pub step: f64,
    pub array: Vec<f64>,
}

impl Linspace {
    /// Divides `[start, stop]` into `nsteps` intervals, returning `nsteps + 1`
    /// boundary points including both endpoints. `nsteps` must be positive.
    pub fn new(start: f64, stop: f64, nsteps: usize) -> Self {
        assert!(nsteps > 0, "Linspace requires at least one interval");
        let step: f64 = (stop - start) / (nsteps as f64);
        Self {
            step,
            array: (0..=nsteps).map(|x| start + (x as f64) * step).collect(),
        }
    }
}

/// Returns the `nsteps + 1` boundary points of `nsteps` intervals, including
/// both endpoints. `nsteps` must be positive.
pub fn linspace(start: f64, stop: f64, nsteps: usize) -> Vec<f64> {
    Linspace::new(start, stop, nsteps).array
}
