//! Stationary Gaussian noise processes, time grids, and online statistics.
//!
//! Samplers accept a caller-supplied RNG. Every path starts in the stationary
//! distribution, with no warmup, and successive calls generate independent
//! realizations rather than continuing the preceding path. Preserve the RNG
//! algorithm, seed, and sampling requests when reproducing a realization.
//!
//! Times and correlation times use the same unit. Spectral frequencies are
//! ordinary frequencies (cycles per time unit), and one-sided PSDs integrate
//! over nonnegative frequency to the stationary variance. With times in seconds,
//! frequencies are in hertz and PSDs have units of signal squared per hertz.
//! The process amplitude can represent any scalar quantity, including frequency
//! deviations or fractional drive-amplitude fluctuations.

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt::{self, Display, Formatter};

use rand::Rng;

/// Invalid parameters supplied to a stochastic process or its sampling grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StochasticError {
    NonFiniteStartTime,
    EmptyOutputTimes,
    InvalidMaximumStep,
    ZeroMaximumIntervalCount,
    InvalidOutputTime(usize),
    NonIncreasingOutputTimes(usize),
    TooManyIntervals(usize),
    InvalidStationaryStandardDeviation,
    InvalidCorrelationTime,
    InvalidPeakFrequency,
    InvalidFullWidthAtHalfMaximum,
    NotUnderdamped,
    InvalidSampleTime(usize),
    NonIncreasingSampleTimes(usize),
    OutputLengthMismatch {
        sample_count: usize,
        output_count: usize,
    },
}

impl Display for StochasticError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteStartTime => formatter.write_str("grid start time must be finite"),
            Self::EmptyOutputTimes => formatter.write_str("at least one output time is required"),
            Self::InvalidMaximumStep => {
                formatter.write_str("maximum grid step must be positive and finite")
            }
            Self::ZeroMaximumIntervalCount => {
                formatter.write_str("maximum grid interval count must be positive")
            }
            Self::InvalidOutputTime(index) => {
                write!(formatter, "output time {index} must be finite")
            }
            Self::NonIncreasingOutputTimes(index) => write!(
                formatter,
                "output time {index} must be greater than the preceding grid boundary"
            ),
            Self::TooManyIntervals(maximum) => {
                write!(formatter, "grid needs more than {maximum} time intervals")
            }
            Self::InvalidStationaryStandardDeviation => {
                formatter.write_str("stationary standard deviation must be finite and nonnegative")
            }
            Self::InvalidCorrelationTime => {
                formatter.write_str("correlation time must be positive and finite")
            }
            Self::InvalidPeakFrequency => {
                formatter.write_str("peak frequency must be positive and finite")
            }
            Self::InvalidFullWidthAtHalfMaximum => {
                formatter.write_str("FWHM must be positive and finite")
            }
            Self::NotUnderdamped => {
                formatter.write_str("FWHM must be less than twice the peak frequency")
            }
            Self::InvalidSampleTime(index) => {
                write!(formatter, "sample time {index} must be finite")
            }
            Self::NonIncreasingSampleTimes(index) => write!(
                formatter,
                "sample time {index} must be greater than the preceding sample time"
            ),
            Self::OutputLengthMismatch {
                sample_count,
                output_count,
            } => write!(
                formatter,
                "sample-time count {sample_count} does not match output count {output_count}"
            ),
        }
    }
}

impl Error for StochasticError {}

/// A time grid refined between requested output times to respect a maximum step.
///
/// Each interval has one midpoint for noise sampling. Requested output times
/// correspond to indices in [`Self::boundaries`], including its initial boundary
/// at index zero, rather than to indices in the midpoint array.
#[derive(Clone, Debug)]
pub struct RefinedTimeGrid {
    boundaries: Vec<f64>,
    midpoints: Vec<f64>,
    output_boundary_indices: Vec<usize>,
}

impl RefinedTimeGrid {
    /// Refines strictly increasing output times after `start_time`.
    ///
    /// Rejects invalid inputs or refinement exceeding `maximum_interval_count`.
    pub fn new(
        start_time: f64,
        output_times: &[f64],
        maximum_step: f64,
        maximum_interval_count: usize,
    ) -> Result<Self, StochasticError> {
        if !start_time.is_finite() {
            return Err(StochasticError::NonFiniteStartTime);
        }
        if output_times.is_empty() {
            return Err(StochasticError::EmptyOutputTimes);
        }
        if !maximum_step.is_finite() || maximum_step <= 0.0 {
            return Err(StochasticError::InvalidMaximumStep);
        }
        if maximum_interval_count == 0 {
            return Err(StochasticError::ZeroMaximumIntervalCount);
        }

        let mut boundaries = vec![start_time];
        let mut output_boundary_indices = Vec::with_capacity(output_times.len());
        let mut previous = start_time;
        for (output_index, &output_time) in output_times.iter().enumerate() {
            if !output_time.is_finite() {
                return Err(StochasticError::InvalidOutputTime(output_index));
            }
            if output_time <= previous {
                return Err(StochasticError::NonIncreasingOutputTimes(output_index));
            }

            let span = output_time - previous;
            let required_intervals = (span / maximum_step).ceil().max(1.0);
            let remaining_intervals = maximum_interval_count - (boundaries.len() - 1);
            if !required_intervals.is_finite() || required_intervals > remaining_intervals as f64 {
                return Err(StochasticError::TooManyIntervals(maximum_interval_count));
            }
            let interval_count = required_intervals as usize;
            for interval in 1..=interval_count {
                boundaries.push(previous + span * interval as f64 / interval_count as f64);
            }
            output_boundary_indices.push(boundaries.len() - 1);
            previous = output_time;
        }

        let midpoints = boundaries
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();

        Ok(Self {
            boundaries,
            midpoints,
            output_boundary_indices,
        })
    }

    /// Interval boundaries, beginning with the start time.
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    /// One noise-sampling time per interval.
    pub fn midpoints(&self) -> &[f64] {
        &self.midpoints
    }

    /// Boundary index for each requested output time, in request order.
    pub fn output_boundary_indices(&self) -> &[usize] {
        &self.output_boundary_indices
    }

    /// Number of propagation intervals and midpoint samples.
    pub fn interval_count(&self) -> usize {
        self.midpoints.len()
    }
}

/// Box-Muller sampler that preserves the spare standard-normal variate.
struct NormalGenerator<R> {
    rng: R,
    spare: Option<f64>,
}

impl<R: Rng> NormalGenerator<R> {
    fn new(rng: R) -> Self {
        Self { rng, spare: None }
    }

    fn next_standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }

        // 1 - U is in (0, 1], avoiding ln(0) without rejection or variable
        // random-number consumption.
        let radius = (-2.0 * (1.0 - self.rng.random::<f64>()).ln()).sqrt();
        let angle = TAU * self.rng.random::<f64>();
        let (sin, cos) = angle.sin_cos();
        self.spare = Some(radius * sin);
        radius * cos
    }
}

fn validate_path_request(sample_times: &[f64], output: &[f64]) -> Result<(), StochasticError> {
    if sample_times.len() != output.len() {
        return Err(StochasticError::OutputLengthMismatch {
            sample_count: sample_times.len(),
            output_count: output.len(),
        });
    }
    for (index, &sample_time) in sample_times.iter().enumerate() {
        if !sample_time.is_finite() {
            return Err(StochasticError::InvalidSampleTime(index));
        }
        if index > 0 && sample_time <= sample_times[index - 1] {
            return Err(StochasticError::NonIncreasingSampleTimes(index));
        }
    }
    Ok(())
}

/// A stationary Ornstein-Uhlenbeck process with an exponential covariance.
#[derive(Clone, Copy, Debug)]
pub struct OrnsteinUhlenbeck {
    stationary_standard_deviation: f64,
    correlation_time: f64,
}

impl OrnsteinUhlenbeck {
    /// Creates a process with nonnegative RMS amplitude and positive correlation time.
    pub fn new(
        stationary_standard_deviation: f64,
        correlation_time: f64,
    ) -> Result<Self, StochasticError> {
        if !stationary_standard_deviation.is_finite() || stationary_standard_deviation < 0.0 {
            return Err(StochasticError::InvalidStationaryStandardDeviation);
        }
        if !correlation_time.is_finite() || correlation_time <= 0.0 {
            return Err(StochasticError::InvalidCorrelationTime);
        }
        Ok(Self {
            stationary_standard_deviation,
            correlation_time,
        })
    }

    /// Correlation time in the same unit used for sample times.
    pub fn correlation_time(self) -> f64 {
        self.correlation_time
    }

    /// Stationary covariance at a finite positive or negative time lag.
    pub fn covariance(self, lag: f64) -> f64 {
        assert!(lag.is_finite(), "OU covariance lag must be finite");
        self.stationary_standard_deviation.powi(2) * (-lag.abs() / self.correlation_time).exp()
    }

    /// One-sided PSD as a function of nonnegative ordinary frequency.
    ///
    /// Integrating this spectrum from zero to infinity gives the stationary
    /// variance of the process.
    pub fn one_sided_psd(self, frequency: f64) -> f64 {
        assert!(
            frequency.is_finite() && frequency >= 0.0,
            "OU PSD frequency must be finite and nonnegative"
        );
        4.0 * self.stationary_standard_deviation.powi(2) * self.correlation_time
            / (1.0 + (TAU * frequency * self.correlation_time).powi(2))
    }

    /// Creates a sampler using the supplied RNG without drawing any samples yet.
    pub fn sampler<R: Rng>(self, rng: R) -> OuSampler<R> {
        OuSampler {
            process: self,
            normal: NormalGenerator::new(rng),
        }
    }
}

/// Stateful sampler for successive independent stationary OU paths.
pub struct OuSampler<R> {
    process: OrnsteinUhlenbeck,
    normal: NormalGenerator<R>,
}

impl<R: Rng> OuSampler<R> {
    /// Draws a new stationary path at finite, strictly increasing sample times.
    /// An empty request returns an empty path.
    pub fn next_path(&mut self, sample_times: &[f64]) -> Result<Vec<f64>, StochasticError> {
        let mut path = vec![0.0; sample_times.len()];
        self.fill_path(sample_times, &mut path)?;
        Ok(path)
    }

    /// Draws a new stationary path into an equally sized output buffer.
    /// Sample times must be finite and strictly increasing; empty buffers are valid.
    pub fn fill_path(
        &mut self,
        sample_times: &[f64],
        output: &mut [f64],
    ) -> Result<(), StochasticError> {
        validate_path_request(sample_times, output)?;
        if output.is_empty() {
            return Ok(());
        }
        if self.process.stationary_standard_deviation == 0.0 {
            output.fill(0.0);
            return Ok(());
        }

        let sigma = self.process.stationary_standard_deviation;
        let correlation_time = self.process.correlation_time;
        let mut value = sigma * self.normal.next_standard_normal();
        output[0] = value;
        for (index, pair) in sample_times.windows(2).enumerate() {
            let separation = pair[1] - pair[0];
            let decay = (-separation / correlation_time).exp();
            let innovation_scale =
                sigma * (-(-2.0 * separation / correlation_time).exp_m1()).sqrt();
            value = decay * value + innovation_scale * self.normal.next_standard_normal();
            output[index + 1] = value;
        }
        Ok(())
    }
}

/// Stationary velocity of an underdamped, white-noise-driven harmonic oscillator.
///
/// The scalar process returned by the sampler is the oscillator's velocity. Its
/// one-sided PSD is zero at DC, peaks at `peak_frequency`, has the requested
/// FWHM, and integrates to `stationary_standard_deviation.powi(2)`.
#[derive(Clone, Copy, Debug)]
pub struct UnderdampedBrownianOscillator {
    stationary_standard_deviation: f64,
    peak_frequency: f64,
    peak_angular_frequency: f64,
    damping_angular_rate: f64,
    damped_angular_frequency: f64,
}

impl UnderdampedBrownianOscillator {
    /// Creates a process specified by its RMS amplitude, PSD peak, and PSD FWHM.
    ///
    /// Peak frequency and FWHM use ordinary frequency. Both must be positive,
    /// and FWHM must be less than twice the peak frequency.
    pub fn new(
        stationary_standard_deviation: f64,
        peak_frequency: f64,
        full_width_at_half_maximum: f64,
    ) -> Result<Self, StochasticError> {
        if !stationary_standard_deviation.is_finite() || stationary_standard_deviation < 0.0 {
            return Err(StochasticError::InvalidStationaryStandardDeviation);
        }
        if !peak_frequency.is_finite()
            || peak_frequency <= 0.0
            || !(TAU * peak_frequency).is_finite()
        {
            return Err(StochasticError::InvalidPeakFrequency);
        }
        if !full_width_at_half_maximum.is_finite()
            || full_width_at_half_maximum <= 0.0
            || !(0.5 * TAU * full_width_at_half_maximum).is_finite()
        {
            return Err(StochasticError::InvalidFullWidthAtHalfMaximum);
        }
        if 0.5 * full_width_at_half_maximum >= peak_frequency {
            return Err(StochasticError::NotUnderdamped);
        }

        let peak_angular_frequency = TAU * peak_frequency;
        let damping_angular_rate = 0.5 * TAU * full_width_at_half_maximum;
        let damping_ratio = damping_angular_rate / peak_angular_frequency;
        let damped_angular_frequency =
            peak_angular_frequency * (1.0 - damping_ratio.powi(2)).sqrt();
        Ok(Self {
            stationary_standard_deviation,
            peak_frequency,
            peak_angular_frequency,
            damping_angular_rate,
            damped_angular_frequency,
        })
    }

    /// Stationary covariance at a finite positive or negative time lag.
    pub fn covariance(self, lag: f64) -> f64 {
        assert!(
            lag.is_finite(),
            "Brownian-oscillator covariance lag must be finite"
        );
        let absolute_lag = lag.abs();
        let scaled_damping = self.damping_angular_rate * absolute_lag;
        let decay = (-scaled_damping).exp();
        if decay == 0.0 {
            return 0.0;
        }
        let phase = self.damped_angular_frequency * absolute_lag;
        self.stationary_standard_deviation.powi(2)
            * decay
            * (phase.cos() - scaled_damping * sinc(phase))
    }

    /// One-sided PSD as a function of nonnegative ordinary frequency.
    ///
    /// Integrating over nonnegative frequency gives the stationary variance.
    pub fn one_sided_psd(self, frequency: f64) -> f64 {
        assert!(
            frequency.is_finite() && frequency >= 0.0,
            "Brownian-oscillator PSD frequency must be finite and nonnegative"
        );
        if frequency == 0.0 || self.stationary_standard_deviation == 0.0 {
            return 0.0;
        }

        let damping_ratio = self.damping_angular_rate / self.peak_angular_frequency;
        let frequency_ratio = frequency / self.peak_frequency;
        let shape = if frequency_ratio <= 1.0 {
            let squared_ratio = frequency_ratio.powi(2);
            squared_ratio
                / ((1.0 - squared_ratio).powi(2) + (2.0 * damping_ratio * frequency_ratio).powi(2))
        } else {
            let inverse_ratio = frequency_ratio.recip();
            let squared_inverse = inverse_ratio.powi(2);
            squared_inverse
                / ((squared_inverse - 1.0).powi(2) + (2.0 * damping_ratio * inverse_ratio).powi(2))
        };
        8.0 * damping_ratio * self.stationary_standard_deviation.powi(2)
            / self.peak_angular_frequency
            * shape
    }

    /// Creates a sampler using the supplied RNG without drawing any samples yet.
    pub fn sampler<R: Rng>(self, rng: R) -> UnderdampedBrownianOscillatorSampler<R> {
        UnderdampedBrownianOscillatorSampler {
            process: self,
            normal: NormalGenerator::new(rng),
        }
    }
}

fn sinc(value: f64) -> f64 {
    if value.abs() < 1.0e-4 {
        let squared = value * value;
        1.0 - squared / 6.0 + squared * squared / 120.0 - squared * squared * squared / 5_040.0
    } else {
        value.sin() / value
    }
}

#[derive(Clone, Copy)]
struct NormalizedOscillatorStep {
    position_position: f64,
    position_velocity: f64,
    velocity_velocity: f64,
    position_variance: f64,
    position_velocity_covariance: f64,
    velocity_variance: f64,
}

impl NormalizedOscillatorStep {
    fn new(process: UnderdampedBrownianOscillator, duration: f64) -> Self {
        let scaled_peak = process.peak_angular_frequency * duration;
        let damping_ratio = process.damping_angular_rate / process.peak_angular_frequency;
        let scaled_damping = damping_ratio * scaled_peak;
        let decay = (-scaled_damping).exp();
        if decay == 0.0 {
            return Self {
                position_position: 0.0,
                position_velocity: 0.0,
                velocity_velocity: 0.0,
                position_variance: 1.0,
                position_velocity_covariance: 0.0,
                velocity_variance: 1.0,
            };
        }

        let phase = process.damped_angular_frequency * duration;
        let scaled_sine = scaled_peak * sinc(phase);
        let damping_sine = damping_ratio * scaled_sine;
        let cosine = phase.cos();
        let position_position = decay * (cosine + damping_sine);
        let position_velocity = decay * scaled_sine;
        let velocity_velocity = decay * (cosine - damping_sine);

        // The normalized stationary state covariance is the identity. These
        // are the entries of I - Phi Phi^T. The short-step series avoids
        // cancellation in the position variance, whose leading term is cubic.
        let position_variance = if scaled_peak.abs() < 1.0e-3 {
            let bracket = 1.0 / 3.0 - damping_ratio * scaled_peak / 2.0
                + (2.0 * damping_ratio.powi(2) - (1.0 - damping_ratio.powi(2)) / 3.0)
                    * scaled_peak.powi(2)
                    / 5.0;
            4.0 * damping_ratio * scaled_peak.powi(3) * bracket
        } else {
            1.0 - position_position.powi(2) - position_velocity.powi(2)
        }
        .max(0.0);
        let position_velocity_covariance =
            position_velocity * (position_position - velocity_velocity);
        let velocity_variance = -(-2.0 * scaled_damping).exp_m1()
            + 2.0 * decay.powi(2) * damping_sine * (cosine - damping_sine);

        Self {
            position_position,
            position_velocity,
            velocity_velocity,
            position_variance,
            position_velocity_covariance,
            velocity_variance: velocity_variance.max(0.0),
        }
    }
}

/// Stateful sampler for successive independent stationary oscillator paths.
pub struct UnderdampedBrownianOscillatorSampler<R> {
    process: UnderdampedBrownianOscillator,
    normal: NormalGenerator<R>,
}

impl<R: Rng> UnderdampedBrownianOscillatorSampler<R> {
    /// Draws a new stationary path at finite, strictly increasing sample times.
    /// An empty request returns an empty path.
    pub fn next_path(&mut self, sample_times: &[f64]) -> Result<Vec<f64>, StochasticError> {
        let mut path = vec![0.0; sample_times.len()];
        self.fill_path(sample_times, &mut path)?;
        Ok(path)
    }

    /// Draws a new stationary path into an equally sized output buffer.
    /// Sample times must be finite and strictly increasing; empty buffers are valid.
    pub fn fill_path(
        &mut self,
        sample_times: &[f64],
        output: &mut [f64],
    ) -> Result<(), StochasticError> {
        validate_path_request(sample_times, output)?;
        if output.is_empty() {
            return Ok(());
        }
        let sigma = self.process.stationary_standard_deviation;
        if sigma == 0.0 {
            output.fill(0.0);
            return Ok(());
        }

        // In these normalized coordinates the stationary covariance is I.
        let mut position = self.normal.next_standard_normal();
        let mut velocity = self.normal.next_standard_normal();
        output[0] = sigma * velocity;
        for (index, pair) in sample_times.windows(2).enumerate() {
            let duration = pair[1] - pair[0];
            if !duration.is_finite() {
                position = self.normal.next_standard_normal();
                velocity = self.normal.next_standard_normal();
                output[index + 1] = sigma * velocity;
                continue;
            }
            let step = NormalizedOscillatorStep::new(self.process, duration);

            // Pivot the Cholesky factor on the velocity variance, which is
            // linear rather than cubic in short durations.
            let velocity_noise_scale = step.velocity_variance.sqrt();
            let shared_position_scale = if velocity_noise_scale == 0.0 {
                0.0
            } else {
                step.position_velocity_covariance / velocity_noise_scale
            };
            let independent_position_scale = (step.position_variance
                - shared_position_scale.powi(2))
            .max(0.0)
            .sqrt();
            let shared_normal = self.normal.next_standard_normal();
            let position_normal = self.normal.next_standard_normal();
            let position_noise = shared_position_scale * shared_normal
                + independent_position_scale * position_normal;
            let velocity_noise = velocity_noise_scale * shared_normal;

            let next_position = step.position_position * position
                + step.position_velocity * velocity
                + position_noise;
            let next_velocity = -step.position_velocity * position
                + step.velocity_velocity * velocity
                + velocity_noise;
            position = next_position;
            velocity = next_velocity;
            output[index + 1] = sigma * velocity;
        }
        Ok(())
    }
}

/// Online componentwise means and standard errors for fixed-length vectors.
///
/// Each pushed vector is one independent observation. If paths are combined
/// into antithetic pairs, push each pair's average as one observation. Correlation
/// between components of a vector is allowed; standard errors are pointwise.
pub struct RunningVectorStats {
    count: usize,
    mean: Vec<f64>,
    squared_deviation_sum: Vec<f64>,
}

impl RunningVectorStats {
    /// Creates an accumulator with zero observations and zero initial means.
    pub fn new(length: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; length],
            squared_deviation_sum: vec![0.0; length],
        }
    }

    /// Adds one observation; panics if its length differs from the configured length.
    pub fn push(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.mean.len());
        self.count += 1;
        let count = self.count as f64;
        for ((mean, squared_deviation_sum), &value) in self
            .mean
            .iter_mut()
            .zip(&mut self.squared_deviation_sum)
            .zip(values)
        {
            let difference = value - *mean;
            *mean += difference / count;
            *squared_deviation_sum += difference * (value - *mean);
        }
    }

    /// Componentwise sample means, or zeros before the first observation.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Componentwise standard errors from the unbiased sample variance.
    /// Returns zeros until two observations have been added.
    pub fn standard_errors(&self) -> Vec<f64> {
        if self.count < 2 {
            return vec![0.0; self.mean.len()];
        }
        let scale = 1.0 / ((self.count - 1) * self.count) as f64;
        self.squared_deviation_sum
            .iter()
            .map(|sum| (scale * sum).sqrt())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }

    #[test]
    fn refined_grid_contains_outputs_and_respects_the_maximum_step() {
        let output_times = [2.0e-6, 11.0e-6, 20.0e-6];
        let grid = RefinedTimeGrid::new(0.0, &output_times, 4.0e-6, 100).unwrap();

        for (&boundary_index, &output_time) in
            grid.output_boundary_indices().iter().zip(&output_times)
        {
            assert_close(grid.boundaries()[boundary_index], output_time, 1.0e-18);
        }
        assert!(
            grid.boundaries()
                .windows(2)
                .all(|pair| pair[1] - pair[0] <= 4.0e-6 * (1.0 + 1.0e-12))
        );
    }

    #[test]
    fn grid_rejects_invalid_output_times_and_excessive_refinement() {
        assert_eq!(
            RefinedTimeGrid::new(0.0, &[1.0, 1.0], 0.1, 100).unwrap_err(),
            StochasticError::NonIncreasingOutputTimes(1)
        );
        assert_eq!(
            RefinedTimeGrid::new(0.0, &[1.0], 0.1, 9).unwrap_err(),
            StochasticError::TooManyIntervals(9)
        );
    }

    #[test]
    fn seeded_ou_paths_are_reproducible_and_stationary() {
        let sigma = 2_300.0;
        let correlation_time = 17.0e-6;
        let process = OrnsteinUhlenbeck::new(sigma, correlation_time).unwrap();
        let sample_times: Vec<_> = (0..40_000).map(|index| index as f64 * 5.0e-6).collect();
        let mut first = process.sampler(ChaCha8Rng::seed_from_u64(123_456));
        let mut second = process.sampler(ChaCha8Rng::seed_from_u64(123_456));
        let first_path = first.next_path(&sample_times).unwrap();
        let second_path = second.next_path(&sample_times).unwrap();

        assert_eq!(first_path, second_path);
        let mean = first_path.iter().sum::<f64>() / first_path.len() as f64;
        let variance = first_path
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / first_path.len() as f64;
        let lag_one = first_path
            .windows(2)
            .map(|pair| (pair[0] - mean) * (pair[1] - mean))
            .sum::<f64>()
            / (first_path.len() - 1) as f64;

        assert!(mean.abs() < 0.04 * sigma);
        assert_close(variance, process.covariance(0.0), 0.04 * sigma.powi(2));
        assert_close(
            lag_one / variance,
            process.covariance(5.0e-6) / process.covariance(0.0),
            0.025,
        );
    }

    #[test]
    fn ou_parameters_and_sample_times_are_validated() {
        assert_eq!(
            OrnsteinUhlenbeck::new(-1.0, 1.0).unwrap_err(),
            StochasticError::InvalidStationaryStandardDeviation
        );
        assert_eq!(
            OrnsteinUhlenbeck::new(1.0, 0.0).unwrap_err(),
            StochasticError::InvalidCorrelationTime
        );

        let process = OrnsteinUhlenbeck::new(1.0, 1.0).unwrap();
        let mut sampler = process.sampler(ChaCha8Rng::seed_from_u64(7));
        assert_eq!(
            sampler.next_path(&[0.0, 0.5, 0.5]).unwrap_err(),
            StochasticError::NonIncreasingSampleTimes(2)
        );
    }

    #[test]
    fn one_sided_ou_psd_has_the_expected_dc_level_and_corner() {
        let sigma = 2_300.0;
        let correlation_time = 17.0e-6;
        let process = OrnsteinUhlenbeck::new(sigma, correlation_time).unwrap();
        let dc_level = 4.0 * sigma.powi(2) * correlation_time;
        let corner_frequency = 1.0 / (TAU * correlation_time);

        assert_close(process.one_sided_psd(0.0), dc_level, 1.0e-12);
        assert_close(
            process.one_sided_psd(corner_frequency),
            0.5 * dc_level,
            1.0e-12,
        );
    }

    #[test]
    fn brownian_oscillator_psd_has_the_requested_peak_and_fwhm() {
        let sigma = 2_300.0;
        let peak_frequency = 1_700.0;
        let fwhm = 420.0;
        let process = UnderdampedBrownianOscillator::new(sigma, peak_frequency, fwhm).unwrap();
        let peak_angular_frequency = TAU * peak_frequency;
        let damping_angular_rate = 0.5 * TAU * fwhm;
        let half_maximum_centre =
            (peak_angular_frequency.powi(2) + damping_angular_rate.powi(2)).sqrt();
        let lower_half_maximum = (half_maximum_centre - damping_angular_rate) / TAU;
        let upper_half_maximum = (half_maximum_centre + damping_angular_rate) / TAU;
        let peak_psd = 2.0 * sigma.powi(2) / damping_angular_rate;

        assert_close(process.covariance(0.0), sigma.powi(2), 1.0e-12);
        assert_eq!(process.one_sided_psd(0.0), 0.0);
        assert_close(
            process.one_sided_psd(peak_frequency),
            peak_psd,
            1.0e-12 * peak_psd,
        );
        assert_close(
            process.one_sided_psd(lower_half_maximum),
            0.5 * peak_psd,
            1.0e-12 * peak_psd,
        );
        assert_close(
            process.one_sided_psd(upper_half_maximum),
            0.5 * peak_psd,
            1.0e-12 * peak_psd,
        );
        assert_close(upper_half_maximum - lower_half_maximum, fwhm, 1.0e-12);
    }

    #[test]
    fn seeded_brownian_oscillator_paths_are_reproducible_and_stationary() {
        let sigma = 2_300.0;
        let process = UnderdampedBrownianOscillator::new(sigma, 1_700.0, 420.0).unwrap();
        let sample_times = [0.0, 17.0e-6, 83.0e-6, 190.0e-6];
        let mut first = process.sampler(ChaCha8Rng::seed_from_u64(987_654));
        let mut second = process.sampler(ChaCha8Rng::seed_from_u64(987_654));

        assert_eq!(
            first.next_path(&sample_times).unwrap(),
            second.next_path(&sample_times).unwrap()
        );

        let lag = 83.0e-6;
        let mut sampler = process.sampler(ChaCha8Rng::seed_from_u64(246_810));
        let mut pair = [0.0; 2];
        let mut first_sum = 0.0;
        let mut second_sum = 0.0;
        let mut first_squared_sum = 0.0;
        let mut product_sum = 0.0;
        let sample_count = 50_000;
        for _ in 0..sample_count {
            sampler.fill_path(&[0.0, lag], &mut pair).unwrap();
            first_sum += pair[0];
            second_sum += pair[1];
            first_squared_sum += pair[0].powi(2);
            product_sum += pair[0] * pair[1];
        }
        let count = sample_count as f64;
        let first_mean = first_sum / count;
        let second_mean = second_sum / count;
        let variance = first_squared_sum / count - first_mean.powi(2);
        let covariance = product_sum / count - first_mean * second_mean;

        assert_close(variance, sigma.powi(2), 0.02 * sigma.powi(2));
        assert_close(covariance, process.covariance(lag), 0.02 * sigma.powi(2));
    }

    #[test]
    fn brownian_oscillator_parameters_and_sample_times_are_validated() {
        assert_eq!(
            UnderdampedBrownianOscillator::new(-1.0, 1.0, 0.5).unwrap_err(),
            StochasticError::InvalidStationaryStandardDeviation
        );
        assert_eq!(
            UnderdampedBrownianOscillator::new(1.0, 0.0, 0.5).unwrap_err(),
            StochasticError::InvalidPeakFrequency
        );
        assert_eq!(
            UnderdampedBrownianOscillator::new(1.0, 1.0, 0.0).unwrap_err(),
            StochasticError::InvalidFullWidthAtHalfMaximum
        );
        assert_eq!(
            UnderdampedBrownianOscillator::new(1.0, 1.0, 2.0).unwrap_err(),
            StochasticError::NotUnderdamped
        );

        let process = UnderdampedBrownianOscillator::new(1.0, 1.0, 0.5).unwrap();
        let mut sampler = process.sampler(ChaCha8Rng::seed_from_u64(7));
        assert_eq!(
            sampler.next_path(&[0.0, 0.5, 0.5]).unwrap_err(),
            StochasticError::NonIncreasingSampleTimes(2)
        );
    }

    #[test]
    fn brownian_oscillator_short_steps_remain_finite_near_critical_damping() {
        let process = UnderdampedBrownianOscillator::new(1.0, 1.0, 1.999_999).unwrap();
        let sample_times: Vec<_> = (0..100).map(|index| index as f64 * 1.0e-12).collect();
        let path = process
            .sampler(ChaCha8Rng::seed_from_u64(123))
            .next_path(&sample_times)
            .unwrap();

        assert!(path.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn brownian_oscillator_innovation_covariance_stays_positive_semidefinite() {
        for damping_ratio in [1.0e-8, 1.0e-4, 0.05, 0.5, 0.999_999] {
            let process =
                UnderdampedBrownianOscillator::new(1.0, 1.0, 2.0 * damping_ratio).unwrap();
            for scaled_peak in [1.0e-12, 1.0e-8, 1.0e-4, 1.0e-2, 1.0, 10.0, 1.0e3] {
                let step = NormalizedOscillatorStep::new(process, scaled_peak / TAU);
                let determinant = step.position_variance * step.velocity_variance
                    - step.position_velocity_covariance.powi(2);

                assert!(step.position_variance.is_finite());
                assert!(step.velocity_variance.is_finite());
                assert!(step.position_velocity_covariance.is_finite());
                assert!(step.position_variance >= 0.0);
                assert!(step.velocity_variance >= 0.0);
                assert!(
                    determinant >= -1.0e-14,
                    "innovation determinant {determinant} for damping ratio {damping_ratio} and scaled peak {scaled_peak}"
                );
            }
        }
    }

    #[test]
    fn running_vector_statistics_match_direct_mean_and_standard_error() {
        let mut statistics = RunningVectorStats::new(2);
        statistics.push(&[1.0, 8.0]);
        statistics.push(&[3.0, 4.0]);
        statistics.push(&[5.0, 0.0]);

        assert_eq!(statistics.mean(), [3.0, 4.0]);
        let standard_errors = statistics.standard_errors();
        assert_close(standard_errors[0], (4.0_f64 / 3.0).sqrt(), 1.0e-14);
        assert_close(standard_errors[1], (16.0_f64 / 3.0).sqrt(), 1.0e-14);
    }
}
