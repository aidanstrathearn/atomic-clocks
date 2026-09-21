use std::f64::consts::PI;

use crate::maths::demodulation::{Demodulation, lockin_period};
use crate::maths::linspace;
use crate::twolevel::{
    BlochVec, Decay, Hamiltonian, Liouvillian, Observable, Process, TimeDependentHamiltonian, Vec3,
    steps,
};

use super::hamiltonian::HamiltonianParams;

/// Sampling for the spatially averaged detuning response at one velocity.
#[derive(Clone, Copy, Debug)]
pub struct LinearResponseSolverParams {
    pub kr_n: usize,
    pub steps_per_period: usize,
    /// Ground-state preparation time before the earliest possible kick, in periods.
    /// Increase this to check convergence to the settled response; zero is allowed.
    pub warmup_periods: usize,
    /// Maximum delay in periods. Zero returns only equal-time kick responses.
    pub delay_periods: usize,
}

impl Default for LinearResponseSolverParams {
    fn default() -> Self {
        Self {
            kr_n: 5,
            steps_per_period: 100,
            warmup_periods: 10,
            delay_periods: 3,
        }
    }
}

struct LinearResponseGrid {
    total_steps: usize,
    warmup_steps: usize,
    delay_steps: usize,
    dt: f64,
}

impl LinearResponseSolverParams {
    /// Validates sampling and derives grid dimensions with checked arithmetic.
    fn validate(&self, period: f64) -> Result<LinearResponseGrid, String> {
        if self.kr_n == 0 || self.steps_per_period == 0 {
            return Err("kr_n and steps_per_period must be positive".to_string());
        }
        let total_steps = self
            .warmup_periods
            .checked_add(self.delay_periods)
            .and_then(|periods| periods.checked_add(1))
            .and_then(|periods| periods.checked_mul(self.steps_per_period))
            .filter(|&count| count < usize::MAX)
            .ok_or("response time grid is too large")?;
        let dt = period / self.steps_per_period as f64;
        if !dt.is_finite() || dt <= 0.0 || !(total_steps as f64 * dt).is_finite() {
            return Err(
                "response time grid must have finite positive spacing and duration".to_string(),
            );
        }
        Ok(LinearResponseGrid {
            total_steps,
            warmup_steps: self.warmup_periods * self.steps_per_period,
            delay_steps: self.delay_periods * self.steps_per_period,
            dt,
        })
    }
}

#[derive(Clone, Debug)]
pub struct LinearResponseOutput {
    /// Absolute observation times from 0 to T, including both endpoints.
    pub times: Vec<f64>,
    /// Delays from 0 to `delay_periods * T`, including both endpoints.
    pub delays: Vec<f64>,
    /// `response[i][j] = chi(times[i], times[i] - delays[j])`.
    /// These are signed kick responses, without time-integration weights.
    pub response: Vec<Vec<f64>>,
}

impl LinearResponseOutput {
    /// Demodulates observation time at each fixed delay using [`lockin_period`].
    /// Returns one signed cosine/sine coefficient pair per entry in `delays`.
    /// The reference phase follows observation time, not the earlier kick time.
    ///
    /// As in [`lockin_period`], harmonic zero returns twice the signed mean in
    /// `in_phase` and zero `quadrature`. Divide by two for the mean response.
    /// Results remain delay-dependent kernels, without delay-integration weights.
    ///
    /// Assumes uniform observation times spanning one complete modulation period,
    /// including both endpoints, as supplied by [`compute_linear_response`].
    /// Panics if the response dimensions do not match the axes or there are fewer
    /// than two observation times.
    pub fn demodulate(&self, harmonic: usize) -> Vec<Demodulation> {
        assert!(
            self.times.len() >= 2,
            "lock-in requires at least two samples"
        );
        assert_eq!(
            self.response.len(),
            self.times.len(),
            "response must match observation times"
        );
        assert!(
            self.response
                .iter()
                .all(|row| row.len() == self.delays.len()),
            "response rows must match delays",
        );
        let mut column = vec![0.0; self.times.len()];
        (0..self.delays.len())
            .map(|delay| {
                for (sample, row) in column.iter_mut().zip(&self.response) {
                    *sample = row[delay];
                }
                lockin_period(&column, harmonic)
            })
            .collect()
    }
}

/// Spatially averages the response of `observable · sigma` to a change in `delta`.
/// Uses exactly `hamiltonian.kv` and `hamiltonian.delta`, with no detuning scan or
/// velocity averaging. The observable is constant in `hamiltonian.frame` and its
/// magnitude scales the result. Use `(0, 1, 0)` to match the MTS example.
///
/// The perturbation is `H = H0 + f(t) sigma_z / 2`, with decay held fixed:
/// `delta<O(t)> = integral chi(t, s) f(s) ds` over `s <= t`. Thus each entry is
/// the derivative after a kick `exp(-i epsilon sigma_z / 2)` at `s`, not an
/// ordinary two-time correlation. The zero-delay entry includes the full kick.
///
/// Samples `kr + 2*pi*k/kr_n` for `k = 1..=kr_n`, with equal weights, as in
/// [`super::compute_demod`]. Evolution starts in the ground state at
/// `-(warmup_periods + delay_periods) * T`. Each atom is evolved at its actual
/// absolute times using left-endpoint Hamiltonians; individual trajectories are
/// never wrapped at T, since nonzero Doppler shifts can make them nonperiodic.
/// Only after sufficient warmup and spatial convergence should the probe-frame
/// averaged response be treated as periodic in observation phase. Check warmup,
/// time resolution, spatial sampling and the delay cutoff for convergence.
pub fn compute_linear_response(
    hamiltonian: &HamiltonianParams,
    decay: Decay,
    solver: &LinearResponseSolverParams,
    observable: Vec3,
) -> Result<LinearResponseOutput, String> {
    hamiltonian.validate()?;
    if ![observable.x, observable.y, observable.z]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("Hamiltonian and observable coefficients must be finite".to_string());
    }
    decay.validate()?;
    let LinearResponseGrid {
        total_steps,
        warmup_steps,
        delay_steps,
        dt,
    } = solver.validate(hamiltonian.modulation.period())?;

    let origin = warmup_steps + delay_steps;
    let grid: Vec<_> = (0..=total_steps)
        .map(|i| (i as f64 - origin as f64) * dt)
        .collect();
    let times: Vec<_> = (0..=solver.steps_per_period)
        .map(|i| i as f64 * dt)
        .collect();
    let delays: Vec<_> = (0..=delay_steps).map(|i| i as f64 * dt).collect();
    let mut response = vec![vec![0.0; delays.len()]; times.len()];
    let measurement = Observable::new(0.0, observable);
    let perturbation = Observable::from(Hamiltonian::new(0.0, 0.0, 1.0));

    for phase in linspace(0.0, 2.0 * PI, solver.kr_n).into_iter().skip(1) {
        let atom = HamiltonianParams {
            kr: hamiltonian.kr + phase,
            ..*hamiltonian
        };
        let step_at = |t, duration| {
            Liouvillian {
                hamiltonian: atom.h(t),
                decay,
            }
            .for_duration(duration)
        };
        let mut channel_steps = steps(&grid, &step_at);
        let warmed = Process::new(channel_steps.by_ref().take(warmup_steps))
            .propagate_to_final(BlochVec::ground());

        let channels: Vec<_> = channel_steps.collect();
        let initial_states = Process::new(&channels).trajectory(warmed);

        for (i, (row, initial)) in response.iter_mut().zip(initial_states).enumerate() {
            let kicks = Process::new(&channels[i..i + delay_steps]).linear_response(
                initial,
                measurement,
                perturbation,
            );
            for (sum, kick) in row.iter_mut().zip(kicks.into_iter().rev()) {
                *sum += kick / solver.kr_n as f64;
            }
        }
    }
    if response.iter().flatten().any(|value| !value.is_finite()) {
        return Err("non-finite response: the dissipative propagator may be singular".to_string());
    }
    Ok(LinearResponseOutput {
        times,
        delays,
        response,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{Frame, MtsParams};
    use super::*;
    use crate::maths::demodulation::ModulationParams;
    use crate::twolevel::Channel;

    #[test]
    fn response_demodulation_resolves_time_harmonics_at_each_delay() {
        let times = linspace(0.0, 2.0 * PI, 16);
        let coefficients = [
            (1.2, -0.3, 0.7, 0.4, -0.9),
            (-0.4, 1.1, -0.2, -0.6, 0.3),
            (0.0, -0.8, -0.6, 0.2, 0.5),
        ];
        let response = times
            .iter()
            .map(|t| {
                coefficients
                    .iter()
                    .map(|&(mean, cosine, sine, cosine2, sine2)| {
                        mean + cosine * t.cos()
                            + sine * t.sin()
                            + cosine2 * (2.0 * t).cos()
                            + sine2 * (2.0 * t).sin()
                    })
                    .collect()
            })
            .collect();
        let output = LinearResponseOutput {
            times,
            delays: vec![0.0, 0.3, 1.7],
            response,
        };
        let dc = output.demodulate(0);
        let harmonic = output.demodulate(1);
        let second_harmonic = output.demodulate(2);
        assert_eq!(dc.len(), output.delays.len());
        assert_eq!(harmonic.len(), output.delays.len());
        assert_eq!(second_harmonic.len(), output.delays.len());
        for (j, (mean, cosine, sine, cosine2, sine2)) in coefficients.into_iter().enumerate() {
            assert!((dc[j].in_phase - 2.0 * mean).abs() < 1e-12);
            assert_eq!(dc[j].quadrature, 0.0);
            assert!((harmonic[j].in_phase - cosine).abs() < 1e-12);
            assert!((harmonic[j].quadrature - sine).abs() < 1e-12);
            assert!((second_harmonic[j].in_phase - cosine2).abs() < 1e-12);
            assert!((second_harmonic[j].quadrature - sine2).abs() < 1e-12);
        }
    }

    #[test]
    fn spatial_response_matches_finite_kicks_at_absolute_times() {
        let solver = LinearResponseSolverParams {
            kr_n: 3,
            steps_per_period: 8,
            warmup_periods: 2,
            delay_periods: 2,
        };
        let observable = Vec3 {
            x: 0.3,
            y: 1.0,
            z: -0.2,
        };
        let decay = MtsParams::default().decay;
        for frame in [Frame::Probe, Frame::Pump, Frame::Atom] {
            let atom = HamiltonianParams {
                frame,
                kv: 0.37,
                delta: -0.6,
                kr: 0.23,
                modulation: ModulationParams {
                    shift: 0.19,
                    ..Default::default()
                },
                ..Default::default()
            };
            let result = compute_linear_response(&atom, decay, &solver, observable).unwrap();
            let period = atom.modulation.period();
            let dt = period / solver.steps_per_period as f64;
            assert_eq!(result.times.len(), 9);
            assert_eq!(result.delays.len(), 17);
            assert_eq!(result.times[0], 0.0);
            assert_eq!(result.times[8], period);
            assert_eq!(result.delays[0], 0.0);
            assert_eq!(result.delays[16], 2.0 * period);

            // Independently evolve finite z rotations from the preparation
            // time, including kicks at both ends and more than a cycle ago.
            for i in [0, 3, 8] {
                for j in [0, 1, 11, 16] {
                    let origin =
                        (solver.warmup_periods + solver.delay_periods) * solver.steps_per_period;
                    let end = origin + i;
                    let kick_at = end - j;
                    let measure = |epsilon: f64| {
                        let mut sum = 0.0;
                        for k in 1..=solver.kr_n {
                            let sample = HamiltonianParams {
                                kr: atom.kr + 2.0 * PI * k as f64 / solver.kr_n as f64,
                                ..atom
                            };
                            let mut state = BlochVec::ground();
                            for boundary in 0..=end {
                                if boundary == kick_at {
                                    state.r = state.r.rotate(
                                        Vec3 {
                                            x: 0.0,
                                            y: 0.0,
                                            z: 1.0,
                                        },
                                        epsilon,
                                    );
                                }
                                if boundary < end {
                                    let t = (boundary as f64 - origin as f64) * dt;
                                    state = Liouvillian {
                                        hamiltonian: sample.h(t),
                                        decay,
                                    }
                                    .for_duration(dt)
                                    .apply_to(state);
                                }
                            }
                            sum += state.r.dot(observable) / solver.kr_n as f64;
                        }
                        sum
                    };
                    let epsilon = 1e-5;
                    let expected = (measure(epsilon) - measure(-epsilon)) / (2.0 * epsilon);
                    assert!(
                        (result.response[i][j] - expected).abs() < 1e-9,
                        "{frame:?}, observation {i}, delay {j}: {} != {expected}",
                        result.response[i][j],
                    );
                }
            }
        }
    }

    #[test]
    fn integrated_response_matches_analytic_static_detuning_slope() {
        let atom = HamiltonianParams {
            r_pump: 0.0,
            r_prbe: 0.8,
            delta: 0.7,
            ..Default::default()
        };
        let decay = Decay {
            gamma_up: 0.0,
            gamma_down: 1.0,
            gamma_phi: 0.2,
        };
        let solver = LinearResponseSolverParams {
            kr_n: 2,
            steps_per_period: 128,
            warmup_periods: 6,
            delay_periods: 6,
        };
        let observable = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let result = compute_linear_response(&atom, decay, &solver, observable).unwrap();
        // For the constant probe drive, r_y = gamma2 * Omega / denominator
        // and the immediate z-kick response is r_x = -Omega * delta / denominator.
        let denominator =
            decay.gamma2().powi(2) + atom.delta.powi(2) + decay.gamma2() * atom.r_prbe.powi(2);
        let expected_kick = -atom.r_prbe * atom.delta / denominator;
        let expected_slope = -2.0 * decay.gamma2() * atom.r_prbe * atom.delta / denominator.powi(2);
        let dt = result.delays[1];
        for row in &result.response {
            assert!((row[0] - expected_kick).abs() < 1e-10);
            let integral: f64 = row
                .windows(2)
                .map(|pair| 0.5 * dt * (pair[0] + pair[1]))
                .sum();
            assert!(
                (integral - expected_slope).abs() < 2e-6,
                "{integral} != {expected_slope}"
            );
        }
    }

    #[test]
    fn response_accepts_zero_delay_and_rejects_invalid_sampling() {
        let atom = HamiltonianParams::default();
        let decay = MtsParams::default().decay;
        let observable = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let solver = LinearResponseSolverParams {
            kr_n: 1,
            steps_per_period: 4,
            warmup_periods: 0,
            delay_periods: 0,
        };
        let result = compute_linear_response(&atom, decay, &solver, observable).unwrap();
        assert_eq!(result.delays, [0.0]);
        assert_eq!(result.response.len(), 5);
        assert!(result.response.iter().all(|row| row.len() == 1));
        assert_eq!(result.response[0][0], 0.0); // Ground state commutes with sigma_z.
        for invalid in [
            LinearResponseSolverParams { kr_n: 0, ..solver },
            LinearResponseSolverParams {
                steps_per_period: 0,
                ..solver
            },
            LinearResponseSolverParams {
                delay_periods: usize::MAX,
                ..solver
            },
        ] {
            assert!(compute_linear_response(&atom, decay, &invalid, observable).is_err());
        }
        for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let invalid = HamiltonianParams {
                modulation: ModulationParams {
                    frequency,
                    ..atom.modulation
                },
                ..atom
            };
            assert!(compute_linear_response(&invalid, decay, &solver, observable).is_err());
        }
    }
}
