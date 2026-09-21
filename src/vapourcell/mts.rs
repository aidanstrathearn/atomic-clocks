use std::f64::consts::PI;

use crate::maths::demodulation::{Demodulation, lockin_period};
use crate::maths::linspace;
use crate::twolevel::{
    BlochVec, Decay, Liouvillian, Process, TimeDependentHamiltonian, Vec3, steps,
};

use super::hamiltonian::HamiltonianParams;

/// Spatial sampling, integration grid, and detuning scan settings for MTS.
#[derive(Clone, Copy, Debug)]
pub struct MtsSolverParams {
    pub kr_n: usize,
    /// Integration intervals per modulation period; the final window has one more sample.
    pub steps_per_period: usize,
    pub n_periods: usize,
    pub hz_lim: f64,
    pub hz_num: usize,
}

impl Default for MtsSolverParams {
    fn default() -> Self {
        Self {
            kr_n: 5,
            steps_per_period: 100,
            n_periods: 3,
            hz_lim: 5.0,
            hz_num: 50,
        }
    }
}

impl MtsSolverParams {
    /// Validates spatial sampling, integration, and detuning scan settings.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_propagation()?;
        self.validate_scan()
    }

    fn validate_propagation(&self) -> Result<(), String> {
        if self.kr_n == 0 {
            return Err("kr_n must be positive".to_string());
        }
        if self.n_periods == 0 {
            return Err("n_periods must be positive".to_string());
        }
        if self.steps_per_period == 0 {
            return Err("steps_per_period must be positive".to_string());
        }
        self.n_periods
            .checked_mul(self.steps_per_period)
            .and_then(|steps| steps.checked_add(1))
            .ok_or("MTS time grid is too large")?;
        Ok(())
    }

    fn validate_scan(&self) -> Result<(), String> {
        if self.hz_num == 0 {
            return Err("hz_num must be positive".to_string());
        }
        if !self.hz_lim.is_finite() || self.hz_lim < 0.0 {
            return Err("hz_lim must be finite and nonnegative".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MtsParams {
    pub hamiltonian: HamiltonianParams,
    pub decay: Decay,
    pub solver: MtsSolverParams,
}

impl Default for MtsParams {
    fn default() -> Self {
        Self {
            hamiltonian: HamiltonianParams::default(),
            decay: Decay {
                gamma_up: 0.0,
                gamma_down: 1.0,
                gamma_phi: 0.0,
            },
            solver: MtsSolverParams::default(),
        }
    }
}

impl MtsParams {
    /// Validates the Hamiltonian, decay, and solver configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_propagation()?;
        self.solver.validate_scan()
    }

    fn validate_propagation(&self) -> Result<(), String> {
        self.hamiltonian.validate()?;
        self.decay.validate()?;
        self.solver.validate_propagation()
    }
}

#[derive(Clone, Debug)]
pub struct DemodOutput {
    /// Detuning offsets relative to the Hamiltonian's `delta`.
    pub hz: Vec<f64>,
    /// Zero-frequency lock-in results: twice the signed mean in `in_phase`,
    /// with zero `quadrature`.
    pub dc: Vec<Demodulation>,
    /// Signed cosine and sine coefficients at the modulation frequency.
    pub harmonic: Vec<Demodulation>,
    /// Signed cosine and sine coefficients at twice the modulation frequency.
    pub second_harmonic: Vec<Demodulation>,
    /// Signed cosine and sine coefficients at three times the modulation frequency.
    pub third_harmonic: Vec<Demodulation>,
}

struct Demodulator {
    params: MtsParams,
    observable: Vec3,
    time_grid: Vec<f64>,
    last_period_start: usize,
    spatial_phases: Vec<f64>,
}

impl Demodulator {
    fn new(params: &MtsParams, observable: Vec3) -> Result<Self, String> {
        params.validate_propagation()?;
        Ok(Self::from_validated(params, observable))
    }

    fn from_validated(params: &MtsParams, observable: Vec3) -> Self {
        Self {
            params: *params,
            observable,
            time_grid: time_grid(
                params.hamiltonian.modulation.period(),
                params.solver.steps_per_period,
                params.solver.n_periods,
            ),
            last_period_start: (params.solver.n_periods - 1) * params.solver.steps_per_period,
            spatial_phases: spatial_phases(params),
        }
    }

    fn projected_at_offset(&self, hz_offset: f64) -> Vec<f64> {
        let mut projected = vec![0.0; self.params.solver.steps_per_period + 1];
        for &kr_phase in &self.spatial_phases {
            let atom = HamiltonianParams {
                delta: self.params.hamiltonian.delta + hz_offset,
                kr: kr_phase,
                ..self.params.hamiltonian
            };
            accumulate_projected_trajectory(
                &atom,
                self.params.decay,
                &self.time_grid,
                self.last_period_start,
                self.observable,
                &mut projected,
            );
        }

        let kr_scale = 1.0 / self.spatial_phases.len() as f64;
        for value in &mut projected {
            *value *= kr_scale;
        }
        projected
    }

    fn demodulate<const N: usize>(
        &self,
        hz_offset: f64,
        harmonics: [usize; N],
    ) -> Result<[Demodulation; N], String> {
        validate_harmonics(&harmonics, self.params.solver.steps_per_period)?;
        let projected = self.projected_at_offset(hz_offset);
        Ok(harmonics.map(|harmonic| lockin_period(&projected, harmonic)))
    }

    fn demodulate_harmonics(
        &self,
        hz_offset: f64,
        harmonics: &[usize],
    ) -> Result<Vec<Demodulation>, String> {
        validate_harmonics(harmonics, self.params.solver.steps_per_period)?;
        let projected = self.projected_at_offset(hz_offset);
        Ok(harmonics
            .iter()
            .map(|&harmonic| lockin_period(&projected, harmonic))
            .collect())
    }
}

/// Demodulates selected harmonics at the Hamiltonian's exact detuning.
///
/// Unlike [`compute_demod`], this evaluates only `hamiltonian.delta` rather than
/// building a detuning scan. Harmonics are returned in the same order as the
/// supplied indices. The observable and lock-in conventions match
/// [`compute_demod`]. The solver's `hz_lim` and `hz_num` scan settings are ignored.
pub fn compute_demod_harmonics(
    params: &MtsParams,
    observable: Vec3,
    harmonics: &[usize],
) -> Result<Vec<Demodulation>, String> {
    Demodulator::new(params, observable)?.demodulate_harmonics(0.0, harmonics)
}

/// Demodulates the expectation of `observable · sigma` in the selected frame.
/// The observable is constant in that frame and is not normalized; its magnitude
/// scales the measured signal.
pub fn compute_demod(params: &MtsParams, observable: Vec3) -> Result<DemodOutput, String> {
    params.validate()?;
    let demodulator = Demodulator::from_validated(params, observable);
    const HARMONICS: [usize; 4] = [0, 1, 2, 3];
    let hz_array = if params.solver.hz_num == 1 {
        vec![-params.solver.hz_lim]
    } else {
        linspace(
            -params.solver.hz_lim,
            params.solver.hz_lim,
            params.solver.hz_num - 1,
        )
    };
    let mut dc = Vec::with_capacity(hz_array.len());
    let mut harmonic = Vec::with_capacity(hz_array.len());
    let mut second_harmonic = Vec::with_capacity(hz_array.len());
    let mut third_harmonic = Vec::with_capacity(hz_array.len());

    for &hz_offset in &hz_array {
        let [
            dc_value,
            harmonic_value,
            second_harmonic_value,
            third_harmonic_value,
        ] = demodulator.demodulate(hz_offset, HARMONICS)?;
        dc.push(dc_value);
        harmonic.push(harmonic_value);
        second_harmonic.push(second_harmonic_value);
        third_harmonic.push(third_harmonic_value);
    }

    Ok(DemodOutput {
        hz: hz_array,
        dc,
        harmonic,
        second_harmonic,
        third_harmonic,
    })
}

fn validate_harmonics(harmonics: &[usize], steps_per_period: usize) -> Result<(), String> {
    if let Some(&harmonic) = harmonics
        .iter()
        .find(|&&harmonic| harmonic > steps_per_period / 2)
    {
        return Err(format!(
            "harmonic {harmonic} exceeds the Nyquist limit for {steps_per_period} steps per period"
        ));
    }
    Ok(())
}

fn spatial_phases(params: &MtsParams) -> Vec<f64> {
    linspace(0.0, 2.0 * PI, params.solver.kr_n)
        .into_iter()
        .skip(1)
        .map(|phase| phase + params.hamiltonian.kr)
        .collect()
}

fn time_grid(period: f64, steps_per_period: usize, n_periods: usize) -> Vec<f64> {
    let dt = period / steps_per_period as f64;
    (0..=n_periods * steps_per_period)
        .map(|i| i as f64 * dt)
        .collect()
}

fn accumulate_projected_trajectory(
    params: &HamiltonianParams,
    decay: Decay,
    t_array: &[f64],
    last_start: usize,
    observable: Vec3,
    projected: &mut [f64],
) {
    let step_at = |t, dt| {
        Liouvillian {
            hamiltonian: params.h(t),
            decay,
        }
        .for_duration(dt)
    };
    let mut channels = steps(t_array, &step_at);
    let warmed =
        Process::new(channels.by_ref().take(last_start)).propagate_to_final(BlochVec::ground());
    let window = Process::new(channels).trajectory(warmed);
    for (sum, state) in projected.iter_mut().zip(window) {
        *sum += state.r.dot(observable);
    }
}

#[cfg(test)]
mod tests {
    use super::super::Frame;
    use super::*;
    use crate::maths::demodulation::ModulationParams;

    #[test]
    fn rejects_invalid_modulation_frequency() {
        for mod_freq in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let params = MtsParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        frequency: mod_freq,
                        ..ModulationParams::default()
                    },
                    ..HamiltonianParams::default()
                },
                ..MtsParams::default()
            };
            assert_eq!(
                compute_demod(&params, Vec3::from_angles(PI / 2.0, PI / 2.0)).unwrap_err(),
                "mod_freq must be positive and finite"
            );
        }
    }

    #[test]
    fn default_scan_is_finite_in_every_frame() {
        for frame in [Frame::Pump, Frame::Atom, Frame::Probe] {
            let params = MtsParams {
                hamiltonian: HamiltonianParams {
                    frame,
                    ..HamiltonianParams::default()
                },
                ..MtsParams::default()
            };
            let output = compute_demod(&params, Vec3::from_angles(PI / 2.0, PI / 2.0))
                .expect("default scan succeeds");
            assert_eq!(output.hz.len(), params.solver.hz_num);
            assert!(output.hz.iter().all(|value| value.is_finite()));
            for values in [
                &output.dc,
                &output.harmonic,
                &output.second_harmonic,
                &output.third_harmonic,
            ] {
                assert_eq!(values.len(), params.solver.hz_num);
                assert!(
                    values
                        .iter()
                        .all(|value| value.in_phase.is_finite() && value.quadrature.is_finite()),
                    "{frame:?}"
                );
            }
        }
    }

    #[test]
    fn selected_harmonics_match_the_existing_demodulation_output() {
        let params = MtsParams {
            solver: MtsSolverParams {
                kr_n: 3,
                steps_per_period: 40,
                n_periods: 2,
                hz_lim: 0.0,
                hz_num: 1,
            },
            ..MtsParams::default()
        };
        let observable = Vec3::from_angles(PI / 2.0, PI / 2.0);
        let output = compute_demod(&params, observable).unwrap();
        let harmonics = [3, 1, 0, 2, 4];
        let selected = compute_demod_harmonics(&params, observable, &harmonics).unwrap();
        let fixed = Demodulator::new(&params, observable)
            .unwrap()
            .demodulate(0.0, harmonics)
            .unwrap();
        let expected = [
            output.third_harmonic[0],
            output.harmonic[0],
            output.dc[0],
            output.second_harmonic[0],
        ];

        for (actual, expected) in selected.iter().zip(expected) {
            assert_eq!(actual.in_phase, expected.in_phase);
            assert_eq!(actual.quadrature, expected.quadrature);
        }
        for (actual, expected) in fixed.iter().zip(&selected) {
            assert_eq!(actual.in_phase, expected.in_phase);
            assert_eq!(actual.quadrature, expected.quadrature);
        }
        assert!(selected[4].in_phase.is_finite());
        assert!(selected[4].quadrature.is_finite());
    }

    #[test]
    fn exact_demodulation_does_not_require_scan_settings() {
        let params = MtsParams {
            solver: MtsSolverParams {
                hz_lim: f64::NAN,
                hz_num: 0,
                ..MtsSolverParams::default()
            },
            ..MtsParams::default()
        };
        let observable = Vec3::from_angles(PI / 2.0, PI / 2.0);

        let output = compute_demod_harmonics(&params, observable, &[0, 1]).unwrap();
        assert_eq!(output.len(), 2);
        assert!(
            output
                .iter()
                .all(|value| value.in_phase.is_finite() && value.quadrature.is_finite())
        );
        assert!(compute_demod(&params, observable).is_err());
    }

    #[test]
    fn selected_harmonics_reject_frequencies_above_nyquist() {
        let params = MtsParams {
            solver: MtsSolverParams {
                steps_per_period: 20,
                ..MtsSolverParams::default()
            },
            ..MtsParams::default()
        };
        let error = compute_demod_harmonics(&params, Vec3::from_angles(PI / 2.0, PI / 2.0), &[11])
            .unwrap_err();
        assert_eq!(
            error,
            "harmonic 11 exceeds the Nyquist limit for 20 steps per period"
        );
    }

    #[test]
    fn raw_observable_coefficients_match_a_damped_rabi_trajectory() {
        // Isotropic unit-rate relaxation and a constant x drive give
        // r_y(t) = drive * exp(-t) * sin(t), r_z(t) = -exp(-t)*cos(t).
        for (drive, periods) in [(-1.0, 1), (1.0, 1), (1.0, 2)] {
            for observable in [
                Vec3::from_angles(PI / 2.0, PI / 2.0),
                Vec3 {
                    x: 0.5,
                    y: -2.0,
                    z: 3.0,
                },
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            ] {
                let params = MtsParams {
                    hamiltonian: HamiltonianParams {
                        r_pump: 0.0,
                        r_prbe: drive,
                        modulation: ModulationParams {
                            depth: 0.0,
                            ..ModulationParams::default()
                        },
                        ..HamiltonianParams::default()
                    },
                    decay: Decay {
                        gamma_up: 0.5,
                        gamma_down: 0.5,
                        gamma_phi: 0.5,
                    },
                    solver: MtsSolverParams {
                        kr_n: 1,
                        n_periods: periods,
                        steps_per_period: 8,
                        hz_num: 3,
                        ..MtsSolverParams::default()
                    },
                };
                let output = compute_demod(&params, observable).unwrap();
                assert_eq!(output.hz[1], 0.0);
                let mut expected = [0.0; 5];
                for i in 0..=8 {
                    let t = (periods - 1) as f64 * 2.0 * PI + i as f64 * PI / 4.0;
                    let signal =
                        (-t).exp() * (observable.y * drive * t.sin() - observable.z * t.cos());
                    // Trapezoidal weight, including the lock-in factor 2 / duration.
                    let weight = if i == 0 || i == 8 { 0.125 } else { 0.25 };
                    expected[0] += weight * signal;
                    expected[1] += weight * signal * t.cos();
                    expected[2] += weight * signal * t.sin();
                    expected[3] += weight * signal * (2.0 * t).cos();
                    expected[4] += weight * signal * (2.0 * t).sin();
                }
                for (actual, expected) in [
                    (output.dc[1].in_phase, expected[0]),
                    (output.harmonic[1].in_phase, expected[1]),
                    (output.harmonic[1].quadrature, expected[2]),
                    (output.second_harmonic[1].in_phase, expected[3]),
                    (output.second_harmonic[1].quadrature, expected[4]),
                ] {
                    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
                }
                assert_eq!(output.dc[1].quadrature, 0.0);
            }
        }
    }

    #[test]
    fn aligned_scan_converges_to_continuous_damped_rabi_demodulation() {
        // Analytic integrals of exp(-t)*sin(t), multiplied by 1, cos(t), sin(t),
        // over [0, 2*pi], including the lock-in normalization 2 / period.
        let scale = (1.0 - (-2.0 * PI).exp()) / (2.0 * PI);
        let expected = [scale, 0.4 * scale, 0.8 * scale];
        let mut previous = [f64::INFINITY; 3];
        for steps in [16, 32, 64] {
            let output = compute_demod(
                &MtsParams {
                    hamiltonian: HamiltonianParams {
                        r_pump: 0.0,
                        r_prbe: 1.0,
                        modulation: ModulationParams {
                            depth: 0.0,
                            ..ModulationParams::default()
                        },
                        ..HamiltonianParams::default()
                    },
                    decay: Decay {
                        gamma_up: 0.5,
                        gamma_down: 0.5,
                        gamma_phi: 0.5,
                    },
                    solver: MtsSolverParams {
                        kr_n: 1,
                        n_periods: 1,
                        steps_per_period: steps,
                        hz_lim: 0.0,
                        hz_num: 1,
                        ..MtsSolverParams::default()
                    },
                },
                Vec3::from_angles(PI / 2.0, PI / 2.0),
            )
            .unwrap();
            let actual = [
                output.dc[0].in_phase,
                output.harmonic[0].in_phase,
                output.harmonic[0].quadrature,
            ];
            for i in 0..3 {
                let error = (actual[i] - expected[i]).abs();
                assert!(
                    error < previous[i] / 3.0,
                    "component {i}, steps {steps}: {error}"
                );
                previous[i] = error;
            }
        }
        assert!(previous.iter().all(|&error| error < 3e-4));
    }

    #[test]
    fn time_grid_aligns_period_boundaries_and_observation_window() {
        for period in [2.0 * PI, 2.0 * PI / 1e-8] {
            for (periods, steps) in [(1, 1), (1, 100), (3, 100)] {
                let times = time_grid(period, steps, periods);
                let last_start = (periods - 1) * steps;
                let dt = period / steps as f64;
                assert_eq!(times.len(), periods * steps + 1);
                assert_eq!(times.len() - last_start, steps + 1);
                for boundary in 0..=periods {
                    assert!((times[boundary * steps] / period - boundary as f64).abs() < 1e-12);
                }
                for interval in times.windows(2) {
                    assert!(((interval[1] - interval[0]) / dt - 1.0).abs() < 1e-12);
                }
            }
        }
    }
}
