use std::f64::consts::PI;

use crate::common::DetuningScanParams;
use crate::maths::demodulation::{Demodulation, lockin_period};
use crate::maths::linspace;
use crate::twolevel::{BlochVec, Decay, Process, Vec3, steps};

use super::hamiltonian::{DrivenAtomParams, HamiltonianParams};

/// Spatial sampling and integration grid settings for MTS.
#[derive(Clone, Copy, Debug)]
pub struct MtsSolverParams {
    pub kr_n: usize,
    /// Integration intervals per modulation period; the final window has one more sample.
    pub steps_per_period: usize,
    pub n_periods: usize,
}

impl Default for MtsSolverParams {
    fn default() -> Self {
        Self {
            kr_n: 5,
            steps_per_period: 100,
            n_periods: 3,
        }
    }
}

impl MtsSolverParams {
    /// Validates spatial sampling and integration settings.
    pub fn validate(&self) -> Result<(), String> {
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
}

#[derive(Clone, Copy, Debug)]
pub struct MtsParams {
    pub atom: DrivenAtomParams,
    pub solver: MtsSolverParams,
    pub scan: DetuningScanParams,
}

impl Default for MtsParams {
    fn default() -> Self {
        Self {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams::default(),
                decay: Decay {
                    gamma_up: 0.0,
                    gamma_down: 1.0,
                    gamma_phi: 0.0,
                },
            },
            solver: MtsSolverParams::default(),
            scan: DetuningScanParams::default(),
        }
    }
}

impl MtsParams {
    /// Validates the Hamiltonian, decay, solver, and detuning scan configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_propagation()?;
        self.scan.validate()
    }

    fn validate_propagation(&self) -> Result<(), String> {
        self.atom.validate()?;
        self.solver.validate()
    }
}

/// The raw spatially averaged observable over the final modulation period.
#[derive(Clone, Debug)]
pub struct RawSignalOutput {
    /// Times relative to the start of the final observation window.
    pub times: Vec<f64>,
    /// Expectations of `observable · sigma` at each time.
    pub values: Vec<f64>,
}

/// Raw demodulated observable values for a detuning scan.
#[derive(Clone, Debug)]
pub struct DemodOutput<const N: usize> {
    /// Detuning offsets relative to the Hamiltonian's `delta`.
    pub hz: Vec<f64>,
    /// Harmonic indices corresponding to each position in `values`.
    pub harmonics: [usize; N],
    /// One array of lock-in results per detuning offset, indexed like `harmonics`.
    pub values: Vec<[Demodulation; N]>,
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
                params.atom.hamiltonian.modulation.period(),
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
            let atom = DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    delta: self.params.atom.hamiltonian.delta + hz_offset,
                    kr: kr_phase,
                    ..self.params.atom.hamiltonian
                },
                ..self.params.atom
            };
            accumulate_projected_trajectory(
                &atom,
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

    fn raw_signal(&self, hz_offset: f64) -> Result<RawSignalOutput, String> {
        if !hz_offset.is_finite() {
            return Err("detuning offset must be finite".to_string());
        }
        let period = self.params.atom.hamiltonian.modulation.period();
        let steps = self.params.solver.steps_per_period;
        let times = (0..=steps)
            .map(|i| i as f64 * period / steps as f64)
            .collect();
        Ok(RawSignalOutput {
            times,
            values: self.projected_at_offset(hz_offset),
        })
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

/// Computes the raw expectation of `observable · sigma` over the final period.
///
/// The result uses the same warmup, spatial-phase average, and observation
/// window as [`compute_demod`]. `hz_offset` is added to the Hamiltonian's
/// `delta`. Returned times run from zero to one modulation period relative to
/// the start of that final window. The detuning scan settings are ignored.
pub fn compute_raw_signal(
    params: &MtsParams,
    observable: Vec3,
    hz_offset: f64,
) -> Result<RawSignalOutput, String> {
    Demodulator::new(params, observable)?.raw_signal(hz_offset)
}

/// Demodulates raw observable harmonics at the Hamiltonian's exact detuning.
///
/// Unlike [`compute_demod`], this evaluates only `hamiltonian.delta` rather than
/// building a detuning scan. Harmonics are returned in the same order as the
/// supplied indices. The observable and lock-in conventions match
/// [`compute_demod`]. The detuning scan settings are ignored.
pub fn compute_demod_harmonics(
    params: &MtsParams,
    observable: Vec3,
    harmonics: &[usize],
) -> Result<Vec<Demodulation>, String> {
    Demodulator::new(params, observable)?.demodulate_harmonics(0.0, harmonics)
}

/// Demodulates the raw expectation of `observable · sigma` in the selected frame.
/// The observable is constant in that frame and is not normalized; its magnitude
/// scales the measured signal. Harmonics are returned in the same order as the
/// supplied indices.
pub fn compute_demod<const N: usize>(
    params: &MtsParams,
    observable: Vec3,
    harmonics: [usize; N],
) -> Result<DemodOutput<N>, String> {
    params.validate_propagation()?;
    let hz_array = params.scan.angular_offsets()?;
    let demodulator = Demodulator::from_validated(params, observable);
    let values = hz_array
        .iter()
        .map(|&hz_offset| demodulator.demodulate(hz_offset, harmonics))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(DemodOutput {
        hz: hz_array,
        harmonics,
        values,
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
        .map(|phase| phase + params.atom.hamiltonian.kr)
        .collect()
}

fn time_grid(period: f64, steps_per_period: usize, n_periods: usize) -> Vec<f64> {
    let dt = period / steps_per_period as f64;
    (0..=n_periods * steps_per_period)
        .map(|i| i as f64 * dt)
        .collect()
}

fn accumulate_projected_trajectory(
    params: &DrivenAtomParams,
    t_array: &[f64],
    last_start: usize,
    observable: Vec3,
    projected: &mut [f64],
) {
    let step_at = |t, dt| params.liouvillian_at(t).for_duration(dt);
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

    const SIGNAL_HARMONICS: [usize; 4] = [0, 1, 2, 3];

    #[test]
    fn rejects_invalid_modulation_frequency() {
        for mod_freq in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let params = MtsParams {
                atom: DrivenAtomParams {
                    hamiltonian: HamiltonianParams {
                        modulation: ModulationParams {
                            frequency: mod_freq,
                            ..ModulationParams::default()
                        },
                        ..HamiltonianParams::default()
                    },
                    ..MtsParams::default().atom
                },
                ..MtsParams::default()
            };
            assert_eq!(
                compute_demod(
                    &params,
                    Vec3::from_angles(PI / 2.0, PI / 2.0),
                    SIGNAL_HARMONICS,
                )
                .unwrap_err(),
                "mod_freq must be positive and finite"
            );
        }
    }

    #[test]
    fn default_scan_is_finite_in_every_frame() {
        for frame in [Frame::Pump, Frame::Atom, Frame::Probe] {
            let params = MtsParams {
                atom: DrivenAtomParams {
                    hamiltonian: HamiltonianParams {
                        frame,
                        ..HamiltonianParams::default()
                    },
                    ..MtsParams::default().atom
                },
                ..MtsParams::default()
            };
            let output = compute_demod(
                &params,
                Vec3::from_angles(PI / 2.0, PI / 2.0),
                SIGNAL_HARMONICS,
            )
            .expect("default scan succeeds");
            assert_eq!(output.hz.len(), params.scan.hz_num);
            assert_eq!(output.harmonics, SIGNAL_HARMONICS);
            assert!(output.hz.iter().all(|value| value.is_finite()));
            assert_eq!(output.values.len(), params.scan.hz_num);
            assert!(
                output
                    .values
                    .iter()
                    .flatten()
                    .all(|value| value.in_phase.is_finite() && value.quadrature.is_finite()),
                "{frame:?}"
            );
        }
    }

    #[test]
    fn selected_harmonics_match_the_existing_demodulation_output() {
        let params = MtsParams {
            solver: MtsSolverParams {
                kr_n: 3,
                steps_per_period: 40,
                n_periods: 2,
            },
            scan: DetuningScanParams {
                hz_lim: 0.0,
                hz_num: 1,
            },
            ..MtsParams::default()
        };
        let observable = Vec3::from_angles(PI / 2.0, PI / 2.0);
        let output = compute_demod(&params, observable, SIGNAL_HARMONICS).unwrap();
        let harmonics = [3, 1, 0, 2, 4];
        let selected = compute_demod_harmonics(&params, observable, &harmonics).unwrap();
        let fixed = compute_demod(&params, observable, harmonics).unwrap();
        let [dc, first, second, third] = output.values[0];
        let expected = [third, first, dc, second];

        for (actual, expected) in selected.iter().zip(expected) {
            assert_eq!(actual.in_phase, expected.in_phase);
            assert_eq!(actual.quadrature, expected.quadrature);
        }
        assert_eq!(fixed.harmonics, harmonics);
        for (actual, expected) in fixed.values[0].iter().zip(&selected) {
            assert_eq!(actual.in_phase, expected.in_phase);
            assert_eq!(actual.quadrature, expected.quadrature);
        }
        assert!(selected[4].in_phase.is_finite());
        assert!(selected[4].quadrature.is_finite());
    }

    #[test]
    fn raw_signal_reproduces_exact_detuning_harmonics() {
        let params = MtsParams {
            solver: MtsSolverParams {
                kr_n: 3,
                steps_per_period: 40,
                n_periods: 2,
            },
            scan: DetuningScanParams {
                hz_lim: f64::NAN,
                hz_num: 0,
            },
            ..MtsParams::default()
        };
        let observable = Vec3::from_angles(PI / 2.0, PI / 2.0);
        let harmonics = [0, 1, 3, 7];
        let raw = compute_raw_signal(&params, observable, 0.0).unwrap();
        let demodulated = compute_demod_harmonics(&params, observable, &harmonics).unwrap();

        assert_eq!(raw.times.len(), params.solver.steps_per_period + 1);
        assert_eq!(raw.values.len(), raw.times.len());
        assert_eq!(raw.times[0], 0.0);
        assert_eq!(
            raw.times[params.solver.steps_per_period],
            params.atom.hamiltonian.modulation.period()
        );
        for (&harmonic, expected) in harmonics.iter().zip(demodulated) {
            let actual = lockin_period(&raw.values, harmonic);
            assert_eq!(actual.in_phase, expected.in_phase);
            assert_eq!(actual.quadrature, expected.quadrature);
        }
    }

    #[test]
    fn raw_signal_rejects_a_non_finite_detuning_offset() {
        let error = compute_raw_signal(
            &MtsParams::default(),
            Vec3::from_angles(PI / 2.0, PI / 2.0),
            f64::NAN,
        )
        .unwrap_err();
        assert_eq!(error, "detuning offset must be finite");
    }

    #[test]
    fn exact_demodulation_does_not_require_scan_settings() {
        let params = MtsParams {
            scan: DetuningScanParams {
                hz_lim: f64::NAN,
                hz_num: 0,
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
        assert!(compute_demod(&params, observable, SIGNAL_HARMONICS).is_err());
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
                    atom: DrivenAtomParams {
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
                    },
                    solver: MtsSolverParams {
                        kr_n: 1,
                        n_periods: periods,
                        steps_per_period: 8,
                    },
                    scan: DetuningScanParams {
                        hz_num: 3,
                        ..DetuningScanParams::default()
                    },
                };
                let output = compute_demod(&params, observable, SIGNAL_HARMONICS).unwrap();
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
                let [dc, first, second, _third] = output.values[1];
                for (actual, expected) in [
                    (dc.in_phase, expected[0]),
                    (first.in_phase, expected[1]),
                    (first.quadrature, expected[2]),
                    (second.in_phase, expected[3]),
                    (second.quadrature, expected[4]),
                ] {
                    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
                }
                assert_eq!(dc.quadrature, 0.0);
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
                    atom: DrivenAtomParams {
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
                    },
                    solver: MtsSolverParams {
                        kr_n: 1,
                        n_periods: 1,
                        steps_per_period: steps,
                    },
                    scan: DetuningScanParams {
                        hz_lim: 0.0,
                        hz_num: 1,
                    },
                },
                Vec3::from_angles(PI / 2.0, PI / 2.0),
                SIGNAL_HARMONICS,
            )
            .unwrap();
            let [dc, first, _second, _third] = output.values[0];
            let actual = [dc.in_phase, first.in_phase, first.quadrature];
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
