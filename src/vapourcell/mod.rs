use std::f64::consts::PI;

use crate::maths::demodulation::{Demodulation, ModulationParams, lockin_period};
use crate::twolevel::{BlochVec, Decay, Hamiltonian, Liouvillian, Vec3, propagate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Frame {
    Atom,
    Pump,
    Probe,
}

impl Frame {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "atom" => Ok(Self::Atom),
            "pump" => Ok(Self::Pump),
            "probe" => Ok(Self::Probe),
            _ => Err(format!("unknown frame '{value}'")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Atom => "atom",
            Self::Pump => "pump",
            Self::Probe => "probe",
        }
    }
}

/// Hamiltonian parameters for a single atom in the chosen rotating frame.
#[derive(Clone, Copy, Debug)]
pub struct HamiltonianParams {
    pub modulation: ModulationParams,
    pub delta: f64,
    pub r_pump: f64,
    pub r_prbe: f64,
    pub kv: f64,
    pub kr: f64,
    pub frame: Frame,
}

impl Default for HamiltonianParams {
    fn default() -> Self {
        Self {
            modulation: ModulationParams::default(),
            delta: 0.0,
            r_pump: 1.0,
            r_prbe: 0.1,
            kv: 0.0,
            kr: 0.0,
            frame: Frame::Probe,
        }
    }
}

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

#[derive(Clone, Debug)]
pub struct DemodOutput {
    /// Detuning offsets relative to the Hamiltonian's `delta`.
    pub hz: Vec<f64>,
    /// Zero-frequency lock-in results: twice the signed mean in `in_phase`,
    /// with zero `quadrature`.
    pub dc: Vec<Demodulation>,
    /// Signed cosine and sine coefficients at the modulation frequency.
    pub harmonic: Vec<Demodulation>,
}

pub fn compute_demod(params: &MtsParams) -> Result<DemodOutput, String> {
    validate_params(params)?;

    let t_array = time_grid(
        params.hamiltonian.modulation.period(),
        params.solver.steps_per_period,
        params.solver.n_periods,
    );
    let hz_array = linspace(
        -params.solver.hz_lim,
        params.solver.hz_lim,
        params.solver.hz_num,
    );
    let last_start = (params.solver.n_periods - 1) * params.solver.steps_per_period;
    let last_period_samples = params.solver.steps_per_period + 1;

    let kr_array = kr_array(params.solver.kr_n, params.hamiltonian.kr);
    let kr_harmonic = match params.hamiltonian.frame {
        Frame::Probe => 1.0,
        Frame::Atom => 0.0,
        Frame::Pump => -1.0,
    };

    let theta = PI / 2.0;
    let phi = PI / 2.0;
    let rot = Vec3 {
        x: theta.sin() * phi.cos(),
        y: theta.sin() * phi.sin(),
        z: theta.cos(),
    };

    let mut dc = Vec::with_capacity(hz_array.len());
    let mut harmonic = Vec::with_capacity(hz_array.len());

    for &hz_offset in &hz_array {
        let mut projected = vec![0.0; last_period_samples];

        for &kr_phase in &kr_array {
            accumulate_projected_trajectory(
                &params.hamiltonian,
                params.decay,
                &t_array,
                last_start,
                hz_offset,
                kr_phase,
                kr_harmonic,
                rot,
                &mut projected,
            );
        }

        let kr_scale = 1.0 / kr_array.len() as f64;
        for value in &mut projected {
            *value *= kr_scale;
        }

        dc.push(lockin_period(&projected, 0));
        harmonic.push(lockin_period(&projected, 1));
    }

    Ok(DemodOutput {
        hz: hz_array,
        dc,
        harmonic,
    })
}

fn validate_params(params: &MtsParams) -> Result<(), String> {
    if !params.hamiltonian.modulation.frequency.is_finite()
        || params.hamiltonian.modulation.frequency <= 0.0
    {
        return Err("mod_freq must be positive and finite".to_string());
    }
    if params.solver.kr_n == 0 {
        return Err("kr_n must be positive".to_string());
    }
    if params.solver.n_periods == 0 {
        return Err("n_periods must be positive".to_string());
    }
    if params.solver.steps_per_period == 0 {
        return Err("steps_per_period must be positive".to_string());
    }
    if params.solver.hz_num == 0 {
        return Err("hz_num must be positive".to_string());
    }
    Ok(())
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![start];
    }
    let step = (end - start) / (count - 1) as f64;
    (0..count).map(|idx| start + step * idx as f64).collect()
}

fn kr_array(kr_n: usize, kr_offset: f64) -> Vec<f64> {
    (1..=kr_n)
        .map(|idx| 2.0 * PI * idx as f64 / kr_n as f64 + kr_offset)
        .collect()
}

fn time_grid(period: f64, steps_per_period: usize, n_periods: usize) -> Vec<f64> {
    let dt = period / steps_per_period as f64;
    (0..=n_periods * steps_per_period)
        .map(|i| i as f64 * dt)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn accumulate_projected_trajectory(
    params: &HamiltonianParams,
    decay: Decay,
    t_array: &[f64],
    last_start: usize,
    hz_offset: f64,
    kr_phase: f64,
    kr_harmonic: f64,
    rot: Vec3,
    projected: &mut [f64],
) {
    let mut state = BlochVec::ground();

    for idx in 0..t_array.len() {
        if idx >= last_start {
            projected[idx - last_start] += state.r.dot(rot);
        }
        if idx + 1 == t_array.len() {
            break;
        }

        let hamiltonian =
            pump_probe_hamiltonian_sample(params, t_array[idx], hz_offset, kr_phase, kr_harmonic);
        let liouvillian = Liouvillian { hamiltonian, decay };
        let dt = t_array[idx + 1] - t_array[idx];
        state = propagate(liouvillian, state, dt);
    }
}

fn pump_probe_hamiltonian_sample(
    params: &HamiltonianParams,
    t: f64,
    hz_offset: f64,
    kr_phase: f64,
    kr_harmonic: f64,
) -> Hamiltonian {
    let phase = params.modulation.phase(t);
    let freq = params.modulation.freq(t);
    let kvt = t * params.kv;

    let (hz_base, pump_phase, probe_phase) = match params.frame {
        Frame::Atom => (params.delta, kvt + phase, -kvt),
        Frame::Pump => (params.delta - params.kv - freq, 0.0, -2.0 * kvt - phase),
        Frame::Probe => (params.delta + params.kv, 2.0 * kvt + phase, 0.0),
    };

    let pump_kr = (kr_harmonic + 1.0) * kr_phase;
    let probe_kr = (kr_harmonic - 1.0) * kr_phase;
    let pu_phase = pump_phase + pump_kr;
    let pr_phase = probe_phase + probe_kr;

    Hamiltonian::new(
        params.r_pump * pu_phase.cos() + params.r_prbe * pr_phase.cos(),
        params.r_pump * pu_phase.sin() + params.r_prbe * pr_phase.sin(),
        hz_base + hz_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
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
                compute_demod(&params).unwrap_err(),
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
            let output = compute_demod(&params).expect("default scan succeeds");
            assert_eq!(output.hz.len(), params.solver.hz_num);
            assert!(output.hz.iter().all(|value| value.is_finite()));
            for values in [&output.dc, &output.harmonic] {
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
    fn raw_central_coefficients_match_a_damped_rabi_trajectory() {
        // Isotropic unit-rate relaxation and a constant x drive give
        // r_y(t) = drive * exp(-t) * sin(t) for drive = +/-1.
        for (drive, periods) in [(-1.0, 1), (1.0, 1), (1.0, 2)] {
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
            let output = compute_demod(&params).unwrap();
            assert_eq!(output.hz[1], 0.0);
            let mut expected = [0.0; 3];
            for i in 0..=8 {
                let t = (periods - 1) as f64 * 2.0 * PI + i as f64 * PI / 4.0;
                let signal = drive * (-t).exp() * t.sin();
                // Trapezoidal weight, including the lock-in factor 2 / duration.
                let weight = if i == 0 || i == 8 { 0.125 } else { 0.25 };
                expected[0] += weight * signal;
                expected[1] += weight * signal * t.cos();
                expected[2] += weight * signal * t.sin();
            }
            for (actual, expected) in [
                (output.dc[1].in_phase, expected[0]),
                (output.harmonic[1].in_phase, expected[1]),
                (output.harmonic[1].quadrature, expected[2]),
            ] {
                assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
                assert!(actual * drive > 1e-5);
            }
            assert_eq!(output.dc[1].quadrature, 0.0);
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
            let output = compute_demod(&MtsParams {
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
            })
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
