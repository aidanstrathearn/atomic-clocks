use std::f64::consts::PI;

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
    pub mod_freq: f64,
    pub mod_depth: f64,
    pub mod_shift: f64,
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
            mod_freq: 1.0,
            mod_depth: 1.0,
            mod_shift: 0.0,
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
    pub hz: Vec<f64>,
    pub amp0: Vec<f64>,
    pub proj0: Vec<f64>,
    pub amp1: Vec<f64>,
    pub proj1: Vec<f64>,
    pub time_samples: usize,
    pub last_period_samples: usize,
}

pub fn compute_demod(params: &MtsParams) -> Result<DemodOutput, String> {
    validate_params(params)?;

    let period = 2.0 * PI / params.hamiltonian.mod_freq.max(1e-6);
    let time_samples = params.solver.n_periods * params.solver.steps_per_period;
    let t_array = linspace(0.0, params.solver.n_periods as f64 * period, time_samples);
    let hz_array = linspace(
        -params.solver.hz_lim,
        params.solver.hz_lim,
        params.solver.hz_num,
    );
    let last_start = last_period_start(&t_array, period);
    let t_last = &t_array[last_start..];
    let last_period_samples = t_last.len();

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

    let mut amp0 = Vec::with_capacity(hz_array.len());
    let mut proj0 = Vec::with_capacity(hz_array.len());
    let mut amp1 = Vec::with_capacity(hz_array.len());
    let mut proj1 = Vec::with_capacity(hz_array.len());

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

        let (raw_amp0, _) = lockin(&projected, t_last, 0.0);
        let out_amp0 = raw_amp0 / 2.0;
        amp0.push(out_amp0);
        proj0.push(out_amp0);

        let (raw_amp1, phase1) = lockin(&projected, t_last, params.hamiltonian.mod_freq);
        let hz_sign = -np_sign(hz_offset);
        let out_amp1 = raw_amp1 * hz_sign;
        let out_cos1 = phase1.cos() * hz_sign;
        amp1.push(out_amp1);
        proj1.push(out_amp1 * out_cos1);
    }

    Ok(DemodOutput {
        hz: hz_array,
        amp0,
        proj0,
        amp1,
        proj1,
        time_samples,
        last_period_samples,
    })
}

fn validate_params(params: &MtsParams) -> Result<(), String> {
    if !params.hamiltonian.mod_freq.is_finite() || params.hamiltonian.mod_freq <= 0.0 {
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
    if params.solver.n_periods * params.solver.steps_per_period < 2 {
        return Err("time grid must contain at least two samples".to_string());
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

fn last_period_start(t_array: &[f64], period: f64) -> usize {
    let last = t_array[t_array.len() - 1];
    let dt = t_array[t_array.len() - 1] - t_array[t_array.len() - 2];
    let threshold = last - period - dt;
    t_array
        .iter()
        .position(|&value| value > threshold)
        .unwrap_or(t_array.len() - 1)
}

fn np_sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
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
    let mod_index = params.mod_depth / params.mod_freq;
    let phase = mod_index * (params.mod_freq * t).cos() + params.mod_shift * t;
    let freq = params.mod_depth * (params.mod_freq * t).sin() + params.mod_shift;
    let kvt = t * params.kv;

    let (hz_base, pump_phase, probe_phase) = match params.frame {
        Frame::Atom => (params.delta, kvt + phase, -kvt),
        Frame::Pump => (params.delta - params.kv + freq, 0.0, -2.0 * kvt - phase),
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

fn lockin(traj: &[f64], t_array: &[f64], freq: f64) -> (f64, f64) {
    let t_total = t_array[t_array.len() - 1] - t_array[0];
    let mut x_integral = 0.0;
    let mut y_integral = 0.0;

    for idx in 0..(traj.len() - 1) {
        let t0 = t_array[idx];
        let t1 = t_array[idx + 1];
        let dt = t1 - t0;

        let x0 = traj[idx] * (freq * t0).cos();
        let x1 = traj[idx + 1] * (freq * t1).cos();
        x_integral += 0.5 * (x0 + x1) * dt;

        let y0 = traj[idx] * (freq * t0).sin();
        let y1 = traj[idx + 1] * (freq * t1).sin();
        y_integral += 0.5 * (y0 + y1) * dt;
    }

    let x_val = x_integral / t_total;
    let y_val = y_integral / t_total;
    let amplitude = 2.0 * (x_val * x_val + y_val * y_val).sqrt();
    let phase = y_val.atan2(x_val);
    (amplitude, phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct Fixture {
        params: MtsParams,
        arrays: HashMap<String, Vec<f64>>,
    }

    #[test]
    fn baseline_fixture_matches_python() {
        assert_fixture(include_str!("fixtures/baseline.fixture"));
    }

    #[test]
    fn pump_shifted_fixture_matches_python() {
        assert_fixture(include_str!("fixtures/pump_shifted.fixture"));
    }

    #[test]
    fn rejects_invalid_modulation_frequency() {
        for mod_freq in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let params = MtsParams {
                hamiltonian: HamiltonianParams {
                    mod_freq,
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
            for values in [&output.hz, &output.amp0, &output.amp1, &output.proj1] {
                assert_eq!(values.len(), params.solver.hz_num);
                assert!(values.iter().all(|value| value.is_finite()), "{frame:?}");
            }
        }
    }

    fn assert_fixture(text: &str) {
        let fixture = parse_fixture(text);
        let output = compute_demod(&fixture.params).expect("compute_demod succeeds");
        assert_close("hz", &output.hz, fixture.array("hz"), 2e-8, 2e-10);
        assert_close("amp0", &output.amp0, fixture.array("amp0"), 2e-8, 2e-10);
        assert_close("proj0", &output.proj0, fixture.array("proj0"), 2e-8, 2e-10);
        assert_close("amp1", &output.amp1, fixture.array("amp1"), 2e-8, 2e-10);
        assert_close("proj1", &output.proj1, fixture.array("proj1"), 2e-8, 2e-10);
    }

    impl Fixture {
        fn array(&self, name: &str) -> &[f64] {
            self.arrays
                .get(name)
                .unwrap_or_else(|| panic!("missing fixture array '{name}'"))
        }
    }

    fn parse_fixture(text: &str) -> Fixture {
        let mut params = MtsParams::default();
        let mut arrays = HashMap::new();

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<_> = line.split_whitespace().collect();
            match parts.as_slice() {
                ["case", _name] => {}
                ["param", name, value] => set_param(&mut params, name, value),
                ["array", name, len, values @ ..] => {
                    let expected_len = len
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("invalid array length '{len}'"));
                    let parsed = values
                        .iter()
                        .map(|value| {
                            value
                                .parse::<f64>()
                                .unwrap_or_else(|_| panic!("invalid float '{value}'"))
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(parsed.len(), expected_len, "array '{name}' length mismatch");
                    arrays.insert((*name).to_string(), parsed);
                }
                _ => panic!("invalid fixture line: {line}"),
            }
        }

        Fixture { params, arrays }
    }

    fn set_param(params: &mut MtsParams, name: &str, value: &str) {
        match name {
            "mod_freq" => params.hamiltonian.mod_freq = parse_f64(value),
            "mod_depth" => params.hamiltonian.mod_depth = parse_f64(value),
            "mod_shift" => params.hamiltonian.mod_shift = parse_f64(value),
            "delta" => params.hamiltonian.delta = parse_f64(value),
            "r_pump" => params.hamiltonian.r_pump = parse_f64(value),
            "r_prbe" => params.hamiltonian.r_prbe = parse_f64(value),
            "kv" => params.hamiltonian.kv = parse_f64(value),
            "kr" => params.hamiltonian.kr = parse_f64(value),
            "frame" => params.hamiltonian.frame = Frame::parse(value).expect("valid frame"),
            "kr_n" => params.solver.kr_n = parse_usize(value),
            "steps_per_period" => params.solver.steps_per_period = parse_usize(value),
            "n_periods" => params.solver.n_periods = parse_usize(value),
            "hz_lim" => params.solver.hz_lim = parse_f64(value),
            "hz_num" => params.solver.hz_num = parse_usize(value),
            "gamma_up" => params.decay.gamma_up = parse_f64(value),
            "gamma_down" => params.decay.gamma_down = parse_f64(value),
            "gamma_phi" => params.decay.gamma_phi = parse_f64(value),
            _ => panic!("unknown fixture param '{name}'"),
        }
    }

    fn parse_f64(value: &str) -> f64 {
        value
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("invalid float param '{value}'"))
    }

    fn parse_usize(value: &str) -> usize {
        value
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("invalid usize param '{value}'"))
    }

    fn assert_close(name: &str, actual: &[f64], expected: &[f64], rtol: f64, atol: f64) {
        assert_eq!(actual.len(), expected.len(), "{name} length mismatch");

        let mut worst_abs = 0.0;
        let mut worst_rel = 0.0;
        let mut worst_idx = 0;
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (a - e).abs();
            let rel = if e == 0.0 { abs } else { abs / e.abs() };
            if abs > worst_abs {
                worst_abs = abs;
                worst_rel = rel;
                worst_idx = idx;
            }
            assert!(
                abs <= atol + rtol * e.abs(),
                "{name}[{idx}] mismatch: actual={a:.17e}, expected={e:.17e}, abs={abs:.3e}, rel={rel:.3e}"
            );
        }

        eprintln!(
            "{name}: worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}, worst_idx={worst_idx}"
        );
    }
}
