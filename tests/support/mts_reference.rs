use std::f64::consts::PI;

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

#[derive(Clone, Debug)]
pub struct MtsParams {
    pub mod_freq: f64,
    pub mod_depth: f64,
    pub mod_shift: f64,
    pub delta: f64,
    pub r_pump: f64,
    pub r_prbe: f64,
    pub kv: f64,
    pub kr: f64,
    pub frame: Frame,
    pub kr_n: usize,
    pub steps_per_period: usize,
    pub n_periods: usize,
    pub hz_lim: f64,
    pub hz_num: usize,
    pub gamma_up: f64,
    pub gamma_down: f64,
    pub gamma_phi: f64,
}

impl Default for MtsParams {
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
            kr_n: 5,
            steps_per_period: 100,
            n_periods: 3,
            hz_lim: 5.0,
            hz_num: 50,
            gamma_up: 0.0,
            gamma_down: 1.0,
            gamma_phi: 0.0,
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

    let period = 2.0 * PI / params.mod_freq;
    let time_samples = params.n_periods * params.steps_per_period + 1;
    let dt = period / params.steps_per_period as f64;
    let t_array: Vec<_> = (0..time_samples).map(|i| i as f64 * dt).collect();
    let hz_array = linspace(-params.hz_lim, params.hz_lim, params.hz_num);
    let last_start = (params.n_periods - 1) * params.steps_per_period;
    let t_last = &t_array[last_start..];
    let last_period_samples = t_last.len();

    let kr_array = kr_array(params.kr_n, params.kr);
    let kr_harmonic = match params.frame {
        Frame::Probe => 1.0,
        Frame::Atom => 0.0,
        Frame::Pump => -1.0,
    };

    let theta = PI / 2.0;
    let phi = PI / 2.0;
    let rot = [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ];

    let mut amp0 = Vec::with_capacity(hz_array.len());
    let mut proj0 = Vec::with_capacity(hz_array.len());
    let mut amp1 = Vec::with_capacity(hz_array.len());
    let mut proj1 = Vec::with_capacity(hz_array.len());

    for &hz_offset in &hz_array {
        let mut projected = vec![0.0; last_period_samples];

        for &kr_phase in &kr_array {
            accumulate_projected_trajectory(
                params,
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

        let (raw_amp1, phase1) = lockin(&projected, t_last, params.mod_freq);
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
    if !params.mod_freq.is_finite() || params.mod_freq <= 0.0 {
        return Err("mod_freq must be positive and finite".to_string());
    }
    if params.kr_n == 0 {
        return Err("kr_n must be positive".to_string());
    }
    if params.n_periods == 0 {
        return Err("n_periods must be positive".to_string());
    }
    if params.steps_per_period == 0 {
        return Err("steps_per_period must be positive".to_string());
    }
    if params.hz_num == 0 {
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
    params: &MtsParams,
    t_array: &[f64],
    last_start: usize,
    hz_offset: f64,
    kr_phase: f64,
    kr_harmonic: f64,
    rot: [f64; 3],
    projected: &mut [f64],
) {
    let rates = [params.gamma_up, params.gamma_down, params.gamma_phi];
    let mut state = [0.0, 0.0, -1.0];

    for idx in 0..t_array.len() {
        if idx >= last_start {
            projected[idx - last_start] +=
                state[0] * rot[0] + state[1] * rot[1] + state[2] * rot[2];
        }
        if idx + 1 == t_array.len() {
            break;
        }

        let h_eff =
            pump_probe_hamiltonian_sample(params, t_array[idx], hz_offset, kr_phase, kr_harmonic);
        let dt = t_array[idx + 1] - t_array[idx];
        state = propagate(h_eff, rates, state, dt);
    }
}

fn pump_probe_hamiltonian_sample(
    params: &MtsParams,
    t: f64,
    hz_offset: f64,
    kr_phase: f64,
    kr_harmonic: f64,
) -> [f64; 3] {
    let mod_index = params.mod_depth / params.mod_freq;
    let phase = mod_index * (params.mod_freq * t).cos() + params.mod_shift * t;
    let freq = -params.mod_depth * (params.mod_freq * t).sin() + params.mod_shift;
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

    [
        params.r_pump * pu_phase.cos() + params.r_prbe * pr_phase.cos(),
        params.r_pump * pu_phase.sin() + params.r_prbe * pr_phase.sin(),
        hz_base + hz_offset,
    ]
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

fn r_ss(p: [f64; 6]) -> [f64; 3] {
    let [hx, hy, hz, gamma_up, gamma_down, gamma_phi] = p;
    let gamma1 = gamma_up + gamma_down;
    let gamma2 = 0.5 * (gamma_up + gamma_down) + gamma_phi;
    let numerator = gamma_up - gamma_down;
    let denominator = gamma1 * (gamma2 * gamma2 + hz * hz) + gamma2 * (hx * hx + hy * hy);

    let v0 = hx * hz + hy * gamma2;
    let v1 = hy * hz - hx * gamma2;
    let v2 = hz * hz + gamma2 * gamma2;

    let scale = numerator / denominator;
    [scale * v0, scale * v1, scale * v2]
}

fn symmetric_polynomials(p: [f64; 6]) -> (f64, f64, f64) {
    let [hx, hy, hz, gamma_up, gamma_down, gamma_phi] = p;
    let gamma1 = gamma_up + gamma_down;
    let gamma2 = 0.5 * (gamma_up + gamma_down) + gamma_phi;
    let h2 = hx * hx + hy * hy + hz * hz;
    let a1 = gamma1 + 2.0 * gamma2;
    let a2 = 2.0 * gamma1 * gamma2 + gamma2 * gamma2 + h2;
    let a3 = hz * hz * gamma1 + gamma2 * (hx * hx + hy * hy + gamma1 * gamma2);
    (a1, a2, a3)
}

fn propagate(h_eff: [f64; 3], rates: [f64; 3], r0: [f64; 3], t: f64) -> [f64; 3] {
    let p = [h_eff[0], h_eff[1], h_eff[2], rates[0], rates[1], rates[2]];
    let rss = r_ss(p);
    let w = [r0[0] - rss[0], r0[1] - rss[1], r0[2] - rss[2]];

    let [hx, hy, hz, gamma_up, gamma_down, gamma_phi] = p;
    let gamma1 = gamma_up + gamma_down;
    let gamma2 = 0.5 * (gamma_up + gamma_down) + gamma_phi;

    let sym = symmetric_polynomials(p);
    let eigs = cubic_roots(sym);
    let (c0, c1, c2) = cayley_coeffs(sym, eigs, t);

    let h2 = hx * hx + hy * hy + hz * hz;
    let d = [-gamma2, -gamma2, -gamma1];

    let w_circ = [d[0] * w[0], d[1] * w[1], d[2] * w[2]];
    let w_cross = [
        hy * w[2] - hz * w[1],
        hz * w[0] - hx * w[2],
        hx * w[1] - hy * w[0],
    ];
    let m_w = [
        w_circ[0] + w_cross[0],
        w_circ[1] + w_cross[1],
        w_circ[2] + w_cross[2],
    ];

    let h_dot_w = hx * w[0] + hy * w[1] + hz * w[2];
    let h_cross_w_circ = [
        hy * w_circ[2] - hz * w_circ[1],
        hz * w_circ[0] - hx * w_circ[2],
        hx * w_circ[1] - hy * w_circ[0],
    ];
    let d_w_circ = [d[0] * w_circ[0], d[1] * w_circ[1], d[2] * w_circ[2]];
    let d_w_cross = [d[0] * w_cross[0], d[1] * w_cross[1], d[2] * w_cross[2]];

    let m2_w = [
        d_w_circ[0] + d_w_cross[0] + h_cross_w_circ[0] + h_dot_w * hx - h2 * w[0],
        d_w_circ[1] + d_w_cross[1] + h_cross_w_circ[1] + h_dot_w * hy - h2 * w[1],
        d_w_circ[2] + d_w_cross[2] + h_cross_w_circ[2] + h_dot_w * hz - h2 * w[2],
    ];

    [
        c0 * w[0] + c1 * m_w[0] + c2 * m2_w[0] + rss[0],
        c0 * w[1] + c1 * m_w[1] + c2 * m2_w[1] + rss[1],
        c0 * w[2] + c1 * m_w[2] + c2 * m2_w[2] + rss[2],
    ]
}

fn cubic_roots((a1, a2, a3): (f64, f64, f64)) -> [Complex; 3] {
    let p = a2 - (a1 * a1) / 3.0;
    let q = a3 - a1 * a2 / 3.0 + 2.0 * (a1 * a1 * a1) / 27.0;
    let d = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);

    let sqrt_d = Complex::new(d, 0.0).sqrt();
    let u = cbrt_cardano(Complex::new(-0.5 * q, 0.0) + sqrt_d);
    let v = cbrt_cardano(Complex::new(-0.5 * q, 0.0) - sqrt_d);
    let u_plus_v = u + v;
    let u_minus_v = u - v;
    let sqrt3 = 3.0_f64.sqrt();

    let x1 = u_plus_v;
    let x2 = u_plus_v * -0.5 + u_minus_v * Complex::new(0.0, 0.5 * sqrt3);
    let x3 = u_plus_v * -0.5 - u_minus_v * Complex::new(0.0, 0.5 * sqrt3);
    let shift = Complex::new(a1 / 3.0, 0.0);

    [x1 - shift, x2 - shift, x3 - shift]
}

fn cayley_coeffs((a1, a2, _a3): (f64, f64, f64), eigvals: [Complex; 3], t: f64) -> (f64, f64, f64) {
    let mut c0 = Complex::new(0.0, 0.0);
    let mut c1 = Complex::new(0.0, 0.0);
    let mut c2 = Complex::new(0.0, 0.0);

    for lambda in eigvals {
        let xj = Complex::new(a1, 0.0) + lambda;
        let dj = lambda * lambda * 3.0 + lambda * (2.0 * a1) + Complex::new(a2, 0.0);
        let exp_term = (lambda * t).exp();
        c0 = c0 + ((Complex::new(a2, 0.0) + xj * lambda) / dj) * exp_term;
        c1 = c1 + (xj / dj) * exp_term;
        c2 = c2 + (Complex::new(1.0, 0.0) / dj) * exp_term;
    }

    (c0.re, c1.re, c2.re)
}

fn cbrt_cardano(value: Complex) -> Complex {
    if value.im == 0.0 {
        if value.re >= 0.0 {
            Complex::new(value.re.powf(1.0 / 3.0), 0.0)
        } else {
            Complex::new(-(-value.re).powf(1.0 / 3.0), 0.0)
        }
    } else {
        value.powf(1.0 / 3.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn sqrt(self) -> Self {
        if self.im == 0.0 {
            if self.re >= 0.0 {
                return Self::new(self.re.sqrt(), 0.0);
            }
            return Self::new(0.0, (-self.re).sqrt());
        }

        let r = self.re.hypot(self.im);
        let re = ((r + self.re) / 2.0).sqrt();
        let im = self.im.signum() * ((r - self.re) / 2.0).sqrt();
        Self::new(re, im)
    }

    fn exp(self) -> Self {
        let scale = self.re.exp();
        Self::new(scale * self.im.cos(), scale * self.im.sin())
    }

    fn powf(self, exponent: f64) -> Self {
        let radius = self.re.hypot(self.im);
        let angle = self.im.atan2(self.re);
        let scaled_radius = radius.powf(exponent);
        let scaled_angle = angle * exponent;
        Self::new(
            scaled_radius * scaled_angle.cos(),
            scaled_radius * scaled_angle.sin(),
        )
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl std::ops::Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_invalid_modulation_frequency() {
        use super::*;

        for mod_freq in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let params = MtsParams {
                mod_freq,
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
        use super::*;

        for frame in [Frame::Pump, Frame::Atom, Frame::Probe] {
            let params = MtsParams {
                frame,
                ..MtsParams::default()
            };
            let output = compute_demod(&params).expect("default scan succeeds");
            for values in [&output.hz, &output.amp0, &output.amp1, &output.proj1] {
                assert_eq!(values.len(), params.hz_num);
                assert!(values.iter().all(|value| value.is_finite()), "{frame:?}");
            }
        }
    }
}
