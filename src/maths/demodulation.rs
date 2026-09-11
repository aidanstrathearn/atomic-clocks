/// Sinusoidal phase modulation with a constant frequency shift.
/// `frequency` must be positive for the modulation index and phase to be defined.
#[derive(Clone, Copy, Debug)]
pub struct ModulationParams {
    /// Modulation angular frequency, in radians per unit time.
    pub frequency: f64,
    /// Angular-frequency deviation amplitude, in radians per unit time.
    pub depth: f64,
    /// Constant angular-frequency shift, in radians per unit time.
    pub shift: f64,
}

impl Default for ModulationParams {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            depth: 1.0,
            shift: 0.0,
        }
    }
}

impl ModulationParams {
    /// Dimensionless phase-modulation amplitude.
    pub fn mod_index(&self) -> f64 {
        self.depth / self.frequency
    }

    /// Modulation phase in radians, including the linear phase from `shift`.
    pub fn phase(&self, t: f64) -> f64 {
        self.mod_index() * (self.frequency * t).cos() + self.shift * t
    }

    /// Instantaneous angular-frequency offset: the time derivative of `phase(t)`.
    pub fn freq(&self, t: f64) -> f64 {
        -self.depth * (self.frequency * t).sin() + self.shift
    }
}

/// Cosine and sine coefficients relative to the demodulation reference.
#[derive(Clone, Copy, Debug)]
pub struct Demodulation {
    /// Cosine coefficient.
    pub in_phase: f64,
    /// Sine coefficient.
    pub quadrature: f64,
}

impl Demodulation {
    pub fn amplitude(&self) -> f64 {
        self.in_phase.hypot(self.quadrature)
    }

    /// Phase lag in radians for `amplitude * cos(reference_phase - phase)`.
    pub fn phase(&self) -> f64 {
        self.quadrature.atan2(self.in_phase)
    }

    /// Projects onto `cos(reference_phase - phase_offset)`.
    pub fn at_phase(&self, phase_offset: f64) -> f64 {
        self.in_phase * phase_offset.cos() + self.quadrature * phase_offset.sin()
    }
}

/// Demodulates equally weighted measurements against cosine and sine references.
pub fn demodulate_measurements(
    measurement_start_times: &[f64],
    modulation_frequency: f64,
    reference_time_offset: f64,
    mut measure: impl FnMut(f64) -> f64,
) -> Demodulation {
    assert!(
        !measurement_start_times.is_empty(),
        "demodulation requires at least one measurement"
    );

    let (in_phase, quadrature) =
        measurement_start_times
            .iter()
            .fold((0.0, 0.0), |(in_phase, quadrature), &start_time| {
                let phase = modulation_frequency * (start_time + reference_time_offset);
                let (reference_sin, reference_cos) = phase.sin_cos();
                let signal = measure(start_time);

                (
                    in_phase + signal * reference_cos,
                    quadrature + signal * reference_sin,
                )
            });
    let scale = 2.0 / measurement_start_times.len() as f64;

    Demodulation {
        in_phase: scale * in_phase,
        quadrature: scale * quadrature,
    }
}

/// Fits cosine and sine coefficients together with a constant background.
pub fn demodulate_measurements_fitted(
    measurement_start_times: &[f64],
    modulation_frequency: f64,
    reference_time_offset: f64,
    mut measure: impl FnMut(f64) -> f64,
) -> Demodulation {
    assert!(
        measurement_start_times.len() >= 3,
        "demodulation requires at least three measurements"
    );

    let samples: Vec<_> = measurement_start_times
        .iter()
        .map(|&start_time| {
            let phase = modulation_frequency * (start_time + reference_time_offset);
            let (reference_sin, reference_cos) = phase.sin_cos();
            (measure(start_time), reference_sin, reference_cos)
        })
        .collect();

    let sample_count = samples.len() as f64;
    let mean_signal = samples.iter().map(|sample| sample.0).sum::<f64>() / sample_count;
    let mean_sin = samples.iter().map(|sample| sample.1).sum::<f64>() / sample_count;
    let mean_cos = samples.iter().map(|sample| sample.2).sum::<f64>() / sample_count;

    let (sin_sin, sin_cos, cos_cos, signal_sin, signal_cos) = samples.iter().fold(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        |(sin_sin, sin_cos, cos_cos, signal_sin, signal_cos), sample| {
            let signal = sample.0 - mean_signal;
            let reference_sin = sample.1 - mean_sin;
            let reference_cos = sample.2 - mean_cos;

            (
                sin_sin + reference_sin * reference_sin,
                sin_cos + reference_sin * reference_cos,
                cos_cos + reference_cos * reference_cos,
                signal_sin + signal * reference_sin,
                signal_cos + signal * reference_cos,
            )
        },
    );

    let determinant = sin_sin * cos_cos - sin_cos * sin_cos;
    assert!(
        determinant.abs() > 1.0e-12,
        "measurement phases do not span both demodulation quadratures"
    );

    Demodulation {
        in_phase: (signal_cos * sin_sin - signal_sin * sin_cos) / determinant,
        quadrature: (signal_sin * cos_cos - signal_cos * sin_cos) / determinant,
    }
}

/// Demodulates trajectory samples using trapezoidal integration over time.
/// Times must be finite and strictly increasing, with at least two samples.
/// Returns twice the time-averaged cosine and sine correlations; at zero
/// frequency, `in_phase` is twice the signed time average and `quadrature` is zero.
pub fn lockin(traj: &[f64], t_array: &[f64], freq: f64) -> Demodulation {
    assert_eq!(
        traj.len(),
        t_array.len(),
        "one value is required per sample time"
    );
    assert!(t_array.len() >= 2, "lock-in requires at least two samples");
    assert!(
        t_array.iter().all(|t| t.is_finite()),
        "sample times must be finite"
    );
    assert!(
        t_array.windows(2).all(|pair| pair[1] > pair[0]),
        "sample times must be strictly increasing"
    );
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
    Demodulation {
        in_phase: 2.0 * x_val,
        quadrature: 2.0 * y_val,
    }
}
