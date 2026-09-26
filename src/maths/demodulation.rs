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
    /// Validates the modulation frequency, depth, and shift.
    pub fn validate(&self) -> Result<(), String> {
        if !self.frequency.is_finite() || self.frequency <= 0.0 {
            return Err("mod_freq must be positive and finite".to_string());
        }
        if !self.depth.is_finite() || !self.shift.is_finite() {
            return Err("modulation depth and shift must be finite".to_string());
        }
        Ok(())
    }

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

    pub fn period(&self) -> f64 {
        std::f64::consts::TAU / self.frequency
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

/// Demodulates uniform samples spanning exactly one period, including both endpoints.
/// The first sample defines phase zero; the last is at phase `2*pi`.
/// Endpoint values need not match. Trapezoidal endpoint weights are one half.
/// `harmonic = 0` returns twice the signed mean in `in_phase` and zero quadrature;
/// positive harmonics return the cosine and sine coefficients at that harmonic.
/// The caller supplies the sampling convention; it cannot be checked from values alone.
pub fn lockin_period(traj: &[f64], harmonic: usize) -> Demodulation {
    assert!(traj.len() >= 2, "lock-in requires at least two samples");
    let intervals = (traj.len() - 1) as f64;
    let mut in_phase = 0.0;
    let mut quadrature = 0.0;
    for (i, &sample) in traj.iter().enumerate() {
        let phase = std::f64::consts::TAU * harmonic as f64 * i as f64 / intervals;
        let (sin, cos) = phase.sin_cos();
        let weight = if i == 0 || i + 1 == traj.len() {
            0.5
        } else {
            1.0
        };
        in_phase += weight * sample * cos;
        quadrature += weight * sample * sin;
    }
    Demodulation {
        in_phase: 2.0 * in_phase / intervals,
        quadrature: 2.0 * quadrature / intervals,
    }
}
