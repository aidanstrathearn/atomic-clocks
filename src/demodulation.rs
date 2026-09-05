pub struct Demodulation {
    pub in_phase: f64,
    pub quadrature: f64,
}

impl Demodulation {
    pub fn at_phase(&self, phase_offset: f64) -> f64 {
        self.in_phase * phase_offset.cos() + self.quadrature * phase_offset.sin()
    }
}

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
                    in_phase + signal * reference_sin,
                    quadrature + signal * reference_cos,
                )
            });
    let scale = 2.0 / measurement_start_times.len() as f64;

    Demodulation {
        in_phase: scale * in_phase,
        quadrature: scale * quadrature,
    }
}

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
        in_phase: (signal_sin * cos_cos - signal_cos * sin_cos) / determinant,
        quadrature: (signal_cos * sin_sin - signal_sin * sin_cos) / determinant,
    }
}
