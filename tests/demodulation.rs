use std::f64::consts::PI;

use atomic_clocks::maths::demodulation::{
    demodulate_measurements, demodulate_measurements_fitted, lockin,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn measurement_correlations_use_cosine_as_in_phase() {
    let frequency = 1.7;
    let offset = 0.3;
    let times: Vec<_> = (0..16)
        .map(|i| 2.0 * PI * i as f64 / (16.0 * frequency) - offset)
        .collect();
    let result = demodulate_measurements(&times, frequency, offset, |t| {
        let phase = frequency * (t + offset);
        5.0 + 2.0 * phase.cos() + 3.0 * phase.sin()
    });

    assert_close(result.in_phase, 2.0);
    assert_close(result.quadrature, 3.0);
    assert_close(result.at_phase(0.0), 2.0);
    assert_close(result.at_phase(PI / 2.0), 3.0);
    assert_close(result.amplitude(), 13.0_f64.sqrt());
    assert_close(result.phase(), 3.0_f64.atan2(2.0));
    assert_close(result.at_phase(result.phase()), result.amplitude());
}

#[test]
fn fitted_measurements_recover_coefficients_with_uneven_phases_and_dc() {
    let result = demodulate_measurements_fitted(&[0.0, 0.2, 0.9, 1.6, 2.8], 1.7, 0.3, |t| {
        let phase = 1.7 * (t + 0.3);
        5.0 + 2.0 * phase.cos() + 3.0 * phase.sin()
    });
    assert_close(result.in_phase, 2.0);
    assert_close(result.quadrature, 3.0);
}

#[test]
fn lockin_weights_nonuniform_trajectory_intervals() {
    // For 2*cos(t) + 3*sin(t), trapezoidal sample weights are
    // pi/4, 3*pi/4, 3*pi/4, pi/4 on this deliberately coarse grid.
    let result = lockin(
        &[2.0, 3.0, -3.0, 2.0],
        &[0.0, PI / 2.0, 3.0 * PI / 2.0, 2.0 * PI],
        1.0,
    );
    assert_close(result.in_phase, 1.0);
    assert_close(result.quadrature, 4.5);
}

#[test]
fn lockin_retains_signed_dc_with_factor_two_normalization() {
    let result = lockin(&[-3.0, -3.0, -3.0], &[0.0, 0.2, 1.0], 0.0);
    assert_close(result.in_phase, -6.0);
    assert_close(result.quadrature, 0.0);
    assert_close(result.amplitude(), 6.0);
}
