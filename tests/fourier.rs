use std::f64::consts::PI;

use atomic_clocks::maths::fourier_transform;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn constant_signal_has_integrated_dc_amplitude() {
    let spectrum = fourier_transform(&[2.0; 7], 0.25, -1.0, 100.0);
    assert_eq!(spectrum.amplitudes.len(), 4);
    assert_close(spectrum.angular_frequencies[0], 0.0);
    assert_close(spectrum.amplitudes[0].re, 3.5);
    assert_close(spectrum.amplitudes[0].im, 0.0);
    for value in &spectrum.amplitudes[1..] {
        assert_close(value.norm(), 0.0);
    }
    assert_close(spectrum.angular_frequencies[3], 6.0 * PI / 1.75);
}

#[test]
fn sine_has_correct_frequency_sign_and_absolute_time_phase() {
    for n in [15, 32] {
        let step = 0.2;
        let start_time = -0.37;
        let omega = 2.0 * PI * 3.0 / (n as f64 * step);
        let samples: Vec<_> = (0..n)
            .map(|i| (omega * (start_time + i as f64 * step)).sin())
            .collect();
        let spectrum = fourier_transform(&samples, step, start_time, 100.0);
        assert_close(spectrum.angular_frequencies[3], omega);
        for (k, value) in spectrum.amplitudes.iter().enumerate() {
            assert_close(value.re, 0.0);
            assert_close(value.im, if k == 3 { -0.5 * n as f64 * step } else { 0.0 });
        }
    }
}

#[test]
fn nyquist_bin_is_included_without_doubling() {
    let spectrum = fourier_transform(&[1.0, -1.0, 1.0, -1.0], 0.5, 0.0, 100.0);
    assert_eq!(spectrum.amplitudes.len(), 3);
    assert_close(spectrum.angular_frequencies[2], 2.0 * PI);
    assert_close(spectrum.amplitudes[2].re, 2.0);
    assert_close(spectrum.amplitudes[2].im, 0.0);
    assert_close(spectrum.amplitudes[0].norm(), 0.0);
    assert_close(spectrum.amplitudes[1].norm(), 0.0);
}

#[test]
fn empty_and_single_sample_inputs_are_supported() {
    let empty = fourier_transform(&[], 0.1, 0.0, 100.0);
    assert!(empty.angular_frequencies.is_empty());
    assert!(empty.amplitudes.is_empty());

    let single = fourier_transform(&[3.0], 0.2, -1.0, 100.0);
    assert_eq!(single.angular_frequencies, vec![0.0]);
    assert_close(single.amplitudes[0].re, 0.6);
    assert_close(single.amplitudes[0].im, 0.0);
}

#[test]
#[should_panic(expected = "Fourier sample step must be finite and positive")]
fn zero_sample_step_is_rejected() {
    fourier_transform(&[1.0], 0.0, 0.0, 100.0);
}

#[test]
fn requested_spacing_rounds_up_padding_and_preserves_dc() {
    // 2*pi / (0.25 * 2.0) = 12.566..., so pad seven samples to thirteen.
    let spectrum = fourier_transform(&[2.0; 7], 0.25, -1.0, 2.0);
    assert_eq!(spectrum.amplitudes.len(), 7);
    assert_close(spectrum.angular_frequencies[1], 2.0 * PI / (13.0 * 0.25));
    assert!(spectrum.angular_frequencies[1] <= 2.0);
    assert_close(spectrum.amplitudes[0].re, 3.5);
    assert_close(spectrum.amplitudes[0].im, 0.0);
}

#[test]
fn padding_preserves_impulse_magnitude_and_absolute_time_phase() {
    // The only nonzero sample is at t = 0.5; padding must not shift it or scale it.
    let spectrum = fourier_transform(&[0.0, 3.0, 0.0], 0.2, 0.3, 0.5);
    assert!(spectrum.amplitudes.len() > 2);
    for (&omega, value) in spectrum
        .angular_frequencies
        .iter()
        .zip(&spectrum.amplitudes)
    {
        assert_close(value.norm(), 0.6);
        assert_close(value.re, 0.6 * (omega * 0.5).cos());
        assert_close(value.im, -0.6 * (omega * 0.5).sin());
    }
}

#[test]
fn padding_preserves_original_bins_and_nyquist_limit() {
    let samples = [1.0, -0.3, 0.7, 2.0, -1.0, 0.4, 0.2, -0.8];
    let original = fourier_transform(&samples, 0.25, -0.37, 100.0);
    // Round up from 31.5 to 32 samples, giving four times as fine a grid.
    let padded = fourier_transform(&samples, 0.25, -0.37, 2.0 * PI / (31.5 * 0.25));
    assert_eq!(padded.amplitudes.len(), 17);
    assert_close(*padded.angular_frequencies.last().unwrap(), PI / 0.25);
    for (k, value) in original.amplitudes.iter().enumerate() {
        assert_close(
            padded.angular_frequencies[4 * k],
            original.angular_frequencies[k],
        );
        assert_close(padded.amplitudes[4 * k].re, value.re);
        assert_close(padded.amplitudes[4 * k].im, value.im);
    }
}

#[test]
fn invalid_frequency_spacing_is_rejected() {
    for df in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| fourier_transform(&[1.0], 0.1, 0.0, df)).is_err());
    }
}
