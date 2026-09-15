use atomic_clocks::maths::FourierSpectrum;
use atomic_clocks::signal_processing::{
    AngularFrequencyGrid, Delay, FunctionalPsd, Gain, Integrator, LtiFeedback, LtiFilter, Psd,
    PsdSamples, SpectrumError, TransferFunctionSamples,
};
use rustfft::num_complex::Complex64;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn frequency_grid_rejects_invalid_samples() {
    assert_eq!(
        AngularFrequencyGrid::new(vec![]),
        Err(SpectrumError::EmptyFrequencyGrid)
    );
    assert_eq!(
        AngularFrequencyGrid::new(vec![0.0, f64::NAN]),
        Err(SpectrumError::NonFiniteFrequency { index: 1 })
    );
    assert_eq!(
        AngularFrequencyGrid::new(vec![0.0, 1.0, 1.0]),
        Err(SpectrumError::FrequenciesNotStrictlyIncreasing { lower_index: 1 })
    );
}

#[test]
fn sampled_transfer_converts_fourier_data_and_interpolates_complex_response() {
    let transfer = TransferFunctionSamples::try_from(FourierSpectrum {
        angular_frequencies: vec![0.0, 1.0, 2.0],
        amplitudes: vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -4.0),
            Complex64::new(4.0, -8.0),
        ],
    })
    .unwrap();

    let interpolated = transfer.response_at(0.5).unwrap();
    assert_close(interpolated.re, 1.0);
    assert_close(interpolated.im, -2.0);
    assert_eq!(
        transfer.response_at(3.0),
        Err(SpectrumError::FrequencyOutsideRange {
            frequency: 3.0,
            minimum: 0.0,
            maximum: 2.0,
        })
    );
    assert_eq!(transfer.frequency_response(1.0), Complex64::new(2.0, -4.0));
}

#[test]
fn sampled_psd_is_real_nonnegative_data() {
    let grid = AngularFrequencyGrid::new(vec![0.0, 1.0, 2.0]).unwrap();
    let source = FunctionalPsd {
        spectrum: |omega| 1.0 + 2.0 * omega,
    };
    let sampled = source.sample(&grid).unwrap();

    assert_eq!(sampled.density_values(), &[1.0, 3.0, 5.0]);
    assert_close(sampled.density_at(0.5).unwrap(), 2.0);
    assert_close(sampled.spectrum(1.5), 4.0);

    assert_eq!(
        PsdSamples::new(grid, vec![1.0, -1.0, 2.0]).unwrap_err(),
        SpectrumError::InvalidPsdValue { index: 1 }
    );
}

#[test]
fn sampled_types_validate_grid_compatibility() {
    let grid = AngularFrequencyGrid::new(vec![0.0, 1.0]).unwrap();
    assert_eq!(
        TransferFunctionSamples::new(grid.clone(), vec![Complex64::new(1.0, 0.0)]).unwrap_err(),
        SpectrumError::LengthMismatch {
            frequencies: 2,
            values: 1,
        }
    );

    let negative_grid = AngularFrequencyGrid::new(vec![-1.0, 0.0]).unwrap();
    assert_eq!(
        PsdSamples::new(negative_grid, vec![1.0, 1.0]).unwrap_err(),
        SpectrumError::NegativePsdFrequency { index: 0 }
    );
}

#[test]
fn primitive_filters_have_the_expected_complex_responses() {
    assert_eq!(
        Gain::new(-2.0).frequency_response(3.0),
        Complex64::new(-2.0, 0.0)
    );
    assert_eq!(
        Integrator::new(3.0).frequency_response(2.0),
        Complex64::new(0.0, -1.5)
    );

    let delayed = Delay::new(2.0).frequency_response(0.5 * std::f64::consts::PI);
    assert_close(delayed.re, -1.0);
    assert_close(delayed.im, 0.0);
}

#[test]
fn filters_compose_in_series() {
    let controller = Gain::new(-2.0).then(Delay::new(0.5));
    let response = controller.frequency_response(std::f64::consts::PI);
    assert_close(response.re, 0.0);
    assert_close(response.im, 2.0);
}

#[test]
fn primitive_filters_reject_invalid_parameters() {
    for invalid_gain in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| Gain::new(invalid_gain)).is_err());
        assert!(std::panic::catch_unwind(|| Integrator::new(invalid_gain)).is_err());
    }
    for invalid_delay in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| Delay::new(invalid_delay)).is_err());
    }
}

#[test]
fn feedback_separates_signal_and_measurement_noise_contributions() {
    let signal = FunctionalPsd { spectrum: |_| 4.0 };
    let noise = FunctionalPsd { spectrum: |_| 9.0 };
    let feedback = LtiFeedback::new(signal, noise, Gain::new(2.0), Gain::new(3.0));
    let omega = 1.0;

    assert_close(
        feedback.sensitivity().frequency_response(omega).re,
        1.0 / 7.0,
    );
    assert_close(
        feedback
            .measurement_noise_response()
            .frequency_response(omega)
            .re,
        -3.0 / 7.0,
    );
    assert_close(feedback.free_running_psd().spectrum(omega), 4.0);
    assert_close(feedback.residual_signal_psd().spectrum(omega), 4.0 / 49.0);
    assert_close(
        feedback.injected_measurement_noise_psd().spectrum(omega),
        81.0 / 49.0,
    );
    assert_close(feedback.output_psd().spectrum(omega), 85.0 / 49.0);
}
