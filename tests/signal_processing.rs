use atomic_clocks::maths::FourierSpectrum;
use atomic_clocks::signal_processing::{
    AngularFrequencyGrid, FunctionalPsd, LtiFilter, Psd, PsdSamples, SpectrumError,
    TransferFunctionSamples,
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
