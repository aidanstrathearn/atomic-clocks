mod support;

use atomic_clocks::vapourcell as current;
use support::{assert_outputs_close, current_params, mts_reference as reference, scan_cases};

#[test]
fn scans_match_reference_across_frames_and_parameters() {
    for case in scan_cases() {
        let expected = reference::compute_demod(&case.params).expect("reference scan succeeds");
        let actual =
            current::compute_demod(&current_params(&case.params)).expect("current scan succeeds");
        let error = assert_outputs_close(&case.name, &actual, &expected);
        eprintln!("{}: max absolute error {error:.3e}", case.name);
    }
}

#[test]
fn default_scan_matches_frozen_defaults() {
    let expected = reference::compute_demod(&reference::MtsParams::default()).unwrap();
    let actual = current::compute_demod(&current::MtsParams::default()).unwrap();
    assert_outputs_close("default configuration", &actual, &expected);
}

#[test]
fn invalid_parameters_match_reference_errors() {
    let mut cases = Vec::new();
    for mod_freq in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.push(reference::MtsParams {
            mod_freq,
            ..reference::MtsParams::default()
        });
    }
    cases.extend([
        reference::MtsParams {
            kr_n: 0,
            ..reference::MtsParams::default()
        },
        reference::MtsParams {
            n_periods: 0,
            ..reference::MtsParams::default()
        },
        reference::MtsParams {
            steps_per_period: 0,
            ..reference::MtsParams::default()
        },
        reference::MtsParams {
            n_periods: 1,
            steps_per_period: 1,
            ..reference::MtsParams::default()
        },
        reference::MtsParams {
            hz_num: 0,
            ..reference::MtsParams::default()
        },
    ]);

    for params in cases {
        let expected = reference::compute_demod(&params).unwrap_err();
        let actual = current::compute_demod(&current_params(&params)).unwrap_err();
        assert_eq!(actual, expected, "{params:?}");
    }
}
