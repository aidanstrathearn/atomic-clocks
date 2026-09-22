use atomic_clocks::common::DetuningScanParams;
use atomic_clocks::maths::demodulation::ModulationParams;
use atomic_clocks::maths::normalised_gaussian;
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DrivenAtomParams, Frame, HamiltonianParams, MtsParams, MtsSolverParams, VelocityParams,
    compute_demod,
};

use crate::app::Params;
use crate::gradients::{harmonic_gradients, normalised_adjacent_gradient_sums};
use crate::probe::{MtsCurves, SIGNAL_HARMONICS, SignalOutput, velocity_average};
use crate::response::{response_at_velocity, response_output};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected} within {tolerance}"
    );
}

fn demod_at_velocity(params: &Params, kv: f64) -> SignalOutput {
    compute_demod(
        &MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: params.probe_hamiltonian(kv),
                ..params.mts.atom
            },
            ..params.mts
        },
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        SIGNAL_HARMONICS,
    )
    .unwrap()
}

#[test]
fn integrated_demodulated_response_matches_probe_curve_slopes() {
    // Compare a single velocity, using a settled observation window and a
    // long delay cutoff. The generic Doppler shift also needs converged kr
    // averaging because a single atom is not modulation-periodic.
    let epsilon = 1e-4;
    for (delta, kv) in [(0.6, 0.0), (-0.4, 0.37)] {
        let mut previous_errors = [f64::INFINITY; 4];
        for steps_per_period in [128, 256] {
            let params = Params {
                mts: MtsParams {
                    atom: DrivenAtomParams {
                        hamiltonian: HamiltonianParams {
                            delta,
                            kv,
                            ..Default::default()
                        },
                        ..MtsParams::default().atom
                    },
                    solver: MtsSolverParams {
                        kr_n: 16,
                        steps_per_period,
                        n_periods: 13,
                    },
                    scan: DetuningScanParams {
                        hz_lim: epsilon,
                        hz_num: 3,
                    },
                    ..Default::default()
                },
                response_delay_periods: 6,
                ..Params::default()
            };
            let output = response_at_velocity(&params, kv).unwrap();
            let integrate = |harmonic| {
                output
                    .demodulate(harmonic)
                    .windows(2)
                    .zip(output.delays.windows(2))
                    .map(|(values, times)| {
                        0.5 * (times[1] - times[0]) * (values[0].in_phase + values[1].in_phase)
                    })
                    .sum::<f64>()
            };
            let raw = demod_at_velocity(&params, kv);
            let dc_sign = raw.values[1][0].in_phase.signum();
            let curves = MtsCurves::from(raw);
            // Probe plots -abs(mean) for DC; harmonic zero returns 2*mean.
            let integrated = [
                -0.5 * dc_sign * integrate(0),
                integrate(1),
                integrate(2),
                integrate(3),
            ];
            let slopes = [
                (curves.amp0[0] - curves.amp0[2]) / (2.0 * epsilon),
                (curves.proj1[2] - curves.proj1[0]) / (2.0 * epsilon),
                (curves.proj2[2] - curves.proj2[0]) / (2.0 * epsilon),
                (curves.proj3[2] - curves.proj3[0]) / (2.0 * epsilon),
            ];
            eprintln!(
                "delta={delta}, kv={kv}, steps={steps_per_period}: integrals={integrated:?}, slopes={slopes:?}"
            );
            for j in 0..4 {
                let error = (integrated[j] - slopes[j]).abs();
                assert!(
                    error < previous_errors[j],
                    "component {j} did not converge: {error}"
                );
                if steps_per_period == 256 {
                    assert!(
                        error < 2e-5,
                        "component {j}: {} != {}, error={error}",
                        integrated[j],
                        slopes[j]
                    );
                }
                previous_errors[j] = error;
            }
        }
    }
}

fn small_params() -> Params {
    Params {
        mts: MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    frame: Frame::Pump,
                    ..HamiltonianParams::default()
                },
                ..MtsParams::default().atom
            },
            solver: MtsSolverParams {
                kr_n: 3,
                steps_per_period: 40,
                n_periods: 2,
            },
            scan: DetuningScanParams {
                hz_num: 7,
                ..DetuningScanParams::default()
            },
            ..MtsParams::default()
        },
        velocity: VelocityParams {
            sigma: 2.0,
            half_width: 1.5,
            sample_count: 3,
            ..VelocityParams::default()
        },
        ..Params::default()
    }
}

#[test]
fn weighted_signal_matches_serial_signed_coefficients() {
    let params = small_params();
    let average = velocity_average(&params).unwrap();
    let samples = params.velocity_params().samples().unwrap();
    let reference: Vec<_> = samples
        .iter()
        .map(|sample| (sample.weight, demod_at_velocity(&params, sample.kv)))
        .collect();

    assert_eq!(average.hz, reference[0].1.hz);
    assert_eq!(average.harmonics, SIGNAL_HARMONICS);
    for index in 0..average.hz.len() {
        for harmonic_index in 0..SIGNAL_HARMONICS.len() {
            let expected = reference
                .iter()
                .map(|(weight, output)| {
                    let value = output.values[index][harmonic_index];
                    (weight * value.in_phase, weight * value.quadrature)
                })
                .fold((0.0, 0.0), |sum, value| (sum.0 + value.0, sum.1 + value.1));
            let actual = average.values[index][harmonic_index];
            assert_close(actual.in_phase, expected.0, 1e-12);
            assert_close(actual.quadrature, expected.1, 1e-12);
        }
    }
}

#[test]
fn harmonic_gradients_match_the_plotted_in_phase_signal_slopes() {
    let epsilon = 1e-4;
    let params = Params {
        mts: MtsParams {
            scan: DetuningScanParams {
                hz_lim: epsilon,
                hz_num: 2,
                ..small_params().mts.scan
            },
            ..small_params().mts
        },
        gradient_epsilon: epsilon,
        gradient_harmonics: 3,
        ..small_params()
    };
    let gradients = harmonic_gradients(&params).unwrap();
    let signal = velocity_average(&params).unwrap();

    for (actual, harmonic_index) in gradients.into_iter().skip(1).zip(1..=3) {
        let expected = (signal.values[1][harmonic_index].in_phase
            - signal.values[0][harmonic_index].in_phase)
            / (2.0 * epsilon);
        assert_close(actual, expected, 1e-12);
    }
}

#[test]
fn harmonic_gradients_validate_the_difference_and_sampling() {
    for params in [
        Params {
            gradient_epsilon: 0.0,
            ..small_params()
        },
        Params {
            gradient_harmonics: 0,
            ..small_params()
        },
        Params {
            gradient_harmonics: 21,
            ..small_params()
        },
    ] {
        assert!(harmonic_gradients(&params).is_err());
    }
}

#[test]
fn adjacent_gradient_sums_use_signed_neighbours_and_double_dc() {
    let values = normalised_adjacent_gradient_sums(&[2.0, 4.0, -3.0, 5.0, -7.0]).unwrap();
    assert_eq!(values.len(), 3);
    assert_close(values[0], 1.0 / 16.0, 1e-15);
    assert_close(values[1], 81.0 / 16.0, 1e-15);
    assert_close(values[2], 25.0 / 4.0, 1e-15);
}

#[test]
fn adjacent_gradient_sums_require_two_harmonics_and_nonzero_normalisation() {
    assert!(normalised_adjacent_gradient_sums(&[1.0, 2.0]).is_err());
    assert!(normalised_adjacent_gradient_sums(&[1.0, 0.0, 2.0]).is_err());
    assert!(normalised_adjacent_gradient_sums(&[1.0, f64::NAN, 2.0]).is_err());
}

#[test]
fn weighted_response_matches_serial_kernels() {
    let params = Params {
        response_delay_periods: 1,
        ..small_params()
    };
    let sum = response_output(&params).unwrap();
    let samples = params.velocity_params().samples().unwrap();
    let reference: Vec<_> = samples
        .iter()
        .map(|sample| {
            (
                sample.weight,
                response_at_velocity(&params, sample.kv).unwrap(),
            )
        })
        .collect();

    assert_eq!(sum.times, reference[0].1.times);
    assert_eq!(sum.delays, reference[0].1.delays);
    for (i, row) in sum.response.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            let expected = reference
                .iter()
                .map(|(weight, output)| weight * output.response[i][j])
                .sum::<f64>();
            assert_close(value, expected, 1e-12);
        }
    }
}

#[test]
fn shift_is_absorbed_into_velocity_and_detuning_coordinates() {
    let shift = 1.4;
    let relative_delta = 0.3;
    let sigma = 2.7;
    let params = Params {
        mts: MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        shift,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..MtsParams::default().atom
            },
            ..Default::default()
        },
        velocity: VelocityParams {
            sigma,
            half_width: 1.5,
            sample_count: 3,
            ..VelocityParams::default()
        },
        ..Default::default()
    };
    assert_eq!(params.probe_hamiltonian(0.0).modulation.shift, 0.0);
    assert_close(
        crate::probe::display_detuning(&params.mts, relative_delta),
        relative_delta + shift / 2.0,
        1e-15,
    );
    for sample in params.velocity_params().samples().unwrap() {
        assert_close(
            sample.weight,
            normalised_gaussian(sample.kv, shift / 2.0, sigma),
            1e-15,
        );
    }

    let solver = MtsSolverParams {
        kr_n: 3,
        steps_per_period: 40,
        n_periods: 3,
    };
    let scan = DetuningScanParams {
        hz_num: 5,
        ..Default::default()
    };
    for shifted_kv in [-0.8, 0.2, 1.1] {
        let physical_kv = shifted_kv - shift / 2.0;
        assert_close(
            normalised_gaussian(shifted_kv, shift / 2.0, sigma),
            normalised_gaussian(physical_kv, 0.0, sigma),
            1e-15,
        );
        let finite_shift = MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        shift,
                        ..Default::default()
                    },
                    delta: relative_delta + shift / 2.0,
                    kv: physical_kv,
                    ..Default::default()
                },
                ..MtsParams::default().atom
            },
            solver,
            scan,
            ..Default::default()
        };
        let zero_shift = MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        shift: 0.0,
                        ..finite_shift.atom.hamiltonian.modulation
                    },
                    delta: relative_delta,
                    kv: shifted_kv,
                    ..finite_shift.atom.hamiltonian
                },
                ..finite_shift.atom
            },
            ..finite_shift
        };
        let finite_shift = compute_demod(
            &finite_shift,
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            SIGNAL_HARMONICS,
        )
        .unwrap();
        let zero_shift = compute_demod(
            &zero_shift,
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            SIGNAL_HARMONICS,
        )
        .unwrap();
        for (actual, expected) in finite_shift.values.iter().zip(zero_shift.values) {
            let actual = actual[1];
            let expected = expected[1];
            assert_close(actual.in_phase, expected.in_phase, 1e-12);
            assert_close(actual.quadrature, expected.quadrature, 1e-12);
        }
    }
}

#[test]
fn integrated_weighted_response_matches_weighted_signal_slope() {
    let epsilon = 1e-4;
    let params = Params {
        mts: MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    delta: 0.4,
                    kv: 0.7,
                    ..Default::default()
                },
                ..MtsParams::default().atom
            },
            solver: MtsSolverParams {
                kr_n: 8,
                steps_per_period: 128,
                n_periods: 13,
            },
            scan: DetuningScanParams {
                hz_lim: epsilon,
                hz_num: 3,
            },
            ..Default::default()
        },
        velocity: VelocityParams {
            sigma: 2.0,
            half_width: 1.5,
            sample_count: 5,
            ..VelocityParams::default()
        },
        response_delay_periods: 6,
        ..Params::default()
    };
    let response = response_output(&params).unwrap();
    let integral = response
        .demodulate(1)
        .windows(2)
        .zip(response.delays.windows(2))
        .map(|(values, times)| {
            0.5 * (times[1] - times[0]) * (values[0].in_phase + values[1].in_phase)
        })
        .sum::<f64>();
    let signal = velocity_average(&params).unwrap();
    let slope = (signal.values[2][1].in_phase - signal.values[0][1].in_phase) / (2.0 * epsilon);
    assert_close(integral, slope, 2e-5);
}

#[test]
fn default_velocity_scan_is_finite() {
    let params = Params::default();
    let average = velocity_average(&params).unwrap();
    assert_eq!(average.harmonics, SIGNAL_HARMONICS);
    assert_eq!(average.values.len(), params.mts.scan.hz_num);
    assert!(
        average
            .values
            .iter()
            .flatten()
            .all(|value| value.in_phase.is_finite() && value.quadrature.is_finite())
    );
}
