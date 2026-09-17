use atomic_clocks::maths::demodulation::{Demodulation, ModulationParams};
use atomic_clocks::maths::normalised_gaussian;
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DemodOutput, Frame, HamiltonianParams, MtsParams, MtsSolverParams, compute_demod,
};

use super::Params;
use super::gradients::harmonic_gradients;
use super::probe::velocity_average;
use super::response::{response_at_velocity, response_output};
use super::velocity::velocity_samples;
use crate::mts::MtsCurves;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected} within {tolerance}"
    );
}

fn demod_at_velocity(params: &Params, kv: f64) -> DemodOutput {
    compute_demod(
        &MtsParams {
            hamiltonian: HamiltonianParams {
                frame: Frame::Probe,
                kv,
                ..params.mts.hamiltonian
            },
            ..params.mts
        },
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
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
                    hamiltonian: HamiltonianParams {
                        delta,
                        kv,
                        ..Default::default()
                    },
                    solver: MtsSolverParams {
                        kr_n: 16,
                        steps_per_period,
                        n_periods: 13,
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
            let dc_sign = raw.dc[1].in_phase.signum();
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
            hamiltonian: HamiltonianParams {
                frame: Frame::Pump,
                kv: 0.75,
                ..HamiltonianParams::default()
            },
            solver: MtsSolverParams {
                hz_num: 7,
                kr_n: 3,
                steps_per_period: 40,
                n_periods: 2,
                ..MtsSolverParams::default()
            },
            ..MtsParams::default()
        },
        kv_sigma: 2.0,
        kv_window_half_width: 1.5,
        kv_samples: 3,
        ..Params::default()
    }
}

fn curves(output: &MtsCurves) -> [&[f64]; 4] {
    [&output.amp0, &output.proj1, &output.proj2, &output.proj3]
}

#[test]
fn gaussian_quadrature_uses_a_fixed_zero_centered_window() {
    let params = small_params();
    let samples = velocity_samples(&params).unwrap();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].kv, -1.0);
    assert_eq!(samples[1].kv, 0.0);
    assert_eq!(samples[2].kv, 1.0);
    for sample in samples {
        assert_close(
            sample.weight,
            normalised_gaussian(sample.kv, 0.75, 2.0),
            1e-15,
        );
    }
}

#[test]
fn quadrature_retains_the_gaussian_mass_inside_the_window() {
    let params = Params {
        kv_sigma: 1.0,
        kv_window_half_width: 1.0,
        kv_samples: 1_000,
        ..Params::default()
    };
    let mass: f64 = velocity_samples(&params)
        .unwrap()
        .iter()
        .map(|sample| sample.weight)
        .sum();
    assert_close(mass, 0.682_689_492_137, 1e-6);
}

#[test]
fn weighted_signal_matches_serial_signed_coefficients() {
    let params = small_params();
    let average = velocity_average(&params).unwrap();
    let samples = velocity_samples(&params).unwrap();
    let reference: Vec<_> = samples
        .iter()
        .map(|sample| (sample.weight, demod_at_velocity(&params, sample.kv)))
        .collect();

    assert_eq!(average.hz, reference[0].1.hz);
    for index in 0..average.hz.len() {
        let sum = |select: fn(&DemodOutput) -> &[Demodulation]| {
            reference
                .iter()
                .map(|(weight, output)| {
                    let value = &select(output)[index];
                    (weight * value.in_phase, weight * value.quadrature)
                })
                .fold((0.0, 0.0), |sum, value| (sum.0 + value.0, sum.1 + value.1))
        };
        let dc = sum(|output| &output.dc);
        let harmonic = sum(|output| &output.harmonic);
        let second = sum(|output| &output.second_harmonic);
        let third = sum(|output| &output.third_harmonic);
        assert_close(average.amp0[index], dc.0.hypot(dc.1) / 2.0, 1e-12);
        assert_close(average.proj1[index], harmonic.0, 1e-12);
        assert_close(average.proj2[index], second.0, 1e-12);
        assert_close(average.proj3[index], third.0, 1e-12);
    }
}

#[test]
fn harmonic_gradients_match_the_plotted_in_phase_signal_slopes() {
    let epsilon = 1e-4;
    let params = Params {
        mts: MtsParams {
            solver: MtsSolverParams {
                hz_lim: epsilon,
                hz_num: 2,
                ..small_params().mts.solver
            },
            ..small_params().mts
        },
        gradient_epsilon: epsilon,
        gradient_harmonics: 3,
        ..small_params()
    };
    let gradients = harmonic_gradients(&params).unwrap();
    let signal = velocity_average(&params).unwrap();

    for (actual, values) in
        gradients
            .into_iter()
            .skip(1)
            .zip([signal.proj1, signal.proj2, signal.proj3])
    {
        let expected = (values[1] - values[0]) / (2.0 * epsilon);
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
fn weighted_response_matches_serial_kernels() {
    let params = Params {
        response_delay_periods: 1,
        ..small_params()
    };
    let sum = response_output(&params).unwrap();
    let samples = velocity_samples(&params).unwrap();
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
fn shifted_gaussian_matches_the_finite_frequency_shift_coordinates() {
    let shift = 1.4;
    let relative_delta = 0.3;
    let sigma = 2.7;
    let solver = MtsSolverParams {
        hz_num: 5,
        kr_n: 3,
        steps_per_period: 40,
        n_periods: 3,
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
            hamiltonian: HamiltonianParams {
                modulation: ModulationParams {
                    shift,
                    ..Default::default()
                },
                delta: relative_delta + shift / 2.0,
                kv: physical_kv,
                ..Default::default()
            },
            solver,
            ..Default::default()
        };
        let zero_shift = MtsParams {
            hamiltonian: HamiltonianParams {
                modulation: ModulationParams {
                    shift: 0.0,
                    ..finite_shift.hamiltonian.modulation
                },
                delta: relative_delta,
                kv: shifted_kv,
                ..finite_shift.hamiltonian
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
        )
        .unwrap();
        let zero_shift = compute_demod(
            &zero_shift,
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
        )
        .unwrap();
        for (actual, expected) in finite_shift.harmonic.iter().zip(zero_shift.harmonic) {
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
            hamiltonian: HamiltonianParams {
                delta: 0.4,
                kv: 0.7,
                ..Default::default()
            },
            solver: MtsSolverParams {
                kr_n: 8,
                steps_per_period: 128,
                n_periods: 13,
                hz_lim: epsilon,
                hz_num: 3,
            },
            ..Default::default()
        },
        kv_sigma: 2.0,
        kv_window_half_width: 1.5,
        kv_samples: 5,
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
    let slope = (signal.proj1[2] - signal.proj1[0]) / (2.0 * epsilon);
    assert_close(integral, slope, 2e-5);
}

#[test]
fn rejects_invalid_gaussian_quadrature() {
    for params in [
        Params {
            kv_sigma: 0.0,
            ..Params::default()
        },
        Params {
            kv_window_half_width: 0.0,
            ..Params::default()
        },
        Params {
            kv_samples: 0,
            ..Params::default()
        },
    ] {
        assert!(velocity_samples(&params).is_err());
    }
}

#[test]
fn default_velocity_scan_is_finite() {
    let params = Params::default();
    let average = velocity_average(&params).unwrap();
    for values in curves(&average) {
        assert_eq!(values.len(), params.mts.solver.hz_num);
        assert!(values.iter().all(|value| value.is_finite()));
    }
}
