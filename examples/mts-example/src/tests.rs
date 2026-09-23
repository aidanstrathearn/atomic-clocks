use std::f64::consts::TAU;

use atomic_clocks::common::DetuningScanParams;
use atomic_clocks::maths::demodulation::{Demodulation, ModulationParams};
use atomic_clocks::maths::normalised_gaussian;
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DemodOutput, DrivenAtomParams, HamiltonianParams, LinearResponseOutput, MtsParams,
    MtsSolverParams, VelocityParams, compute_demod,
};

use crate::app::Params;
use crate::gradients::{harmonic_gradients, normalised_adjacent_gradient_sums};
use crate::probe::{DC_HARMONICS, DcCurve, HarmonicCurves, MODULATION_HARMONICS, velocity_average};
use crate::response::{demodulated_transmission_response, response_at_velocity, response_output};

const TEST_HARMONICS: [usize; 4] = [0, 1, 2, 3];

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected} within {tolerance}"
    );
}

#[test]
fn default_experiment_reproduces_the_previous_model_rates() {
    let params = Params::default();
    let mts = params.mts_params().unwrap();

    assert_eq!(params.experiment.cell.transition.wavelength_nm, 556.0);
    assert_eq!(params.experiment.pump.waist_radius_mm, 1.0);
    assert_eq!(params.experiment.probe.waist_radius_mm, 1.0);
    assert_eq!(params.experiment.cell.density_per_m3, 1.0e14);
    assert_eq!(params.experiment.cell.length_m, 0.1);
    assert_close(mts.atom.hamiltonian.r_pump, 1.0, 1e-14);
    assert_close(mts.atom.hamiltonian.r_prbe, 0.1, 1e-14);
    assert_close(mts.atom.decay.gamma_down, 1.0, 1e-14);
    assert_close(
        params.experiment.pump.power_milliwatts,
        0.003_802_164_024_35,
        1e-12,
    );
    assert_close(
        params.experiment.probe.power_milliwatts,
        0.000_038_021_640_243_5,
        1e-14,
    );
}

#[test]
fn default_experiment_has_a_finite_nonzero_transmission_gain() {
    let gain = Params::default().transmission_gain().unwrap();
    assert!(gain.is_finite());
    assert!(gain < 0.0);
}

fn demod_at_velocity<const N: usize>(
    params: &Params,
    kv: f64,
    harmonics: [usize; N],
) -> DemodOutput<N> {
    let mts = params.mts_params().unwrap();
    compute_demod(
        &MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: params.probe_hamiltonian(&mts, kv),
                ..mts.atom
            },
            ..mts
        },
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        harmonics,
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
            let mut params = Params::default();
            params.experiment.probe_detuning = delta / TAU;
            params.experiment.scan_half_range = epsilon / TAU;
            params.experiment.scan_samples = 3;
            params.solver = MtsSolverParams {
                kr_n: 16,
                steps_per_period,
                n_periods: 13,
            };
            params.response_delay_periods = 6;
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
            let raw = demod_at_velocity(&params, kv, TEST_HARMONICS);
            // Harmonic zero returns twice the mean.
            let integrated = [0.5 * integrate(0), integrate(1), integrate(2), integrate(3)];
            let slopes: [f64; 4] = std::array::from_fn(|index| {
                let dc_scale = if index == 0 { 0.5 } else { 1.0 };
                dc_scale * (raw.values[2][index].in_phase - raw.values[0][index].in_phase)
                    / (2.0 * epsilon)
            });
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

#[test]
fn transmission_curves_apply_one_gain_to_signed_components() {
    let dc_raw = DemodOutput {
        hz: vec![0.0],
        harmonics: DC_HARMONICS,
        values: vec![[Demodulation {
            in_phase: 0.8,
            quadrature: 0.0,
        }]],
    };
    let harmonic_raw = DemodOutput {
        hz: vec![0.0],
        harmonics: MODULATION_HARMONICS,
        values: vec![[
            Demodulation {
                in_phase: -0.4,
                quadrature: 0.1,
            },
            Demodulation {
                in_phase: 0.2,
                quadrature: -0.3,
            },
            Demodulation {
                in_phase: -0.1,
                quadrature: 0.5,
            },
        ]],
    };
    let dc = DcCurve::from_demodulated(dc_raw, -0.25);
    let harmonics = HarmonicCurves::from_demodulated(harmonic_raw, -0.25);
    assert_eq!(dc.values, [-0.1]);
    assert_eq!(harmonics.proj1, [0.1]);
    assert_eq!(harmonics.proj2, [-0.05]);
    assert_eq!(harmonics.proj3, [0.025]);
}

fn small_params() -> Params {
    let mut params = Params::default();
    params.solver = MtsSolverParams {
        kr_n: 3,
        steps_per_period: 40,
        n_periods: 2,
    };
    params.experiment.scan_samples = 7;
    params.velocity = VelocityParams {
        sigma: 2.0,
        half_width: 1.5,
        sample_count: 3,
        ..VelocityParams::default()
    };
    params
}

#[test]
fn weighted_signal_matches_serial_signed_coefficients() {
    let params = small_params();
    let average = velocity_average(&params, TEST_HARMONICS).unwrap();
    let mts = params.mts_params().unwrap();
    let samples = params.velocity_params(&mts).samples().unwrap();
    let reference: Vec<_> = samples
        .iter()
        .map(|sample| {
            (
                sample.weight,
                demod_at_velocity(&params, sample.kv, TEST_HARMONICS),
            )
        })
        .collect();

    assert_eq!(average.hz, reference[0].1.hz);
    assert_eq!(average.harmonics, TEST_HARMONICS);
    for index in 0..average.hz.len() {
        for harmonic_index in 0..TEST_HARMONICS.len() {
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
    let mut params = small_params();
    params.experiment.scan_half_range = epsilon / TAU;
    params.experiment.scan_samples = 2;
    params.gradient_epsilon = epsilon;
    params.gradient_harmonics = 3;
    let gradients = harmonic_gradients(&params).unwrap();
    let signal = velocity_average(&params, MODULATION_HARMONICS).unwrap();

    for (actual, harmonic_index) in gradients.into_iter().skip(1).zip(0..3) {
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
    let gradients = [2.0, 4.0, -3.0, 5.0, -7.0];
    let values = normalised_adjacent_gradient_sums(&gradients).unwrap();
    assert_eq!(values.len(), 3);
    assert_close(values[0], 1.0 / 16.0, 1e-15);
    assert_close(values[1], 81.0 / 16.0, 1e-15);
    assert_close(values[2], 25.0 / 4.0, 1e-15);

    let scaled: Vec<_> = gradients.iter().map(|value| -0.37 * value).collect();
    let scaled_values = normalised_adjacent_gradient_sums(&scaled).unwrap();
    for (actual, expected) in scaled_values.into_iter().zip(values) {
        assert_close(actual, expected, 1e-14);
    }
}

#[test]
fn transmission_response_uses_mean_dc_and_standard_harmonics() {
    let output = LinearResponseOutput {
        times: vec![0.0, std::f64::consts::PI, std::f64::consts::TAU],
        delays: vec![0.0],
        response: vec![vec![1.0], vec![-1.0], vec![1.0]],
    };
    let gain = -0.25;
    let raw_dc = output.demodulate(0)[0].in_phase;
    let raw_first = output.demodulate(1)[0].in_phase;
    assert_eq!(
        demodulated_transmission_response(&output, 0, gain),
        [0.5 * gain * raw_dc]
    );
    assert_eq!(
        demodulated_transmission_response(&output, 1, gain),
        [gain * raw_first]
    );
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
    let mts = params.mts_params().unwrap();
    let samples = params.velocity_params(&mts).samples().unwrap();
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
    let mut params = Params::default();
    params.experiment.pump_probe_offset = shift / TAU;
    params.velocity = VelocityParams {
        sigma,
        half_width: 1.5,
        sample_count: 3,
        ..VelocityParams::default()
    };
    let mts = params.mts_params().unwrap();
    assert_eq!(params.probe_hamiltonian(&mts, 0.0).modulation.shift, 0.0);
    assert_close(
        crate::probe::display_detuning(&mts, relative_delta),
        relative_delta + shift / 2.0,
        1e-15,
    );
    for sample in params.velocity_params(&mts).samples().unwrap() {
        assert_close(
            sample.weight,
            normalised_gaussian(sample.kv, shift / 2.0, sigma),
            1e-15,
        );
    }
    params.velocity.sigma = 0.0;
    let zero_width_samples = params.velocity_params(&mts).samples().unwrap();
    assert_eq!(zero_width_samples.len(), 1);
    assert_close(zero_width_samples[0].kv, shift / 2.0, 1e-15);
    assert_eq!(zero_width_samples[0].weight, 1.0);

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
            TEST_HARMONICS,
        )
        .unwrap();
        let zero_shift = compute_demod(
            &zero_shift,
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            TEST_HARMONICS,
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
    let mut params = Params::default();
    params.experiment.probe_detuning = 0.4 / TAU;
    params.experiment.scan_half_range = epsilon / TAU;
    params.experiment.scan_samples = 3;
    params.solver = MtsSolverParams {
        kr_n: 8,
        steps_per_period: 128,
        n_periods: 13,
    };
    params.velocity = VelocityParams {
        sigma: 2.0,
        half_width: 1.5,
        sample_count: 5,
        ..VelocityParams::default()
    };
    params.response_delay_periods = 6;
    let response = response_output(&params).unwrap();
    let integral = response
        .demodulate(1)
        .windows(2)
        .zip(response.delays.windows(2))
        .map(|(values, times)| {
            0.5 * (times[1] - times[0]) * (values[0].in_phase + values[1].in_phase)
        })
        .sum::<f64>();
    let signal = velocity_average(&params, [1]).unwrap();
    let slope = (signal.values[2][0].in_phase - signal.values[0][0].in_phase) / (2.0 * epsilon);
    assert_close(integral, slope, 2e-5);
}

#[test]
fn default_velocity_scan_is_finite() {
    let params = Params::default();
    let dc = velocity_average(&params, DC_HARMONICS).unwrap();
    let harmonics = velocity_average(&params, MODULATION_HARMONICS).unwrap();
    assert_eq!(dc.harmonics, DC_HARMONICS);
    assert_eq!(harmonics.harmonics, MODULATION_HARMONICS);
    assert_eq!(dc.values.len(), params.experiment.scan_samples);
    assert_eq!(harmonics.values.len(), params.experiment.scan_samples);
    assert!(
        dc.values
            .iter()
            .flatten()
            .chain(harmonics.values.iter().flatten())
            .all(|value| value.in_phase.is_finite() && value.quadrature.is_finite())
    );
}
