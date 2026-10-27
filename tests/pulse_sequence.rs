use std::f64::consts::{PI, TAU};

use atomic_clocks::twolevel::{
    BlochVec, Channel, DephasingLiouvillian, Hamiltonian, PauliDephasingLiouvillian,
};

use atomic_clocks::interferometer::pulse_sequence::*;
use atomic_clocks::twolevel::pulse_sequence::*;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}"
    );
}

fn assert_state_close(actual: BlochVec, expected: BlochVec, tolerance: f64) {
    assert_close(actual.r.x, expected.r.x, tolerance);
    assert_close(actual.r.y, expected.r.y, tolerance);
    assert_close(actual.r.z, expected.r.z, tolerance);
}

#[test]
fn momentum_branch_detunings_include_doppler_and_recoil_shifts() {
    let detunings =
        MomentumBranchDetunings::from_carrier_doppler_recoil(12_345.0, 2_300.0, 1_750.0);

    assert_close(detunings.forward_hz, 8_295.0, 0.0);
    assert_close(detunings.backward_lower_hz, 12_895.0, 0.0);
    assert_close(detunings.backward_upper_hz, 19_895.0, 0.0);
}

#[test]
fn prepared_pulse_matches_direct_evolution_for_multiple_states() {
    let duration_s = 310.0e-6;
    let rabi_angular_frequency = TAU * 1_900.0;
    let detuning_angular_frequency = TAU * 4_700.0;
    let mut mixed = BlochVec::ground();
    mixed.r.x = 0.24;
    mixed.r.y = -0.31;
    mixed.r.z = 0.42;

    for linewidth_hz in [0.0, 1_300.0] {
        let parameters = PulseParameters {
            rabi_angular_frequency,
            detuning_angular_frequency,
            gamma_x: 0.0,
            gamma_phi: PI * linewidth_hz,
        };
        let hamiltonian = Hamiltonian::new(
            parameters.rabi_angular_frequency,
            0.0,
            parameters.detuning_angular_frequency,
        );
        let pulse = prepare_pulse(parameters, duration_s);

        match pulse {
            PreparedPulse::Unitary(_) => assert_eq!(linewidth_hz, 0.0),
            PreparedPulse::Dephasing(_) => assert_ne!(linewidth_hz, 0.0),
        }

        for initial in [BlochVec::ground(), mixed] {
            let expected = if linewidth_hz == 0.0 {
                hamiltonian.for_duration(duration_s).apply_to(initial)
            } else {
                DephasingLiouvillian {
                    hamiltonian,
                    gamma_phi: PI * linewidth_hz,
                }
                .for_duration(duration_s)
                .apply_to(initial)
            };

            assert_state_close(pulse.apply_to(initial), expected, 1.0e-14);
        }

        let z_response = pulse.apply_to(BlochVec::excited()).r.z;
        assert_close(
            pulse.apply_to(BlochVec::ground()).ground_probability(),
            0.5 * (1.0 + z_response),
            1.0e-14,
        );
        assert_close(
            pulse.apply_to(BlochVec::excited()).ground_probability(),
            0.5 * (1.0 - z_response),
            1.0e-14,
        );
    }
}

#[test]
fn prepared_pulse_combines_drive_axis_and_frequency_dephasing() {
    let duration_s = 310.0e-6;
    let parameters = PulseParameters {
        rabi_angular_frequency: TAU * 1_900.0,
        detuning_angular_frequency: TAU * 4_700.0,
        gamma_x: 430.0,
        gamma_phi: PI * 1_300.0,
    };
    let hamiltonian = Hamiltonian::new(
        parameters.rabi_angular_frequency,
        0.0,
        parameters.detuning_angular_frequency,
    );
    let expected = PauliDephasingLiouvillian {
        hamiltonian,
        gamma_x: parameters.gamma_x,
        gamma_y: 0.0,
        gamma_z: parameters.gamma_phi,
    }
    .for_duration(duration_s);
    let actual = prepare_pulse(parameters, duration_s);
    let mut mixed = BlochVec::ground();
    mixed.r.x = 0.24;
    mixed.r.y = -0.31;
    mixed.r.z = 0.42;

    for initial in [BlochVec::ground(), mixed] {
        assert_state_close(
            actual.apply_to(initial),
            expected.apply_to(initial),
            1.0e-14,
        );
    }
}

#[test]
fn resonant_white_rabi_noise_exponentially_damps_population_oscillations() {
    let duration_s = 310.0e-6;
    let rabi_angular_frequency = TAU * 1_900.0;
    let gamma_x = 430.0;
    let parameters = PulseParameters {
        rabi_angular_frequency,
        detuning_angular_frequency: 0.0,
        gamma_x,
        gamma_phi: 0.0,
    };

    let actual = SinglePulse::new(duration_s).ground_probability(parameters);
    let expected =
        0.5 * (1.0 + (-gamma_x * duration_s).exp() * (rabi_angular_frequency * duration_s).cos());

    assert_close(actual, expected, 1.0e-14);
}

#[test]
fn phased_application_matches_a_directly_phased_hamiltonian() {
    let duration_s = 310.0e-6;
    let rabi_angular_frequency = TAU * 1_900.0;
    let detuning_angular_frequency = TAU * 4_700.0;
    let mut mixed = BlochVec::ground();
    mixed.r.x = 0.24;
    mixed.r.y = -0.31;
    mixed.r.z = 0.42;

    for linewidth_hz in [0.0, 1_300.0] {
        let parameters = PulseParameters {
            rabi_angular_frequency,
            detuning_angular_frequency,
            gamma_x: 0.0,
            gamma_phi: PI * linewidth_hz,
        };
        let pulse = prepare_pulse(parameters, duration_s);

        for phase_rad in [-PI, -0.73, 0.0, 0.41, 0.5 * PI] {
            let (sin_phase, cos_phase) = phase_rad.sin_cos();
            let phased_hamiltonian = Hamiltonian::new(
                rabi_angular_frequency * cos_phase,
                rabi_angular_frequency * sin_phase,
                detuning_angular_frequency,
            );

            for initial in [BlochVec::ground(), mixed] {
                let expected = if linewidth_hz == 0.0 {
                    phased_hamiltonian
                        .for_duration(duration_s)
                        .apply_to(initial)
                } else {
                    DephasingLiouvillian {
                        hamiltonian: phased_hamiltonian,
                        gamma_phi: PI * linewidth_hz,
                    }
                    .for_duration(duration_s)
                    .apply_to(initial)
                };

                assert_state_close(
                    pulse.apply_to_with_phase(initial, phase_rad),
                    expected,
                    1.0e-12,
                );
            }
        }
    }
}

#[test]
fn phased_application_rotates_rabi_dephasing_with_the_drive_axis() {
    let duration_s = 310.0e-6;
    let rabi_angular_frequency = TAU * 1_900.0;
    let detuning_angular_frequency = TAU * 4_700.0;
    let gamma_x = 430.0;
    let gamma_z = PI * 1_300.0;
    let parameters = PulseParameters {
        rabi_angular_frequency,
        detuning_angular_frequency,
        gamma_x,
        gamma_phi: gamma_z,
    };
    let pulse = prepare_pulse(parameters, duration_s);
    let mut mixed = BlochVec::ground();
    mixed.r.x = 0.24;
    mixed.r.y = -0.31;
    mixed.r.z = 0.42;

    for (phase_rad, h_x, h_y, dephasing_x, dephasing_y) in [
        (0.0, rabi_angular_frequency, 0.0, gamma_x, 0.0),
        (PI, -rabi_angular_frequency, 0.0, gamma_x, 0.0),
        (0.5 * PI, 0.0, rabi_angular_frequency, 0.0, gamma_x),
        (-0.5 * PI, 0.0, -rabi_angular_frequency, 0.0, gamma_x),
    ] {
        let expected = PauliDephasingLiouvillian {
            hamiltonian: Hamiltonian::new(h_x, h_y, detuning_angular_frequency),
            gamma_x: dephasing_x,
            gamma_y: dephasing_y,
            gamma_z,
        }
        .for_duration(duration_s);

        for initial in [BlochVec::ground(), mixed] {
            assert_state_close(
                pulse.apply_to_with_phase(initial, phase_rad),
                expected.apply_to(initial),
                1.0e-12,
            );
        }
    }
}

#[test]
fn corpse_pi_uses_the_canonical_angles_and_phases() {
    let rabi_angular_frequency = TAU * 1_000.0;
    let sequence = Corpse3::new(PI / rabi_angular_frequency);
    let expected_angles_rad = [7.0 * PI / 3.0, 5.0 * PI / 3.0, PI / 3.0];

    for (actual_duration_s, expected_angle_rad) in sequence
        .segment_durations_s()
        .into_iter()
        .zip(expected_angles_rad)
    {
        assert_close(
            actual_duration_s * rabi_angular_frequency,
            expected_angle_rad,
            1.0e-14,
        );
    }
    assert_eq!(Corpse3::SEGMENT_PHASES_RAD, [0.0, PI, 0.0]);
}

#[test]
fn resonant_corpse_matches_the_target_rotation() {
    let rabi_angular_frequency = TAU * 1_900.0;
    let parameters = PulseParameters {
        rabi_angular_frequency,
        detuning_angular_frequency: 0.0,
        gamma_x: 0.0,
        gamma_phi: 0.0,
    };
    let mut mixed = BlochVec::ground();
    mixed.r.x = 0.24;
    mixed.r.y = -0.31;
    mixed.r.z = 0.42;

    for target_angle_rad in [0.2 * PI, 0.5 * PI, PI, 1.7 * PI, 2.0 * PI] {
        let target_duration_s = target_angle_rad / rabi_angular_frequency;
        let corpse = Corpse3::new(target_duration_s);
        let single = SinglePulse::new(target_duration_s);

        for initial in [BlochVec::ground(), mixed] {
            assert_state_close(
                corpse.apply_to(parameters, initial),
                single.apply_to(parameters, initial),
                2.0e-14,
            );
        }
    }
}

#[test]
fn corpse_pi_suppresses_small_static_detuning_error() {
    let rabi_angular_frequency = TAU * 1_900.0;
    let target_duration_s = PI / rabi_angular_frequency;
    let resonant_parameters = PulseParameters {
        rabi_angular_frequency,
        detuning_angular_frequency: 0.0,
        gamma_x: 0.0,
        gamma_phi: 0.0,
    };
    let detuned_parameters = PulseParameters {
        detuning_angular_frequency: 0.05 * rabi_angular_frequency,
        ..resonant_parameters
    };
    let initial = BlochVec::ground();
    let target = SinglePulse::new(target_duration_s).apply_to(resonant_parameters, initial);
    let primitive = SinglePulse::new(target_duration_s).apply_to(detuned_parameters, initial);
    let corpse = Corpse3::new(target_duration_s).apply_to(detuned_parameters, initial);
    let squared_error = |actual: BlochVec| {
        (actual.r.x - target.r.x).powi(2)
            + (actual.r.y - target.r.y).powi(2)
            + (actual.r.z - target.r.z).powi(2)
    };

    assert!(squared_error(corpse) < 0.01 * squared_error(primitive));
}

#[test]
fn zero_target_duration_is_the_identity_with_detuning_and_dephasing() {
    let parameters = PulseParameters {
        rabi_angular_frequency: TAU * 1_900.0,
        detuning_angular_frequency: TAU * 4_700.0,
        gamma_x: 700.0,
        gamma_phi: PI * 1_300.0,
    };
    let mut mixed = BlochVec::ground();
    mixed.r.x = 0.24;
    mixed.r.y = -0.31;
    mixed.r.z = 0.42;

    assert_state_close(
        Corpse3::new(0.0).apply_to(parameters, mixed),
        mixed,
        2.0e-15,
    );
}

#[test]
fn zero_dark_time_ramsey_pair_matches_one_full_duration_pulse() {
    let parameters = PulseParameters {
        rabi_angular_frequency: TAU * 1_900.0,
        detuning_angular_frequency: TAU * 4_700.0,
        gamma_x: 430.0,
        gamma_phi: PI * 1_300.0,
    };
    let duration_s = 310.0e-6;
    let initial = BlochVec::ground();

    assert_state_close(
        RamseyPair::new(duration_s, 0.0).apply_to(parameters, initial),
        SinglePulse::new(duration_s).apply_to(parameters, initial),
        0.0,
    );
}

#[test]
fn counterpropagating_sequence_matches_explicit_branch_evolution() {
    let forward_rabi = 11_000.0;
    let backward_rabi = 8_000.0;
    let gamma_phi = PI * 1_300.0;
    let white_rin_per_hz = 3.0e-10;
    let forward_drive = PulseDrive::new(
        forward_rabi,
        forward_rabi.powi(2) * white_rin_per_hz / 16.0,
        gamma_phi,
    );
    let backward_drive = PulseDrive::new(
        backward_rabi,
        backward_rabi.powi(2) * white_rin_per_hz / 16.0,
        gamma_phi,
    );
    let forward = SinglePulse::new(90.0e-6);
    let backward = RamseyPair::new(140.0e-6, 60.0e-6);
    let sequence = CounterpropagatingPulseSequence::new(
        DrivenPulseSequence::new(forward, forward_drive),
        DrivenPulseSequence::new(backward, backward_drive),
    );

    for detuning_hz in [-9_000.0, 0.0, 13_000.0] {
        for doppler_hz in [-18_000.0, 0.0, 21_000.0] {
            let detunings = MomentumBranchDetunings::from_carrier_doppler_recoil(
                detuning_hz,
                doppler_hz,
                1_750.0,
            );
            let forward_parameters = forward_drive.at_detuning_hz(detunings.forward_hz);
            let backward_lower_parameters =
                backward_drive.at_detuning_hz(detunings.backward_lower_hz);
            let backward_upper_parameters =
                backward_drive.at_detuning_hz(detunings.backward_upper_hz);
            let after_forward = forward.apply_to(forward_parameters, BlochVec::ground());
            let after_backward_from_ground =
                backward.apply_to(backward_lower_parameters, BlochVec::ground());
            let after_backward_from_excited =
                backward.apply_to(backward_upper_parameters, BlochVec::excited());
            let expected = after_forward.ground_probability()
                * after_backward_from_ground.ground_probability()
                + after_forward.excited_probability()
                    * after_backward_from_excited.ground_probability();
            let actual = sequence.ground_probability(detunings);

            assert_close(actual, expected, 1.0e-14);
            assert!((0.0..=1.0).contains(&actual));
        }
    }
}
