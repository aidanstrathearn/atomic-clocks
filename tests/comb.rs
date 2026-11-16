use std::f64::consts::PI;

use atomic_clocks::common::{Comb, CombMode, CombRabiMode, Laser, YB171_CLOCK, YB174_CLOCK};

fn comb() -> Comb {
    Comb {
        laser: Laser {
            power_milliwatts: 5.0,
            waist_radius_mm: 0.6,
            wavelength_nm: 578.0,
            linewidth_fwhm_hz: 200.0,
            white_rin_per_hz: 0.0,
        },
        spacing_hz: 1_000.0,
        modes: vec![
            CombMode {
                index: 2,
                amplitude: 1.0,
                phase_rad: -0.2,
            },
            CombMode {
                index: -3,
                amplitude: 2.0,
                phase_rad: 7.3,
            },
            CombMode {
                index: 0,
                amplitude: 0.0,
                phase_rad: 0.4,
            },
        ],
    }
}

fn assert_close(actual: f64, expected: f64, relative_tolerance: f64) {
    assert!(actual.is_finite(), "nonfinite actual value {actual}");
    let scale = actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= relative_tolerance * scale,
        "actual {actual}, expected {expected}"
    );
}

fn assert_invalid_comb(comb: &Comb) {
    assert!(comb.validate().is_err());
    assert!(comb.mode_offsets_hz().is_err());
    assert!(comb.mode_power_fractions().is_err());
    assert!(comb.mode_powers_milliwatts().is_err());
    assert!(comb.power_spectrum_watts_per_hz(&[0.0]).is_err());
    assert!(YB171_CLOCK.transition.comb_rabi_freqs(comb).is_err());
}

#[test]
fn schroeder_comb_centers_the_teeth_and_uses_ordinals_for_unwrapped_phases() {
    let comb = Comb::schroeder(comb().laser, 1_000.0, 2).unwrap();
    assert_eq!(comb.modes.len(), 5);
    assert_eq!(
        comb.mode_offsets_hz().unwrap(),
        [-2_000.0, -1_000.0, 0.0, 1_000.0, 2_000.0]
    );
    for ((mode, index), phase_in_pi) in comb
        .modes
        .iter()
        .zip([-2, -1, 0, 1, 2])
        .zip([0.0, -0.4, -1.2, -2.4, -4.0])
    {
        assert_eq!(mode.index, index);
        assert_eq!(mode.amplitude, 1.0);
        assert_close(mode.phase_rad, phase_in_pi * PI, 2e-15);
    }
}

#[test]
fn schroeder_comb_distributes_power_and_preserves_phases_in_rabi_couplings() {
    let transition = YB171_CLOCK.transition;
    for side_modes in [0, 1, 3, 8] {
        for power_milliwatts in [0.0, 5.0] {
            let laser = Laser {
                power_milliwatts,
                ..comb().laser
            };
            let comb = Comb::schroeder(laser, 2_500.0, side_modes).unwrap();
            let mode_count = 2 * side_modes + 1;
            assert_eq!(comb.modes.len(), mode_count);
            assert_eq!(comb.modes[side_modes].index, 0);
            let expected_rabi =
                transition.rabi_freq_radians_per_s(&laser) / (mode_count as f64).sqrt();
            for ((resolved, mode), power) in transition
                .comb_rabi_freqs(&comb)
                .unwrap()
                .iter()
                .zip(&comb.modes)
                .zip(comb.mode_powers_milliwatts().unwrap())
            {
                assert_close(power, power_milliwatts / mode_count as f64, 2e-15);
                assert_close(resolved.rabi_angular_frequency, expected_rabi, 2e-15);
                assert_eq!(resolved.index, mode.index);
                assert_eq!(resolved.phase_rad, mode.phase_rad);
            }
        }
    }
}

#[test]
fn schroeder_comb_with_zero_side_modes_matches_a_single_carrier() {
    let laser = Laser {
        linewidth_fwhm_hz: 0.0,
        ..comb().laser
    };
    let comb = Comb::schroeder(laser, 1_000.0, 0).unwrap();
    assert_eq!(comb.modes.len(), 1);
    assert_eq!(comb.modes[0].index, 0);
    assert_eq!(comb.modes[0].phase_rad, 0.0);
    assert_eq!(
        comb.mode_powers_milliwatts().unwrap(),
        [laser.power_milliwatts]
    );
    let transition = YB171_CLOCK.transition;
    assert_eq!(
        transition.comb_rabi_freqs(&comb).unwrap()[0].rabi_angular_frequency,
        transition.rabi_freq_radians_per_s(&laser)
    );
}

#[test]
fn schroeder_constructor_rejects_unrepresentable_counts_and_invalid_parameters() {
    let laser = comb().laser;
    for side_modes in [i32::MAX as usize + 1, usize::MAX] {
        assert!(Comb::schroeder(laser, 1_000.0, side_modes).is_err());
    }
    for spacing in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(Comb::schroeder(laser, spacing, 2).is_err());
    }
    // Reject invalid endpoints before attempting an enormous mode allocation.
    assert!(Comb::schroeder(laser, f64::MAX, i32::MAX as usize).is_err());
    assert!(
        Comb::schroeder(
            Laser {
                waist_radius_mm: 0.0,
                ..laser
            },
            1_000.0,
            1
        )
        .is_err()
    );
}

#[test]
fn discrete_modes_preserve_order_and_distribute_total_power_by_squared_weights() {
    let comb = comb();
    assert_eq!(comb.mode_offsets_hz().unwrap(), [2_000.0, -3_000.0, 0.0]);
    for (fraction, expected) in comb
        .mode_power_fractions()
        .unwrap()
        .into_iter()
        .zip([0.2, 0.8, 0.0])
    {
        assert_close(fraction, expected, 2e-15);
    }
    let powers = comb.mode_powers_milliwatts().unwrap();
    for (&power, expected) in powers.iter().zip([1.0, 4.0, 0.0]) {
        assert_close(power, expected, 2e-15);
    }
    assert_close(powers.iter().sum(), comb.laser.power_milliwatts, 2e-15);
}

#[test]
fn normalization_is_invariant_to_large_and_small_common_amplitude_scales() {
    let reference = comb().mode_power_fractions().unwrap();
    for scale in [1e-300, f64::MAX / 2.0] {
        let mut comb = comb();
        for mode in &mut comb.modes {
            mode.amplitude *= scale;
        }
        for (fraction, &expected) in comb.mode_power_fractions().unwrap().iter().zip(&reference) {
            assert_close(*fraction, expected, 2e-15);
        }
    }
}

#[test]
fn one_tooth_matches_the_existing_laser_coupling_at_any_weight() {
    for amplitude in [1e-300, 1.0, f64::MAX] {
        let mut comb = comb();
        comb.modes.truncate(1);
        comb.modes[0].amplitude = amplitude;
        for transition in [YB171_CLOCK.transition, YB174_CLOCK.transition] {
            let modes: Vec<CombRabiMode> = transition.comb_rabi_freqs(&comb).unwrap();
            assert_eq!(modes.len(), 1);
            assert_eq!(
                modes[0].rabi_angular_frequency,
                transition.rabi_freq_radians_per_s(&comb.laser)
            );
            assert_eq!(modes[0].offset_hz, 2_000.0);
        }
    }
}

#[test]
fn equal_power_teeth_have_inverse_square_root_rabi_scaling() {
    let mut comb = comb();
    for mode in &mut comb.modes {
        mode.amplitude = 1.0;
    }
    let transition = YB171_CLOCK.transition;
    let total_rabi = transition.rabi_freq_radians_per_s(&comb.laser);
    let modes = transition.comb_rabi_freqs(&comb).unwrap();
    for mode in &modes {
        assert_close(
            mode.rabi_angular_frequency,
            total_rabi / 3.0_f64.sqrt(),
            2e-15,
        );
    }
    assert_close(
        modes
            .iter()
            .map(|mode| mode.rabi_angular_frequency.powi(2))
            .sum(),
        total_rabi.powi(2),
        2e-15,
    );
}

#[test]
fn unequal_mode_couplings_match_separate_lasers_and_preserve_metadata() {
    let comb = comb();
    let transition = YB171_CLOCK.transition;
    for ((resolved, mode), power_milliwatts) in transition
        .comb_rabi_freqs(&comb)
        .unwrap()
        .iter()
        .zip(&comb.modes)
        .zip([1.0, 4.0, 0.0])
    {
        assert_eq!(resolved.index, mode.index);
        assert_eq!(resolved.offset_hz, f64::from(mode.index) * comb.spacing_hz);
        assert_eq!(resolved.phase_rad, mode.phase_rad);
        assert_close(
            resolved.rabi_angular_frequency,
            transition.rabi_freq_radians_per_s(&Laser {
                power_milliwatts,
                ..comb.laser
            }),
            2e-15,
        );
    }
}

#[test]
fn very_weak_teeth_retain_representable_field_couplings() {
    let mut comb = comb();
    comb.modes[0].amplitude = 1e-200;
    comb.modes[1].amplitude = 1.0;
    let transition = YB171_CLOCK.transition;
    let weak_rabi = transition.comb_rabi_freqs(&comb).unwrap()[0].rabi_angular_frequency;
    assert!(weak_rabi > 0.0);
    assert_close(
        weak_rabi,
        transition.rabi_freq_radians_per_s(&comb.laser) * 1e-200,
        2e-15,
    );
}

#[test]
fn phases_change_only_the_resolved_phases() {
    let reference = comb();
    let mut changed = reference.clone();
    for (mode, phase) in changed.modes.iter_mut().zip([4.3, -8.9, 10.0]) {
        mode.phase_rad = phase;
    }
    assert_eq!(
        reference.mode_powers_milliwatts().unwrap(),
        changed.mode_powers_milliwatts().unwrap()
    );
    let offsets = [-3_000.0, 0.0, 2_000.0];
    assert_eq!(
        reference.power_spectrum_watts_per_hz(&offsets).unwrap(),
        changed.power_spectrum_watts_per_hz(&offsets).unwrap()
    );
    let transition = YB171_CLOCK.transition;
    for ((reference, changed), mode) in transition
        .comb_rabi_freqs(&reference)
        .unwrap()
        .iter()
        .zip(transition.comb_rabi_freqs(&changed).unwrap())
        .zip(&changed.modes)
    {
        assert_eq!(
            reference.rabi_angular_frequency,
            changed.rabi_angular_frequency
        );
        assert_eq!(changed.phase_rad, mode.phase_rad);
    }
}

#[test]
fn single_tooth_spectrum_has_the_inherited_linewidth_and_correct_peak() {
    let mut comb = comb();
    comb.modes = vec![comb.modes[1]];
    let center = -3_000.0;
    let density = comb
        .power_spectrum_watts_per_hz(&[center, center - 100.0, center + 100.0, center + 200.0])
        .unwrap();
    assert_close(density[0], 0.005 / (PI * 100.0), 2e-15);
    assert_close(density[1], density[0] / 2.0, 2e-15);
    assert_close(density[2], density[0] / 2.0, 2e-15);
    assert_close(density[3], density[0] / 5.0, 2e-15);
}

#[test]
fn spectrum_sums_overlapping_teeth_and_preserves_sampling_order() {
    let comb = comb();
    let offsets = [0.0, -3_000.0, 2_000.0, -3_000.0, 13.0];
    let density = comb.power_spectrum_watts_per_hz(&offsets).unwrap();
    for (&actual, offset) in density.iter().zip(offsets) {
        let expected = 0.001 / (PI * 100.0) / (1.0 + ((offset - 2_000.0) / 100.0).powi(2))
            + 0.004 / (PI * 100.0) / (1.0 + ((offset + 3_000.0) / 100.0).powi(2));
        assert_close(actual, expected, 2e-15);
    }
    assert_eq!(density[1], density[3]);
    assert!(comb.power_spectrum_watts_per_hz(&[]).unwrap().is_empty());
}

#[test]
fn integrated_spectrum_preserves_power_and_does_not_renormalize_a_finite_window() {
    let comb = comb();
    let limit = 20_000.0;
    let offsets: Vec<_> = (-20_000..=20_000).map(f64::from).collect();
    let density = comb.power_spectrum_watts_per_hz(&offsets).unwrap();
    let captured_power: f64 = density
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .sum();
    // The analytic finite-window integral includes the Lorentzian tails omitted
    // by the sampled window, independently of the spectrum implementation.
    let expected_power: f64 = [(2_000.0, 0.001), (-3_000.0, 0.004)]
        .iter()
        .map(|&(center, power)| {
            power / PI
                * (((limit - center) / 100.0_f64).atan() - ((-limit - center) / 100.0_f64).atan())
        })
        .sum();
    assert_close(captured_power, expected_power, 1e-9);
    let total_power = comb.laser.power_milliwatts * 1e-3;
    assert!(captured_power < total_power);
    assert_close(captured_power, total_power, 0.004);
}

#[test]
fn extreme_finite_linewidths_still_produce_the_correct_line_shape() {
    let mut comb = comb();
    comb.modes = vec![CombMode {
        index: 0,
        amplitude: 1.0,
        phase_rad: 0.0,
    }];
    for linewidth in [1e-300, 200.0, 1e300] {
        comb.laser.linewidth_fwhm_hz = linewidth;
        let density = comb
            .power_spectrum_watts_per_hz(&[0.0, linewidth / 2.0, linewidth])
            .unwrap();
        assert_close(density[0], (0.005 * 2.0 / PI) / linewidth, 2e-15);
        assert_close(density[1], density[0] / 2.0, 2e-15);
        assert_close(density[2], density[0] / 5.0, 2e-15);
    }
}

#[test]
fn zero_total_power_gives_zero_mode_powers_spectrum_and_couplings() {
    let mut comb = comb();
    comb.laser.power_milliwatts = 0.0;
    assert!(comb.validate().is_ok());
    assert_eq!(comb.mode_powers_milliwatts().unwrap(), [0.0; 3]);
    assert_eq!(
        comb.power_spectrum_watts_per_hz(&[-3_000.0, 0.0, 2_000.0])
            .unwrap(),
        [0.0; 3]
    );
    for mode in YB171_CLOCK.transition.comb_rabi_freqs(&comb).unwrap() {
        assert_eq!(mode.rabi_angular_frequency, 0.0);
    }
}

#[test]
fn zero_linewidth_supports_discrete_modes_and_couplings_but_not_pointwise_density() {
    let mut comb = comb();
    let transition = YB171_CLOCK.transition;
    let expected = transition.comb_rabi_freqs(&comb).unwrap();
    comb.laser.linewidth_fwhm_hz = 0.0;
    assert!(comb.validate().is_ok());
    assert!(comb.mode_powers_milliwatts().is_ok());
    assert!(comb.power_spectrum_watts_per_hz(&[0.0]).is_err());
    for (mode, expected) in transition
        .comb_rabi_freqs(&comb)
        .unwrap()
        .iter()
        .zip(expected)
    {
        assert_eq!(mode.rabi_angular_frequency, expected.rabi_angular_frequency);
    }
}

#[test]
fn invalid_mode_sets_and_parameters_are_rejected_by_all_calculations() {
    let mut empty = comb();
    empty.modes.clear();
    assert_invalid_comb(&empty);
    let mut duplicate = comb();
    duplicate.modes[0].index = duplicate.modes[1].index;
    assert_invalid_comb(&duplicate);
    let mut all_zero = comb();
    for mode in &mut all_zero.modes {
        mode.amplitude = 0.0;
    }
    assert_invalid_comb(&all_zero);

    for amplitude in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut comb = comb();
        comb.modes[0].amplitude = amplitude;
        assert_invalid_comb(&comb);
    }
    for phase_rad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut comb = comb();
        comb.modes[0].phase_rad = phase_rad;
        assert_invalid_comb(&comb);
    }
    for spacing_hz in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        let mut comb = comb();
        comb.spacing_hz = spacing_hz;
        assert_invalid_comb(&comb);
    }
}

#[test]
fn invalid_embedded_lasers_and_spectrum_queries_are_rejected() {
    let reference = comb();
    for laser in [
        Laser {
            power_milliwatts: -1.0,
            ..reference.laser
        },
        Laser {
            waist_radius_mm: 0.0,
            ..reference.laser
        },
        Laser {
            wavelength_nm: f64::NAN,
            ..reference.laser
        },
        Laser {
            linewidth_fwhm_hz: -1.0,
            ..reference.laser
        },
        Laser {
            white_rin_per_hz: -1.0,
            ..reference.laser
        },
    ] {
        assert_invalid_comb(&Comb {
            laser,
            ..reference.clone()
        });
    }
    for offset in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            reference
                .power_spectrum_watts_per_hz(&[0.0, offset])
                .is_err()
        );
    }
}

#[test]
fn rabi_table_rejects_invalid_transitions_and_nonfinite_couplings() {
    let mut transition = YB171_CLOCK.transition;
    transition.clebsch_gordan_squared = -1.0;
    assert!(transition.comb_rabi_freqs(&comb()).is_err());
    let mut comb = comb();
    comb.laser.waist_radius_mm = f64::MIN_POSITIVE;
    assert!(comb.validate().is_ok());
    assert!(YB171_CLOCK.transition.comb_rabi_freqs(&comb).is_err());
}
