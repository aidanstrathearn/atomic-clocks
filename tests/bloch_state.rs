use atomic_clocks::twolevel::{BlochVec, Channel, DephasingLiouvillian, Hamiltonian, Vec3};

fn state(x: f64, y: f64, z: f64) -> BlochVec {
    BlochVec {
        r: Vec3 { x, y, z },
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "actual {actual}, expected {expected}"
    );
}

fn assert_state(actual: BlochVec, expected: BlochVec) {
    assert_close(actual.r.x, expected.r.x);
    assert_close(actual.r.y, expected.r.y);
    assert_close(actual.r.z, expected.r.z);
}

#[test]
fn z_rotation_uses_the_hamiltonian_sign_convention() {
    let initial = state(0.4, -0.2, 0.7);

    for angle in [0.0, 0.37, -1.2, std::f64::consts::FRAC_PI_2] {
        let expected = Hamiltonian::new(0.0, 0.0, 1.0)
            .for_duration(angle)
            .apply_to(initial);
        assert_state(initial.rotate_about_z(angle), expected);
    }
}

#[test]
fn specialized_and_generic_z_rotations_agree() {
    let z_axis = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    for initial in [
        BlochVec::ground(),
        BlochVec::excited(),
        state(0.4, -0.2, 0.7),
    ] {
        for angle in [0.0, 1.0e-12, 0.37, -1.2] {
            let specialized = initial.rotate_about_z(angle);
            assert_state(specialized, initial.rotate(z_axis, angle));
            assert_state(
                specialized,
                BlochVec {
                    r: initial.r.rotate_about_z(angle),
                },
            );
            assert_eq!(initial.rotate_about_z(angle).r.z, initial.r.z);
        }
    }
}

#[test]
fn rotation_about_an_arbitrary_unit_axis_preserves_length() {
    let axis = Vec3 {
        x: 2.0 / 3.0,
        y: -1.0 / 3.0,
        z: 2.0 / 3.0,
    };
    let initial = state(0.4, -0.2, 0.7);

    for angle in [0.0, 0.37, -1.2, 5.0] {
        assert_close(initial.rotate(axis, angle).r.norm(), initial.r.norm());
    }
}

#[test]
fn dephasing_scales_only_transverse_coherence() {
    let initial = state(0.4, -0.2, 0.7);

    assert_state(initial.dephase(1.0), initial);
    assert_state(initial.dephase(0.5), state(0.2, -0.1, 0.7));
    assert_state(initial.dephase(0.0), state(0.0, 0.0, 0.7));
    assert_eq!(
        initial.dephase(0.5).ground_probability(),
        initial.ground_probability()
    );
    assert_eq!(
        initial.dephase(0.5).excited_probability(),
        initial.excited_probability()
    );
}

#[test]
fn z_rotation_and_dephasing_commute() {
    let initial = state(0.4, -0.2, 0.7);
    let angle = 0.73;
    let coherence = 0.41;

    assert_state(
        initial.rotate_about_z(angle).dephase(coherence),
        initial.dephase(coherence).rotate_about_z(angle),
    );
}

#[test]
fn primitives_match_longitudinal_dephasing_evolution() {
    let initial = state(0.4, -0.2, 0.7);
    let angular_detuning = 1.1;
    let gamma_phi = 0.7;

    for duration in [1.0e-4, 0.2, 1.0, 5.0] {
        let expected = DephasingLiouvillian {
            hamiltonian: Hamiltonian::new(0.0, 0.0, angular_detuning),
            gamma_phi,
        }
        .for_duration(duration)
        .apply_to(initial);
        let coherence = (-gamma_phi * duration).exp();
        let actual = initial
            .rotate_about_z(angular_detuning * duration)
            .dephase(coherence);

        assert_state(actual, expected);
    }
}
