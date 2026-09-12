use atomic_clocks::twolevel::{
    BlochVec, Channel, Decay, DissipativeStep, Hamiltonian, Liouvillian, Vec3, compose_channels,
    propagate, propagate_to_final, steps,
};

fn state(x: f64, y: f64, z: f64) -> BlochVec {
    BlochVec {
        r: Vec3 { x, y, z },
    }
}

fn initial_states() -> [BlochVec; 4] {
    [
        BlochVec::ground(),
        state(0.0, 0.0, 1.0),
        state(0.2, -0.3, 0.4),
        state(0.0, 0.0, 0.0),
    ]
}

fn assert_state(actual: BlochVec, expected: BlochVec) {
    for (a, e) in [actual.r.x, actual.r.y, actual.r.z].into_iter().zip([
        expected.r.x,
        expected.r.y,
        expected.r.z,
    ]) {
        assert!((a - e).abs() < 1e-11, "{a} != {e}");
    }
}

#[test]
fn longitudinal_drive_matches_analytic_relaxation_and_precession() {
    // Nonzero precession separates the transverse eigenvalues, avoiding the
    // degenerate cases whose numerical treatment is deferred.
    let liouvillian = Liouvillian {
        hamiltonian: Hamiltonian::new(0.0, 0.0, 1.1),
        decay: Decay {
            gamma_up: 0.2,
            gamma_down: 0.8,
            gamma_phi: 0.3,
        },
    };
    for dt in [1e-4, 0.2, 1.0, 5.0] {
        let step: DissipativeStep = liouvillian.for_duration(dt);
        let affine = step.to_affine();
        let (sin, cos) = (1.1 * dt).sin_cos();
        let transverse = (-0.8 * dt).exp();
        let longitudinal = (-dt).exp();
        for initial in initial_states() {
            let r = initial.r;
            let expected = state(
                transverse * (r.x * cos - r.y * sin),
                transverse * (r.x * sin + r.y * cos),
                -0.6 + (r.z + 0.6) * longitudinal,
            );
            assert_state(step.apply_to(initial), expected);
            assert_state(affine.apply_to(initial), expected);
        }
    }
}

#[test]
fn isotropic_relaxation_matches_a_damped_unitary_rotation() {
    let decay = Decay {
        gamma_up: 0.5,
        gamma_down: 0.5,
        gamma_phi: 0.5,
    };
    for hamiltonian in [
        Hamiltonian::new(1.0, 0.0, 0.0),
        Hamiltonian::new(0.4, -0.7, 1.1),
    ] {
        for dt in [1e-4, 0.2, 1.0, 5.0] {
            let step = Liouvillian { hamiltonian, decay }.for_duration(dt);
            let affine = step.to_affine();
            let unitary = hamiltonian.for_duration(dt);
            for initial in initial_states() {
                let expected = BlochVec {
                    r: unitary.apply_to(initial).r * (-dt).exp(),
                };
                assert_state(step.apply_to(initial), expected);
                assert_state(affine.apply_to(initial), expected);
            }
        }
    }
}

#[test]
fn driven_steps_match_affine_conversion_and_constant_generator_composition() {
    let liouvillian = Liouvillian {
        hamiltonian: Hamiltonian::new(1.3, -0.4, 0.7),
        decay: Decay {
            gamma_up: 0.2,
            gamma_down: 0.8,
            gamma_phi: 0.3,
        },
    };
    let times = [2.0, 2.1, 2.35, 2.9];
    let channels: Vec<_> = steps(&times, |_, dt| liouvillian.for_duration(dt)).collect();
    let composed = compose_channels(channels.iter().map(Channel::to_affine));
    let one_interval = liouvillian.for_duration(times[3] - times[0]);
    for initial in initial_states() {
        let trajectory: Vec<_> = propagate(&channels, initial).collect();
        assert_eq!(trajectory.len(), times.len() - 1);
        for (&time, actual) in times[1..].iter().zip(&trajectory) {
            assert_state(
                *actual,
                liouvillian.for_duration(time - times[0]).apply_to(initial),
            );
        }
        let expected = one_interval.apply_to(initial);
        assert_state(*trajectory.last().unwrap(), expected);
        assert_state(propagate_to_final(&channels, initial), expected);
        assert_state(composed.apply_to(initial), expected);
        assert_state(one_interval.to_affine().apply_to(initial), expected);
    }
}

#[test]
fn different_dissipative_steps_compose_with_unitary_channels() {
    let first = Liouvillian {
        hamiltonian: Hamiltonian::new(1.3, -0.4, 0.7),
        decay: Decay {
            gamma_up: 0.2,
            gamma_down: 0.8,
            gamma_phi: 0.3,
        },
    }
    .for_duration(0.3);
    let second = Liouvillian {
        hamiltonian: Hamiltonian::new(-0.6, 0.9, -0.5),
        decay: Decay {
            gamma_up: 0.6,
            gamma_down: 0.1,
            gamma_phi: 0.2,
        },
    }
    .for_duration(0.7);
    let pulse = Hamiltonian::new(1.0, 0.0, 0.0).for_duration(0.4);
    let channels: [&dyn Channel; 3] = [&first, &pulse, &second];
    let composed = compose_channels(channels.map(|channel| channel.to_affine()));
    for initial in initial_states() {
        assert_state(
            composed.apply_to(initial),
            propagate_to_final(channels, initial),
        );
    }
}
