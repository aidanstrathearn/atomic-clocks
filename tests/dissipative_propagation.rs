use atomic_clocks::twolevel::{
    BlochVec, Channel, Decay, DephasingLiouvillian, DephasingStep, DissipativeStep, Hamiltonian,
    Liouvillian, Process, Vec3, steps,
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
    let composed = Process::new(channels.iter().map(Channel::to_affine)).compose();
    let one_interval = liouvillian.for_duration(times[3] - times[0]);
    for initial in initial_states() {
        let trajectory: Vec<_> = Process::new(&channels).propagate(initial).collect();
        assert_eq!(trajectory.len(), times.len() - 1);
        for (&time, actual) in times[1..].iter().zip(&trajectory) {
            assert_state(
                *actual,
                liouvillian.for_duration(time - times[0]).apply_to(initial),
            );
        }
        let expected = one_interval.apply_to(initial);
        assert_state(*trajectory.last().unwrap(), expected);
        assert_state(
            Process::new(&channels).propagate_to_final(initial),
            expected,
        );
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
    let composed = Process::new(channels.map(|channel| channel.to_affine())).compose();
    for initial in initial_states() {
        assert_state(
            composed.apply_to(initial),
            Process::new(channels).propagate_to_final(initial),
        );
    }
}

#[test]
fn pure_dephasing_without_a_hamiltonian_damps_only_transverse_components() {
    let initial = state(0.2, -0.3, 0.4);
    for dt in [0.0, 0.2, 1.0, 5.0] {
        let step = DephasingLiouvillian {
            hamiltonian: Hamiltonian::default(),
            gamma_phi: 0.7,
        }
        .for_duration(dt);
        let transverse = (-0.7 * dt).exp();
        let expected = state(
            transverse * initial.r.x,
            transverse * initial.r.y,
            initial.r.z,
        );
        assert_state(step.apply_to(initial), expected);
        assert_state(step.to_affine().apply_to(initial), expected);
    }
}

#[test]
fn longitudinal_hamiltonian_gives_damped_transverse_precession() {
    let initial = state(0.2, -0.3, 0.4);
    let h = 1.1;
    let gamma_phi = 0.7;
    for dt in [1e-4, 0.2, 1.0, 5.0] {
        let step = DephasingLiouvillian {
            hamiltonian: Hamiltonian::new(0.0, 0.0, h),
            gamma_phi,
        }
        .for_duration(dt);
        let (sin, cos) = (h * dt).sin_cos();
        let transverse = (-gamma_phi * dt).exp();
        let expected = state(
            transverse * (initial.r.x * cos - initial.r.y * sin),
            transverse * (initial.r.x * sin + initial.r.y * cos),
            initial.r.z,
        );
        assert_state(step.apply_to(initial), expected);
    }
}

#[test]
fn pure_dephasing_handles_a_defective_repeated_eigenvalue() {
    // For gamma_phi = 2 and H = sigma_x / 2, the yz block has a repeated
    // eigenvalue -1 and is not diagonalisable.
    let initial = state(0.2, -0.3, 0.4);
    for dt in [1e-4, 0.2, 1.0, 5.0] {
        let step = DephasingLiouvillian {
            hamiltonian: Hamiltonian::new(1.0, 0.0, 0.0),
            gamma_phi: 2.0,
        }
        .for_duration(dt);
        let block_scale = (-dt).exp();
        let expected = state(
            (-2.0 * dt).exp() * initial.r.x,
            block_scale * ((1.0 - dt) * initial.r.y - dt * initial.r.z),
            block_scale * (dt * initial.r.y + (1.0 + dt) * initial.r.z),
        );
        assert_state(step.apply_to(initial), expected);
    }
}

#[test]
fn general_pure_dephasing_is_unital_composable_and_matches_its_affine_form() {
    let hamiltonian = Hamiltonian::new(1.3, -0.4, 0.7);
    let gamma_phi = 0.3;
    let liouvillian = DephasingLiouvillian {
        hamiltonian,
        gamma_phi,
    };
    let first: DephasingStep = liouvillian.for_duration(0.3);
    let second = liouvillian.for_duration(0.7);
    let combined = liouvillian.for_duration(1.0);

    assert_state(first.apply_to(state(0.0, 0.0, 0.0)), state(0.0, 0.0, 0.0));
    for initial in initial_states() {
        assert_state(
            second.apply_to(first.apply_to(initial)),
            combined.apply_to(initial),
        );
        assert_state(first.to_affine().apply_to(initial), first.apply_to(initial));
    }

    let r = Vec3 {
        x: 0.2,
        y: -0.3,
        z: 0.4,
    };
    let o = Vec3 {
        x: -0.5,
        y: 0.7,
        z: 0.1,
    };
    let forward = first.apply_traceless(r);
    let backward = first.pull_back_traceless_observable(o);
    assert!((o.dot(forward) - backward.dot(r)).abs() < 1e-14);
}

#[test]
fn pure_dephasing_matches_the_nonsingular_dissipative_limit() {
    let hamiltonian = Hamiltonian::new(1.3, -0.4, 0.7);
    let gamma_phi = 0.3;
    let dephasing = DephasingLiouvillian {
        hamiltonian,
        gamma_phi,
    };
    let dissipative = Liouvillian {
        hamiltonian,
        decay: Decay {
            gamma_up: 0.0,
            gamma_down: 0.0,
            gamma_phi,
        },
    };
    let traceless = Vec3 {
        x: 0.2,
        y: -0.3,
        z: 0.4,
    };
    let observable = Vec3 {
        x: -0.5,
        y: 0.7,
        z: 0.1,
    };

    for dt in [0.0, 1e-4, 0.2, 1.0, 5.0] {
        let dephasing_step = dephasing.for_duration(dt);
        let dissipative_step = dissipative.for_duration(dt);

        for initial in initial_states() {
            assert_state(
                dephasing_step.apply_to(initial),
                dissipative_step.apply_to(initial),
            );
            assert_state(
                dephasing_step.to_affine().apply_to(initial),
                dissipative_step.to_affine().apply_to(initial),
            );
        }
        assert_state(
            BlochVec {
                r: dephasing_step.apply_traceless(traceless),
            },
            BlochVec {
                r: dissipative_step.apply_traceless(traceless),
            },
        );
        assert_state(
            BlochVec {
                r: dephasing_step.pull_back_traceless_observable(observable),
            },
            BlochVec {
                r: dissipative_step.pull_back_traceless_observable(observable),
            },
        );
    }
}

#[test]
fn zero_dephasing_matches_unitary_evolution() {
    let hamiltonian = Hamiltonian::new(0.4, -0.7, 1.1);
    let liouvillian = DephasingLiouvillian {
        hamiltonian,
        gamma_phi: 0.0,
    };
    for dt in [0.0, 1e-4, 0.2, 1.0, 5.0] {
        let actual = liouvillian.for_duration(dt);
        let expected = hamiltonian.for_duration(dt);
        for initial in initial_states() {
            assert_state(actual.apply_to(initial), expected.apply_to(initial));
        }
    }
}
