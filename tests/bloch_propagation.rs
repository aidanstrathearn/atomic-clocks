use atomic_clocks::maths::Linspace;
use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver, Unitary};
use atomic_clocks::maths::vec3::Vec3;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn unitary_rotates_by_the_physical_angle() {
    use std::f64::consts::{FRAC_PI_2, PI};

    let h = Hamiltonian::new(1.0, 0.0, 0.0);
    let initial = BlochVec::ground();
    for (dt, expected_y, expected_z) in [
        (0.0, 0.0, -1.0),
        (FRAC_PI_2, 1.0, 0.0),
        (-FRAC_PI_2, -1.0, 0.0),
        (PI, 0.0, 1.0),
    ] {
        let actual = Unitary::from_hamiltonian(h, dt).apply_to(initial);
        assert_close(actual.r.x, 0.0);
        assert_close(actual.r.y, expected_y);
        assert_close(actual.r.z, expected_z);
    }
}

#[test]
fn unitary_preserves_length_and_inverse_restores_state() {
    for h in [
        Hamiltonian::new(0.0, 0.0, 0.0),
        Hamiltonian::new(0.4, -0.7, 1.1),
    ] {
        for initial in [
            BlochVec::ground(),
            BlochVec {
                r: Vec3 {
                    x: 0.2,
                    y: -0.3,
                    z: 0.4,
                },
            },
            BlochVec {
                r: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
        ] {
            for dt in [0.0, 1.0e-8, 0.1, 1.7, -0.6] {
                let unitary = Unitary::from_hamiltonian(h, dt);
                let rotated = unitary.apply_to(initial);
                assert_close(rotated.r.norm(), initial.r.norm());
                let restored = unitary.inverse().apply_to(rotated);
                assert_close(restored.r.x, initial.r.x);
                assert_close(restored.r.y, initial.r.y);
                assert_close(restored.r.z, initial.r.z);
            }
        }
    }
}

#[test]
fn composition_applies_the_earlier_rotation_first() {
    let x = Unitary::from_hamiltonian(Hamiltonian::new(1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
    let y = Unitary::from_hamiltonian(Hamiltonian::new(0.0, 1.0, 0.0), std::f64::consts::FRAC_PI_2);
    let result = y.compose(x).apply_to(BlochVec::ground());
    assert_close(result.r.x, 0.0);
    assert_close(result.r.y, 1.0);
    assert_close(result.r.z, 0.0);
}

#[test]
fn composed_propagation_matches_sequential_for_full_and_reduced_grids() {
    let ramsey = Ramsey {
        pulse_area: 1.2,
        detuning: 0.7,
        pulse_width: 0.2,
        pulse_separation: 2.0,
        phase_diff: 0.8,
    };
    for samples in [2, 501, 10_000] {
        for reduced in [false, true] {
            let mut solver = Solver::from(&ramsey, Linspace::new(-1.0, 3.0, samples - 1));
            if reduced {
                solver.trotter_reduce(1.0e-6);
            }
            for initial in [
                BlochVec::ground(),
                BlochVec {
                    r: Vec3 {
                        x: 0.2,
                        y: -0.3,
                        z: 0.4,
                    },
                },
                BlochVec {
                    r: Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                },
            ] {
                let expected = solver.propagate_to_final(initial);
                let actual = solver.propagate_to_final_composed(initial);
                assert_close(actual.r.x, expected.r.x);
                assert_close(actual.r.y, expected.r.y);
                assert_close(actual.r.z, expected.r.z);
                assert_close(actual.r.norm(), initial.r.norm());
            }
        }
    }

    let (mut solver, _) = response_solver();
    solver.h_t.clear();
    let actual = solver.propagate_to_final_composed(BlochVec::ground());
    assert_close(actual.r.x, 0.0);
    assert_close(actual.r.y, 0.0);
    assert_close(actual.r.z, -1.0);
}

fn response_solver() -> (Solver, f64) {
    let times = Linspace::new(-1.0, 3.0, 80);
    let dt = times.step;
    let solver = Solver::from(
        &Ramsey {
            pulse_area: 1.2,
            detuning: 0.7,
            pulse_width: 0.2,
            pulse_separation: 2.0,
            phase_diff: 0.8,
        },
        times,
    );
    (solver, dt)
}

#[test]
fn linear_response_matches_finite_kicks_for_pure_and_mixed_states() {
    let (solver, dt) = response_solver();
    let perturbation = Hamiltonian::new(0.4, -0.7, 1.1);
    let epsilon = 1.0e-5;
    for initial in [
        BlochVec::ground(),
        BlochVec {
            r: Vec3 {
                x: 0.2,
                y: -0.3,
                z: 0.4,
            },
        },
        BlochVec {
            r: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
    ] {
        let responses = solver.linear_response(initial, perturbation);
        for (kick_index, response) in responses.into_iter().enumerate() {
            let probability = |strength| {
                let mut state = initial;
                for (i, &h) in solver.h_t.iter().enumerate() {
                    state = Unitary::from_hamiltonian(h, dt).apply_to(state);
                    if i == kick_index {
                        state = Unitary::from_hamiltonian(perturbation, strength).apply_to(state);
                    }
                }
                state.ground_probability()
            };
            let derivative = (probability(epsilon) - probability(-epsilon)) / (2.0 * epsilon);
            assert!(
                (response - derivative).abs() < 1.0e-8,
                "kick {kick_index}: response {response}, finite difference {derivative}"
            );
        }
    }
}

#[test]
fn linear_response_handles_identity_and_empty_evolution() {
    let (mut solver, _) = response_solver();
    solver.h_t = vec![Hamiltonian::new(0.0, 0.0, 0.0)];
    let initial = BlochVec {
        r: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    };
    let perturbation = Hamiltonian::new(0.0, 1.0, 0.0);
    let response = solver.linear_response(initial, perturbation);
    assert_eq!(response.len(), 1);
    assert_close(response[0], 0.5);

    solver.h_t.clear();
    assert!(solver.linear_response(initial, perturbation).is_empty());
}
