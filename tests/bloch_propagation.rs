use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::maths::Linspace;
use atomic_clocks::maths::vec3::Vec3;
use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver, TimeDependentHamiltonian, Unitary};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn propagation_covers_exactly_the_requested_intervals() {
    struct ConstantDrive;
    impl TimeDependentHamiltonian for ConstantDrive {
        fn h(&self, _t: f64) -> Hamiltonian {
            Hamiltonian::new(1.0, 0.0, 0.0)
        }
    }

    for nsteps in [1, 2, 80] {
        let times = Linspace::new(-1.0, 1.0, nsteps);
        assert_eq!(times.array.len(), nsteps + 1);
        assert_eq!(
            times.array,
            atomic_clocks::maths::linspace(-1.0, 1.0, nsteps)
        );
        let solver = Solver::from(&ConstantDrive, times.clone());
        let initial = BlochVec::ground();
        let trajectory = solver.propagate(initial);
        assert_eq!(trajectory.len(), nsteps);
        assert_eq!(solver.h_t.len(), nsteps);
        assert_eq!(
            solver
                .linear_response(initial, Hamiltonian::new(0.0, 0.0, 1.0))
                .len(),
            nsteps
        );
        for (state, &time) in trajectory.iter().zip(&times.array[1..]) {
            let elapsed = time + 1.0;
            assert_close(state.r.x, 0.0);
            assert_close(state.r.y, elapsed.sin());
            assert_close(state.r.z, -elapsed.cos());
        }
        for state in [
            solver.propagate_to_final(initial),
            solver.propagate_to_final_composed(initial),
        ] {
            assert_close(state.r.x, 0.0);
            assert_close(state.r.y, 2.0_f64.sin());
            assert_close(state.r.z, -2.0_f64.cos());
        }
    }
}

#[test]
fn propagation_samples_left_endpoints_only() {
    struct RampDrive;
    impl TimeDependentHamiltonian for RampDrive {
        fn h(&self, t: f64) -> Hamiltonian {
            assert!(t < 1.0, "the final boundary must not be sampled");
            Hamiltonian::new(t + 2.0, 0.0, 0.0)
        }
    }

    // Four half-unit intervals sample drive strengths 1, 1.5, 2, and 2.5.
    let solver = Solver::from(&RampDrive, Linspace::new(-1.0, 1.0, 4));
    let state = solver.propagate_to_final(BlochVec::ground());
    assert_close(state.r.x, 0.0);
    assert_close(state.r.y, 3.5_f64.sin());
    assert_close(state.r.z, -3.5_f64.cos());
}

#[test]
fn reduced_step_times_match_post_step_states_and_end_at_stop() {
    struct SwitchedDrive;
    impl TimeDependentHamiltonian for SwitchedDrive {
        fn h(&self, t: f64) -> Hamiltonian {
            if t < 0.0 {
                Hamiltonian::new(1.0, 0.0, 0.0)
            } else {
                Hamiltonian::new(0.0, 1.0, 0.0)
            }
        }
    }

    for nsteps in [1, 4] {
        let times = Linspace::new(-1.0, 1.0, nsteps);
        let expected = Solver::from(&SwitchedDrive, times.clone()).propagate(BlochVec::ground());
        for tolerance in [0.0, 1.0e-6] {
            let mut solver = Solver::from(&SwitchedDrive, times.clone());
            let end_times = solver.trotter_reduce(tolerance);
            let states = solver.propagate(BlochVec::ground());
            assert_eq!(end_times.len(), states.len());
            assert_eq!(*end_times.last().unwrap(), 1.0);
            if tolerance == 0.0 {
                assert_eq!(end_times, times.array[1..]);
            } else if nsteps == 4 {
                assert_eq!(end_times, vec![0.0, 1.0]);
            }
            for (state, time) in states.iter().zip(end_times) {
                let index = times.array[1..].iter().position(|&t| t == time).unwrap();
                assert_close(state.r.x, expected[index].r.x);
                assert_close(state.r.y, expected[index].r.y);
                assert_close(state.r.z, expected[index].r.z);
            }
        }
    }
}

#[test]
#[should_panic(expected = "Linspace requires at least one interval")]
fn linspace_rejects_zero_intervals() {
    Linspace::new(0.0, 1.0, 0);
}

#[test]
#[should_panic(expected = "Linspace requires at least one interval")]
fn linspace_function_rejects_zero_intervals() {
    atomic_clocks::maths::linspace(0.0, 1.0, 0);
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
    for nsteps in [1, 2, 501, 10_000] {
        for reduced in [false, true] {
            let mut solver = Solver::from(&ramsey, Linspace::new(-1.0, 3.0, nsteps));
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
