use atomic_clocks::maths::Linspace;
use atomic_clocks::ramsey::Ramsey;
use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver, Unitary};
use atomic_clocks::vec3::Vec3;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "actual {actual}, expected {expected}"
    );
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
