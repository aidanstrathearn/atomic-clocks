use std::f64::consts::PI;

use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver};

fn assert_state(state: BlochVec, expected: [f64; 3]) {
    for (actual, expected) in [state.r.x, state.r.y, state.r.z].into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-11,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn nonuniform_intervals_produce_the_expected_rotations() {
    // Quarter turns about x, z, then y, over three different durations.
    let solver = Solver::new(
        vec![
            Hamiltonian::new(2.0 * PI, 0.0, 0.0),
            Hamiltonian::new(0.0, 0.0, 2.0 * PI / 3.0),
            Hamiltonian::new(0.0, PI / 4.0, 0.0),
        ],
        vec![2.0, 2.25, 3.0, 5.0],
    );
    let trajectory = solver.propagate(BlochVec::ground());
    assert_eq!(trajectory.len(), 3);
    for (state, expected) in
        trajectory
            .into_iter()
            .zip([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    {
        assert_state(state, expected);
    }
}

#[test]
fn empty_evolution_preserves_its_time_and_initial_state() {
    let solver = Solver::new(vec![], vec![7.0]);
    assert_eq!(solver.times(), &[7.0]);
    assert!(solver.hamiltonians().is_empty());
    assert!(solver.propagate(BlochVec::ground()).is_empty());
}

#[test]
fn invalid_boundaries_and_mismatched_lengths_are_rejected() {
    for times in [
        vec![],
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, f64::NAN],
        vec![0.0, f64::INFINITY],
        vec![-f64::MAX, f64::MAX],
        vec![0.0],
    ] {
        assert!(
            std::panic::catch_unwind(|| Solver::new(vec![Hamiltonian::new(1.0, 0.0, 0.0)], times))
                .is_err()
        );
    }
}
