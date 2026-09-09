use std::f64::consts::PI;

use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver, Unitary};

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
    assert_state(
        solver.propagate_to_final(BlochVec::ground()),
        [0.0, 0.0, 1.0],
    );
    assert_state(
        solver.propagate_to_final_composed(BlochVec::ground()),
        [0.0, 0.0, 1.0],
    );
}

#[test]
fn reduction_weights_durations_and_handles_cancelling_and_zero_blocks() {
    let solver = Solver::new(
        vec![
            Hamiltonian::new(2.0, 0.0, 0.0),
            Hamiltonian::new(-1.0, 0.0, 0.0),
            Hamiltonian::new(0.0, 0.0, 0.0),
            Hamiltonian::new(3.0, 0.0, 0.0),
        ],
        vec![1.0, 1.25, 1.75, 3.0, 5.0],
    );
    let reduced = solver.trotter_reduce(1.0e-8);
    assert_eq!(reduced.times(), &[1.0, 5.0]);
    assert_eq!(reduced.hamiltonians().len(), 1);
    // Integrated drive = 2*0.25 - 1*0.5 + 0*1.25 + 3*2 = 6; mean = 6/4.
    assert!((reduced.hamiltonians()[0].norm() - 1.5).abs() < 1.0e-12);
    assert_state(
        reduced.propagate_to_final(BlochVec::ground()),
        [0.0, 6.0_f64.sin(), -6.0_f64.cos()],
    );
    assert_state(
        solver.propagate_to_final(BlochVec::ground()),
        [0.0, 6.0_f64.sin(), -6.0_f64.cos()],
    );
    assert_eq!(solver.times(), &[1.0, 1.25, 1.75, 3.0, 5.0]);
    assert_eq!(solver.hamiltonians().len(), 4);
    assert_eq!(solver.hamiltonians()[0].norm(), 2.0);
}

#[test]
fn reduction_can_be_applied_again_to_nonuniform_blocks() {
    let solver = Solver::new(
        vec![
            Hamiltonian::new(2.0, 0.0, 0.0),
            Hamiltonian::new(1.0, 0.0, 0.0),
            Hamiltonian::new(0.0, 0.5, 0.0),
            Hamiltonian::new(0.0, 1.0, 0.0),
        ],
        vec![0.0, 0.25, 1.0, 1.5, 3.0],
    );
    let reduced = solver.trotter_reduce(1.0e-8);
    assert_eq!(reduced.times(), &[0.0, 1.0, 3.0]);
    assert_eq!(reduced.trotter_reduce(1.0e-8).times(), reduced.times());
    let merged = reduced.trotter_reduce(100.0);
    assert_eq!(merged.times(), &[0.0, 3.0]);
    // The integrated x and y drives are 1.25 and 1.75 respectively.
    let expected = Unitary::from_hamiltonian(Hamiltonian::new(1.25, 1.75, 0.0), 1.0)
        .apply_to(BlochVec::ground());
    assert_state(
        merged.propagate_to_final(BlochVec::ground()),
        [expected.r.x, expected.r.y, expected.r.z],
    );
    assert_eq!(reduced.times(), &[0.0, 1.0, 3.0]);
}

#[test]
fn merge_threshold_accounts_for_both_durations() {
    let make_solver = |times| {
        Solver::new(
            vec![
                Hamiltonian::new(1.0, 0.0, 0.0),
                Hamiltonian::new(0.0, 1.0, 0.0),
            ],
            times,
        )
    };
    // Threshold = 4 * tolerance / 2 = 0.2. The cross-product norm is 1,
    // so the duration products 0.01*10 and 0.1*10 lie on opposite sides.
    assert_eq!(
        make_solver(vec![0.0, 0.01, 10.01])
            .trotter_reduce(0.1)
            .hamiltonians()
            .len(),
        1
    );
    assert_eq!(
        make_solver(vec![0.0, 0.1, 10.1])
            .trotter_reduce(0.1)
            .hamiltonians()
            .len(),
        2
    );
}

#[test]
fn empty_evolution_preserves_its_time_and_initial_state() {
    let solver = Solver::new(vec![], vec![7.0]);
    let reduced = solver.trotter_reduce(1.0e-6);
    assert_eq!(reduced.times(), &[7.0]);
    assert!(reduced.hamiltonians().is_empty());
    assert!(reduced.propagate(BlochVec::ground()).is_empty());
    assert_state(
        reduced.propagate_to_final(BlochVec::ground()),
        [0.0, 0.0, -1.0],
    );
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

#[test]
fn invalid_reduction_tolerances_are_rejected() {
    let solver = Solver::new(vec![], vec![0.0]);
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| solver.trotter_reduce(tolerance)).is_err());
    }
}
