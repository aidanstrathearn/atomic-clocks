use std::f64::consts::PI;

use atomic_clocks::twolevel::{
    AffineChannel, BlochVec, Channel, ComposableChannel, Hamiltonian, Unitary, compose_channels,
    propagate, propagate_to_final, steps,
};

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
    let hamiltonians = [
        Hamiltonian::new(2.0 * PI, 0.0, 0.0),
        Hamiltonian::new(0.0, 0.0, 2.0 * PI / 3.0),
        Hamiltonian::new(0.0, PI / 4.0, 0.0),
    ];
    let times = [2.0, 2.25, 3.0, 5.0];
    let mut generators = hamiltonians.into_iter();
    let channels = steps(&times, |_, dt| generators.next().unwrap().for_duration(dt));
    let trajectory: Vec<_> = propagate(channels, BlochVec::ground()).collect();
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
    let channels = || {
        steps(&[7.0], |_, _| -> Unitary {
            panic!("no interval to sample")
        })
    };
    assert_eq!(channels().len(), 0);
    assert_eq!(propagate(channels(), BlochVec::ground()).count(), 0);
    assert_state(
        propagate_to_final(channels(), BlochVec::ground()),
        [0.0, 0.0, -1.0],
    );
    assert_state(
        compose_channels(channels()).apply_to(BlochVec::ground()),
        [0.0, 0.0, -1.0],
    );
    assert_state(
        compose_channels::<AffineChannel>([]).apply_to(BlochVec::ground()),
        [0.0, 0.0, -1.0],
    );
}

#[test]
fn invalid_boundaries_are_rejected_before_sampling() {
    for times in [
        vec![],
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, f64::NAN],
        vec![0.0, f64::INFINITY],
        vec![-f64::MAX, f64::MAX],
        vec![f64::NAN],
        vec![f64::NEG_INFINITY],
    ] {
        let sampled = std::cell::Cell::new(false);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _channels = steps(&times, |_, _| {
                    sampled.set(true);
                    Unitary::identity()
                });
            }))
            .is_err()
        );
        assert!(!sampled.get());
    }
}

#[test]
fn sampling_and_propagation_are_lazy_and_use_absolute_left_endpoints() {
    let sampled = std::cell::RefCell::new(Vec::new());
    let times = [2.0, 2.25, 3.0, 5.0];
    let channels = steps(&times, |t, dt| {
        sampled.borrow_mut().push((t, dt));
        Hamiltonian::new(1.0, 0.0, 0.0).for_duration(dt)
    });
    assert!(sampled.borrow().is_empty());
    let mut trajectory = propagate(channels, BlochVec::ground());
    assert!(sampled.borrow().is_empty());
    assert_state(
        trajectory.next().unwrap(),
        [0.0, 0.25_f64.sin(), -0.25_f64.cos()],
    );
    assert_eq!(*sampled.borrow(), [(2.0, 0.25)]);
    assert_state(
        trajectory.last().unwrap(),
        [0.0, 3.0_f64.sin(), -3.0_f64.cos()],
    );
    assert_eq!(*sampled.borrow(), [(2.0, 0.25), (2.25, 0.75), (3.0, 2.0)]);
}

#[test]
fn warmup_and_recording_share_the_window_boundary() {
    let times = [2.0, 2.25, 3.0, 5.0];
    let step_at = |t: f64, dt| Hamiltonian::new(t, 0.0, 0.0).for_duration(dt);
    let initial = BlochVec::ground();
    let full: Vec<_> = std::iter::once(initial)
        .chain(propagate(steps(&times, step_at), initial))
        .collect();
    for start in 0..times.len() {
        let warmed = propagate_to_final(steps(&times[..=start], step_at), initial);
        let window: Vec<_> = std::iter::once(warmed)
            .chain(propagate(steps(&times[start..], step_at), warmed))
            .collect();
        assert_eq!(window.len(), times.len() - start);
        for (actual, expected) in window.into_iter().zip(&full[start..]) {
            assert_state(actual, [expected.r.x, expected.r.y, expected.r.z]);
        }
    }
}

#[test]
fn propagation_accepts_borrowed_channels_without_composition() {
    // This representation deliberately implements Channel alone and isn't Copy.
    struct PreparedRotation(Unitary);
    impl Channel for PreparedRotation {
        fn apply_to(&self, state: BlochVec) -> BlochVec {
            self.0.apply_to(state)
        }
        fn to_affine(&self) -> AffineChannel {
            self.0.to_affine()
        }
    }
    let channels = [PreparedRotation(
        Hamiltonian::new(1.0, 0.0, 0.0).for_duration(PI / 2.0),
    )];
    assert_state(
        propagate_to_final(&channels, BlochVec::ground()),
        [0.0, 1.0, 0.0],
    );
    assert_state(
        propagate(&channels, BlochVec::ground()).next().unwrap(),
        [0.0, 1.0, 0.0],
    );
}
