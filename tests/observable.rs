use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Observable, Process, Vec3};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn projector_readout_matches_probabilities_along_a_trajectory() {
    let ground = Observable::ground_projector();
    let excited = Observable::new(
        0.5,
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.5,
        },
    );
    let channels = [
        Hamiltonian::new(0.4, -0.7, 1.1).for_duration(0.3),
        Hamiltonian::new(-0.8, 0.3, 0.2).for_duration(0.7),
    ];
    for initial in [
        BlochVec::ground(),
        BlochVec {
            r: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        },
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
        for state in Process::new(&channels).trajectory(initial) {
            assert_close(ground.expectation(state), state.ground_probability());
            assert_close(excited.expectation(state), state.excited_probability());
            assert_close(ground.expectation(state) + excited.expectation(state), 1.0);
        }
    }
}

#[test]
fn traceless_pairing_matches_a_difference_of_state_readouts() {
    let first = BlochVec {
        r: Vec3 {
            x: 0.2,
            y: -0.3,
            z: 0.4,
        },
    };
    let second = BlochVec::ground();
    for scalar in [0.0, 0.5, -2.0] {
        let observable = Observable::new(
            scalar,
            Vec3 {
                x: 0.7,
                y: -0.4,
                z: 0.2,
            },
        );
        assert_close(
            observable.pair_traceless(first.r - second.r),
            observable.expectation(first) - observable.expectation(second),
        );
    }
}

#[test]
fn hamiltonian_conversion_preserves_energy_eigenvalues() {
    let h = Vec3 {
        x: 0.4,
        y: -0.7,
        z: 1.1,
    };
    let hamiltonian = Hamiltonian::new(h.x, h.y, h.z);
    let observable = Observable::from(hamiltonian);
    assert_eq!(observable.scalar(), 0.0);
    let axis = h * (1.0 / h.norm());
    assert_close(
        observable.expectation(BlochVec { r: axis }),
        hamiltonian.norm() / 2.0,
    );
    assert_close(
        observable.expectation(BlochVec { r: axis * -1.0 }),
        -hamiltonian.norm() / 2.0,
    );
}
