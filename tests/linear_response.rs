use atomic_clocks::maths::Mat3;
use atomic_clocks::twolevel::{
    AffineChannel, BlochVec, Channel, Decay, Hamiltonian, Liouvillian, Observable, Process,
    Unitary, Vec3,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

fn finite_kick_response<C: Channel>(
    channels: &[C],
    kick_index: usize,
    initial: BlochVec,
    observable: Observable,
    perturbation: Hamiltonian,
) -> f64 {
    let measure = |duration| {
        let before = Process::new(&channels[..kick_index]).propagate_to_final(initial);
        let kicked = perturbation.for_duration(duration).apply_to(before);
        let state = Process::new(&channels[kick_index..]).propagate_to_final(kicked);
        observable.expectation(state)
    };
    let epsilon = 1e-5;
    (measure(epsilon) - measure(-epsilon)) / (2.0 * epsilon)
}

fn assert_response_contractions<C: Channel>(channels: &[C]) {
    // Prepare once, then vary all three inputs independently.
    let pairs = Process::new(channels.iter().map(Channel::to_affine)).insertion_pairs();
    assert_eq!(pairs.len(), channels.len());
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
        for observable in [
            Observable::ground_projector(),
            Observable::new(
                0.3,
                Vec3 {
                    x: 0.7,
                    y: -0.4,
                    z: 0.2,
                },
            ),
            Observable::new(
                0.0,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            ),
        ] {
            for perturbation in [
                Hamiltonian::new(0.4, -0.7, 1.1),
                Hamiltonian::new(-0.8, 0.3, 0.2),
            ] {
                let direct = Process::new(channels).linear_response(
                    initial,
                    observable,
                    perturbation.into(),
                );
                assert_eq!(direct.len(), channels.len() + 1);
                for (i, &response) in direct.iter().enumerate() {
                    assert_close(
                        response,
                        finite_kick_response(channels, i, initial, observable, perturbation),
                    );
                }
                for (i, pair) in pairs.iter().enumerate() {
                    assert_close(
                        pair.linear_response(initial, observable, perturbation.into()),
                        direct[i + 1],
                    );

                    // The open slot must be at the specified boundary, even
                    // when tested with a finite kick instead of a derivative.
                    let kick = perturbation.for_duration(0.3);
                    let split = pair
                        .after
                        .apply_to(kick.apply_to(pair.before.apply_to(initial)));
                    let before = Process::new(&channels[..=i]).propagate_to_final(initial);
                    let full =
                        Process::new(&channels[i + 1..]).propagate_to_final(kick.apply_to(before));
                    assert_close((split.r - full.r).norm(), 0.0);
                }
            }
        }
    }
}

#[test]
fn dissipative_and_mixed_channel_responses_match_finite_kicks() {
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
    assert_response_contractions(&[first, second]);

    let pulse = Hamiltonian::new(0.5, -0.3, 1.2).for_duration(0.2);
    let mixed: [&dyn Channel; 3] = [&first, &pulse, &second];
    assert_response_contractions(&mixed);
    assert_response_contractions(&mixed.map(Channel::to_affine));
}

#[test]
fn reset_channel_erases_earlier_responses_without_an_inverse() {
    let pulse = Hamiltonian::new(0.5, -0.3, 1.2)
        .for_duration(0.2)
        .to_affine();
    let reset = AffineChannel::new(Mat3::zero(), BlochVec::ground().r);
    let channels = [pulse, reset, pulse];
    assert_response_contractions(&channels);
    let response = Process::new(channels).linear_response(
        BlochVec::ground(),
        Observable::new(
            0.0,
            Vec3 {
                x: 0.7,
                y: -0.4,
                z: 0.2,
            },
        ),
        Hamiltonian::new(0.4, -0.7, 1.1).into(),
    );
    assert_eq!(response[0], 0.0);
    assert_eq!(response[1], 0.0);
    assert!(response[2].abs() > 1e-3);
}

#[test]
fn direct_response_uses_borrowed_channel_vector_operations() {
    // No Copy, Clone, or ComposableChannel implementation. Affine conversion
    // must not be needed when a channel supplies the vector operations.
    struct PreparedRotation(Unitary);
    impl Channel for PreparedRotation {
        fn apply_to(&self, state: BlochVec) -> BlochVec {
            self.0.apply_to(state)
        }
        fn to_affine(&self) -> AffineChannel {
            panic!("direct response must not construct matrices")
        }
        fn apply_traceless(&self, r: Vec3) -> Vec3 {
            self.0.apply_traceless(r)
        }
        fn pull_back_traceless_observable(&self, o: Vec3) -> Vec3 {
            self.0.pull_back_traceless_observable(o)
        }
    }
    let channels = [
        PreparedRotation(Hamiltonian::new(0.3, -0.2, 1.1).for_duration(0.3)),
        PreparedRotation(Hamiltonian::new(-0.8, 0.9, 0.1).for_duration(0.7)),
    ];
    let initial = BlochVec::ground();
    let observable = Observable::new(
        0.0,
        Vec3 {
            x: 0.7,
            y: -0.4,
            z: 0.2,
        },
    );
    let perturbation = Hamiltonian::new(0.4, -0.7, 1.1);
    let response =
        Process::new(&channels).linear_response(initial, observable, perturbation.into());
    assert_eq!(response.len(), channels.len() + 1);
    for (i, actual) in response.into_iter().enumerate() {
        assert_close(
            actual,
            finite_kick_response(&channels, i, initial, observable, perturbation),
        );
    }
}

#[test]
fn identity_terms_do_not_contribute_to_readout_or_kick_response() {
    let damping = Liouvillian {
        hamiltonian: Hamiltonian::new(1.3, -0.4, 0.7),
        decay: Decay {
            gamma_up: 0.2,
            gamma_down: 0.8,
            gamma_phi: 0.3,
        },
    }
    .for_duration(0.3);
    let pulse = Hamiltonian::new(-0.6, 0.9, -0.5).for_duration(0.7);
    let channels: [&dyn Channel; 2] = [&damping, &pulse];
    let pairs = Process::new(channels.map(Channel::to_affine)).insertion_pairs();
    let initial = BlochVec::ground();
    let observable = Observable::ground_projector();
    let hamiltonian = Hamiltonian::new(0.4, -0.7, 1.1);
    let kick = Observable::from(hamiltonian);
    let identity = Observable::new(
        1.0,
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    );
    let expected: Vec<_> = (0..=channels.len())
        .map(|i| finite_kick_response(&channels, i, initial, observable, hamiltonian))
        .collect();
    assert!(expected.iter().any(|r| r.abs() > 1e-3));

    for scalar in [0.0, 0.5, -2.0] {
        let shifted_readout = Observable::new(scalar, observable.vector());
        let shifted_kick = Observable::new(scalar, kick.vector());
        let direct = Process::new(channels).linear_response(initial, shifted_readout, shifted_kick);
        assert_eq!(direct.len(), expected.len());
        for (actual, expected) in direct.into_iter().zip(&expected) {
            assert_close(actual, *expected);
        }
        for (pair, expected) in pairs.iter().zip(&expected[1..]) {
            assert_close(
                pair.linear_response(initial, shifted_readout, shifted_kick),
                *expected,
            );
            assert_eq!(pair.linear_response(initial, identity, shifted_kick), 0.0);
            assert_eq!(
                pair.linear_response(initial, shifted_readout, identity),
                0.0
            );
        }
    }
    assert!(
        Process::new(channels)
            .linear_response(initial, identity, kick)
            .iter()
            .all(|r| *r == 0.0)
    );
    assert!(
        Process::new(channels)
            .linear_response(initial, observable, identity)
            .iter()
            .all(|r| *r == 0.0)
    );
}
