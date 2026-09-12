use atomic_clocks::maths::Mat3;
use atomic_clocks::twolevel::{
    AffineChannel, BlochVec, Channel, Decay, Hamiltonian, Liouvillian, Process, Unitary, Vec3,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

fn finite_kick_response<C: Channel>(
    channels: &[C],
    kick_index: usize,
    initial: BlochVec,
    observable: Vec3,
    perturbation: Hamiltonian,
) -> f64 {
    let measure = |duration| {
        let mut state = initial;
        for (i, channel) in channels.iter().enumerate() {
            state = channel.apply_to(state);
            if i == kick_index {
                state = perturbation.for_duration(duration).apply_to(state);
            }
        }
        observable.dot(state.r)
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
            BlochVec::ground().r * 0.5,
            Vec3 {
                x: 0.7,
                y: -0.4,
                z: 0.2,
            },
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        ] {
            for perturbation in [
                Hamiltonian::new(0.4, -0.7, 1.1),
                Hamiltonian::new(-0.8, 0.3, 0.2),
            ] {
                let direct =
                    Process::new(channels).linear_response(initial, observable, perturbation);
                assert_eq!(direct.len(), channels.len());
                for (i, (pair, response)) in pairs.iter().zip(direct).enumerate() {
                    let expected =
                        finite_kick_response(channels, i, initial, observable, perturbation);
                    assert_close(response, expected);
                    assert_close(
                        pair.linear_response(initial, observable, perturbation),
                        expected,
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
        Vec3 {
            x: 0.7,
            y: -0.4,
            z: 0.2,
        },
        Hamiltonian::new(0.4, -0.7, 1.1),
    );
    assert_eq!(response[0], 0.0);
    assert!(response[1].abs() > 1e-3);
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
    let observable = Vec3 {
        x: 0.7,
        y: -0.4,
        z: 0.2,
    };
    let perturbation = Hamiltonian::new(0.4, -0.7, 1.1);
    let response = Process::new(&channels).linear_response(initial, observable, perturbation);
    for (i, actual) in response.into_iter().enumerate() {
        assert_close(
            actual,
            finite_kick_response(&channels, i, initial, observable, perturbation),
        );
    }
}
