use std::f64::consts::FRAC_PI_2;

use atomic_clocks::maths::Mat3;
use atomic_clocks::twolevel::{
    AffineChannel, BlochVec, Channel, ComposableChannel, Hamiltonian, Process, Unitary, Vec3,
};

fn assert_state(actual: BlochVec, expected: BlochVec) {
    for (a, e) in [actual.r.x, actual.r.y, actual.r.z].into_iter().zip([
        expected.r.x,
        expected.r.y,
        expected.r.z,
    ]) {
        assert!((a - e).abs() < 1e-11, "{a} != {e}");
    }
}

fn states() -> [BlochVec; 4] {
    [
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
    ]
}

fn amplitude_damping(probability: f64) -> AffineChannel {
    let remaining = 1.0 - probability;
    AffineChannel::new(
        Mat3::diagonal(Vec3 {
            x: remaining.sqrt(),
            y: remaining.sqrt(),
            z: remaining,
        }),
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: -probability,
        },
    )
}

#[test]
fn unitary_affine_conversion_preserves_rotations_of_pure_and_mixed_states() {
    for h in [Hamiltonian::default(), Hamiltonian::new(0.4, -0.7, 1.1)] {
        for dt in [0.0, 1e-8, 0.1, 1.7, -0.6] {
            let unitary = h.for_duration(dt);
            let affine = unitary.to_affine();
            let rotation = affine.linear();
            let orthogonal = rotation.transpose() * rotation;
            for (row, expected) in orthogonal.rows().iter().zip(Mat3::identity().rows()) {
                for (a, e) in row.iter().zip(expected) {
                    assert!((a - e).abs() < 1e-12);
                }
            }
            let shift = affine.shift();
            assert_eq!([shift.x, shift.y, shift.z], [0.0; 3]);
            for initial in states() {
                assert_state(affine.apply_to(initial), unitary.apply_to(initial));
                assert_state(
                    Unitary::from_hamiltonian(h, dt).apply_to(initial),
                    unitary.apply_to(initial),
                );
            }
        }
    }
}

#[test]
fn affine_translation_describes_relaxation_towards_ground() {
    let decay = amplitude_damping(0.75);
    let [ground, excited, mixed, unpolarized] = states();
    assert_state(decay.apply_to(ground), ground);
    assert_state(
        decay.apply_to(excited),
        BlochVec {
            r: Vec3 {
                x: 0.0,
                y: 0.0,
                z: -0.5,
            },
        },
    );
    assert_state(
        decay.apply_to(mixed),
        BlochVec {
            r: Vec3 {
                x: 0.1,
                y: -0.15,
                z: -0.65,
            },
        },
    );
    assert_state(
        decay.apply_to(unpolarized),
        BlochVec {
            r: Vec3 {
                x: 0.0,
                y: 0.0,
                z: -0.75,
            },
        },
    );
}

#[test]
fn composition_rotates_the_earlier_translation() {
    let reset = amplitude_damping(1.0);
    let rotate = Hamiltonian::new(1.0, 0.0, 0.0)
        .for_duration(FRAC_PI_2)
        .to_affine();
    for initial in states() {
        assert_state(
            rotate.compose(&reset).apply_to(initial),
            BlochVec {
                r: Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            },
        );
        assert_state(reset.compose(&rotate).apply_to(initial), BlochVec::ground());
    }
}

#[test]
fn affine_composition_matches_sequential_evolution_and_is_associative() {
    let a = amplitude_damping(0.3);
    let b = Hamiltonian::new(0.4, -0.7, 1.1)
        .for_duration(0.8)
        .to_affine();
    let c = amplitude_damping(0.6);
    let total = Process::new([a, b, c]).compose();
    for initial in states() {
        let expected = Process::new([a, b, c]).propagate_to_final(initial);
        assert_state(total.apply_to(initial), expected);
        assert_state(c.compose(&b.compose(&a)).apply_to(initial), expected);
        assert_state(c.compose(&b).compose(&a).apply_to(initial), expected);
        assert_state(
            total.compose(&AffineChannel::identity()).apply_to(initial),
            expected,
        );
        assert_state(
            AffineChannel::identity().compose(&total).apply_to(initial),
            expected,
        );
        assert_state(total.to_affine().apply_to(initial), expected);
    }
}

#[test]
fn different_channel_representations_can_be_applied_or_composed_as_affine() {
    let rotate = Hamiltonian::new(1.0, 0.0, 0.0).for_duration(FRAC_PI_2);
    let decay = amplitude_damping(0.75);
    let channels: [&dyn Channel; 2] = [&rotate, &decay];
    let total = Process::new(channels.map(|channel| channel.to_affine())).compose();
    for initial in states() {
        assert_state(
            total.apply_to(initial),
            Process::new(channels).propagate_to_final(initial),
        );
    }
}

#[test]
fn composing_many_unitaries_preserves_norm_and_matches_affine_composition() {
    let channels = || {
        (0..20_000)
            .map(|i| Hamiltonian::new(0.4, (i as f64 * 0.001).sin(), 1.1).for_duration(0.001))
    };
    let unitary = Process::new(channels()).compose();
    let affine = Process::new(channels().map(|channel| channel.to_affine())).compose();
    for initial in states() {
        let actual = unitary.apply_to(initial);
        assert!((actual.r.norm() - initial.r.norm()).abs() < 1e-12);
        assert_state(actual, Process::new(channels()).propagate_to_final(initial));
        assert_state(actual, affine.apply_to(initial));
        assert_state(unitary.inverse().apply_to(actual), initial);
    }
}
