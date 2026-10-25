use atomic_clocks::twolevel::{
    AffineChannel, BlochLiouvillian, BlochVec, Channel, ComposableChannel, Decay, Hamiltonian,
    Liouvillian, Observable, PauliDephasingLiouvillian, Process, Vec3,
};

fn state(x: f64, y: f64, z: f64) -> BlochVec {
    BlochVec {
        r: Vec3 { x, y, z },
    }
}

fn initial_states() -> [BlochVec; 4] {
    [
        BlochVec::ground(),
        BlochVec::excited(),
        state(0.2, -0.3, 0.4),
        state(0.0, 0.0, 0.0),
    ]
}

fn mixed_generator() -> BlochLiouvillian {
    BlochLiouvillian {
        hamiltonian: Hamiltonian::new(1.3, -0.4, 0.7),
        gamma_up: 0.2,
        gamma_down: 0.8,
        gamma_x: 0.35,
        gamma_y: 0.2,
        gamma_z: 0.3,
    }
}

fn assert_vec(actual: Vec3, expected: Vec3, tolerance: f64) {
    for (a, e) in [actual.x, actual.y, actual.z]
        .into_iter()
        .zip([expected.x, expected.y, expected.z])
    {
        assert!(
            (a - e).abs() <= tolerance,
            "{a} != {e} (tolerance {tolerance})"
        );
    }
}

#[test]
fn zero_duration_is_exactly_the_identity() {
    let channel: AffineChannel = mixed_generator().for_duration(0.0);
    assert_eq!(
        channel.linear().rows(),
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    assert_vec(channel.shift(), state(0.0, 0.0, 0.0).r, 0.0);
    for initial in initial_states() {
        assert_vec(channel.apply_to(initial).r, initial.r, 0.0);
    }
}

#[test]
fn zero_dissipation_matches_unitary_evolution() {
    for hamiltonian in [Hamiltonian::default(), Hamiltonian::new(0.4, -0.7, 1.1)] {
        let generator = BlochLiouvillian {
            hamiltonian,
            gamma_up: 0.0,
            gamma_down: 0.0,
            gamma_x: 0.0,
            gamma_y: 0.0,
            gamma_z: 0.0,
        };
        for dt in [0.0, 1e-12, 0.2, 5.0, 1_000.0] {
            let actual = generator.for_duration(dt);
            let expected = hamiltonian.for_duration(dt);
            for initial in initial_states() {
                assert_vec(
                    actual.apply_to(initial).r,
                    expected.apply_to(initial).r,
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn undriven_population_transfer_and_pauli_damping_match_analytic_evolution() {
    for (gamma_up, gamma_down) in [(0.0, 0.0), (0.0, 0.8), (0.8, 0.0), (0.2, 0.8)] {
        for (gamma_x, gamma_y, gamma_z) in [(0.0, 0.0, 0.0), (0.2, 0.3, 0.5)] {
            let generator = BlochLiouvillian {
                hamiltonian: Hamiltonian::default(),
                gamma_up,
                gamma_down,
                gamma_x,
                gamma_y,
                gamma_z,
            };
            let dx = 0.5 * (gamma_up + gamma_down) + gamma_y + gamma_z;
            let dy = 0.5 * (gamma_up + gamma_down) + gamma_x + gamma_z;
            let dz = gamma_up + gamma_down + gamma_x + gamma_y;
            for dt in [0.0, 1e-12, 0.2, 5.0, 1_000.0] {
                let shift = if dz == 0.0 {
                    0.0
                } else {
                    (gamma_up - gamma_down) / dz * -(-dz * dt).exp_m1()
                };
                let channel = generator.for_duration(dt);
                for initial in initial_states() {
                    let r = initial.r;
                    let expected = state(
                        (-dx * dt).exp() * r.x,
                        (-dy * dt).exp() * r.y,
                        (-dz * dt).exp() * r.z + shift,
                    );
                    assert_vec(channel.apply_to(initial).r, expected.r, 1e-11);
                }
            }
        }
    }
}

#[test]
fn pauli_only_limit_matches_the_existing_propagator() {
    for hamiltonian in [
        Hamiltonian::default(),
        Hamiltonian::new(0.0, 0.0, 1.1),
        Hamiltonian::new(0.4, -0.7, 1.1),
    ] {
        for (gamma_x, gamma_y, gamma_z) in [(0.0, 0.0, 0.7), (0.2, 0.3, 0.5)] {
            let generator = BlochLiouvillian {
                hamiltonian,
                gamma_up: 0.0,
                gamma_down: 0.0,
                gamma_x,
                gamma_y,
                gamma_z,
            };
            let legacy = PauliDephasingLiouvillian {
                hamiltonian,
                gamma_x,
                gamma_y,
                gamma_z,
            };
            for dt in [1e-8, 0.2, 5.0] {
                let actual = generator.for_duration(dt);
                let expected = legacy.for_duration(dt);
                for initial in initial_states() {
                    assert_vec(
                        actual.apply_to(initial).r,
                        expected.apply_to(initial).r,
                        1e-11,
                    );
                }
            }
        }
    }
}

#[test]
fn decay_and_z_dephasing_match_the_well_conditioned_legacy_limit() {
    let hamiltonian = Hamiltonian::new(1.3, -0.4, 0.7);
    for (gamma_up, gamma_down, gamma_z) in [(0.2, 0.8, 0.3), (0.5, 0.5, 0.5)] {
        let generator = BlochLiouvillian {
            hamiltonian,
            gamma_up,
            gamma_down,
            gamma_x: 0.0,
            gamma_y: 0.0,
            gamma_z,
        };
        let legacy = Liouvillian {
            hamiltonian,
            decay: Decay {
                gamma_up,
                gamma_down,
                gamma_phi: gamma_z,
            },
        };
        for dt in [1e-4, 0.2, 1.0, 5.0] {
            let actual = generator.for_duration(dt);
            let expected = legacy.for_duration(dt);
            for initial in initial_states() {
                assert_vec(
                    actual.apply_to(initial).r,
                    expected.apply_to(initial).r,
                    1e-11,
                );
            }
        }
    }
}

#[test]
fn critical_damping_with_a_population_source_matches_the_jordan_solution() {
    // Omega = 1 and gamma_down = 4 give a defective yz block with eigenvalue -3.
    // A_yz + 3 I has square zero. The affine steady state is (0, 4/9, -8/9).
    let generator = BlochLiouvillian {
        hamiltonian: Hamiltonian::new(1.0, 0.0, 0.0),
        gamma_up: 0.0,
        gamma_down: 4.0,
        gamma_x: 0.0,
        gamma_y: 0.0,
        gamma_z: 0.0,
    };
    for dt in [0.0, 1e-9, 0.2, 1.0, 10.0] {
        let channel = generator.for_duration(dt);
        for initial in initial_states() {
            let wy = initial.r.y - 4.0 / 9.0;
            let wz = initial.r.z + 8.0 / 9.0;
            let decay = (-3.0 * dt).exp();
            let expected = state(
                initial.r.x * (-2.0 * dt).exp(),
                4.0 / 9.0 + decay * (wy + dt * (wy - wz)),
                -8.0 / 9.0 + decay * (wz + dt * (wy - wz)),
            );
            assert_vec(channel.apply_to(initial).r, expected.r, 1e-12);
            for perturbation in [-1e-10, 1e-10] {
                let nearby = BlochLiouvillian {
                    gamma_down: generator.gamma_down * (1.0 + perturbation),
                    ..generator
                };
                assert_vec(
                    nearby.for_duration(dt).apply_to(initial).r,
                    expected.r,
                    1e-9,
                );
            }
        }
    }
}

#[test]
fn time_unit_rescaling_preserves_the_channel() {
    for base in [
        mixed_generator(),
        BlochLiouvillian {
            gamma_up: 0.0,
            gamma_down: 0.0,
            gamma_x: 0.0,
            gamma_y: 0.0,
            gamma_z: 0.0,
            ..mixed_generator()
        },
    ] {
        let expected = base.for_duration(0.7);
        for scale in [1e-300, 1e-12, 1.0, 1e12, 1e300] {
            let h = Observable::from(base.hamiltonian).vector() * 2.0;
            let scaled = BlochLiouvillian {
                hamiltonian: Hamiltonian::new(h.x * scale, h.y * scale, h.z * scale),
                gamma_up: base.gamma_up * scale,
                gamma_down: base.gamma_down * scale,
                gamma_x: base.gamma_x * scale,
                gamma_y: base.gamma_y * scale,
                gamma_z: base.gamma_z * scale,
            };
            let actual = scaled.for_duration(0.7 / scale);
            for initial in initial_states() {
                assert_vec(
                    actual.apply_to(initial).r,
                    expected.apply_to(initial).r,
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn affine_channels_compose_and_keep_traceless_propagation_separate_from_the_source() {
    let generator = mixed_generator();
    let first = generator.for_duration(0.37);
    let second = generator.for_duration(0.83);
    let combined = generator.for_duration(1.2);
    let composed = second.compose(&first);
    for initial in initial_states() {
        let expected = combined.apply_to(initial);
        assert_vec(composed.apply_to(initial).r, expected.r, 1e-12);
        assert_vec(
            Process::new([first, second]).propagate_to_final(initial).r,
            expected.r,
            1e-12,
        );
        assert_vec(combined.to_affine().apply_to(initial).r, expected.r, 0.0);
    }
    let r = state(0.2, -0.3, 0.4).r;
    let observable = state(-0.5, 0.7, 0.1).r;
    let traceless = combined.apply_traceless(r);
    let difference =
        combined.apply_to(BlochVec { r }).r - combined.apply_to(state(0.0, 0.0, 0.0)).r;
    assert_vec(traceless, difference, 1e-15);
    let forward = observable.dot(traceless);
    let backward = combined.pull_back_traceless_observable(observable).dot(r);
    assert!((forward - backward).abs() < 1e-15);
}

#[test]
fn invalid_rates_hamiltonians_and_durations_are_rejected() {
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        for index in 0..5 {
            let mut generator = mixed_generator();
            match index {
                0 => generator.gamma_up = value,
                1 => generator.gamma_down = value,
                2 => generator.gamma_x = value,
                3 => generator.gamma_y = value,
                _ => generator.gamma_z = value,
            }
            assert!(generator.validate().is_err());
            assert!(std::panic::catch_unwind(|| generator.for_duration(0.0)).is_err());
        }
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..3 {
            let mut generator = mixed_generator();
            let mut components = [1.3, -0.4, 0.7];
            components[index] = value;
            generator.hamiltonian = Hamiltonian::new(components[0], components[1], components[2]);
            assert!(generator.validate().is_err());
            assert!(std::panic::catch_unwind(|| generator.for_duration(0.1)).is_err());
        }
    }
    let overflow = BlochLiouvillian {
        gamma_up: f64::MAX,
        gamma_down: f64::MAX,
        ..mixed_generator()
    };
    assert!(overflow.validate().is_err());
    for dt in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| mixed_generator().for_duration(dt)).is_err());
    }
}

// Independent reference: integrate the 2x2 density-matrix Lindblad equation.
// Use (excited, ground) basis order and explicit complex matrix operations,
// rather than constructing the production Bloch generator.
type Complex = (f64, f64);
type Matrix = [[Complex; 2]; 2];

fn complex_add(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn matrix_add(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| complex_add(a[i][j], b[i][j])))
}

fn matrix_scale(a: Matrix, factor: Complex) -> Matrix {
    a.map(|row| row.map(|entry| complex_mul(entry, factor)))
}

fn matrix_mul(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            complex_add(complex_mul(a[i][0], b[0][j]), complex_mul(a[i][1], b[1][j]))
        })
    })
}

fn adjoint(a: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| (a[j][i].0, -a[j][i].1)))
}

fn density_derivative(generator: BlochLiouvillian, rho: Matrix) -> Matrix {
    let h = Observable::from(generator.hamiltonian).vector() * 2.0;
    let hamiltonian = [
        [(0.5 * h.z, 0.0), (0.5 * h.x, -0.5 * h.y)],
        [(0.5 * h.x, 0.5 * h.y), (-0.5 * h.z, 0.0)],
    ];
    let commutator = matrix_add(
        matrix_mul(hamiltonian, rho),
        matrix_scale(matrix_mul(rho, hamiltonian), (-1.0, 0.0)),
    );
    let mut derivative = matrix_scale(commutator, (0.0, -1.0));
    let zero = (0.0, 0.0);
    let one = (1.0, 0.0);
    let jumps = [
        ([[zero, one], [zero, zero]], generator.gamma_up),
        ([[zero, zero], [one, zero]], generator.gamma_down),
        ([[zero, one], [one, zero]], 0.5 * generator.gamma_x),
        (
            [[zero, (0.0, -1.0)], [(0.0, 1.0), zero]],
            0.5 * generator.gamma_y,
        ),
        ([[one, zero], [zero, (-1.0, 0.0)]], 0.5 * generator.gamma_z),
    ];
    for (jump, rate) in jumps {
        let jump_adjoint = adjoint(jump);
        let product = matrix_mul(jump_adjoint, jump);
        let gain = matrix_mul(matrix_mul(jump, rho), jump_adjoint);
        let anticommutator = matrix_add(matrix_mul(product, rho), matrix_mul(rho, product));
        let dissipator = matrix_add(gain, matrix_scale(anticommutator, (-0.5, 0.0)));
        derivative = matrix_add(derivative, matrix_scale(dissipator, (rate, 0.0)));
    }
    derivative
}

fn reference_density(generator: BlochLiouvillian, initial: BlochVec, duration: f64) -> Matrix {
    let r = initial.r;
    let mut rho = [
        [((1.0 + r.z) / 2.0, 0.0), (r.x / 2.0, -r.y / 2.0)],
        [(r.x / 2.0, r.y / 2.0), ((1.0 - r.z) / 2.0, 0.0)],
    ];
    let dt = duration / 2_048.0;
    for _ in 0..2_048 {
        let k1 = density_derivative(generator, rho);
        let k2 = density_derivative(
            generator,
            matrix_add(rho, matrix_scale(k1, (dt / 2.0, 0.0))),
        );
        let k3 = density_derivative(
            generator,
            matrix_add(rho, matrix_scale(k2, (dt / 2.0, 0.0))),
        );
        let k4 = density_derivative(generator, matrix_add(rho, matrix_scale(k3, (dt, 0.0))));
        let weighted = matrix_add(
            matrix_add(k1, matrix_scale(k2, (2.0, 0.0))),
            matrix_add(matrix_scale(k3, (2.0, 0.0)), k4),
        );
        rho = matrix_add(rho, matrix_scale(weighted, (dt / 6.0, 0.0)));
    }
    rho
}

#[test]
fn combined_driven_dynamics_match_an_independent_density_matrix_reference() {
    for generator in [
        mixed_generator(),
        BlochLiouvillian {
            hamiltonian: Hamiltonian::new(-0.3, 1.7, -1.1),
            gamma_up: 0.0,
            gamma_down: 1.2,
            gamma_x: 0.0,
            gamma_y: 0.4,
            gamma_z: 0.2,
        },
    ] {
        let channel = generator.for_duration(1.7);
        for initial in initial_states() {
            let rho = reference_density(generator, initial, 1.7);
            let expected = state(
                2.0 * rho[0][1].0,
                -2.0 * rho[0][1].1,
                rho[0][0].0 - rho[1][1].0,
            );
            assert!((rho[0][0].0 + rho[1][1].0 - 1.0).abs() < 1e-12);
            assert_vec(channel.apply_to(initial).r, expected.r, 1e-10);
        }
        for initial in [
            state(1.0, 0.0, 0.0),
            state(-1.0, 0.0, 0.0),
            state(0.0, 1.0, 0.0),
            state(0.0, -1.0, 0.0),
            BlochVec::ground(),
            BlochVec::excited(),
        ] {
            for dt in [1e-12, 0.01, 1.0, 100.0] {
                let evolved = generator.for_duration(dt).apply_to(initial);
                assert!(evolved.r.norm() <= 1.0 + 1e-12);
            }
        }
    }
}
