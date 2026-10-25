use std::{f64::consts::FRAC_PI_2, hint::black_box, time::Duration};

use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::twolevel::{
    BlochLiouvillian, BlochVec, Channel, Decay, Hamiltonian, Liouvillian,
    PauliDephasingLiouvillian, TrotterConfig, Unitary,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn propagation(c: &mut Criterion) {
    let ramsey = Ramsey {
        pulse_area: FRAC_PI_2,
        detuning: 0.7,
        pulse_width: 0.05,
        pulse_separation: 2.0,
        phase_diff: 0.8,
    };
    let initial = BlochVec::ground();

    for nsteps in [32, 501, 10_000] {
        for reduced in [false, true] {
            let tolerance = if reduced { 1.0e-6 } else { 0.0 };
            let mode = if reduced { "reduced" } else { "full" };
            let mut group = c.benchmark_group(format!("from_system/{mode}/{nsteps}"));
            group.throughput(Throughput::Elements(nsteps as u64));
            group.bench_function("composed", |b| {
                b.iter(|| {
                    black_box(
                        Unitary::from_system(
                            black_box(&ramsey),
                            TrotterConfig {
                                start: black_box(-1.0),
                                stop: black_box(3.0),
                                nsteps: black_box(nsteps),
                                tolerance: black_box(tolerance),
                            },
                        )
                        .apply_to(black_box(initial)),
                    )
                });
            });
            group.finish();
        }
    }
}

fn constant_generator_preparation(c: &mut Criterion) {
    let hamiltonian = Hamiltonian::new(1.3, -0.4, 0.7);
    let bloch = BlochLiouvillian {
        hamiltonian,
        gamma_up: 0.2,
        gamma_down: 0.8,
        gamma_x: 0.0,
        gamma_y: 0.0,
        gamma_z: 0.3,
    };
    let legacy_decay = Liouvillian {
        hamiltonian,
        decay: Decay {
            gamma_up: bloch.gamma_up,
            gamma_down: bloch.gamma_down,
            gamma_phi: bloch.gamma_z,
        },
    };
    let pauli = BlochLiouvillian {
        gamma_up: 0.0,
        gamma_down: 0.0,
        gamma_x: 0.2,
        gamma_y: 0.3,
        ..bloch
    };
    let legacy_pauli = PauliDephasingLiouvillian {
        hamiltonian,
        gamma_x: pauli.gamma_x,
        gamma_y: pauli.gamma_y,
        gamma_z: pauli.gamma_z,
    };
    let combined = BlochLiouvillian {
        gamma_x: 0.2,
        gamma_y: 0.3,
        ..bloch
    };
    let critical = BlochLiouvillian {
        hamiltonian: Hamiltonian::new(1.0, 0.0, 0.0),
        gamma_up: 0.0,
        gamma_down: 4.0,
        gamma_x: 0.0,
        gamma_y: 0.0,
        gamma_z: 0.0,
    };

    // Each case prepares the same affine representation, including its source.
    // This small interval is representative of a sampled noise trajectory.
    let mut group = c.benchmark_group("constant_generator_preparation");
    group.bench_function("legacy_decay", |b| {
        b.iter(|| {
            black_box(
                black_box(legacy_decay)
                    .for_duration(black_box(0.025))
                    .to_affine(),
            )
        });
    });
    group.bench_function("bloch_decay", |b| {
        b.iter(|| black_box(black_box(bloch).for_duration(black_box(0.025))));
    });
    group.bench_function("legacy_pauli", |b| {
        b.iter(|| {
            black_box(
                black_box(legacy_pauli)
                    .for_duration(black_box(0.025))
                    .to_affine(),
            )
        });
    });
    group.bench_function("bloch_pauli", |b| {
        b.iter(|| black_box(black_box(pauli).for_duration(black_box(0.025))));
    });
    group.bench_function("bloch_decay_pauli", |b| {
        b.iter(|| black_box(black_box(combined).for_duration(black_box(0.025))));
    });
    group.bench_function("bloch_critical", |b| {
        b.iter(|| black_box(black_box(critical).for_duration(black_box(0.025))));
    });
    group.finish();

    // Include one state update and let the legacy step apply directly, without
    // requiring an affine conversion. Every step is freshly prepared.
    let initial = BlochVec::ground();
    let mut group = c.benchmark_group("constant_generator_step");
    group.bench_function("legacy_decay", |b| {
        b.iter(|| {
            black_box(
                black_box(legacy_decay)
                    .for_duration(black_box(0.025))
                    .apply_to(black_box(initial)),
            )
        });
    });
    group.bench_function("legacy_pauli", |b| {
        b.iter(|| {
            black_box(
                black_box(legacy_pauli)
                    .for_duration(black_box(0.025))
                    .apply_to(black_box(initial)),
            )
        });
    });
    for (name, generator) in [
        ("bloch_decay", bloch),
        ("bloch_pauli", pauli),
        ("bloch_decay_pauli", combined),
        ("bloch_critical", critical),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    black_box(generator)
                        .for_duration(black_box(0.025))
                        .apply_to(black_box(initial)),
                )
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = propagation, constant_generator_preparation
}
criterion_main!(benches);
