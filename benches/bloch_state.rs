use std::{hint::black_box, time::Duration};

use atomic_clocks::twolevel::{BlochVec, Channel, DephasingLiouvillian, Hamiltonian, Vec3};
use criterion::{Criterion, criterion_group, criterion_main};

fn state() -> BlochVec {
    BlochVec {
        r: Vec3 {
            x: 0.4,
            y: -0.2,
            z: 0.7,
        },
    }
}

fn legacy_rotate_about_z(mut state: BlochVec, angle: f64) -> BlochVec {
    let (sin, cos) = angle.sin_cos();
    let x = state.r.x;
    let y = state.r.y;
    state.r.x = cos * x - sin * y;
    state.r.y = sin * x + cos * y;
    state
}

fn bloch_state(c: &mut Criterion) {
    let initial = state();
    let angle = 0.73;
    let z_axis = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    let unitary = Hamiltonian::new(0.0, 0.0, 1.0).for_duration(angle);

    let mut rotation = c.benchmark_group("bloch_state/rotation");
    rotation.bench_function("legacy_z_formula", |b| {
        b.iter(|| legacy_rotate_about_z(black_box(initial), black_box(angle)))
    });
    rotation.bench_function("rotate_about_z", |b| {
        b.iter(|| black_box(initial).rotate_about_z(black_box(angle)))
    });
    rotation.bench_function("generic_rotate", |b| {
        b.iter(|| black_box(initial).rotate(black_box(z_axis), black_box(angle)))
    });
    rotation.bench_function("prepared_unitary", |b| {
        b.iter(|| black_box(&unitary).apply_to(black_box(initial)))
    });
    rotation.bench_function("construct_unitary_and_apply", |b| {
        b.iter(|| {
            Hamiltonian::new(0.0, 0.0, 1.0)
                .for_duration(black_box(angle))
                .apply_to(black_box(initial))
        })
    });
    rotation.finish();

    let gamma_phi: f64 = 0.7;
    let duration: f64 = 0.37;
    let coherence = (-gamma_phi * duration).exp();
    let prepared_dephasing = DephasingLiouvillian {
        hamiltonian: Hamiltonian::default(),
        gamma_phi,
    }
    .for_duration(duration);

    let mut dephasing = c.benchmark_group("bloch_state/dephasing");
    dephasing.bench_function("direct", |b| {
        b.iter(|| black_box(initial).dephase(black_box(coherence)))
    });
    dephasing.bench_function("prepared_step", |b| {
        b.iter(|| black_box(&prepared_dephasing).apply_to(black_box(initial)))
    });
    dephasing.bench_function("construct_step_and_apply", |b| {
        b.iter(|| {
            DephasingLiouvillian {
                hamiltonian: Hamiltonian::default(),
                gamma_phi: black_box(gamma_phi),
            }
            .for_duration(black_box(duration))
            .apply_to(black_box(initial))
        })
    });
    dephasing.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = bloch_state
}
criterion_main!(benches);
