use std::{f64::consts::FRAC_PI_2, hint::black_box, time::Duration};

use atomic_clocks::maths::Linspace;
use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::twolevel::{BlochVec, Solver};
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

    for samples in [32, 501, 10_000] {
        for reduced in [false, true] {
            // Sample the Hamiltonian and reduce outside the timed region.
            let mut solver = Solver::from(&ramsey, Linspace::new(-1.0, 3.0, samples - 1));
            if reduced {
                solver.trotter_reduce(1.0e-6);
            }
            let expected = solver.propagate_to_final(initial);
            let actual = solver.propagate_to_final_composed(initial);
            for (actual, expected) in [
                (actual.r.x, expected.r.x),
                (actual.r.y, expected.r.y),
                (actual.r.z, expected.r.z),
            ] {
                assert!(
                    (actual - expected).abs() < 1.0e-10,
                    "methods disagree: composed {actual}, sequential {expected}"
                );
            }

            let mode = if reduced { "reduced" } else { "full" };
            let steps = solver.h_t.len();
            eprintln!("{mode}/{samples}: {steps} stored steps; final states agree");
            let mut group = c.benchmark_group(format!("propagation/{mode}/{samples}"));
            group.throughput(Throughput::Elements(steps as u64));
            group.bench_function("sequential", |b| {
                b.iter(|| black_box(black_box(&solver).propagate_to_final(black_box(initial))));
            });
            group.bench_function("composed", |b| {
                b.iter(|| {
                    black_box(black_box(&solver).propagate_to_final_composed(black_box(initial)))
                });
            });
            group.finish();
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = propagation
}
criterion_main!(benches);
