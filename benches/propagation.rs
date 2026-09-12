use std::{f64::consts::FRAC_PI_2, hint::black_box, time::Duration};

use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::twolevel::{BlochVec, Channel, TrotterConfig, Unitary};
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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = propagation
}
criterion_main!(benches);
