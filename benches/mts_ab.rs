use std::{hint::black_box, time::Duration};

use atomic_clocks::vapourcell as current;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

#[path = "../tests/support/mod.rs"]
mod support;

use support::{assert_outputs_close, current_params, mts_reference as reference, scan_cases};

fn mts_ab(c: &mut Criterion) {
    for case in scan_cases().into_iter().filter(|case| {
        matches!(
            case.name.as_str(),
            "default_probe" | "default_pump" | "default_atom" | "shifted_pump" | "dense_probe"
        )
    }) {
        let reference_params = case.params;
        let current_params = current_params(&reference_params);

        // Check correctness and convert inputs outside the timed work. Both
        // timed calls include the complete scan's grid construction/allocations.
        let expected =
            reference::compute_demod(&reference_params).expect("reference scan succeeds");
        let actual = current::compute_demod(&current_params).expect("current scan succeeds");
        let error = assert_outputs_close(&case.name, &actual, &expected);
        eprintln!(
            "{}: A/B parity OK, max absolute error {error:.3e}",
            case.name
        );

        let steps = reference_params.hz_num
            * reference_params.kr_n
            * (reference_params.n_periods * reference_params.steps_per_period - 1);
        let mut group = c.benchmark_group(format!("mts_ab/{}", case.name));
        group.throughput(Throughput::Elements(steps as u64));
        group.bench_function("reference", |b| {
            b.iter(|| {
                black_box(
                    reference::compute_demod(black_box(&reference_params))
                        .expect("reference scan succeeds"),
                )
            });
        });
        group.bench_function("current", |b| {
            b.iter(|| {
                black_box(
                    current::compute_demod(black_box(&current_params))
                        .expect("current scan succeeds"),
                )
            });
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = mts_ab
}
criterion_main!(benches);
