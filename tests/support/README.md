# MTS reference and A/B comparison

`mts_reference.rs` was copied from `src/vapourcell/mod.rs` at commit
`9153052937b574f82097d909153ea63c4d176b16`, before integrating MTS with `twolevel`.
The original snapshot's SHA-256 was `89b5791e797defac7344c558879d5eb2f13efae2fd7014cb7c10d83f6f4648af`.
The reference intentionally shares two corrected conventions with production:

- `freq = phase'(t)` and pump-frame `hz = delta - kv - freq`, correcting the
  sign of the longitudinal `mod_shift` contribution.
- Exactly `steps_per_period` intervals per period, with
  `dt = (2*pi / mod_freq) / steps_per_period`, no frequency clamp, and
  `n_periods * steps_per_period + 1` boundaries. The observation window is the
  final `steps_per_period + 1` states, including both period endpoints. One
  interval is now valid (though too coarse to resolve a harmonic).

Production demodulates with `lockin_period(traj, harmonic)`. The reference keeps
its independent time-based trapezoidal integrator and propagation implementation.
A/B parity establishes agreement under the corrected conventions, not agreement
with the original Python calculation.

The retired Python fixtures used the old grid/window and pump-shift convention
and have been removed along with their tests, parsers, and detuning-adjustment
helpers. Analytic and convergence tests replace those comparisons.

Apart from these documented corrections, keep the reference independent.
The reference includes its own types, defaults,
numerical helpers, and display conventions, including the imposed sign and
exact-zero behaviour. Do not regenerate it from the working solver or make it
call refactored library helpers. Update the adapter in `mod.rs` when the production
API changes. Neither the reference nor the adapter is in the library.

## Parity tests

```sh
cargo test --test mts_parity -- --nocapture
```

The shared cases exercise all three frames with the original defaults, shifted
modulation and positive velocity, negative velocity with excitation/dephasing,
zero modulation depth over one period, and a dense grid. They include odd and
even detuning counts and one or multiple spatial phase samples. Tests also check
the live defaults, invalid-parameter errors, and the minimal one-interval grid.

Production returns raw `dc` and `harmonic` cosine/sine coefficients. The comparison
callers pass `Vec3::from_angles(pi/2, pi/2)` to match the reference's fixed
measurement direction. The reference does not depend on the production observable
API. The comparison adapter reconstructs the reference's four display arrays (`amp0`, `proj0`, `amp1`,
`proj1`), including its imposed detuning sign and forced central zero. These arrays
must be finite and agree within `1e-12 + 1e-10 * abs(reference)`. Detuning grids
must match exactly. The comparison reports the maximum absolute signal error.
Raw coefficients are also checked for finiteness. Signed DC and both harmonic
quadratures at zero detuning are tested separately against an analytic damped
Rabi trajectory, since the historical display arrays discard that information.
An internal grid test verifies interval spacing, period boundaries, and final
window size. Refining the grid is tested against continuous analytic damped-Rabi
integrals. Harmonic extraction, endpoint weights, and DC normalization have
separate tests in `tests/demodulation.rs`. Further intentional behaviour changes
should be documented and tested explicitly.

## Paired speed benchmarks

```sh
cargo bench --bench mts_ab
```

Each case has adjacent `reference` and `current` measurements in the same
optimized benchmark binary. The timed cases are all three default frames,
`shifted_pump`, and `dense_probe`. They check parity before timing. Each timed
call performs the full serial detuning scan, including grid construction,
allocation, propagation, spatial averaging, demodulation, and output disposal.
Input conversion, legacy display reconstruction, and parity checks are outside
the timed region. The reference still constructs its display arrays during the
timed call; production returns raw coefficients without display processing.
Throughput counts `hz_num * kr_n * n_periods * steps_per_period` propagation
steps, summed over detunings and spatial phases.

For example, run only the default probe pair, or save a named baseline:

```sh
cargo bench --bench mts_ab -- default_probe
cargo bench --bench mts_ab -- --save-baseline before_refactor
```

Criterion writes timings and HTML reports under `target/criterion/`. Compare the
current/reference timings within each case; wall-clock speed is not a unit-test
assertion. Small differences are possible even with identical source because of
measurement noise and compilation/inlining across the library boundary.

This harness uses the public `compute_demod` API. Individual-step and trajectory
comparisons can be added separately; Rayon velocity comparisons can be added
with a fixed worker count. Agreement with the reference does not establish
physical correctness where its propagator is singular, such as the
zero-relaxation limit.
