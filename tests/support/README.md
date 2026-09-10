# MTS reference and A/B comparison

`mts_reference.rs` was copied from `src/vapourcell/mod.rs` at commit
`9153052937b574f82097d909153ea63c4d176b16`, before integrating MTS with `twolevel`.
The original snapshot's SHA-256 was `89b5791e797defac7344c558879d5eb2f13efae2fd7014cb7c10d83f6f4648af`.
The two Python fixtures in `fixtures/` were copied with it so its existing tests
continue to work independently of the production fixture paths.

The reference has one intentional physics correction shared with production:
`freq = phase'(t)` and pump-frame `hz = delta - kv - freq`. This changes the
longitudinal `mod_shift` contribution from positive to negative. The Python
fixtures retain the original convention; both fixture test helpers add
`2 * mod_shift` to `delta` for Pump-frame fixtures to reproduce their original
Hamiltonian. For `pump_shifted`, this means testing at `delta = 0.7` instead of
`0.4`. Scan offsets, demodulation signs, expected arrays, and tolerances are
unchanged. The normal A/B scans use identical, unadjusted physical parameters
with the corrected convention in both implementations.

Apart from this documented correction and fixture adaptation, keep the reference
and these fixture copies frozen. It includes its own types,
defaults, numerical helpers, and demodulation conventions, including the imposed
sign and exact-zero behaviour. Do not regenerate it from the working solver or
make it call refactored library helpers. Update the adapter in `mod.rs` when the
production API changes. Neither the snapshot nor the adapter is in the library.

## Parity tests

```sh
cargo test --test mts_parity -- --nocapture
```

The shared cases exercise all three frames with the original defaults, shifted
modulation and positive velocity, negative velocity with excitation/dephasing,
zero modulation depth over one period, and a dense grid. They include odd and
even detuning counts and one or multiple spatial phase samples. Tests also check
the live defaults and invalid-parameter errors. The copied solver's original
fixture tests run as part of this test target; the production fixture tests still
run through the library's normal test target.

All four signal arrays (`amp0`, `proj0`, `amp1`, `proj1`) must be finite and agree
within `1e-12 + 1e-10 * abs(reference)`. Detuning grids and time-sample metadata
must match exactly. The comparison reports the maximum absolute signal error.
An intentional behaviour change should be documented and tested explicitly,
without altering the frozen reference to hide the difference.

## Paired speed benchmarks

```sh
cargo bench --bench mts_ab
```

Each case has adjacent `reference` and `current` measurements in the same
optimized benchmark binary. The timed cases are all three default frames,
`shifted_pump`, and `dense_probe`. They check parity before timing. Each timed
call performs the full serial detuning scan, including grid construction,
allocation, propagation, spatial averaging, demodulation, and output disposal.
Input conversion and parity checks are outside the timed region. Throughput
counts individual propagation steps over all detunings and spatial phases.

For example, run only the default probe pair, or save a named baseline:

```sh
cargo bench --bench mts_ab -- default_probe
cargo bench --bench mts_ab -- --save-baseline before_refactor
```

Criterion writes timings and HTML reports under `target/criterion/`. Compare the
current/reference timings within each case; wall-clock speed is not a unit-test
assertion. Small differences are possible even with identical source because of
measurement noise and compilation/inlining across the library boundary.

This first harness uses the public `compute_demod` API and does not change the
production solver. Individual-step and trajectory comparisons can be added when
the propagator is extracted; Rayon velocity comparisons can be added separately
with a fixed worker count. Agreement with the original preserves behaviour but
does not establish physical correctness where the original is singular, such as
the zero-relaxation limit.
