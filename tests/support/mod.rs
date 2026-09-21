//! Shared cases and comparisons for the frozen MTS solver and the live library.

use atomic_clocks::maths::demodulation::ModulationParams;
use atomic_clocks::vapourcell as current;

// Keep the snapshot unchanged, including helpers unused by the benchmark binary.
#[allow(dead_code)]
pub mod mts_reference;

pub const ABS_TOL: f64 = 1e-12;
pub const REL_TOL: f64 = 1e-10;
pub const SIGNAL_HARMONICS: [usize; 4] = [0, 1, 2, 3];

pub struct Case {
    pub name: String,
    pub params: mts_reference::MtsParams,
}

/// Inputs come from the frozen defaults so future library defaults cannot move
/// both sides of the comparison. This adapter is the place to follow API changes.
pub fn current_params(params: &mts_reference::MtsParams) -> current::MtsParams {
    current::MtsParams {
        hamiltonian: current::HamiltonianParams {
            modulation: ModulationParams {
                frequency: params.mod_freq,
                depth: params.mod_depth,
                shift: params.mod_shift,
            },
            delta: params.delta,
            r_pump: params.r_pump,
            r_prbe: params.r_prbe,
            kv: params.kv,
            kr: params.kr,
            frame: match params.frame {
                mts_reference::Frame::Atom => current::Frame::Atom,
                mts_reference::Frame::Pump => current::Frame::Pump,
                mts_reference::Frame::Probe => current::Frame::Probe,
            },
        },
        decay: atomic_clocks::twolevel::Decay {
            gamma_up: params.gamma_up,
            gamma_down: params.gamma_down,
            gamma_phi: params.gamma_phi,
        },
        solver: current::MtsSolverParams {
            kr_n: params.kr_n,
            steps_per_period: params.steps_per_period,
            n_periods: params.n_periods,
        },
        scan: atomic_clocks::common::DetuningScanParams {
            hz_lim: params.hz_lim,
            hz_num: params.hz_num,
        },
    }
}

pub fn scan_cases() -> Vec<Case> {
    use mts_reference::{Frame, MtsParams};

    let configurations = [
        ("default", MtsParams::default()),
        (
            "shifted",
            MtsParams {
                mod_freq: 1.3,
                mod_depth: 2.4,
                mod_shift: 0.15,
                delta: 0.4,
                r_pump: 2.2,
                r_prbe: 0.35,
                kv: 0.7,
                kr: 0.4,
                kr_n: 6,
                steps_per_period: 120,
                hz_lim: 6.0,
                hz_num: 21,
                gamma_down: 0.8,
                gamma_phi: 0.05,
                ..MtsParams::default()
            },
        ),
        (
            "negative_velocity",
            MtsParams {
                mod_freq: 0.6,
                mod_depth: 0.4,
                mod_shift: -0.2,
                delta: -0.3,
                r_pump: 0.7,
                r_prbe: 0.2,
                kv: -1.2,
                kr: -0.6,
                kr_n: 1,
                steps_per_period: 60,
                n_periods: 5,
                hz_lim: 3.0,
                hz_num: 32,
                gamma_up: 0.15,
                gamma_down: 0.7,
                gamma_phi: 0.3,
                ..MtsParams::default()
            },
        ),
        (
            "unmodulated",
            MtsParams {
                mod_depth: 0.0,
                kv: 0.25,
                kr_n: 3,
                steps_per_period: 80,
                n_periods: 1,
                hz_num: 17,
                ..MtsParams::default()
            },
        ),
        (
            "dense",
            MtsParams {
                kr_n: 8,
                steps_per_period: 200,
                n_periods: 5,
                hz_num: 101,
                ..MtsParams::default()
            },
        ),
    ];

    configurations
        .into_iter()
        .flat_map(|(name, params)| {
            [Frame::Probe, Frame::Pump, Frame::Atom]
                .into_iter()
                .map(move |frame| Case {
                    name: format!("{name}_{}", frame.as_str()),
                    params: MtsParams {
                        frame,
                        ..params.clone()
                    },
                })
        })
        .collect()
}

/// Check the grid exactly and legacy display signals with a mixed absolute/relative
/// tolerance. Return the maximum absolute signal error for test/benchmark logs.
pub fn assert_outputs_close(
    case: &str,
    actual: &current::DemodOutput<{ SIGNAL_HARMONICS.len() }>,
    expected: &mts_reference::DemodOutput,
) -> f64 {
    assert!(
        actual.hz.iter().all(|value| value.is_finite()),
        "{case}: non-finite grid"
    );
    assert_eq!(actual.hz, expected.hz, "{case}: detuning grid changed");
    assert_eq!(
        actual.harmonics, SIGNAL_HARMONICS,
        "{case}: harmonics changed"
    );
    assert_eq!(
        actual.values.len(),
        actual.hz.len(),
        "{case}: coefficient count changed"
    );
    assert!(
        actual
            .values
            .iter()
            .flatten()
            .all(|value| value.in_phase.is_finite() && value.quadrature.is_finite()),
        "{case}: non-finite coefficient"
    );
    // The reference stores magnitudes and signed display projections, not raw I/Q.
    let amp0: Vec<_> = actual
        .values
        .iter()
        .map(|values| values[0].amplitude() / 2.0)
        .collect();
    let mut amp1 = Vec::new();
    let mut proj1 = Vec::new();
    for (&hz, values) in actual.hz.iter().zip(&actual.values) {
        let value = values[1];
        let sign = if hz == 0.0 { 0.0 } else { -hz.signum() };
        let amplitude = value.amplitude() * sign;
        amp1.push(amplitude);
        proj1.push(amplitude * (value.phase().cos() * sign));
    }

    let mut max_abs_error: f64 = 0.0;
    for (name, actual, expected) in [
        ("amp0", &amp0, &expected.amp0),
        ("proj0", &amp0, &expected.proj0),
        ("amp1", &amp1, &expected.amp1),
        ("proj1", &proj1, &expected.proj1),
    ] {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{case}: {name} length changed"
        );
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && e.is_finite(),
                "{case}: {name}[{index}] is non-finite: current={a}, reference={e}"
            );
            let error = (a - e).abs();
            assert!(
                error <= ABS_TOL + REL_TOL * e.abs(),
                "{case}: {name}[{index}] differs: current={a:.17e}, reference={e:.17e}, abs_error={error:.3e}"
            );
            max_abs_error = max_abs_error.max(error);
        }
    }
    max_abs_error
}
