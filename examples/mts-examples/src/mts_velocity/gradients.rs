use std::f64::consts::FRAC_PI_2;

use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{HamiltonianParams, MtsParams, compute_demod_harmonics};
use myplotlib::{AppResult, AxisScale, Plotter, Slider, SliderGrid, SliderGroup};
use rayon::prelude::*;

use super::{Params, common_control_groups, velocity};
use crate::units::{angular_gradient_to_per_mhz, frequency_slider};

pub(super) fn controls(params: &mut Params) -> SliderGrid<'_> {
    let gradient = SliderGroup::new(
        "Harmonic gradient",
        [
            frequency_slider(
                "Finite-difference epsilon",
                &mut params.gradient_epsilon,
                1e-6..=0.1,
            )
            .logarithmic(true),
            Slider::new("Maximum harmonic", &mut params.gradient_harmonics, 1..=50),
        ],
    );
    let common = common_control_groups(
        &mut params.mts.hamiltonian,
        &mut params.mts.decay,
        &mut params.mts.solver.kr_n,
        &mut params.mts.solver.steps_per_period,
        &mut params.mts.solver.n_periods,
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(5, common.into_iter().chain([gradient]))
}

pub(super) fn harmonic_gradients(params: &Params) -> Result<Vec<f64>, String> {
    if !params.gradient_epsilon.is_finite() || params.gradient_epsilon <= 0.0 {
        return Err("finite-difference epsilon must be positive and finite".to_string());
    }
    if params.gradient_harmonics == 0 {
        return Err("maximum harmonic must be positive".to_string());
    }
    if params.gradient_harmonics > params.mts.solver.steps_per_period / 2 {
        return Err(format!(
            "harmonic {} exceeds the Nyquist limit for {} steps per period",
            params.gradient_harmonics, params.mts.solver.steps_per_period
        ));
    }

    let samples = velocity::velocity_samples(params)?;
    let harmonics: Vec<_> = (0..=params.gradient_harmonics).collect();
    let observable = Vec3::from_angles(FRAC_PI_2, FRAC_PI_2);

    // Collect in sample order so the weighted sum is independent of Rayon scheduling.
    let outputs: Result<Vec<_>, String> = samples
        .into_par_iter()
        .map(|sample| {
            let at_offset = |offset| {
                compute_demod_harmonics(
                    &MtsParams {
                        hamiltonian: HamiltonianParams {
                            delta: params.mts.hamiltonian.delta + offset,
                            ..params.probe_hamiltonian(sample.kv)
                        },
                        ..params.mts
                    },
                    observable,
                    &harmonics,
                )
            };
            Ok((
                sample.weight,
                at_offset(-params.gradient_epsilon)?,
                at_offset(params.gradient_epsilon)?,
            ))
        })
        .collect();

    let scale = 1.0 / (2.0 * params.gradient_epsilon);
    let mut gradients = vec![0.0; harmonics.len()];
    for (weight, below, above) in outputs? {
        for ((gradient, below), above) in gradients.iter_mut().zip(below).zip(above) {
            *gradient += weight * (above.in_phase - below.in_phase) * scale;
        }
    }
    Ok(gradients)
}

pub(super) fn plot(params: &mut Params) -> AppResult {
    let magnitudes: Vec<_> = harmonic_gradients(params)?
        .into_iter()
        .map(angular_gradient_to_per_mhz)
        .map(f64::abs)
        .collect();
    let harmonics: Vec<_> = (0..=params.gradient_harmonics)
        .map(|harmonic| harmonic as f64)
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&harmonics, &magnitudes).label("|dI/dΔ|");
    plot.xlabel("Harmonic number");
    plot.ylabel("|d(in-phase signal) / dΔ| (per MHz)");
    plot.xlim(0.0, params.gradient_harmonics as f64 + 0.5);
    plot.yscale(AxisScale::Log10);
    Ok(plot)
}
