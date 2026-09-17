use std::f64::consts::FRAC_PI_2;

use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{Frame, HamiltonianParams, MtsParams, compute_demod_harmonics};
use myplotlib::{AppResult, AxisScale, Plotter, Slider, SliderGrid, SliderGroup};
use rayon::prelude::*;

use super::{Params, velocity};
use crate::mts;
use crate::units::{angular_gradient_to_per_mhz, frequency_slider, to_mhz};

pub(super) fn controls(params: &mut Params) -> SliderGrid<'_> {
    params.mts.hamiltonian.kr = 0.0;
    params.mts.decay.gamma_up = 0.0;
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
            Slider::new("Spatial phase samples", &mut params.mts.solver.kr_n, 1..=21).step_by(2.0),
            Slider::new(
                "Steps per period",
                &mut params.mts.solver.steps_per_period,
                20..=1_000,
            ),
            Slider::new("Periods", &mut params.mts.solver.n_periods, 1..=20),
        ],
    );
    let velocity = velocity::control_group(
        &mut params.mts.hamiltonian.kv,
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(
        5,
        mts::atom_control_groups(
            &mut params.mts.hamiltonian.modulation,
            &mut params.mts.hamiltonian.r_pump,
            &mut params.mts.hamiltonian.r_prbe,
            &mut params.mts.hamiltonian.delta,
            &mut params.mts.decay.gamma_down,
            &mut params.mts.decay.gamma_phi,
        )
        .into_iter()
        .chain([gradient, velocity]),
    )
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
                            frame: Frame::Probe,
                            delta: params.mts.hamiltonian.delta + offset,
                            kv: sample.kv,
                            ..params.mts.hamiltonian
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
    plot.title(format!(
        "Velocity-averaged harmonic gradients at relative detuning 0: Δ = {:.3} MHz, ε = {:.2e} MHz, Gaussian μ = {:.3} MHz, σ = {:.3} MHz, |kv| ≤ {:.3} MHz ({} samples)",
        to_mhz(params.mts.hamiltonian.delta),
        to_mhz(params.gradient_epsilon),
        to_mhz(params.mts.hamiltonian.kv),
        to_mhz(params.kv_sigma),
        to_mhz(params.kv_window_half_width),
        params.kv_samples,
    ));
    plot.xlabel("Harmonic number");
    plot.ylabel("|d(in-phase signal) / dΔ| (per MHz)");
    plot.xlim(0.0, params.gradient_harmonics as f64 + 0.5);
    plot.yscale(AxisScale::Log10);
    Ok(plot)
}
