use std::f64::consts::FRAC_PI_2;

use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DrivenAtomParams, HamiltonianParams, MtsParams, VelocitySample, compute_demod_harmonics,
};
use myplotlib::{AppResult, AxisScale, Plotter};

use crate::app::{Params, add_rabi_summaries};
use crate::parallel::try_fold_chunks_ordered;
use crate::units::angular_gradient_to_per_mhz;

pub(super) fn harmonic_gradients(params: &Params) -> Result<Vec<f64>, String> {
    let mts = params.mts_params()?;
    harmonic_gradients_for_model(params, &mts)
}

fn harmonic_gradients_for_model(params: &Params, mts: &MtsParams) -> Result<Vec<f64>, String> {
    if !params.gradient_epsilon.is_finite() || params.gradient_epsilon <= 0.0 {
        return Err("finite-difference epsilon must be positive and finite".to_string());
    }
    if params.gradient_harmonics == 0 {
        return Err("maximum harmonic must be positive".to_string());
    }
    if params.gradient_harmonics > mts.solver.steps_per_period / 2 {
        return Err(format!(
            "harmonic {} exceeds the Nyquist limit for {} steps per period",
            params.gradient_harmonics, mts.solver.steps_per_period
        ));
    }

    let samples = params.velocity_params(mts)?.samples()?;
    let harmonics: Vec<_> = (0..=params.gradient_harmonics).collect();
    let observable = Vec3::from_angles(FRAC_PI_2, FRAC_PI_2);

    let accumulate_sample =
        |gradients: &mut [f64], sample: &VelocitySample| -> Result<(), String> {
            let at_offset = |offset| {
                compute_demod_harmonics(
                    &MtsParams {
                        atom: DrivenAtomParams {
                            hamiltonian: HamiltonianParams {
                                delta: mts.atom.hamiltonian.delta + offset,
                                ..params.probe_hamiltonian(mts, sample.kv)
                            },
                            ..mts.atom
                        },
                        ..*mts
                    },
                    observable,
                    &harmonics,
                )
            };
            let below = at_offset(-params.gradient_epsilon)?;
            let above = at_offset(params.gradient_epsilon)?;
            for ((gradient, below), above) in gradients.iter_mut().zip(below).zip(above) {
                *gradient += sample.weight * (above.in_phase - below.in_phase);
            }
            Ok(())
        };

    let mut gradients = try_fold_chunks_ordered(
        &samples,
        params.compute_threads,
        |sample| {
            let mut gradients = vec![0.0; harmonics.len()];
            accumulate_sample(&mut gradients, sample)?;
            Ok(gradients)
        },
        |gradients, sample| accumulate_sample(gradients, sample),
        |gradients, partial| {
            for (gradient, partial) in gradients.iter_mut().zip(partial) {
                *gradient += partial;
            }
            Ok(())
        },
    )?
    .ok_or_else(|| "velocity sample count must be positive".to_string())?;
    let scale = 1.0 / (2.0 * params.gradient_epsilon);
    for gradient in &mut gradients {
        *gradient *= scale;
    }
    Ok(gradients)
}

pub(super) fn plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let magnitudes: Vec<_> = harmonic_gradients_for_model(params, &mts)?
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
    add_rabi_summaries(&mut plot, &mts);
    Ok(plot)
}

pub(super) fn normalised_adjacent_gradient_sums(gradients: &[f64]) -> Result<Vec<f64>, String> {
    if gradients.len() < 3 {
        return Err("at least two harmonics are required".to_string());
    }
    let normalisation = gradients[1];
    if !normalisation.is_finite() || normalisation == 0.0 {
        return Err("first-harmonic gradient must be finite and nonzero".to_string());
    }

    (1..gradients.len() - 1)
        .map(|harmonic| {
            let lower = if harmonic == 1 {
                2.0 * gradients[0]
            } else {
                gradients[harmonic - 1]
            };
            let ratio = (lower + gradients[harmonic + 1]) / normalisation;
            let squared = ratio.abs().powi(2);
            if squared.is_finite() {
                Ok(squared)
            } else {
                Err(format!(
                    "normalised adjacent-gradient sum is non-finite at harmonic {harmonic}"
                ))
            }
        })
        .collect()
}

pub(super) fn adjacent_plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let values = normalised_adjacent_gradient_sums(&harmonic_gradients_for_model(params, &mts)?)?;
    let harmonics: Vec<_> = (1..params.gradient_harmonics)
        .map(|harmonic| harmonic as f64)
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&harmonics, &values);
    plot.ylabel("Weight");
    plot.xlabel("Harmonic number k");
    plot.yscale(AxisScale::Log10);
    //plot.ylim(0.5, params.gradient_harmonics as f64 - 0.5);
    add_rabi_summaries(&mut plot, &mts);
    Ok(plot)
}
