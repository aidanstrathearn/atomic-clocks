use std::f64::consts::PI;

use atomic_clocks::maths::fourier_transform;
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DrivenAtomParams, LinearResponseOutput, LinearResponseSolverParams, compute_linear_response,
};
use myplotlib::{AppResult, Plotter};
use rayon::prelude::*;

use crate::app::{Params, add_rabi_summaries};
use crate::units::to_mhz;

const FREQUENCY_STEP: f64 = 0.02; // Maximum internal spacing, in rad / µs.

pub(super) fn response_at_velocity(
    params: &Params,
    kv: f64,
) -> Result<LinearResponseOutput, String> {
    let mts = params.mts_params()?;
    response_at_velocity_for_model(params, &mts, kv)
}

fn response_at_velocity_for_model(
    params: &Params,
    mts: &atomic_clocks::vapourcell::MtsParams,
    kv: f64,
) -> Result<LinearResponseOutput, String> {
    let warmup_periods = mts
        .solver
        .n_periods
        .checked_sub(1)
        .ok_or("MTS period count must be positive")?;
    let atom = DrivenAtomParams {
        hamiltonian: params.probe_hamiltonian(mts, kv),
        ..mts.atom
    };
    let solver = LinearResponseSolverParams {
        kr_n: mts.solver.kr_n,
        steps_per_period: mts.solver.steps_per_period,
        warmup_periods,
        delay_periods: params.response_delay_periods,
    };
    compute_linear_response(
        &atom,
        &solver,
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    )
}

fn add_response(
    sum: &mut LinearResponseOutput,
    output: LinearResponseOutput,
    weight: f64,
) -> Result<(), String> {
    if sum.times != output.times || sum.delays != output.delays {
        return Err("velocity response grids do not match".to_string());
    }
    if sum.response.len() != output.response.len() {
        return Err("velocity response dimensions do not match".to_string());
    }
    for (sum_row, row) in sum.response.iter_mut().zip(output.response) {
        if sum_row.len() != row.len() {
            return Err("velocity response dimensions do not match".to_string());
        }
        for (sum, value) in sum_row.iter_mut().zip(row) {
            *sum += weight * value;
        }
    }
    Ok(())
}

pub(super) fn response_output(params: &Params) -> Result<LinearResponseOutput, String> {
    let mts = params.mts_params()?;
    response_output_for_model(params, &mts)
}

fn response_output_for_model(
    params: &Params,
    mts: &atomic_clocks::vapourcell::MtsParams,
) -> Result<LinearResponseOutput, String> {
    let samples = params.velocity_params(mts)?.samples()?;

    // Bound retained matrices by the worker count, and retain a fixed sample/chunk
    // order so floating-point sums do not depend on scheduling.
    let chunk_count = rayon::current_num_threads().min(samples.len());
    let chunk_size = samples.len().div_ceil(chunk_count);
    let partials: Result<Vec<_>, String> = samples
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk = chunk.iter().copied();
            let first = chunk.next().expect("velocity chunks are non-empty");
            let mut sum = response_at_velocity_for_model(params, mts, first.kv)?;
            for row in &mut sum.response {
                for value in row {
                    *value *= first.weight;
                }
            }
            for sample in chunk {
                add_response(
                    &mut sum,
                    response_at_velocity_for_model(params, mts, sample.kv)?,
                    sample.weight,
                )?;
            }
            Ok(sum)
        })
        .collect();

    let mut partials = partials?.into_iter();
    let mut sum = partials
        .next()
        .expect("a positive velocity sample count produces a response");
    for partial in partials {
        add_response(&mut sum, partial, 1.0)?;
    }
    Ok(sum)
}

pub(super) fn plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let gain = params.transmission_gain()?;
    let mut output = response_output_for_model(params, &mts)?;
    // Select the nearest computed observation time, retaining t = T as a
    // distinct endpoint. Keep the slider aligned when resolution changes.
    let index = (params.response_time_fraction.clamp(0.0, 1.0) * mts.solver.steps_per_period as f64)
        .round() as usize;
    params.response_time_fraction = index as f64 / mts.solver.steps_per_period as f64;
    for value in &mut output.response[index] {
        *value *= gain;
    }

    let mut plot = Plotter::new();
    plot.plot(&output.delays, &output.response[index])
        .label("Transmission response");
    plot.xlabel("Delay τ (µs)");
    plot.ylabel("Normalized transmission response to σz / 2");
    plot.xlim(
        0.0,
        params.response_delay_periods as f64 * mts.atom.hamiltonian.modulation.period(),
    );
    add_rabi_summaries(&mut plot, &mts);
    Ok(plot)
}

pub(super) fn demodulated_transmission_response(
    output: &LinearResponseOutput,
    harmonic: usize,
    gain: f64,
) -> Vec<f64> {
    let scale = if harmonic == 0 { 0.5 * gain } else { gain };
    output
        .demodulate(harmonic)
        .into_iter()
        .map(|value| scale * value.in_phase)
        .collect()
}

pub(super) fn demodulated_plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let output = response_output_for_model(params, &mts)?;
    let gain = params.transmission_gain()?;
    let dc = demodulated_transmission_response(&output, 0, gain);
    let harmonic = demodulated_transmission_response(&output, 1, gain);
    let second_harmonic = demodulated_transmission_response(&output, 2, gain);
    let third_harmonic = demodulated_transmission_response(&output, 3, gain);
    let mut plot = Plotter::new();
    plot.plot(&output.delays, &dc).label("Mean response");
    plot.plot(&output.delays, &harmonic)
        .label("First harmonic (in phase)");
    plot.plot(&output.delays, &second_harmonic)
        .label("Second harmonic (in phase)");
    plot.plot(&output.delays, &third_harmonic)
        .label("Third harmonic (in phase)");
    plot.xlabel("Delay τ (µs)");
    plot.ylabel("Demodulated normalized transmission response to σz / 2");
    plot.xlim(
        0.0,
        params.response_delay_periods as f64 * mts.atom.hamiltonian.modulation.period(),
    );
    add_rabi_summaries(&mut plot, &mts);
    Ok(plot)
}

pub(super) fn frequency_plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let output = response_output_for_model(params, &mts)?;
    let gain = params.transmission_gain()?;
    let atom = &mts.atom.hamiltonian;
    let step = atom.modulation.period() / mts.solver.steps_per_period as f64;
    let mut plot = Plotter::new();
    for (harmonic, label) in [
        (0, "Mean response"),
        (1, "First harmonic (in phase)"),
        (2, "Second harmonic (in phase)"),
        (3, "Third harmonic (in phase)"),
    ] {
        let response = demodulated_transmission_response(&output, harmonic, gain);
        // Match the Ramsey example: transform the real kernel over delay,
        // retaining the helper's one-sided, undoubled amplitudes and dt scaling.
        let spectrum = fourier_transform(&response, step, output.delays[0], FREQUENCY_STEP);
        let magnitude: Vec<_> = spectrum
            .amplitudes
            .iter()
            .map(|value| value.norm())
            .collect();
        let frequencies_mhz: Vec<_> = spectrum
            .angular_frequencies
            .iter()
            .copied()
            .map(to_mhz)
            .collect();
        plot.plot(&frequencies_mhz, &magnitude).label(label);
    }
    plot.xlabel("Frequency (MHz)");
    plot.ylabel("Fourier magnitude of normalized transmission response");
    plot.xlim(0.0, to_mhz(PI / step));
    //plot.yscale(AxisScale::Log10);
    add_rabi_summaries(&mut plot, &mts);
    Ok(plot)
}
