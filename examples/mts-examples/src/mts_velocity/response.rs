use std::f64::consts::PI;

use atomic_clocks::maths::fourier_transform;
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    Frame, HamiltonianParams, LinearResponseOutput, LinearResponseSolverParams,
    compute_linear_response,
};
use myplotlib::{AppResult, Plotter, Slider, SliderGrid, SliderGroup};
use rayon::prelude::*;

use super::{Params, velocity};
use crate::mts;
use crate::units::to_mhz;

const FREQUENCY_STEP: f64 = 0.02; // Maximum internal spacing, in rad / µs.

pub(super) fn controls(params: &mut Params) -> SliderGrid<'_> {
    control_grid(params, true)
}

pub(super) fn demodulated_controls(params: &mut Params) -> SliderGrid<'_> {
    control_grid(params, false)
}

fn control_grid(params: &mut Params, show_time: bool) -> SliderGrid<'_> {
    params.mts.hamiltonian.delta = 0.0;
    params.mts.hamiltonian.kr = 0.0;
    params.mts.decay.gamma_up = 0.0;
    let time_step = 1.0 / params.mts.solver.steps_per_period as f64;
    let time_slider = show_time.then(|| {
        Slider::new(
            "Observation time (t / T)",
            &mut params.response_time_fraction,
            0.0..=1.0,
        )
        .step_by(time_step)
    });
    let response = SliderGroup::new(
        "Linear response",
        time_slider.into_iter().chain([
            Slider::new(
                "Warmup periods",
                &mut params.response_warmup_periods,
                0..=100,
            ),
            Slider::new(
                "Maximum delay (periods)",
                &mut params.response_delay_periods,
                1..=20,
            ),
            Slider::new("Spatial phase samples", &mut params.mts.solver.kr_n, 1..=21).step_by(2.0),
            Slider::new(
                "Steps per period",
                &mut params.mts.solver.steps_per_period,
                20..=1_000,
            ),
        ]),
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
            &mut params.mts.decay.gamma_down,
            &mut params.mts.decay.gamma_phi,
        )
        .into_iter()
        .chain([response, velocity]),
    )
}

pub(super) fn response_at_velocity(
    params: &Params,
    kv: f64,
) -> Result<LinearResponseOutput, String> {
    let atom = HamiltonianParams {
        frame: Frame::Probe,
        kv,
        ..params.mts.hamiltonian
    };
    let solver = LinearResponseSolverParams {
        kr_n: params.mts.solver.kr_n,
        steps_per_period: params.mts.solver.steps_per_period,
        warmup_periods: params.response_warmup_periods,
        delay_periods: params.response_delay_periods,
    };
    compute_linear_response(
        &atom,
        params.mts.decay,
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
    let samples = velocity::velocity_samples(params)?;

    // Bound retained matrices by the worker count, and retain a fixed sample/chunk
    // order so floating-point sums do not depend on scheduling.
    let chunk_count = rayon::current_num_threads().min(samples.len());
    let chunk_size = samples.len().div_ceil(chunk_count);
    let partials: Result<Vec<_>, String> = samples
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk = chunk.iter().copied();
            let first = chunk.next().expect("velocity chunks are non-empty");
            let mut sum = response_at_velocity(params, first.kv)?;
            for row in &mut sum.response {
                for value in row {
                    *value *= first.weight;
                }
            }
            for sample in chunk {
                add_response(
                    &mut sum,
                    response_at_velocity(params, sample.kv)?,
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
    let output = response_output(params)?;
    // Select the nearest computed observation time, retaining t = T as a
    // distinct endpoint. Keep the slider aligned when resolution changes.
    let index = (params.response_time_fraction.clamp(0.0, 1.0)
        * params.mts.solver.steps_per_period as f64)
        .round() as usize;
    params.response_time_fraction = index as f64 / params.mts.solver.steps_per_period as f64;

    let mut plot = Plotter::new();
    plot.plot(&output.delays, &output.response[index])
        .label("C(t, t - τ)");
    plot.xlabel("Delay τ (µs)");
    plot.ylabel("C(t, t - τ): σy response to σz / 2");
    plot.xlim(
        0.0,
        params.response_delay_periods as f64 * params.mts.hamiltonian.modulation.period(),
    );
    Ok(plot)
}

pub(super) fn demodulated_plot(params: &mut Params) -> AppResult {
    let output = response_output(params)?;
    let dc: Vec<_> = output
        .demodulate(0)
        .iter()
        .map(|value| value.in_phase)
        .collect();
    let harmonic: Vec<_> = output
        .demodulate(1)
        .iter()
        .map(|value| value.in_phase)
        .collect();
    let second_harmonic: Vec<_> = output
        .demodulate(2)
        .iter()
        .map(|value| value.in_phase)
        .collect();
    let third_harmonic: Vec<_> = output
        .demodulate(3)
        .iter()
        .map(|value| value.in_phase)
        .collect();
    let mut plot = Plotter::new();
    plot.plot(&output.delays, &dc)
        .label("DC (twice signed mean)");
    plot.plot(&output.delays, &harmonic)
        .label("First harmonic (in phase)");
    plot.plot(&output.delays, &second_harmonic)
        .label("Second harmonic (in phase)");
    plot.plot(&output.delays, &third_harmonic)
        .label("Third harmonic (in phase)");
    plot.xlabel("Delay τ (µs)");
    plot.ylabel("Demodulated σy response to σz / 2");
    plot.xlim(
        0.0,
        params.response_delay_periods as f64 * params.mts.hamiltonian.modulation.period(),
    );
    Ok(plot)
}

pub(super) fn frequency_plot(params: &mut Params) -> AppResult {
    let output = response_output(params)?;
    let atom = &params.mts.hamiltonian;
    let step = atom.modulation.period() / params.mts.solver.steps_per_period as f64;
    let mut plot = Plotter::new();
    for (harmonic, label) in [
        (0, "DC (twice signed mean)"),
        (1, "First harmonic (in phase)"),
        (2, "Second harmonic (in phase)"),
        (3, "Third harmonic (in phase)"),
    ] {
        let response: Vec<_> = output
            .demodulate(harmonic)
            .iter()
            .map(|value| value.in_phase)
            .collect();
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
    plot.ylabel("Fourier magnitude of demodulated response");
    plot.xlim(0.0, to_mhz(PI / step));
    //plot.yscale(AxisScale::Log10);
    Ok(plot)
}
