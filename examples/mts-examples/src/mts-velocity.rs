use std::f64::consts::{FRAC_PI_2, PI};

use atomic_clocks::maths::{fourier_transform, normalised_gaussian};
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DemodOutput, Frame, HamiltonianParams, LinearResponseOutput, LinearResponseSolverParams,
    MtsParams, compute_demod, compute_demod_harmonics, compute_linear_response,
};
use myplotlib::{AppDefinition, AppResult, AxisScale, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};
use rayon::prelude::*;

use crate::mts::{self, MtsCurves};

const FREQUENCY_STEP: f64 = 0.02; // Maximum angular-frequency spacing, in rad / time.

pub(crate) struct Params {
    // In this example, `mts.hamiltonian.kv` is the Gaussian mean. Each solver
    // call replaces it with a quadrature node inside the window around kv = 0.
    mts: MtsParams,
    kv_sigma: f64,
    kv_window_half_width: f64,
    kv_samples: usize,
    gradient_epsilon: f64,
    gradient_harmonics: usize,
    response_time_fraction: f64,
    response_warmup_periods: usize,
    response_delay_periods: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mts: MtsParams::default(),
            kv_sigma: 10.0,
            kv_window_half_width: 5.0,
            kv_samples: 21,
            gradient_epsilon: 1e-3,
            gradient_harmonics: 10,
            response_time_fraction: 0.0,
            response_warmup_periods: LinearResponseSolverParams::default().warmup_periods,
            response_delay_periods: LinearResponseSolverParams::default().delay_periods,
        }
    }
}

fn controls(params: &mut Params) -> SliderGrid<'_> {
    let velocity = velocity_control_group(
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(
        5,
        mts::control_groups(&mut params.mts, "Gaussian mean (kv)")
            .into_iter()
            .chain([velocity]),
    )
}

fn velocity_control_group<'a>(
    kv_sigma: &'a mut f64,
    kv_window_half_width: &'a mut f64,
    kv_samples: &'a mut usize,
) -> SliderGroup<'a> {
    SliderGroup::new(
        "Truncated Gaussian velocity distribution",
        [
            Slider::new("Gaussian sigma (kv)", kv_sigma, 0.1..=100.0).logarithmic(true),
            Slider::new(
                "Integration half-width around kv = 0",
                kv_window_half_width,
                0.1..=20.0,
            ),
            Slider::new("Velocity samples", kv_samples, 1..=201),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
struct VelocitySample {
    kv: f64,
    weight: f64,
}

fn velocity_samples(params: &Params) -> Result<Vec<VelocitySample>, String> {
    if params.kv_samples == 0 {
        return Err("velocity sample count must be positive".to_string());
    }
    if !params.kv_sigma.is_finite() || params.kv_sigma <= 0.0 {
        return Err("Gaussian velocity sigma must be positive and finite".to_string());
    }
    if !params.kv_window_half_width.is_finite() || params.kv_window_half_width <= 0.0 {
        return Err("velocity integration half-width must be positive and finite".to_string());
    }
    let mean = params.mts.hamiltonian.kv;
    if !mean.is_finite() {
        return Err("Gaussian velocity mean must be finite".to_string());
    }

    // Midpoint quadrature over a window fixed around the resonant class kv = 0.
    // Weights retain the Gaussian probability mass inside the truncated window.
    // At zero modulation shift, mean = s/2 reproduces a physical carrier shift s.
    let step = 2.0 * params.kv_window_half_width / params.kv_samples as f64;
    Ok((0..params.kv_samples)
        .map(|index| {
            let kv = -params.kv_window_half_width + (index as f64 + 0.5) * step;
            VelocitySample {
                kv,
                weight: step * normalised_gaussian(kv, mean, params.kv_sigma),
            }
        })
        .collect())
}

fn scale_demod_output(output: &mut DemodOutput, weight: f64) {
    for values in [
        &mut output.dc,
        &mut output.harmonic,
        &mut output.second_harmonic,
        &mut output.third_harmonic,
    ] {
        for value in values {
            value.in_phase *= weight;
            value.quadrature *= weight;
        }
    }
}

fn add_demod_output(sum: &mut DemodOutput, output: DemodOutput, weight: f64) -> Result<(), String> {
    if sum.hz != output.hz {
        return Err("velocity signal grids do not match".to_string());
    }
    for (sums, values) in [
        (&mut sum.dc, output.dc),
        (&mut sum.harmonic, output.harmonic),
        (&mut sum.second_harmonic, output.second_harmonic),
        (&mut sum.third_harmonic, output.third_harmonic),
    ] {
        if sums.len() != values.len() {
            return Err("velocity signal dimensions do not match".to_string());
        }
        for (sum, value) in sums.iter_mut().zip(values) {
            sum.in_phase += weight * value.in_phase;
            sum.quadrature += weight * value.quadrature;
        }
    }
    Ok(())
}

fn velocity_average(params: &Params) -> Result<MtsCurves, String> {
    let samples = velocity_samples(params)?;
    let probe = MtsParams {
        hamiltonian: HamiltonianParams {
            frame: Frame::Probe,
            ..params.mts.hamiltonian
        },
        ..params.mts
    };

    // Collect in sample order so summation is independent of Rayon scheduling.
    let outputs: Result<Vec<_>, String> = samples
        .into_par_iter()
        .map(|sample| {
            compute_demod(
                &MtsParams {
                    hamiltonian: HamiltonianParams {
                        kv: sample.kv,
                        ..probe.hamiltonian
                    },
                    ..probe
                },
                Vec3::from_angles(FRAC_PI_2, FRAC_PI_2),
            )
            .map(|output| (sample.weight, output))
        })
        .collect();

    let mut outputs = outputs?.into_iter();
    let (weight, mut average) = outputs.next().expect("velocity sample count is positive");
    scale_demod_output(&mut average, weight);
    for (weight, output) in outputs {
        add_demod_output(&mut average, output, weight)?;
    }
    Ok(MtsCurves::from(average))
}

fn plot(params: &mut Params) -> AppResult {
    let output = velocity_average(params)?;
    Ok(mts::signal_plot(
        &params.mts,
        &output,
        format!(
            "Probe frame: Gaussian μ = {:.3}, σ = {:.3}, |kv| ≤ {:.3} ({} samples)",
            params.mts.hamiltonian.kv,
            params.kv_sigma,
            params.kv_window_half_width,
            params.kv_samples,
        ),
    ))
}

fn harmonic_gradient_controls(params: &mut Params) -> SliderGrid<'_> {
    let gradient = SliderGroup::new(
        "Harmonic gradient",
        [
            Slider::new(
                "Finite-difference epsilon",
                &mut params.gradient_epsilon,
                1e-6..=0.1,
            )
            .logarithmic(true),
            Slider::new("Maximum harmonic", &mut params.gradient_harmonics, 1..=50),
            Slider::new("Spatial phase samples", &mut params.mts.solver.kr_n, 1..=20),
            Slider::new(
                "Steps per period",
                &mut params.mts.solver.steps_per_period,
                20..=1_000,
            ),
            Slider::new("Periods", &mut params.mts.solver.n_periods, 1..=20),
        ],
    );
    let velocity = velocity_control_group(
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(
        5,
        mts::atom_control_groups(
            &mut params.mts.hamiltonian,
            &mut params.mts.decay,
            "Gaussian mean (kv)",
        )
        .into_iter()
        .chain([gradient, velocity]),
    )
}

fn harmonic_gradients(params: &Params) -> Result<Vec<f64>, String> {
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

    let samples = velocity_samples(params)?;
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

fn harmonic_gradient_plot(params: &mut Params) -> AppResult {
    let magnitudes: Vec<_> = harmonic_gradients(params)?
        .into_iter()
        .map(f64::abs)
        .collect();
    let harmonics: Vec<_> = (0..=params.gradient_harmonics)
        .map(|harmonic| harmonic as f64)
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&harmonics, &magnitudes).label("|dI/dΔ|");
    plot.title(format!(
        "Velocity-averaged harmonic gradients at relative detuning 0: Δ = {:.3}, ε = {:.2e}, Gaussian μ = {:.3}, σ = {:.3}, |kv| ≤ {:.3} ({} samples)",
        params.mts.hamiltonian.delta,
        params.gradient_epsilon,
        params.mts.hamiltonian.kv,
        params.kv_sigma,
        params.kv_window_half_width,
        params.kv_samples,
    ));
    plot.xlabel("Harmonic number");
    plot.ylabel("|d(in-phase signal) / dΔ|");
    plot.xlim(0.5, params.gradient_harmonics as f64 + 0.5);
    plot.yscale(AxisScale::Log10);
    Ok(plot)
}

fn response_controls(params: &mut Params) -> SliderGrid<'_> {
    response_control_grid(params, true)
}

fn demodulated_response_controls(params: &mut Params) -> SliderGrid<'_> {
    response_control_grid(params, false)
}

fn response_control_grid(params: &mut Params, show_time: bool) -> SliderGrid<'_> {
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
            Slider::new("Spatial phase samples", &mut params.mts.solver.kr_n, 1..=20),
            Slider::new(
                "Steps per period",
                &mut params.mts.solver.steps_per_period,
                20..=1_000,
            ),
        ]),
    );
    let velocity = velocity_control_group(
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(
        5,
        mts::atom_control_groups(
            &mut params.mts.hamiltonian,
            &mut params.mts.decay,
            "Gaussian mean (kv)",
        )
        .into_iter()
        .chain([response, velocity]),
    )
}

fn response_at_velocity(params: &Params, kv: f64) -> Result<LinearResponseOutput, String> {
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

fn response_output(params: &Params) -> Result<LinearResponseOutput, String> {
    let samples = velocity_samples(params)?;

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

fn response_plot(params: &mut Params) -> AppResult {
    let output = response_output(params)?;
    let atom = &params.mts.hamiltonian;
    // Select the nearest computed observation time, retaining t = T as a
    // distinct endpoint. Keep the slider aligned when resolution changes.
    let index = (params.response_time_fraction.clamp(0.0, 1.0)
        * params.mts.solver.steps_per_period as f64)
        .round() as usize;
    params.response_time_fraction = index as f64 / params.mts.solver.steps_per_period as f64;

    let mut plot = Plotter::new();
    plot.plot(&output.delays, &output.response[index])
        .label("C(t, t - τ)");
    plot.title(format!(
        "Probe-frame response: t/T = {:.3}, t = {:.3}, detuning = {:.3}, Gaussian μ = {:.3}, σ = {:.3}, |kv| ≤ {:.3} ({} samples)",
        params.response_time_fraction,
        output.times[index],
        atom.delta,
        atom.kv,
        params.kv_sigma,
        params.kv_window_half_width,
        params.kv_samples,
    ));
    plot.xlabel("Delay τ (time)");
    plot.ylabel("C(t, t - τ): σy response to σz / 2");
    plot.xlim(
        0.0,
        params.response_delay_periods as f64 * atom.modulation.period(),
    );
    Ok(plot)
}

fn demodulated_response_plot(params: &mut Params) -> AppResult {
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
    let atom = &params.mts.hamiltonian;

    let mut plot = Plotter::new();
    plot.plot(&output.delays, &dc)
        .label("DC (twice signed mean)");
    plot.plot(&output.delays, &harmonic)
        .label("First harmonic (in phase)");
    plot.plot(&output.delays, &second_harmonic)
        .label("Second harmonic (in phase)");
    plot.plot(&output.delays, &third_harmonic)
        .label("Third harmonic (in phase)");
    plot.title(format!(
        "Demodulated probe-frame response: detuning = {:.3}, Gaussian μ = {:.3}, σ = {:.3}, |kv| ≤ {:.3} ({} samples)",
        atom.delta,
        atom.kv,
        params.kv_sigma,
        params.kv_window_half_width,
        params.kv_samples,
    ));
    plot.xlabel("Delay τ (time)");
    plot.ylabel("Demodulated σy response to σz / 2");
    plot.xlim(
        0.0,
        params.response_delay_periods as f64 * atom.modulation.period(),
    );
    Ok(plot)
}

fn frequency_response_plot(params: &mut Params) -> AppResult {
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
        plot.plot(&spectrum.angular_frequencies, &magnitude)
            .label(label);
    }
    plot.title(format!(
        "Demodulated frequency response: detuning = {:.3}, Gaussian μ = {:.3}, σ = {:.3}, |kv| ≤ {:.3} ({} samples)",
        atom.delta,
        atom.kv,
        params.kv_sigma,
        params.kv_window_half_width,
        params.kv_samples,
    ));
    plot.xlabel("Angular frequency (rad / time)");
    plot.ylabel("Fourier magnitude of demodulated response");
    plot.xlim(0.0, PI / step);
    //plot.yscale(AxisScale::Log10);
    Ok(plot)
}

pub(crate) fn definition() -> AppDefinition<Params> {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Probe", plot, controls),

        ViewOption::new("Linear response", response_plot, response_controls),
        ViewOption::new(
            "Demodulated response",
            demodulated_response_plot,
            demodulated_response_controls,
        ),
        ViewOption::new(
            "Frequency response",
            frequency_response_plot,
            demodulated_response_controls,
        ),
        ViewOption::new(
            "Harmonic gradients",
            harmonic_gradient_plot,
            harmonic_gradient_controls,
        ),
    ];
    AppDefinition::new("MTS velocity", "mts-velocity-canvas", VIEWS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomic_clocks::maths::demodulation::{Demodulation, ModulationParams};
    use atomic_clocks::vapourcell::MtsSolverParams;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }

    fn demod_at_velocity(params: &Params, kv: f64) -> DemodOutput {
        compute_demod(
            &MtsParams {
                hamiltonian: HamiltonianParams {
                    frame: Frame::Probe,
                    kv,
                    ..params.mts.hamiltonian
                },
                ..params.mts
            },
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
        )
        .unwrap()
    }

    #[test]
    fn integrated_demodulated_response_matches_probe_curve_slopes() {
        // Compare a single velocity, using a settled observation window and a
        // long delay cutoff. The generic Doppler shift also needs converged kr
        // averaging because a single atom is not modulation-periodic.
        let epsilon = 1e-4;
        for (delta, kv) in [(0.6, 0.0), (-0.4, 0.37)] {
            let mut previous_errors = [f64::INFINITY; 4];
            for steps_per_period in [128, 256] {
                let params = Params {
                    mts: MtsParams {
                        hamiltonian: HamiltonianParams {
                            delta,
                            kv,
                            ..Default::default()
                        },
                        solver: MtsSolverParams {
                            kr_n: 16,
                            steps_per_period,
                            n_periods: 13,
                            hz_lim: epsilon,
                            hz_num: 3,
                        },
                        ..Default::default()
                    },
                    response_warmup_periods: 6,
                    response_delay_periods: 6,
                    ..Params::default()
                };
                let output = response_at_velocity(&params, kv).unwrap();
                let integrate = |harmonic| {
                    output
                        .demodulate(harmonic)
                        .windows(2)
                        .zip(output.delays.windows(2))
                        .map(|(values, times)| {
                            0.5 * (times[1] - times[0]) * (values[0].in_phase + values[1].in_phase)
                        })
                        .sum::<f64>()
                };
                let raw = demod_at_velocity(&params, kv);
                let dc_sign = raw.dc[1].in_phase.signum();
                let curves = MtsCurves::from(raw);
                // Probe plots -abs(mean) for DC; harmonic zero returns 2*mean.
                let integrated = [
                    -0.5 * dc_sign * integrate(0),
                    integrate(1),
                    integrate(2),
                    integrate(3),
                ];
                let slopes = [
                    (curves.amp0[0] - curves.amp0[2]) / (2.0 * epsilon),
                    (curves.proj1[2] - curves.proj1[0]) / (2.0 * epsilon),
                    (curves.proj2[2] - curves.proj2[0]) / (2.0 * epsilon),
                    (curves.proj3[2] - curves.proj3[0]) / (2.0 * epsilon),
                ];
                eprintln!(
                    "delta={delta}, kv={kv}, steps={steps_per_period}: integrals={integrated:?}, slopes={slopes:?}"
                );
                for j in 0..4 {
                    let error = (integrated[j] - slopes[j]).abs();
                    assert!(
                        error < previous_errors[j],
                        "component {j} did not converge: {error}"
                    );
                    if steps_per_period == 256 {
                        assert!(
                            error < 2e-5,
                            "component {j}: {} != {}, error={error}",
                            integrated[j],
                            slopes[j]
                        );
                    }
                    previous_errors[j] = error;
                }
            }
        }
    }

    fn small_params() -> Params {
        Params {
            mts: MtsParams {
                hamiltonian: HamiltonianParams {
                    frame: Frame::Pump,
                    kv: 0.75,
                    ..HamiltonianParams::default()
                },
                solver: MtsSolverParams {
                    hz_num: 7,
                    kr_n: 3,
                    steps_per_period: 40,
                    n_periods: 2,
                    ..MtsSolverParams::default()
                },
                ..MtsParams::default()
            },
            kv_sigma: 2.0,
            kv_window_half_width: 1.5,
            kv_samples: 3,
            ..Params::default()
        }
    }

    fn curves(output: &MtsCurves) -> [&[f64]; 4] {
        [&output.amp0, &output.proj1, &output.proj2, &output.proj3]
    }

    #[test]
    fn gaussian_quadrature_uses_a_fixed_zero_centered_window() {
        let params = small_params();
        let samples = velocity_samples(&params).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].kv, -1.0);
        assert_eq!(samples[1].kv, 0.0);
        assert_eq!(samples[2].kv, 1.0);
        for sample in samples {
            assert_close(
                sample.weight,
                normalised_gaussian(sample.kv, 0.75, 2.0),
                1e-15,
            );
        }
    }

    #[test]
    fn quadrature_retains_the_gaussian_mass_inside_the_window() {
        let params = Params {
            kv_sigma: 1.0,
            kv_window_half_width: 1.0,
            kv_samples: 1_000,
            ..Params::default()
        };
        let mass: f64 = velocity_samples(&params)
            .unwrap()
            .iter()
            .map(|sample| sample.weight)
            .sum();
        assert_close(mass, 0.682_689_492_137, 1e-6);
    }

    #[test]
    fn weighted_signal_matches_serial_signed_coefficients() {
        let params = small_params();
        let average = velocity_average(&params).unwrap();
        let samples = velocity_samples(&params).unwrap();
        let reference: Vec<_> = samples
            .iter()
            .map(|sample| (sample.weight, demod_at_velocity(&params, sample.kv)))
            .collect();

        assert_eq!(average.hz, reference[0].1.hz);
        for index in 0..average.hz.len() {
            let sum = |select: fn(&DemodOutput) -> &[Demodulation]| {
                reference
                    .iter()
                    .map(|(weight, output)| {
                        let value = &select(output)[index];
                        (weight * value.in_phase, weight * value.quadrature)
                    })
                    .fold((0.0, 0.0), |sum, value| (sum.0 + value.0, sum.1 + value.1))
            };
            let dc = sum(|output| &output.dc);
            let harmonic = sum(|output| &output.harmonic);
            let second = sum(|output| &output.second_harmonic);
            let third = sum(|output| &output.third_harmonic);
            assert_close(average.amp0[index], dc.0.hypot(dc.1) / 2.0, 1e-12);
            assert_close(average.proj1[index], harmonic.0, 1e-12);
            assert_close(average.proj2[index], second.0, 1e-12);
            assert_close(average.proj3[index], third.0, 1e-12);
        }
    }

    #[test]
    fn harmonic_gradients_match_the_plotted_in_phase_signal_slopes() {
        let epsilon = 1e-4;
        let params = Params {
            mts: MtsParams {
                solver: MtsSolverParams {
                    hz_lim: epsilon,
                    hz_num: 2,
                    ..small_params().mts.solver
                },
                ..small_params().mts
            },
            gradient_epsilon: epsilon,
            gradient_harmonics: 3,
            ..small_params()
        };
        let gradients = harmonic_gradients(&params).unwrap();
        let signal = velocity_average(&params).unwrap();

        for (actual, values) in
            gradients
                .into_iter()
                .zip([signal.proj1, signal.proj2, signal.proj3])
        {
            let expected = (values[1] - values[0]) / (2.0 * epsilon);
            assert_close(actual, expected, 1e-12);
        }
    }

    #[test]
    fn harmonic_gradients_validate_the_difference_and_sampling() {
        for params in [
            Params {
                gradient_epsilon: 0.0,
                ..small_params()
            },
            Params {
                gradient_harmonics: 0,
                ..small_params()
            },
            Params {
                gradient_harmonics: 21,
                ..small_params()
            },
        ] {
            assert!(harmonic_gradients(&params).is_err());
        }
    }

    #[test]
    fn weighted_response_matches_serial_kernels() {
        let params = Params {
            response_warmup_periods: 1,
            response_delay_periods: 1,
            ..small_params()
        };
        let sum = response_output(&params).unwrap();
        let samples = velocity_samples(&params).unwrap();
        let reference: Vec<_> = samples
            .iter()
            .map(|sample| {
                (
                    sample.weight,
                    response_at_velocity(&params, sample.kv).unwrap(),
                )
            })
            .collect();

        assert_eq!(sum.times, reference[0].1.times);
        assert_eq!(sum.delays, reference[0].1.delays);
        for (i, row) in sum.response.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let expected = reference
                    .iter()
                    .map(|(weight, output)| weight * output.response[i][j])
                    .sum::<f64>();
                assert_close(value, expected, 1e-12);
            }
        }
    }

    #[test]
    fn shifted_gaussian_matches_the_finite_frequency_shift_coordinates() {
        let shift = 1.4;
        let relative_delta = 0.3;
        let sigma = 2.7;
        let solver = MtsSolverParams {
            hz_num: 5,
            kr_n: 3,
            steps_per_period: 40,
            n_periods: 3,
            ..Default::default()
        };
        for shifted_kv in [-0.8, 0.2, 1.1] {
            let physical_kv = shifted_kv - shift / 2.0;
            assert_close(
                normalised_gaussian(shifted_kv, shift / 2.0, sigma),
                normalised_gaussian(physical_kv, 0.0, sigma),
                1e-15,
            );
            let finite_shift = MtsParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        shift,
                        ..Default::default()
                    },
                    delta: relative_delta + shift / 2.0,
                    kv: physical_kv,
                    ..Default::default()
                },
                solver,
                ..Default::default()
            };
            let zero_shift = MtsParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        shift: 0.0,
                        ..finite_shift.hamiltonian.modulation
                    },
                    delta: relative_delta,
                    kv: shifted_kv,
                    ..finite_shift.hamiltonian
                },
                ..finite_shift
            };
            let finite_shift = compute_demod(
                &finite_shift,
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            )
            .unwrap();
            let zero_shift = compute_demod(
                &zero_shift,
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            )
            .unwrap();
            for (actual, expected) in finite_shift.harmonic.iter().zip(zero_shift.harmonic) {
                assert_close(actual.in_phase, expected.in_phase, 1e-12);
                assert_close(actual.quadrature, expected.quadrature, 1e-12);
            }
        }
    }

    #[test]
    fn integrated_weighted_response_matches_weighted_signal_slope() {
        let epsilon = 1e-4;
        let params = Params {
            mts: MtsParams {
                hamiltonian: HamiltonianParams {
                    delta: 0.4,
                    kv: 0.7,
                    ..Default::default()
                },
                solver: MtsSolverParams {
                    kr_n: 8,
                    steps_per_period: 128,
                    n_periods: 13,
                    hz_lim: epsilon,
                    hz_num: 3,
                },
                ..Default::default()
            },
            kv_sigma: 2.0,
            kv_window_half_width: 1.5,
            kv_samples: 5,
            response_warmup_periods: 6,
            response_delay_periods: 6,
            ..Params::default()
        };
        let response = response_output(&params).unwrap();
        let integral = response
            .demodulate(1)
            .windows(2)
            .zip(response.delays.windows(2))
            .map(|(values, times)| {
                0.5 * (times[1] - times[0]) * (values[0].in_phase + values[1].in_phase)
            })
            .sum::<f64>();
        let signal = velocity_average(&params).unwrap();
        let slope = (signal.proj1[2] - signal.proj1[0]) / (2.0 * epsilon);
        assert_close(integral, slope, 2e-5);
    }

    #[test]
    fn rejects_invalid_gaussian_quadrature() {
        for params in [
            Params {
                kv_sigma: 0.0,
                ..Params::default()
            },
            Params {
                kv_window_half_width: 0.0,
                ..Params::default()
            },
            Params {
                kv_samples: 0,
                ..Params::default()
            },
        ] {
            assert!(velocity_samples(&params).is_err());
        }
    }

    #[test]
    fn default_velocity_scan_is_finite() {
        let params = Params::default();
        let average = velocity_average(&params).unwrap();
        for values in curves(&average) {
            assert_eq!(values.len(), params.mts.solver.hz_num);
            assert!(values.iter().all(|value| value.is_finite()));
        }
    }
}
