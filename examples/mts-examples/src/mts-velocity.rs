use std::f64::consts::{FRAC_PI_2, PI};

use atomic_clocks::maths::fourier_transform;
use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    Frame, HamiltonianParams, LinearResponseOutput, LinearResponseSolverParams, MtsParams,
    compute_demod, compute_linear_response,
};
use myplotlib::{AppDefinition, AppResult, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};
use rayon::prelude::*;

use crate::mts::{self, MtsCurves};

const FREQUENCY_STEP: f64 = 0.02; // Maximum angular-frequency spacing, in rad / time.

pub(crate) struct Params {
    mts: MtsParams,
    kv_half_width: f64,
    kv_samples: usize,
    response_time_fraction: f64,
    response_warmup_periods: usize,
    response_delay_periods: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mts: MtsParams::default(),
            kv_half_width: 0.0,
            kv_samples: 21,
            response_time_fraction: 0.0,
            response_warmup_periods: LinearResponseSolverParams::default().warmup_periods,
            response_delay_periods: LinearResponseSolverParams::default().delay_periods,
        }
    }
}

fn controls(params: &mut Params) -> SliderGrid<'_> {
    let velocity = SliderGroup::new(
        "Uniform velocity distribution",
        [
            Slider::new("Doppler half-width", &mut params.kv_half_width, 0.0..=20.0),
            Slider::new("Velocity samples", &mut params.kv_samples, 1..=201),
        ],
    );
    SliderGrid::new(
        5,
        mts::control_groups(&mut params.mts, "Mean Doppler shift (kv)")
            .into_iter()
            .chain([velocity]),
    )
}

fn velocity_average(params: &Params) -> Result<MtsCurves, String> {
    if params.kv_samples == 0 {
        return Err("velocity sample count must be positive".to_string());
    }
    if !params.kv_half_width.is_finite() || params.kv_half_width < 0.0 {
        return Err("Doppler half-width must be non-negative and finite".to_string());
    }

    let probe = MtsParams {
        hamiltonian: HamiltonianParams {
            frame: Frame::Probe,
            ..params.mts.hamiltonian
        },
        ..params.mts
    };
    if params.kv_half_width == 0.0 || params.kv_samples == 1 {
        return compute_demod(&probe, Vec3::from_angles(FRAC_PI_2, FRAC_PI_2)).map(MtsCurves::from);
    }

    // Equal-weight midpoint quadrature for a uniform distribution on kv +/- half-width.
    // Collect in sample order so summation is independent of Rayon scheduling.
    let outputs: Result<Vec<_>, String> = (0..params.kv_samples)
        .into_par_iter()
        .map(|index| {
            let offset = (2.0 * (index as f64 + 0.5) / params.kv_samples as f64 - 1.0)
                * params.kv_half_width;
            compute_demod(
                &MtsParams {
                    hamiltonian: HamiltonianParams {
                        kv: probe.hamiltonian.kv + offset,
                        ..probe.hamiltonian
                    },
                    ..probe
                },
                Vec3::from_angles(FRAC_PI_2, FRAC_PI_2),
            )
            .map(MtsCurves::from)
        })
        .collect();

    let mut outputs = outputs?.into_iter();
    let mut average = outputs.next().expect("velocity sample count is positive");
    // Preserve the current sum of per-velocity scalar curves, including magnitudes.
    // Conversion must happen before summation; all velocities share the same grid.
    for output in outputs {
        for (sum, values) in [
            (&mut average.amp0, output.amp0),
            (&mut average.proj0, output.proj0),
            (&mut average.amp1, output.amp1),
            (&mut average.proj1, output.proj1),
            (&mut average.proj2, output.proj2),
            (&mut average.proj3, output.proj3),
        ] {
            for (sum, value) in sum.iter_mut().zip(values) {
                *sum += value;
            }
        }
    }
    for values in [
        &mut average.amp0,
        &mut average.proj0,
        &mut average.amp1,
        &mut average.proj1,
        &mut average.proj2,
        &mut average.proj3,
    ] {
        for value in values {
            //*value /= params.kv_samples as f64;
            *value /= 1.0;
        }
    }
    Ok(average)
}

fn plot(params: &mut Params) -> AppResult {
    let output = velocity_average(params)?;
    Ok(mts::signal_plot(
        &params.mts,
        &output,
        "Probe frame".to_string(),
    ))
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
    SliderGrid::new(
        5,
        mts::atom_control_groups(
            &mut params.mts.hamiltonian,
            &mut params.mts.decay,
            "Mean Doppler shift (kv)",
        )
        .into_iter()
        .chain([response]),
    )
}

fn response_output(params: &Params) -> Result<LinearResponseOutput, String> {
    let atom = HamiltonianParams {
        frame: Frame::Probe,
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
        "Probe-frame response: t/T = {:.3}, t = {:.3}, detuning = {:.3}, kv = {:.3}",
        params.response_time_fraction, output.times[index], atom.delta, atom.kv,
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
        "Demodulated probe-frame response: detuning = {:.3}, kv = {:.3}",
        atom.delta, atom.kv,
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
        "Demodulated frequency response: detuning = {:.3}, kv = {:.3}",
        atom.delta, atom.kv,
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
    ];
    AppDefinition::new("MTS velocity", "mts-velocity-canvas", VIEWS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomic_clocks::vapourcell::MtsSolverParams;

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
                let output = response_output(&params).unwrap();
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
                let raw = compute_demod(
                    &params.mts,
                    Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                )
                .unwrap();
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
            kv_half_width: 1.5,
            kv_samples: 3,
            ..Params::default()
        }
    }

    fn curves(output: &MtsCurves) -> [&[f64]; 6] {
        [
            &output.amp0,
            &output.proj0,
            &output.amp1,
            &output.proj1,
            &output.proj2,
            &output.proj3,
        ]
    }

    #[test]
    fn velocity_sum_matches_serial_scalar_curves() {
        let params = small_params();
        let average = velocity_average(&params).unwrap();
        // Midpoints of the three equal bins in [-0.75, 2.25].
        let reference: Vec<_> = [-0.25, 0.75, 1.75]
            .into_iter()
            .map(|kv| {
                compute_demod(
                    &MtsParams {
                        hamiltonian: HamiltonianParams {
                            frame: Frame::Probe,
                            kv,
                            ..params.mts.hamiltonian
                        },
                        ..params.mts
                    },
                    Vec3::from_angles(FRAC_PI_2, FRAC_PI_2),
                )
                .map(MtsCurves::from)
                .unwrap()
            })
            .collect();

        assert_eq!(average.hz, reference[0].hz);
        for (curve, values) in curves(&average).into_iter().enumerate() {
            assert_eq!(values.len(), params.mts.solver.hz_num);
            for (index, &value) in values.iter().enumerate() {
                let expected = reference
                    .iter()
                    .map(|output| curves(output)[curve][index])
                    .sum::<f64>();
                assert!((value - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn zero_width_and_single_sample_recover_the_central_probe_scan() {
        for (kv_half_width, kv_samples) in [(0.0, 21), (5.0, 1)] {
            let params = Params {
                kv_half_width,
                kv_samples,
                ..small_params()
            };
            let average = velocity_average(&params).unwrap();
            let reference = compute_demod(
                &MtsParams {
                    hamiltonian: HamiltonianParams {
                        frame: Frame::Probe,
                        ..params.mts.hamiltonian
                    },
                    ..params.mts
                },
                Vec3::from_angles(FRAC_PI_2, FRAC_PI_2),
            )
            .map(MtsCurves::from)
            .unwrap();
            assert_eq!(average.hz, reference.hz);
            assert_eq!(curves(&average), curves(&reference));
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
