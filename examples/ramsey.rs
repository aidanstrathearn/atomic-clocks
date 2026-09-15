use std::f64::consts::{PI, TAU};

use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::maths::{Linspace, linspace, past_response_transform};
use atomic_clocks::signal_processing::{
    AngularFrequencyGrid, Delay, FunctionalPsd, Integrator, LtiFeedback, LtiFilter, Psd,
    PsdSamples, Series, SpectrumError, TransferFunctionSamples,
};
use atomic_clocks::twolevel::{
    BlochVec, Channel, Hamiltonian, Observable, Process, TimeDependentHamiltonian, TrotterConfig,
    Unitary, steps,
};
use myplotlib::{AppDefinition, AppResult, AxisScale, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};

const N_DETUNINGS: usize = 500;
const FREQUENCY_STEP_KHZ: f64 = 0.1 / TAU;

fn khz_to_angular_frequency(frequency_khz: f64) -> f64 {
    TAU * frequency_khz
}

fn angular_frequency_to_khz(angular_frequency: f64) -> f64 {
    angular_frequency / TAU
}

struct Params {
    pulse_width_ms: f64,
    ramsey_time_ms: f64,
    pulse_area: f64,
    detuning_khz: f64,
    time_steps: usize,
    oscillator_white_psd_log10: f64,
    oscillator_random_walk_log10: f64,
    measurement_white_psd_log10: f64,
    integrator_gain: f64,
    feedback_delay_ms: f64,
}

impl Params {
    fn start_time_ms(&self) -> f64 {
        -4.0 * self.pulse_width_ms
    }

    fn measurement_time_ms(&self) -> f64 {
        self.ramsey_time_ms + 4.0 * self.pulse_width_ms
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            pulse_width_ms: 0.05,
            ramsey_time_ms: 2.0,
            pulse_area: 0.5 * PI,
            detuning_khz: angular_frequency_to_khz(1.0),
            time_steps: 501,
            oscillator_white_psd_log10: -4.0,
            oscillator_random_walk_log10: -6.0,
            measurement_white_psd_log10: -4.0,
            integrator_gain: 0.05,
            feedback_delay_ms: 0.05,
        }
    }
}

fn controls(params: &mut Params) -> SliderGrid<'_> {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    SliderGrid::new(
        1,
        [SliderGroup::new(
            "Ramsey parameters",
            [
                Slider::new("Pulse width (ms)", &mut params.pulse_width_ms, 0.02..=0.5),
                Slider::new("Ramsey time (ms)", &mut params.ramsey_time_ms, 0.02..=4.0),
                Slider::new("Pulse area (rad)", &mut params.pulse_area, 0.0..=2.0 * PI),
                Slider::new(
                    "Detuning (kHz)",
                    &mut params.detuning_khz,
                    -detuning_limit_khz..=detuning_limit_khz,
                ),
                Slider::new("Time steps", &mut params.time_steps, 2..=10_000),
            ],
        )],
    )
}

fn feedback_controls(params: &mut Params) -> SliderGrid<'_> {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    let Params {
        pulse_width_ms,
        ramsey_time_ms,
        pulse_area,
        detuning_khz,
        time_steps,
        oscillator_white_psd_log10,
        oscillator_random_walk_log10,
        measurement_white_psd_log10,
        integrator_gain,
        feedback_delay_ms,
    } = params;
    SliderGrid::new(
        5,
        [
            SliderGroup::new(
                "Ramsey parameters",
                [
                    Slider::new("Pulse width (ms)", pulse_width_ms, 0.02..=0.5),
                    Slider::new("Ramsey time (ms)", ramsey_time_ms, 0.02..=4.0),
                    Slider::new("Pulse area (rad)", pulse_area, 0.0..=2.0 * PI),
                    Slider::new(
                        "Detuning (kHz)",
                        detuning_khz,
                        -detuning_limit_khz..=detuning_limit_khz,
                    ),
                    Slider::new("Time steps", time_steps, 2..=10_000),
                ],
            ),
            SliderGroup::new(
                "Feedback parameters",
                [
                    Slider::new(
                        "log10 oscillator white PSD (kHz^2/kHz)",
                        oscillator_white_psd_log10,
                        -12.0..=2.0,
                    ),
                    Slider::new(
                        "log10 oscillator random walk (kHz^3)",
                        oscillator_random_walk_log10,
                        -12.0..=2.0,
                    ),
                    Slider::new(
                        "log10 measurement white PSD (probability^2/kHz)",
                        measurement_white_psd_log10,
                        -12.0..=2.0,
                    ),
                    Slider::new(
                        "Integrator gain magn. (kHz / probability ms)",
                        integrator_gain,
                        0.0..=5.0,
                    )
                    .step_by(0.001),
                    Slider::new("Feedback delay (ms)", feedback_delay_ms, 0.0..=2.0).step_by(0.001),
                ],
            ),
        ],
    )
}

fn nyquist_controls(params: &mut Params) -> SliderGrid<'_> {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    let Params {
        pulse_width_ms,
        ramsey_time_ms,
        pulse_area,
        detuning_khz,
        time_steps,
        integrator_gain,
        feedback_delay_ms,
        ..
    } = params;
    SliderGrid::new(
        2,
        [
            SliderGroup::new(
                "Ramsey parameters",
                [
                    Slider::new("Pulse width (ms)", pulse_width_ms, 0.02..=0.5),
                    Slider::new("Ramsey time (ms)", ramsey_time_ms, 0.02..=4.0),
                    Slider::new("Pulse area (rad)", pulse_area, 0.0..=2.0 * PI),
                    Slider::new(
                        "Detuning (kHz)",
                        detuning_khz,
                        -detuning_limit_khz..=detuning_limit_khz,
                    ),
                    Slider::new("Time steps", time_steps, 2..=10_000),
                ],
            ),
            SliderGroup::new(
                "Feedback parameters",
                [
                    Slider::new(
                        "Integrator gain magn. (kHz / probability ms)",
                        integrator_gain,
                        0.0..=5.0,
                    )
                    .step_by(0.001),
                    Slider::new("Feedback delay (ms)", feedback_delay_ms, 0.0..=2.0)
                        .step_by(0.001),
                ],
            ),
        ],
    )
}

fn ramsey(params: &Params, detuning_khz: f64) -> Ramsey {
    Ramsey {
        pulse_area: params.pulse_area,
        detuning: khz_to_angular_frequency(detuning_khz),
        pulse_width: params.pulse_width_ms,
        pulse_separation: params.ramsey_time_ms,
        phase_diff: 0.0,
    }
}

fn final_ground_probability(params: &Params, detuning_khz: f64) -> f64 {
    let unitary = Unitary::from_system(
        &ramsey(params, detuning_khz),
        TrotterConfig {
            start: params.start_time_ms(),
            stop: params.measurement_time_ms(),
            nsteps: params.time_steps,
            tolerance: 0.0,
        },
    );
    Observable::ground_projector().expectation(unitary.apply_to(BlochVec::ground()))
}

fn signal_plot(params: &mut Params) -> AppResult {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    params.detuning_khz = params
        .detuning_khz
        .clamp(-detuning_limit_khz, detuning_limit_khz);
    let detunings_khz = linspace(-detuning_limit_khz, detuning_limit_khz, N_DETUNINGS - 1);
    let mut signal = Vec::with_capacity(detunings_khz.len());

    for &detuning_khz in &detunings_khz {
        signal.push(final_ground_probability(params, detuning_khz));
    }

    let mut plot = Plotter::new();
    plot.plot(&detunings_khz, &signal).label("Ramsey signal");
    plot.axvline(params.detuning_khz)
        .label(format!("Selected detuning: {:.3} kHz", params.detuning_khz));
    plot.title("Ramsey signal");
    plot.xlabel("Detuning (kHz)");
    plot.ylabel("Final ground-state probability");
    plot.xlim(-detuning_limit_khz, detuning_limit_khz);
    Ok(plot)
}

fn temporal_response(params: &mut Params) -> (Linspace, Vec<f64>) {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    params.detuning_khz = params
        .detuning_khz
        .clamp(-detuning_limit_khz, detuning_limit_khz);
    let initial = BlochVec::ground();
    let times = Linspace::new(
        params.start_time_ms(),
        params.measurement_time_ms(),
        params.time_steps,
    );
    let ramsey = ramsey(params, params.detuning_khz);
    // Detuning in kHz enters H = 2 pi detuning sigma_z / 2 when time is in ms.
    let perturbation = Hamiltonian::new(0.0, 0.0, TAU);
    let observable = Observable::ground_projector();
    let response = Process::new(steps(&times.array, |t, dt| ramsey.h(t).for_duration(dt)))
        .linear_response(initial, observable, perturbation.into());
    // Responses include kicks at both the initial and final time boundaries.
    let relative_times = Linspace::new(
        params.start_time_ms() - params.measurement_time_ms(),
        0.0,
        params.time_steps,
    );
    (relative_times, response)
}

fn measurement_transfer(params: &mut Params) -> Result<TransferFunctionSamples, SpectrumError> {
    let (relative_times, response) = temporal_response(params);
    TransferFunctionSamples::try_from(past_response_transform(
        &response,
        relative_times.step,
        khz_to_angular_frequency(FREQUENCY_STEP_KHZ),
    ))
}

fn response_plot(params: &mut Params) -> AppResult {
    let (relative_times, response) = temporal_response(params);

    let mut plot = Plotter::new();
    plot.plot(&relative_times.array, &response)
        .label("Linear response");
    plot.axhline(0.0);
    plot.title(format!(
        "Detuning linear response at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Time relative to measurement, tau (ms)");
    plot.ylabel("Final ground probability response per kHz ms impulse");
    plot.xlim(relative_times.array[0], 0.0);
    Ok(plot)
}

fn frequency_response_plot(params: &mut Params) -> AppResult {
    let transfer = measurement_transfer(params)?;
    let frequencies_khz: Vec<_> = transfer
        .grid()
        .values()
        .iter()
        .map(|&frequency| angular_frequency_to_khz(frequency))
        .collect();
    let magnitude: Vec<_> = transfer
        .response_values()
        .iter()
        .map(|value| value.norm())
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&frequencies_khz, &magnitude)
        .label("Response magnitude");
    plot.title(format!(
        "Detuning frequency response at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Frequency (kHz)");
    plot.ylabel("Ground probability response magnitude (1 / kHz)");
    plot.yscale(AxisScale::Log10);
    plot.xlim(0.0, *frequencies_khz.last().unwrap());
    Ok(plot)
}

fn psd_per_khz_to_per_angular_frequency(value: f64) -> f64 {
    value / TAU
}

fn psd_per_angular_frequency_to_per_khz(value: f64) -> f64 {
    TAU * value
}

fn density_per_khz(samples: &PsdSamples) -> Vec<f64> {
    samples
        .density_values()
        .iter()
        .map(|&value| psd_per_angular_frequency_to_per_khz(value))
        .collect()
}

struct FeedbackSpectra {
    frequencies_khz: Vec<f64>,
    free_running: Vec<f64>,
    residual_signal: Vec<f64>,
    injected_measurement_noise: Vec<f64>,
    total: Vec<f64>,
}

fn feedback_controller(
    params: &Params,
    measurement: &TransferFunctionSamples,
) -> Series<Integrator, Delay> {
    let dc_slope = measurement.response_values()[0].re;
    let signed_gain = params.integrator_gain * dc_slope.signum();
    Integrator::new(signed_gain).then(Delay::new(params.feedback_delay_ms))
}

fn feedback_spectra(params: &mut Params) -> Result<FeedbackSpectra, SpectrumError> {
    let measurement = measurement_transfer(params)?;
    let grid = AngularFrequencyGrid::new(measurement.grid().values()[1..].to_vec())?;
    let frequencies_khz = grid
        .values()
        .iter()
        .map(|&omega| angular_frequency_to_khz(omega))
        .collect();

    let oscillator_white_psd = 10.0_f64.powf(params.oscillator_white_psd_log10);
    let oscillator_random_walk = 10.0_f64.powf(params.oscillator_random_walk_log10);
    let measurement_white_psd = 10.0_f64.powf(params.measurement_white_psd_log10);
    let oscillator_noise = FunctionalPsd {
        spectrum: move |omega| {
            let frequency_khz = angular_frequency_to_khz(omega);
            psd_per_khz_to_per_angular_frequency(
                oscillator_white_psd + oscillator_random_walk / frequency_khz.powi(2),
            )
        },
    };
    let measurement_noise = FunctionalPsd {
        spectrum: move |_| psd_per_khz_to_per_angular_frequency(measurement_white_psd),
    };
    let controller = feedback_controller(params, &measurement);
    let feedback = LtiFeedback::new(oscillator_noise, measurement_noise, measurement, controller);

    let free_running = feedback.free_running_psd().sample(&grid)?;
    let residual_signal = feedback.residual_signal_psd().sample(&grid)?;
    let injected_measurement_noise = feedback.injected_measurement_noise_psd().sample(&grid)?;
    let total = feedback.output_psd().sample(&grid)?;

    Ok(FeedbackSpectra {
        frequencies_khz,
        free_running: density_per_khz(&free_running),
        residual_signal: density_per_khz(&residual_signal),
        injected_measurement_noise: density_per_khz(&injected_measurement_noise),
        total: density_per_khz(&total),
    })
}

fn nyquist_plot(params: &mut Params) -> AppResult {
    let measurement = measurement_transfer(params)?;
    let grid = AngularFrequencyGrid::new(measurement.grid().values()[1..].to_vec())?;
    let controller = feedback_controller(params, &measurement);
    let unused_signal = FunctionalPsd { spectrum: |_| 0.0 };
    let unused_measurement_noise = FunctionalPsd { spectrum: |_| 0.0 };
    let feedback = LtiFeedback::new(
        unused_signal,
        unused_measurement_noise,
        measurement,
        controller,
    );
    let open_loop = feedback.open_loop().sample(&grid)?;

    let positive_real: Vec<_> = open_loop
        .response_values()
        .iter()
        .map(|response| response.re)
        .collect();
    let positive_imaginary: Vec<_> = open_loop
        .response_values()
        .iter()
        .map(|response| response.im)
        .collect();
    let negative_real: Vec<_> = positive_real.iter().rev().copied().collect();
    let negative_imaginary: Vec<_> = positive_imaginary
        .iter()
        .rev()
        .map(|imaginary| -imaginary)
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&positive_real, &positive_imaginary)
        .label("Positive frequencies");
    plot.plot(&negative_real, &negative_imaginary)
        .label("Negative frequencies");
    plot.axhline(0.0).label("Im(L) = 0");
    plot.axvline(-1.0).label("Re(L) = -1");
    plot.title(format!(
        "Feedback Nyquist plot at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Re(L)");
    plot.ylabel("Im(L)");
    plot.xlim(-2.0, 1.0);
    plot.ylim(-1.5, 1.5);
    Ok(plot)
}

fn feedback_plot(params: &mut Params) -> AppResult {
    let spectra = feedback_spectra(params)?;

    let mut plot = Plotter::new();
    plot.plot(&spectra.frequencies_khz, &spectra.free_running)
        .label("Input");

    // plot.plot(&spectra.frequencies_khz, &spectra.residual_signal)
    //     .label("Residual oscillator noise");

    plot.plot(
        &spectra.frequencies_khz,
        &spectra.injected_measurement_noise,
    )
    .label("measurement noise");

    plot.plot(&spectra.frequencies_khz, &spectra.total)
        .label("Output");
    plot.title(format!(
        "Feedback PSD at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Frequency (kHz)");
    plot.ylabel("Detuning PSD (kHz^2 / kHz)");
    plot.yscale(AxisScale::Log10);
    // plot.xlim(
    //     spectra.frequencies_khz[0],
    //     *spectra.frequencies_khz.last().unwrap(),
    // );

    plot.xlim(
        0.0,
        10.0 / params.ramsey_time_ms,
    );
    Ok(plot)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10 * (1.0 + expected.abs()),
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn dc_response_matches_detuning_slope() {
        let mut params = Params {
            time_steps: 4_000,
            ..Params::default()
        };
        let (relative_times, response) = temporal_response(&mut params);
        let transfer = TransferFunctionSamples::try_from(past_response_transform(
            &response,
            relative_times.step,
            1.0,
        ))
        .unwrap();
        let dc_response = transfer.response_values()[0].re;

        let delta_khz = 1.0e-5;
        let finite_difference =
            (final_ground_probability(&params, params.detuning_khz + delta_khz)
                - final_ground_probability(&params, params.detuning_khz - delta_khz))
                / (2.0 * delta_khz);

        assert!(
            (dc_response - finite_difference).abs() < 2.0e-3,
            "DC response {dc_response} did not match detuning slope {finite_difference}"
        );
    }

    #[test]
    fn psd_frequency_density_conversion_round_trips() {
        let per_khz = 3.25;
        assert_close(
            psd_per_angular_frequency_to_per_khz(psd_per_khz_to_per_angular_frequency(per_khz)),
            per_khz,
        );
    }

    #[test]
    fn selected_lock_point_sets_the_measurement_response() {
        let mut params = Params::default();
        let positive = measurement_transfer(&mut params).unwrap().response_values()[0].re;
        assert!(params.integrator_gain * positive > 0.0);
        params.detuning_khz = -params.detuning_khz;
        let negative = measurement_transfer(&mut params).unwrap().response_values()[0].re;

        assert!(positive * negative < 0.0);
        assert_close(positive.abs(), negative.abs());
    }

    #[test]
    fn default_feedback_spectra_are_finite_and_add_up() {
        let spectra = feedback_spectra(&mut Params::default()).unwrap();
        assert!(!spectra.frequencies_khz.is_empty());
        assert!(spectra.frequencies_khz.iter().all(|value| *value > 0.0));
        for (((free_running, residual), injected), total) in spectra
            .free_running
            .iter()
            .zip(&spectra.residual_signal)
            .zip(&spectra.injected_measurement_noise)
            .zip(&spectra.total)
        {
            assert!(free_running.is_finite() && *free_running >= 0.0);
            assert!(residual.is_finite() && *residual >= 0.0);
            assert!(injected.is_finite() && *injected >= 0.0);
            assert!(total.is_finite() && *total >= 0.0);
            assert_close(*total, residual + injected);
        }
    }
}

fn main() -> myplotlib::Result {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Ramsey signal", signal_plot, controls),
        ViewOption::new("Linear response", response_plot, controls),
        ViewOption::new("Frequency response", frequency_response_plot, controls),
        ViewOption::new("Feedback Nyquist", nyquist_plot, nyquist_controls),
        ViewOption::new("Feedback PSD", feedback_plot, feedback_controls),
    ];
    myplotlib::run_native(AppDefinition::new("Ramsey", "ramsey-canvas", VIEWS))
}
