use std::f64::consts::{PI, TAU};

use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::maths::{fourier_transform, linspace, Linspace};
use atomic_clocks::twolevel::{
    steps, BlochVec, Channel, Hamiltonian, Observable, Process, TimeDependentHamiltonian,
    TrotterConfig, Unitary,
};
use myplotlib::{AppDefinition, AppResult, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};

const TIME_START_MS: f64 = -1.0;
const TIME_STOP_MS: f64 = 3.0;
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
}

impl Params {
    fn start_time_ms(&self) -> f64{
        - 4.0 * self.pulse_width_ms
    }
    fn stop_time_ms(&self) -> f64{
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

fn signal_plot(params: &mut Params) -> AppResult {
    let initial = BlochVec::ground();
    let observable = Observable::ground_projector();
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    params.detuning_khz = params
        .detuning_khz
        .clamp(-detuning_limit_khz, detuning_limit_khz);
    let detunings_khz = linspace(-detuning_limit_khz, detuning_limit_khz, N_DETUNINGS - 1);
    let mut signal = Vec::with_capacity(detunings_khz.len());

    for &detuning_khz in &detunings_khz {
        let ramsey = Ramsey {
            pulse_area: params.pulse_area,
            detuning: khz_to_angular_frequency(detuning_khz),
            pulse_width: params.pulse_width_ms,
            pulse_separation: params.ramsey_time_ms,
            phase_diff: 0.0,
        };
        let unitary = Unitary::from_system(
            &ramsey,
            TrotterConfig {
                start: params.start_time_ms(),
                stop: params.stop_time_ms(),
                nsteps: params.time_steps,
                tolerance: 0.0,
            },
        );
        signal.push(observable.expectation(unitary.apply_to(initial)));
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
    let times = Linspace::new(params.start_time_ms(), params.stop_time_ms(), params.time_steps);
    let ramsey = Ramsey {
        pulse_area: params.pulse_area,
        detuning: khz_to_angular_frequency(params.detuning_khz),
        pulse_width: params.pulse_width_ms,
        pulse_separation: params.ramsey_time_ms,
        phase_diff: 0.0,
    };
    // Hamiltonian stores h in h.sigma / 2, so h.z = 2 represents sigma_z.
    let perturbation = Hamiltonian::new(0.0, 0.0, 2.0);
    let observable = Observable::ground_projector();
    let response = Process::new(steps(&times.array, |t, dt| ramsey.h(t).for_duration(dt)))
        .linear_response(initial, observable, perturbation.into());
    // Responses include kicks at both the initial and final time boundaries.
    (times, response)
}

fn response_plot(params: &mut Params) -> AppResult {
    let (kick_times, response) = temporal_response(params);

    let mut plot = Plotter::new();
    plot.plot(&kick_times.array, &response)
        .label("Linear response");
    plot.axhline(0.0);
    plot.title(format!(
        "Sigma-z linear response at detuning {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Kick time (ms)");
    plot.ylabel("Final ground probability response per sigma-z kick");
    plot.xlim(params.start_time_ms(), params.stop_time_ms());
    Ok(plot)
}

fn frequency_response_plot(params: &mut Params) -> AppResult {
    let (kick_times, response) = temporal_response(params);
    let spectrum = fourier_transform(
        &response,
        kick_times.step,
        kick_times.array[0],
        khz_to_angular_frequency(FREQUENCY_STEP_KHZ),
    );
    let frequencies_khz: Vec<_> = spectrum
        .angular_frequencies
        .iter()
        .map(|&frequency| angular_frequency_to_khz(frequency))
        .collect();
    let magnitude: Vec<_> = spectrum
        .amplitudes
        .iter()
        .map(|value| value.norm())
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&frequencies_khz, &magnitude)
        .label("Response magnitude");
    plot.title(format!(
        "Sigma-z frequency response at detuning {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Frequency (kHz)");
    plot.ylabel("Fourier magnitude of linear response");
    plot.xlim(0.0, angular_frequency_to_khz(PI / kick_times.step));
    Ok(plot)
}

fn main() -> myplotlib::Result {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Ramsey signal", signal_plot, controls),
        ViewOption::new("Linear response", response_plot, controls),
        ViewOption::new("Frequency response", frequency_response_plot, controls),
    ];
    myplotlib::run_native(AppDefinition::new("Ramsey", "ramsey-canvas", VIEWS))
}
