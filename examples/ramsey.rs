use std::f64::consts::PI;

use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::maths::{Linspace, fourier_transform, linspace};
use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver};
use myplotlib::{AppDefinition, AppResult, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};

const TIME_START: f64 = -1.0;
const TIME_STOP: f64 = 3.0;
const PULSE_SEPARATION: f64 = 2.0;
const N_DETUNINGS: usize = 500;
const FREQUENCY_STEP: f64 = 0.1; // Angular frequency, in rad / time.
struct Params {
    pulse_width: f64,
    pulse_area: f64,
    detuning: f64,
    time_steps: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            pulse_width: 0.05,
            pulse_area: 0.5 * PI,
            detuning: 1.0,
            time_steps: 501,
        }
    }
}

fn controls(params: &mut Params) -> SliderGrid<'_> {
    let detuning_limit = 4.0 / params.pulse_width;
    SliderGrid::new(
        1,
        [SliderGroup::new(
            "Ramsey parameters",
            [
                Slider::new("Pulse width", &mut params.pulse_width, 0.02..=0.5),
                Slider::new("Pulse area (rad)", &mut params.pulse_area, 0.0..=2.0 * PI),
                Slider::new(
                    "Detuning",
                    &mut params.detuning,
                    -detuning_limit..=detuning_limit,
                ),
                Slider::new("Time steps", &mut params.time_steps, 2..=10_000),
            ],
        )],
    )
}

fn signal_plot(params: &mut Params) -> AppResult {
    let initial = BlochVec::ground();
    let detuning_limit = 4.0 / params.pulse_width;
    params.detuning = params.detuning.clamp(-detuning_limit, detuning_limit);
    let detunings = linspace(-detuning_limit, detuning_limit, N_DETUNINGS - 1);
    let mut signal = Vec::with_capacity(detunings.len());

    for &detuning in &detunings {
        let ramsey = Ramsey {
            pulse_area: params.pulse_area,
            detuning,
            pulse_width: params.pulse_width,
            pulse_separation: PULSE_SEPARATION,
            phase_diff: 0.0,
        };
        let solver = Solver::from(
            &ramsey,
            Linspace::new(TIME_START, TIME_STOP, params.time_steps),
        );
        signal.push(solver.propagate_to_final(initial).ground_probability());
    }

    let mut plot = Plotter::new();
    plot.plot(&detunings, &signal).label("Ramsey signal");
    plot.axvline(params.detuning)
        .label(format!("Selected detuning: {:.3}", params.detuning));
    plot.title("Ramsey signal");
    plot.xlabel("Detuning (rad / time)");
    plot.ylabel("Final ground-state probability");
    plot.xlim(-detuning_limit, detuning_limit);
    Ok(plot)
}

fn temporal_response(params: &mut Params) -> (Linspace, Vec<f64>) {
    let detuning_limit = 4.0 / params.pulse_width;
    params.detuning = params.detuning.clamp(-detuning_limit, detuning_limit);
    let initial = BlochVec::ground();
    let times = Linspace::new(TIME_START, TIME_STOP, params.time_steps);
    let ramsey = Ramsey {
        pulse_area: params.pulse_area,
        detuning: params.detuning,
        pulse_width: params.pulse_width,
        pulse_separation: PULSE_SEPARATION,
        phase_diff: 0.0,
    };
    let solver = Solver::from(&ramsey, times.clone());
    // Hamiltonian stores h in h.sigma / 2, so h.z = 2 represents sigma_z.
    let perturbation = Hamiltonian::new(0.0, 0.0, 2.0);
    let response = solver.linear_response(initial, perturbation);
    // Each response is to a kick after its propagation step.
    let kick_times = Linspace {
        step: times.step,
        array: times.array.into_iter().skip(1).collect(),
    };
    (kick_times, response)
}

fn response_plot(params: &mut Params) -> AppResult {
    let (kick_times, response) = temporal_response(params);

    let mut plot = Plotter::new();
    plot.plot(&kick_times.array, &response)
        .label("Linear response");
    plot.axhline(0.0);
    plot.title(format!(
        "Sigma-z linear response at detuning {:.3}",
        params.detuning
    ));
    plot.xlabel("Kick time");
    plot.ylabel("Final ground probability response per sigma-z kick");
    plot.xlim(TIME_START, TIME_STOP);
    Ok(plot)
}

fn frequency_response_plot(params: &mut Params) -> AppResult {
    let (kick_times, response) = temporal_response(params);
    let spectrum = fourier_transform(
        &response,
        kick_times.step,
        kick_times.array[0],
        FREQUENCY_STEP,
    );
    let magnitude: Vec<_> = spectrum
        .amplitudes
        .iter()
        .map(|value| value.norm())
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&spectrum.angular_frequencies, &magnitude)
        .label("Response magnitude");
    plot.title(format!(
        "Sigma-z frequency response at detuning {:.3}",
        params.detuning
    ));
    plot.xlabel("Angular frequency (rad / time)");
    plot.ylabel("Fourier magnitude of linear response");
    plot.xlim(0.0, PI / kick_times.step);
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
