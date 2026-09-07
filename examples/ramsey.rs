use std::f64::consts::PI;

use atomic_clocks::maths::{Linspace, linspace};
use atomic_clocks::ramsey::Ramsey;
use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Solver};
use myplotlib::{AppDefinition, AppResult, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};

const TIME_START: f64 = -1.0;
const TIME_STOP: f64 = 3.0;
const PULSE_SEPARATION: f64 = 2.0;

struct Params {
    pulse_width: f64,
    pulse_area: f64,
    tolerance: f64,
    detuning: f64,
    time_steps: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            pulse_width: 0.05,
            pulse_area: 0.5 * PI,
            tolerance: 1e-8,
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
                Slider::new("Tolerance", &mut params.tolerance, 1e-14..=1e-1)
                    .logarithmic(true)
                    .custom_formatter(|value, _| format!("{value:.1e}")),
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
    let detunings = linspace(-detuning_limit, detuning_limit, 500);
    let mut full_signal = Vec::with_capacity(detunings.len());
    let mut reduced_signal = Vec::with_capacity(detunings.len());
    let mut min_steps = params.time_steps;
    let mut max_steps = 0;
    let mut total_steps = 0;

    for &detuning in &detunings {
        let ramsey = Ramsey {
            pulse_area: params.pulse_area,
            detuning,
            pulse_width: params.pulse_width,
            pulse_separation: PULSE_SEPARATION,
            phase_diff: 0.0,
        };
        // Linspace includes both endpoints, so N samples require N - 1 intervals.
        let mut solver = Solver::from(
            &ramsey,
            Linspace::new(TIME_START, TIME_STOP, params.time_steps - 1),
        );
        full_signal.push(solver.propagate_to_final(initial).ground_probability());

        solver.trotter_reduce(params.tolerance);
        reduced_signal.push(solver.propagate_to_final(initial).ground_probability());

        let steps = solver.h_t.len();
        min_steps = min_steps.min(steps);
        max_steps = max_steps.max(steps);
        total_steps += steps;
    }

    let mean_steps = total_steps as f64 / detunings.len() as f64;
    let mut plot = Plotter::new();
    plot.plot(&detunings, &full_signal)
        .label(format!("Full ({} steps)", params.time_steps));
    plot.plot(&detunings, &reduced_signal).label(format!(
        "Trotter ({min_steps}–{max_steps} steps, mean {mean_steps:.1})"
    ));
    plot.axvline(params.detuning)
        .label(format!("Selected detuning: {:.3}", params.detuning));
    plot.title("Ramsey signal: full vs Trotter reduction");
    plot.xlabel("Detuning (rad / time)");
    plot.ylabel("Final ground-state probability");
    plot.xlim(-detuning_limit, detuning_limit);
    Ok(plot)
}

fn response_plot(params: &mut Params) -> AppResult {
    let detuning_limit = 4.0 / params.pulse_width;
    params.detuning = params.detuning.clamp(-detuning_limit, detuning_limit);
    let initial = BlochVec::ground();
    let times = Linspace::new(TIME_START, TIME_STOP, params.time_steps - 1);
    let step = times.step;
    // Each response is to a kick after its propagation step.
    let full_times: Vec<_> = times.array.iter().map(|&t| t + step).collect();
    let ramsey = Ramsey {
        pulse_area: params.pulse_area,
        detuning: params.detuning,
        pulse_width: params.pulse_width,
        pulse_separation: PULSE_SEPARATION,
        phase_diff: 0.0,
    };
    let mut solver = Solver::from(&ramsey, times);
    // Hamiltonian stores h in h.sigma / 2, so h.z = 2 represents sigma_z.
    let perturbation = Hamiltonian::new(0.0, 0.0, 2.0);
    let full_response = solver.linear_response(initial, perturbation);

    let reduced_times: Vec<_> = solver
        .trotter_reduce(params.tolerance)
        .into_iter()
        .map(|t| t + step)
        .collect();
    let reduced_response = solver.linear_response(initial, perturbation);

    let mut plot = Plotter::new();
    plot.plot(&full_times, &full_response)
        .label(format!("Full ({} steps)", params.time_steps));
    plot.plot(&reduced_times, &reduced_response)
        .label(format!("Trotter ({} steps)", solver.h_t.len()));
    plot.axhline(0.0);
    plot.title(format!(
        "Sigma-z linear response at detuning {:.3}",
        params.detuning
    ));
    plot.xlabel("Kick time");
    plot.ylabel("Final ground probability response per sigma-z kick");
    plot.xlim(TIME_START, TIME_STOP + step);
    Ok(plot)
}

fn main() -> myplotlib::Result {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Ramsey signal", signal_plot, controls),
        ViewOption::new("Linear response", response_plot, controls),
    ];
    myplotlib::run_native(AppDefinition::new("Ramsey", "ramsey-canvas", VIEWS))
}
