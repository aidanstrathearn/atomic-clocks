use std::f64::consts::{PI, TAU};

use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::maths::{linspace, past_response_transform, Linspace};
use atomic_clocks::signal_processing::TransferFunctionSamples;
use atomic_clocks::twolevel::{
    steps, BlochVec, Channel, Hamiltonian, Observable, Process, TimeDependentHamiltonian,
    TrotterConfig, Unitary,
};
use myplotlib::{AppDefinition, AppResult, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};

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
    let (relative_times, response) = temporal_response(params);
    let transfer = TransferFunctionSamples::try_from(past_response_transform(
        &response,
        relative_times.step,
        khz_to_angular_frequency(FREQUENCY_STEP_KHZ),
    ))?;
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
    plot.xlim(0.0, angular_frequency_to_khz(PI / relative_times.step));
    Ok(plot)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

fn main() -> myplotlib::Result {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Ramsey signal", signal_plot, controls),
        ViewOption::new("Linear response", response_plot, controls),
        ViewOption::new("Frequency response", frequency_response_plot, controls),
    ];
    myplotlib::run_native(AppDefinition::new("Ramsey", "ramsey-canvas", VIEWS))
}
