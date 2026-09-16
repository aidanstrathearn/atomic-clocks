use crate::params::RamseyParameters;
use crate::ramsey::{angular_frequency_to_khz, signal};
use atomic_clocks::maths::linspace;
use myplotlib::{AppResult, Plotter};

const N_DETUNINGS: usize = 500;

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    params.detuning_khz = params
        .detuning_khz
        .clamp(-detuning_limit_khz, detuning_limit_khz);
    let detunings_khz = linspace(-detuning_limit_khz, detuning_limit_khz, N_DETUNINGS - 1);
    let signal = signal(params, &detunings_khz)?;

    let mut plot = Plotter::new();
    plot.plot(&signal.detunings, &signal.ground_probabilities)
        .label("Ramsey signal");
    plot.axvline(params.detuning_khz)
        .label(format!("Selected detuning: {:.3} kHz", params.detuning_khz));
    plot.title("Ramsey signal");
    plot.xlabel("Detuning (kHz)");
    plot.ylabel("Final ground-state probability");
    plot.xlim(-detuning_limit_khz, detuning_limit_khz);
    Ok(plot)
}
