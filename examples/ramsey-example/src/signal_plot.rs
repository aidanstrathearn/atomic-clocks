use crate::params::RamseyParameters;
use crate::ramsey::{detuning_limit_khz, signal};
use atomic_clocks::maths::linspace;
use myplotlib::{AppResult, Plotter};

const N_DETUNINGS: usize = 501;

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let detuning_limit = detuning_limit_khz(params.pulse_width_ms);
    let detunings_khz = linspace(-detuning_limit, detuning_limit, N_DETUNINGS - 1);
    let signal = signal(params, &detunings_khz)?;

    let mut plot = Plotter::new();
    plot.plot(&signal.detunings, &signal.ground_probabilities)
        .label("Ramsey signal");
    plot.axvline(params.detuning_khz)
        .label(format!("Selected detuning: {:.3} kHz", params.detuning_khz));
    plot.title("Ramsey signal");
    plot.xlabel("Detuning (kHz)");
    plot.ylabel("Final ground-state probability");
    plot.xlim(-detuning_limit, detuning_limit);
    Ok(plot)
}
