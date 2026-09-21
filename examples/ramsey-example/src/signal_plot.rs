use crate::params::RamseyParameters;
use crate::ramsey::signal;
use myplotlib::{AppResult, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let signal = signal(params)?;

    let mut plot = Plotter::new();
    plot.plot(&signal.detunings, &signal.ground_probabilities)
        .label("Ramsey signal");
    plot.axvline(params.detuning_khz)
        .label(format!("Selected detuning: {:.3} kHz", params.detuning_khz));
    plot.title("Ramsey signal");
    plot.xlabel("Detuning (kHz)");
    plot.ylabel("Final ground-state probability");
    let detuning_limit = params.scan.frequency_half_range();
    plot.xlim(-detuning_limit, detuning_limit);
    Ok(plot)
}
