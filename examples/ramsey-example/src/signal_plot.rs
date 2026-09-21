use crate::params::RamseyParameters;
use crate::ramsey::signal;
use myplotlib::{AppResult, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let detunings_khz = params.scan.offsets()?;
    let signal = signal(params, &detunings_khz)?;

    let mut plot = Plotter::new();
    plot.plot(&signal.detunings, &signal.ground_probabilities)
        .label("Ramsey signal");
    plot.axvline(params.detuning_khz)
        .label(format!("Selected detuning: {:.3} kHz", params.detuning_khz));
    plot.title("Ramsey signal");
    plot.xlabel("Detuning (kHz)");
    plot.ylabel("Final ground-state probability");
    plot.xlim(-params.scan.hz_lim, params.scan.hz_lim);
    Ok(plot)
}
