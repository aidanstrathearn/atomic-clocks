use crate::params::RamseyParameters;
use crate::ramsey::temporal_response;
use myplotlib::{AppResult, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let response = temporal_response(params)?;

    let mut plot = Plotter::new();
    plot.plot(&response.relative_times, &response.values)
        .label("Linear response");
    plot.axhline(0.0);
    plot.title(format!(
        "Detuning linear response at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Time relative to measurement, tau (ms)");
    plot.ylabel("Final ground probability response per kHz ms impulse");
    plot.xlim(response.relative_times[0], 0.0);
    Ok(plot)
}
