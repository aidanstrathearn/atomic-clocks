use crate::feedback::adev;
use crate::params::RamseyParameters;
use myplotlib::{AppResult, AxisScale, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let adev = adev(params)?;

    let mut plot = Plotter::new();
    plot.plot(
        adev.free_running.averaging_times(),
        adev.free_running.deviation_values(),
    )
    .label("Free-running oscillator");
    plot.plot(adev.total.averaging_times(), adev.total.deviation_values())
        .label("Locked oscillator");
    plot.title(format!(
        "Allan deviation at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Averaging time (ms)");
    plot.ylabel("Detuning Allan deviation (kHz)");
    plot.xscale(AxisScale::Log10);
    plot.yscale(AxisScale::Log10);
    Ok(plot)
}
