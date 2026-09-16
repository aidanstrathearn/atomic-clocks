use crate::params::RamseyParameters;
use crate::ramsey::{angular_frequency_to_khz, measurement_transfer};
use myplotlib::{AppResult, AxisScale, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let transfer = measurement_transfer(params)?;
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
    plot.yscale(AxisScale::Log10);
    plot.xlim(0.0, *frequencies_khz.last().unwrap());
    Ok(plot)
}
