use crate::feedback::spectra;
use crate::params::RamseyParameters;
use myplotlib::{AppResult, AxisScale, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let spectra = spectra(params)?;

    let mut plot = Plotter::new();
    plot.plot(&spectra.frequencies_khz, &spectra.free_running)
        .label("Input");

    // plot.plot(&spectra.frequencies_khz, &spectra.residual_signal)
    //     .label("Residual oscillator noise");

    plot.plot(
        &spectra.frequencies_khz,
        &spectra.injected_measurement_noise,
    )
    .label("measurement noise");

    plot.plot(&spectra.frequencies_khz, &spectra.total)
        .label("Output");
    plot.title(format!(
        "Feedback PSD at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Frequency (kHz)");
    plot.ylabel("Detuning PSD (kHz^2 / kHz)");
    plot.yscale(AxisScale::Log10);
    plot.xlim(0.0, 10.0 / params.ramsey_time_ms);
    Ok(plot)
}
