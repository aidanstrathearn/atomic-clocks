use crate::feedback::spectra;
use crate::params::RamseyParameters;
use crate::ramsey::angular_frequency_to_khz;
use atomic_clocks::signal_processing::PsdSamples;
use myplotlib::{AppResult, AxisScale, Plotter};
use std::f64::consts::TAU;

fn density_per_khz(samples: &PsdSamples) -> Vec<f64> {
    samples
        .density_values()
        .iter()
        .map(|&value| TAU * value)
        .collect()
}

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let spectra = spectra(params)?;
    let frequencies_khz: Vec<_> = spectra
        .output()
        .grid()
        .values()
        .iter()
        .map(|&omega| angular_frequency_to_khz(omega))
        .collect();
    let free_running = density_per_khz(spectra.free_running());
    let injected_measurement_noise = density_per_khz(spectra.injected_measurement_noise());
    let total = density_per_khz(spectra.output());

    let mut plot = Plotter::new();
    plot.plot(&frequencies_khz, &free_running).label("Input");
    plot.plot(&frequencies_khz, &injected_measurement_noise)
        .label("measurement noise");
    plot.plot(&frequencies_khz, &total).label("Output");
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
