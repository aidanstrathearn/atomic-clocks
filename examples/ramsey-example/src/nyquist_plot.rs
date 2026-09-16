use crate::feedback::controller;
use crate::params::RamseyParameters;
use crate::ramsey::measurement_transfer;
use atomic_clocks::signal_processing::{
    AngularFrequencyGrid, FunctionalPsd, LtiFeedback, LtiFilter,
};
use myplotlib::{AppResult, Plotter};

pub(crate) fn plot(params: &mut RamseyParameters) -> AppResult {
    let measurement = measurement_transfer(params)?;
    let grid = AngularFrequencyGrid::new(measurement.grid().values()[1..].to_vec())?;
    let controller = controller(params, &measurement);
    let unused_signal = FunctionalPsd { spectrum: |_| 0.0 };
    let unused_measurement_noise = FunctionalPsd { spectrum: |_| 0.0 };
    let feedback = LtiFeedback::new(
        unused_signal,
        unused_measurement_noise,
        measurement,
        controller,
    );
    let open_loop = feedback.open_loop().sample(&grid)?;

    let positive_real: Vec<_> = open_loop
        .response_values()
        .iter()
        .map(|response| response.re)
        .collect();
    let positive_imaginary: Vec<_> = open_loop
        .response_values()
        .iter()
        .map(|response| response.im)
        .collect();
    let negative_real: Vec<_> = positive_real.iter().rev().copied().collect();
    let negative_imaginary: Vec<_> = positive_imaginary
        .iter()
        .rev()
        .map(|imaginary| -imaginary)
        .collect();

    let mut plot = Plotter::new();
    plot.plot(&positive_real, &positive_imaginary)
        .label("Positive frequencies");
    plot.plot(&negative_real, &negative_imaginary)
        .label("Negative frequencies");
    plot.axhline(0.0).label("Im(L) = 0");
    plot.axvline(-1.0).label("Re(L) = -1");
    plot.title(format!(
        "Feedback Nyquist plot at lock point {:.3} kHz",
        params.detuning_khz
    ));
    plot.xlabel("Re(L)");
    plot.ylabel("Im(L)");
    plot.xlim(-2.0, 1.0);
    plot.ylim(-1.5, 1.5);
    Ok(plot)
}
