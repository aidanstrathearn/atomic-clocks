use crate::params::RamseyParameters;
use crate::{
    adev_plot, controls, feedback_plot, frequency_response_plot, nyquist_plot, response_plot,
    signal_plot,
};
use myplotlib::{AppDefinition, ViewOption};

const RAMSEY_VIEWS: &[ViewOption<RamseyParameters>] = &[
    ViewOption::new("Ramsey signal", signal_plot::plot, controls::standard),
    ViewOption::new("Linear response", response_plot::plot, controls::standard),
    ViewOption::new(
        "Frequency response",
        frequency_response_plot::plot,
        controls::standard,
    ),
    ViewOption::new("Feedback Nyquist", nyquist_plot::plot, controls::nyquist),
    ViewOption::new("Feedback PSD", feedback_plot::plot, controls::feedback),
    ViewOption::new("Allan deviation", adev_plot::plot, controls::feedback),
];

const RAMSEY_APP: AppDefinition<RamseyParameters> =
    AppDefinition::new("Ramsey", "ramsey-canvas", RAMSEY_VIEWS);

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_native() -> myplotlib::NativeResult {
    myplotlib::run_native(RAMSEY_APP)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn run_web() -> myplotlib::WebResult {
    myplotlib::run_web(RAMSEY_APP)
}
