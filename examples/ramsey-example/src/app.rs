use crate::params::RamseyParameters;
use crate::{
    adev_plot, controls, feedback_plot, frequency_response_plot, nyquist_plot, response_plot,
    signal_plot,
};
use myplotlib::{AppDefinition, ViewOption};

const RAMSEY_VIEWS: &[ViewOption<RamseyParameters>] = &[
    ViewOption::new("Ramsey signal", signal_plot::plot, controls::signal),
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

pub(crate) fn definition() -> AppDefinition<RamseyParameters> {
    AppDefinition::new("Ramsey", RAMSEY_VIEWS)
}
