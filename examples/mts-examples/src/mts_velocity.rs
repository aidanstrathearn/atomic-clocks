mod gradients;
mod probe;
mod response;
mod velocity;

use atomic_clocks::vapourcell::{LinearResponseSolverParams, MtsParams};
use myplotlib::{AppDefinition, ViewOption};

pub(crate) struct Params {
    // In this example, `mts.hamiltonian.kv` is the Gaussian mean. Each solver
    // call replaces it with a quadrature node inside the window around kv = 0.
    pub(super) mts: MtsParams,
    pub(super) kv_sigma: f64,
    pub(super) kv_window_half_width: f64,
    pub(super) kv_samples: usize,
    pub(super) gradient_epsilon: f64,
    pub(super) gradient_harmonics: usize,
    pub(super) response_time_fraction: f64,
    pub(super) response_warmup_periods: usize,
    pub(super) response_delay_periods: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mts: MtsParams::default(),
            kv_sigma: 10.0,
            kv_window_half_width: 0.01,
            kv_samples: 1,
            gradient_epsilon: 1e-3,
            gradient_harmonics: 10,
            response_time_fraction: 0.0,
            response_warmup_periods: LinearResponseSolverParams::default().warmup_periods,
            response_delay_periods: LinearResponseSolverParams::default().delay_periods,
        }
    }
}

pub(crate) fn definition() -> AppDefinition<Params> {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Probe", probe::plot, probe::controls),
        ViewOption::new("Linear response", response::plot, response::controls),
        ViewOption::new(
            "Demodulated response",
            response::demodulated_plot,
            response::demodulated_controls,
        ),
        ViewOption::new(
            "Frequency response",
            response::frequency_plot,
            response::demodulated_controls,
        ),
        ViewOption::new("Harmonic gradients", gradients::plot, gradients::controls),
    ];
    AppDefinition::new("MTS velocity", "mts-velocity-canvas", VIEWS)
}

#[cfg(test)]
mod tests;
