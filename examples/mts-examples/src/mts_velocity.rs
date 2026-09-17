mod gradients;
mod probe;
mod response;
mod velocity;

use atomic_clocks::twolevel::Decay;
use atomic_clocks::vapourcell::{HamiltonianParams, LinearResponseSolverParams, MtsParams};
use myplotlib::{AppDefinition, Slider, SliderGroup, ViewOption};

use crate::mts;

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
            response_delay_periods: LinearResponseSolverParams::default().delay_periods,
        }
    }
}

pub(super) fn common_control_groups<'a>(
    hamiltonian: &'a mut HamiltonianParams,
    decay: &'a mut Decay,
    kr_n: &'a mut usize,
    steps_per_period: &'a mut usize,
    n_periods: &'a mut usize,
    kv_sigma: &'a mut f64,
    kv_window_half_width: &'a mut f64,
    kv_samples: &'a mut usize,
) -> [SliderGroup<'a>; 4] {
    hamiltonian.delta = 0.0;
    hamiltonian.kr = 0.0;
    decay.gamma_up = 0.0;

    let [modulation, rates] = mts::atom_control_groups(
        &mut hamiltonian.modulation,
        &mut hamiltonian.r_pump,
        &mut hamiltonian.r_prbe,
        &mut decay.gamma_down,
        &mut decay.gamma_phi,
    );
    let velocity = velocity::control_group(
        &mut hamiltonian.kv,
        kv_sigma,
        kv_window_half_width,
        kv_samples,
    );
    let solver = SliderGroup::new(
        "Solver",
        [
            Slider::new("Spatial phase samples", kr_n, 1..=21).step_by(2.0),
            Slider::new("Steps per period", steps_per_period, 20..=1_000),
            Slider::from_get_set("Warmup periods", 0.0..=100.0, move |warmup| {
                if let Some(warmup) = warmup {
                    *n_periods = warmup.round() as usize + 1;
                }
                n_periods.saturating_sub(1) as f64
            })
            .step_by(1.0),
        ],
    );

    [modulation, rates, velocity, solver]
}

pub(crate) fn definition() -> AppDefinition<Params> {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Signal", probe::plot, probe::controls),
        //ViewOption::new("Linear response", response::plot, response::controls),
        ViewOption::new(
            "Linear response",
            response::demodulated_plot,
            response::demodulated_controls,
        ),
        ViewOption::new(
            "Transfer function",
            response::frequency_plot,
            response::demodulated_controls,
        ),
        ViewOption::new("Harmonic gradients", gradients::plot, gradients::controls),
    ];
    AppDefinition::new("MTS velocity", "mts-velocity-canvas", VIEWS)
}

#[cfg(test)]
mod tests;
