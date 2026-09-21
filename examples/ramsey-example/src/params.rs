use std::f64::consts::{PI, TAU};

use atomic_clocks::common::DetuningScanParams;

pub(crate) struct RamseyParameters {
    pub(crate) pulse_width_ms: f64,
    pub(crate) ramsey_time_ms: f64,
    pub(crate) pulse_area: f64,
    pub(crate) detuning_khz: f64,
    pub(crate) time_steps: usize,
    pub(crate) scan: DetuningScanParams,
    pub(crate) oscillator_white_psd_log10: f64,
    pub(crate) oscillator_random_walk_log10: f64,
    pub(crate) measurement_white_psd_log10: f64,
    pub(crate) integrator_gain: f64,
    pub(crate) feedback_delay_ms: f64,
}

impl Default for RamseyParameters {
    fn default() -> Self {
        Self {
            pulse_width_ms: 0.05,
            ramsey_time_ms: 2.0,
            pulse_area: 0.5 * PI,
            detuning_khz: 0.0,
            time_steps: 501,
            scan: DetuningScanParams {
                hz_lim: TAU * 10.0,
                hz_num: 501,
            },
            oscillator_white_psd_log10: -4.0,
            oscillator_random_walk_log10: -6.0,
            measurement_white_psd_log10: -4.0,
            integrator_gain: 0.05,
            feedback_delay_ms: 0.05,
        }
    }
}
