use std::f64::consts::{PI, TAU};

use crate::params::RamseyParameters;
use atomic_clocks::interferometer::{
    Ramsey, RamseyResponse, RamseySignal, RamseySolver, RamseySolverConfig, RamseySolverError,
};
use atomic_clocks::signal_processing::TransferFunctionSamples;

const MAX_ANGULAR_FREQUENCY_STEP: f64 = 0.05;
pub(crate) const DETUNING_STEP_KHZ: f64 = 0.01;
const DETUNING_RANGE_FACTOR: f64 = 3.0;

pub(crate) fn khz_to_angular_frequency(frequency_khz: f64) -> f64 {
    TAU * frequency_khz
}

pub(crate) fn angular_frequency_to_khz(angular_frequency: f64) -> f64 {
    angular_frequency / TAU
}

pub(crate) fn detuning_limit_khz(pulse_width_ms: f64) -> f64 {
    assert!(
        pulse_width_ms.is_finite() && pulse_width_ms > 0.0,
        "pulse width must be finite and positive"
    );
    let raw_limit = angular_frequency_to_khz(DETUNING_RANGE_FACTOR / pulse_width_ms);
    (raw_limit / DETUNING_STEP_KHZ).floor() * DETUNING_STEP_KHZ
}

fn solver(params: &RamseyParameters, detuning_khz: f64) -> Result<RamseySolver, RamseySolverError> {
    Ramsey {
        pulse_area: params.pulse_area,
        detuning: khz_to_angular_frequency(detuning_khz),
        pulse_width: params.pulse_width_ms,
        pulse_separation: params.ramsey_time_ms,
        phase_diff: PI / 2.0,
    }
    .solver(RamseySolverConfig {
        pulse_tail_widths: 4.0,
        integration_steps: params.time_steps,
    })
}

pub(crate) fn signal(
    params: &RamseyParameters,
    detunings_khz: &[f64],
) -> Result<RamseySignal, RamseySolverError> {
    let angular_detunings: Vec<_> = detunings_khz
        .iter()
        .map(|&detuning| khz_to_angular_frequency(detuning))
        .collect();
    let mut signal = solver(params, params.detuning_khz)?.signal(&angular_detunings);
    for detuning in &mut signal.detunings {
        *detuning = angular_frequency_to_khz(*detuning);
    }
    Ok(signal)
}

pub(crate) fn temporal_response(
    params: &RamseyParameters,
) -> Result<RamseyResponse, RamseySolverError> {
    let mut response = solver(params, params.detuning_khz)?.detuning_response();
    // Convert response per angular-detuning impulse to response per kHz ms impulse.
    for value in &mut response.values {
        *value *= TAU;
    }
    Ok(response)
}

pub(crate) fn measurement_transfer(
    params: &RamseyParameters,
) -> Result<TransferFunctionSamples, Box<dyn std::error::Error>> {
    Ok(temporal_response(params)?.transfer_function(MAX_ANGULAR_FREQUENCY_STEP)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detuning_slider_lattice_contains_zero() {
        let limit = detuning_limit_khz(RamseyParameters::default().pulse_width_ms);
        let snapped_zero = -limit + (limit / DETUNING_STEP_KHZ).round() * DETUNING_STEP_KHZ;
        assert_eq!(snapped_zero, 0.0);
    }

    #[test]
    fn zero_detuning_is_a_responsive_lock_point() {
        let params = RamseyParameters::default();
        assert_eq!(params.detuning_khz, 0.0);
        let dc_response = measurement_transfer(&params).unwrap().response_values()[0].re;

        assert!(dc_response.is_finite());
        assert!(dc_response.abs() > 1.0e-3);
    }
}
