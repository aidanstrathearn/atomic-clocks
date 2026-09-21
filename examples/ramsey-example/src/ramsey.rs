use std::{
    error::Error,
    f64::consts::{PI, TAU},
};

use crate::params::RamseyParameters;
use atomic_clocks::interferometer::{
    Ramsey, RamseyResponse, RamseySignal, RamseySolver, RamseySolverConfig, RamseySolverError,
};
use atomic_clocks::signal_processing::TransferFunctionSamples;

const MAX_ANGULAR_FREQUENCY_STEP: f64 = 0.05;

pub(crate) fn angular_frequency_to_khz(angular_frequency: f64) -> f64 {
    angular_frequency / TAU
}

fn solver(params: &RamseyParameters) -> Result<RamseySolver, RamseySolverError> {
    Ramsey {
        pulse_area: params.pulse_area,
        detuning: TAU * params.detuning_khz,
        pulse_width: params.pulse_width_ms,
        pulse_separation: params.ramsey_time_ms,
        phase_diff: PI / 2.0,
    }
    .solver(RamseySolverConfig {
        pulse_tail_widths: 4.0,
        integration_steps: params.time_steps,
    })
}

pub(crate) fn signal(params: &RamseyParameters) -> Result<RamseySignal, Box<dyn Error>> {
    let angular_detunings = params.scan.angular_offsets()?;
    let mut signal = solver(params)?.signal(&angular_detunings);
    signal.detunings = params.scan.frequency_offsets()?;
    Ok(signal)
}

pub(crate) fn temporal_response(
    params: &RamseyParameters,
) -> Result<RamseyResponse, RamseySolverError> {
    let mut response = solver(params)?.detuning_response();
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
    fn default_detuning_scan_contains_zero() {
        let params = RamseyParameters::default();
        let detunings = params.scan.frequency_offsets().unwrap();
        assert_eq!(detunings.len(), params.scan.hz_num);
        assert!(detunings[detunings.len() / 2].abs() < 1.0e-12);
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
