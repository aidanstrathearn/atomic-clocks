use std::f64::consts::{PI, TAU};

use crate::params::RamseyParameters;
use atomic_clocks::interferometer::{
    Ramsey, RamseyResponse, RamseySignal, RamseySolver, RamseySolverConfig, RamseySolverError,
};
use atomic_clocks::signal_processing::TransferFunctionSamples;

const MAX_ANGULAR_FREQUENCY_STEP: f64 = 0.05;

pub(crate) fn khz_to_angular_frequency(frequency_khz: f64) -> f64 {
    TAU * frequency_khz
}

pub(crate) fn angular_frequency_to_khz(angular_frequency: f64) -> f64 {
    angular_frequency / TAU
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
    params: &mut RamseyParameters,
) -> Result<RamseyResponse, RamseySolverError> {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    params.detuning_khz = params
        .detuning_khz
        .clamp(-detuning_limit_khz, detuning_limit_khz);
    let mut response = solver(params, params.detuning_khz)?.detuning_response();
    // Convert response per angular-detuning impulse to response per kHz ms impulse.
    for value in &mut response.values {
        *value *= TAU;
    }
    Ok(response)
}

pub(crate) fn measurement_transfer(
    params: &mut RamseyParameters,
) -> Result<TransferFunctionSamples, Box<dyn std::error::Error>> {
    Ok(temporal_response(params)?.transfer_function(MAX_ANGULAR_FREQUENCY_STEP)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10 * (1.0 + expected.abs()),
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn selected_lock_point_sets_the_measurement_response() {
        let mut params = RamseyParameters::default();
        let positive = measurement_transfer(&mut params).unwrap().response_values()[0].re;
        assert!(params.integrator_gain * positive > 0.0);
        params.detuning_khz = -params.detuning_khz;
        let negative = measurement_transfer(&mut params).unwrap().response_values()[0].re;

        assert!(positive * negative < 0.0);
        assert_close(positive.abs(), negative.abs());
    }
}
