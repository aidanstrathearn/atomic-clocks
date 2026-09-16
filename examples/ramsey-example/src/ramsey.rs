use crate::params::RamseyParameters;
use atomic_clocks::interferometer::ramsey::Ramsey;
use atomic_clocks::maths::{Linspace, past_response_transform};
use atomic_clocks::signal_processing::{SpectrumError, TransferFunctionSamples};
use atomic_clocks::twolevel::{
    BlochVec, Channel, Hamiltonian, Observable, Process, TimeDependentHamiltonian, TrotterConfig,
    Unitary, steps,
};
use std::f64::consts::TAU;

const FREQUENCY_STEP_KHZ: f64 = 0.05 / TAU;

pub(crate) fn khz_to_angular_frequency(frequency_khz: f64) -> f64 {
    TAU * frequency_khz
}

pub(crate) fn angular_frequency_to_khz(angular_frequency: f64) -> f64 {
    angular_frequency / TAU
}

fn interferometer(params: &RamseyParameters, detuning_khz: f64) -> Ramsey {
    Ramsey {
        pulse_area: params.pulse_area,
        detuning: khz_to_angular_frequency(detuning_khz),
        pulse_width: params.pulse_width_ms,
        pulse_separation: params.ramsey_time_ms,
        phase_diff: 0.0,
    }
}

pub(crate) fn final_ground_probability(params: &RamseyParameters, detuning_khz: f64) -> f64 {
    let unitary = Unitary::from_system(
        &interferometer(params, detuning_khz),
        TrotterConfig {
            start: params.start_time_ms(),
            stop: params.measurement_time_ms(),
            nsteps: params.time_steps,
            tolerance: 0.0,
        },
    );
    Observable::ground_projector().expectation(unitary.apply_to(BlochVec::ground()))
}

pub(crate) fn temporal_response(params: &mut RamseyParameters) -> (Linspace, Vec<f64>) {
    let detuning_limit_khz = angular_frequency_to_khz(4.0 / params.pulse_width_ms);
    params.detuning_khz = params
        .detuning_khz
        .clamp(-detuning_limit_khz, detuning_limit_khz);
    let initial = BlochVec::ground();
    let times = Linspace::new(
        params.start_time_ms(),
        params.measurement_time_ms(),
        params.time_steps,
    );
    let ramsey = interferometer(params, params.detuning_khz);
    // Detuning in kHz enters H = 2 pi detuning sigma_z / 2 when time is in ms.
    let perturbation = Hamiltonian::new(0.0, 0.0, TAU);
    let observable = Observable::ground_projector();
    let response = Process::new(steps(&times.array, |t, dt| ramsey.h(t).for_duration(dt)))
        .linear_response(initial, observable, perturbation.into());
    // Responses include kicks at both the initial and final time boundaries.
    let relative_times = Linspace::new(
        params.start_time_ms() - params.measurement_time_ms(),
        0.0,
        params.time_steps,
    );
    (relative_times, response)
}

pub(crate) fn measurement_transfer(
    params: &mut RamseyParameters,
) -> Result<TransferFunctionSamples, SpectrumError> {
    let (relative_times, response) = temporal_response(params);
    TransferFunctionSamples::try_from(past_response_transform(
        &response,
        relative_times.step,
        khz_to_angular_frequency(FREQUENCY_STEP_KHZ),
    ))
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
    fn dc_response_matches_detuning_slope() {
        let mut params = RamseyParameters {
            time_steps: 4_000,
            ..RamseyParameters::default()
        };
        let (relative_times, response) = temporal_response(&mut params);
        let transfer = TransferFunctionSamples::try_from(past_response_transform(
            &response,
            relative_times.step,
            1.0,
        ))
        .unwrap();
        let dc_response = transfer.response_values()[0].re;

        let delta_khz = 1.0e-5;
        let finite_difference =
            (final_ground_probability(&params, params.detuning_khz + delta_khz)
                - final_ground_probability(&params, params.detuning_khz - delta_khz))
                / (2.0 * delta_khz);

        assert!(
            (dc_response - finite_difference).abs() < 2.0e-3,
            "DC response {dc_response} did not match detuning slope {finite_difference}"
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
