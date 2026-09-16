use std::{error::Error, fmt};

use crate::maths::{Linspace, normalised_gaussian, past_response_transform};
use crate::signal_processing::{SpectrumError, TransferFunctionSamples};
use crate::twolevel::{
    BlochVec, Channel, Hamiltonian, Observable, Process, TimeDependentHamiltonian, TrotterConfig,
    Unitary, steps,
};

#[derive(Clone, Copy, Debug)]
pub struct Ramsey {
    pub pulse_area: f64,
    pub detuning: f64,
    pub pulse_width: f64,
    pub pulse_separation: f64,
    pub phase_diff: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct RamseySolverConfig {
    /// Number of pulse widths retained before the first pulse and after the second.
    pub pulse_tail_widths: f64,
    /// Number of uniform integration intervals across the retained time window.
    pub integration_steps: usize,
}

impl Default for RamseySolverConfig {
    fn default() -> Self {
        Self {
            pulse_tail_widths: 4.0,
            integration_steps: 501,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RamseySolverError {
    NonFiniteParameter(&'static str),
    NonPositivePulseWidth,
    NegativePulseSeparation,
    NonPositivePulseTailWidths,
    NoIntegrationSteps,
    NonFiniteTimeWindow,
}

impl fmt::Display for RamseySolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteParameter(parameter) => {
                write!(formatter, "Ramsey parameter {parameter} must be finite")
            }
            Self::NonPositivePulseWidth => {
                write!(formatter, "Ramsey pulse width must be positive")
            }
            Self::NegativePulseSeparation => {
                write!(formatter, "Ramsey pulse separation must be nonnegative")
            }
            Self::NonPositivePulseTailWidths => {
                write!(formatter, "Ramsey pulse-tail cutoff must be positive")
            }
            Self::NoIntegrationSteps => {
                write!(
                    formatter,
                    "Ramsey solver requires at least one integration step"
                )
            }
            Self::NonFiniteTimeWindow => {
                write!(formatter, "Ramsey solver time window must be finite")
            }
        }
    }
}

impl Error for RamseySolverError {}

#[derive(Clone, Copy, Debug)]
pub struct RamseySolver {
    ramsey: Ramsey,
    config: RamseySolverConfig,
    start: f64,
    stop: f64,
}

#[derive(Clone, Debug)]
pub struct RamseySignal {
    /// Angular detunings in radians per model time unit.
    pub detunings: Vec<f64>,
    pub ground_probabilities: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RamseyResponse {
    /// Perturbation times relative to the final measurement, ending at zero.
    pub relative_times: Vec<f64>,
    /// Ground-probability response per unit angular-detuning impulse.
    pub values: Vec<f64>,
    pub time_step: f64,
}

impl Ramsey {
    pub fn solver(self, config: RamseySolverConfig) -> Result<RamseySolver, RamseySolverError> {
        for (name, value) in [
            ("pulse_area", self.pulse_area),
            ("detuning", self.detuning),
            ("pulse_width", self.pulse_width),
            ("pulse_separation", self.pulse_separation),
            ("phase_diff", self.phase_diff),
        ] {
            if !value.is_finite() {
                return Err(RamseySolverError::NonFiniteParameter(name));
            }
        }
        if self.pulse_width <= 0.0 {
            return Err(RamseySolverError::NonPositivePulseWidth);
        }
        if self.pulse_separation < 0.0 {
            return Err(RamseySolverError::NegativePulseSeparation);
        }
        if !config.pulse_tail_widths.is_finite() || config.pulse_tail_widths <= 0.0 {
            return Err(RamseySolverError::NonPositivePulseTailWidths);
        }
        if config.integration_steps == 0 {
            return Err(RamseySolverError::NoIntegrationSteps);
        }

        let tail_duration = config.pulse_tail_widths * self.pulse_width;
        let start = -tail_duration;
        let stop = self.pulse_separation + tail_duration;
        if !start.is_finite() || !stop.is_finite() || !(stop - start).is_finite() {
            return Err(RamseySolverError::NonFiniteTimeWindow);
        }
        Ok(RamseySolver {
            ramsey: self,
            config,
            start,
            stop,
        })
    }
}

impl RamseySolver {
    fn trotter_config(&self) -> TrotterConfig {
        TrotterConfig {
            start: self.start,
            stop: self.stop,
            nsteps: self.config.integration_steps,
            tolerance: 0.0,
        }
    }

    fn final_ground_probability_for(&self, ramsey: &Ramsey) -> f64 {
        let unitary = Unitary::from_system(ramsey, self.trotter_config());
        Observable::ground_projector().expectation(unitary.apply_to(BlochVec::ground()))
    }

    pub fn final_ground_probability(&self) -> f64 {
        self.final_ground_probability_for(&self.ramsey)
    }

    /// Ground-state probability at an angular detuning in radians per model time unit.
    pub fn ground_probability_at(&self, detuning: f64) -> f64 {
        assert!(detuning.is_finite(), "Ramsey detuning must be finite");
        self.final_ground_probability_for(&Ramsey {
            detuning,
            ..self.ramsey
        })
    }

    pub fn signal(&self, detunings: &[f64]) -> RamseySignal {
        RamseySignal {
            detunings: detunings.to_vec(),
            ground_probabilities: detunings
                .iter()
                .map(|&detuning| self.ground_probability_at(detuning))
                .collect(),
        }
    }

    pub fn detuning_response(&self) -> RamseyResponse {
        let times = Linspace::new(self.start, self.stop, self.config.integration_steps);
        let perturbation = Hamiltonian::new(0.0, 0.0, 1.0);
        let values = Process::new(steps(&times.array, |t, dt| {
            self.ramsey.h(t).for_duration(dt)
        }))
        .linear_response(
            BlochVec::ground(),
            Observable::ground_projector(),
            perturbation.into(),
        );
        RamseyResponse {
            relative_times: times
                .array
                .into_iter()
                .map(|time| time - self.stop)
                .collect(),
            values,
            time_step: times.step,
        }
    }
}

impl RamseyResponse {
    pub fn transfer_function(
        &self,
        max_angular_frequency_step: f64,
    ) -> Result<TransferFunctionSamples, SpectrumError> {
        TransferFunctionSamples::try_from(past_response_transform(
            &self.values,
            self.time_step,
            max_angular_frequency_step,
        ))
    }
}

impl TimeDependentHamiltonian for Ramsey {
    fn h(&self, t: f64) -> Hamiltonian {
        let (sin_phi, cos_phi) = self.phase_diff.sin_cos();
        let pulse1 = normalised_gaussian(t, 0.0, self.pulse_width);
        let pulse2 = normalised_gaussian(t, self.pulse_separation, self.pulse_width);
        Hamiltonian::new(
            self.pulse_area * (pulse1 + pulse2 * cos_phi),
            self.pulse_area * pulse2 * sin_phi,
            self.detuning,
        )
    }
}

pub struct ModulatedRamsey {
    pub pulse_area: f64,
    pub detuning: f64,
    pub mod_freq: f64,
    pub mod_depth: f64,
    pub start_time: f64,
    pub pulse_width: f64,
    pub pulse_separation: f64,
}

impl TimeDependentHamiltonian for ModulatedRamsey {
    fn h(&self, t: f64) -> Hamiltonian {
        let hx = self.pulse_area
            * (normalised_gaussian(t, 0.0, self.pulse_width)
                + normalised_gaussian(t, self.pulse_separation, self.pulse_width));
        Hamiltonian::new(
            hx,
            0.0,
            self.detuning + self.mod_depth * (self.mod_freq * (t + self.start_time)).sin(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramsey() -> Ramsey {
        Ramsey {
            pulse_area: 0.5 * std::f64::consts::PI,
            detuning: 1.0,
            pulse_width: 0.05,
            pulse_separation: 2.0,
            phase_diff: 0.0,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance * (1.0 + expected.abs()),
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn solver_uses_the_configured_pulse_window_for_response() {
        let config = RamseySolverConfig {
            pulse_tail_widths: 3.0,
            integration_steps: 80,
        };
        let model = ramsey();
        let response = model.solver(config).unwrap().detuning_response();
        let duration = model.pulse_separation + 2.0 * config.pulse_tail_widths * model.pulse_width;

        assert_eq!(response.relative_times.len(), config.integration_steps + 1);
        assert_eq!(response.values.len(), response.relative_times.len());
        assert_close(response.relative_times[0], -duration, 1.0e-14);
        assert_eq!(*response.relative_times.last().unwrap(), 0.0);
        assert_close(
            response.time_step,
            duration / config.integration_steps as f64,
            1.0e-14,
        );
    }

    #[test]
    fn signal_uses_angular_detunings_without_changing_the_solver_lock_point() {
        let solver = ramsey().solver(RamseySolverConfig::default()).unwrap();
        let detunings = [-0.7, 0.2, 1.4];
        let signal = solver.signal(&detunings);

        assert_eq!(signal.detunings, detunings);
        assert_eq!(signal.ground_probabilities.len(), detunings.len());
        for (&probability, &detuning) in signal.ground_probabilities.iter().zip(&detunings) {
            assert_close(probability, solver.ground_probability_at(detuning), 1.0e-14);
        }
        assert_close(
            solver.final_ground_probability(),
            solver.ground_probability_at(1.0),
            1.0e-14,
        );
    }

    #[test]
    fn dc_response_matches_the_final_probability_detuning_slope() {
        let solver = ramsey()
            .solver(RamseySolverConfig {
                integration_steps: 4_000,
                ..RamseySolverConfig::default()
            })
            .unwrap();
        let response = solver.detuning_response();
        let dc_response = response.transfer_function(1.0).unwrap().response_values()[0].re;

        let delta = 1.0e-5;
        let finite_difference = (solver.ground_probability_at(1.0 + delta)
            - solver.ground_probability_at(1.0 - delta))
            / (2.0 * delta);

        assert_close(dc_response, finite_difference, 5.0e-4);
    }

    #[test]
    fn solver_rejects_invalid_physics_and_sampling() {
        assert_eq!(
            Ramsey {
                pulse_width: 0.0,
                ..ramsey()
            }
            .solver(RamseySolverConfig::default())
            .unwrap_err(),
            RamseySolverError::NonPositivePulseWidth
        );
        assert_eq!(
            ramsey()
                .solver(RamseySolverConfig {
                    pulse_tail_widths: f64::NAN,
                    ..RamseySolverConfig::default()
                })
                .unwrap_err(),
            RamseySolverError::NonPositivePulseTailWidths
        );
        assert_eq!(
            ramsey()
                .solver(RamseySolverConfig {
                    integration_steps: 0,
                    ..RamseySolverConfig::default()
                })
                .unwrap_err(),
            RamseySolverError::NoIntegrationSteps
        );
    }
}
