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

#[derive(Clone, Debug, PartialEq)]
pub enum RamseyError {
    NonFiniteParameter(&'static str),
    NonPositivePulseWidth,
    NegativePulseSeparation,
}

impl fmt::Display for RamseyError {
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
        }
    }
}

impl Error for RamseyError {}

#[derive(Clone, Copy, Debug)]
pub struct RamseySolver {
    ramsey: Ramsey,
    trotter: TrotterConfig,
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

impl Ramsey {
    pub fn validate(&self) -> Result<(), RamseyError> {
        for (name, value) in [
            ("pulse_area", self.pulse_area),
            ("detuning", self.detuning),
            ("pulse_width", self.pulse_width),
            ("pulse_separation", self.pulse_separation),
            ("phase_diff", self.phase_diff),
        ] {
            if !value.is_finite() {
                return Err(RamseyError::NonFiniteParameter(name));
            }
        }
        if self.pulse_width <= 0.0 {
            return Err(RamseyError::NonPositivePulseWidth);
        }
        if self.pulse_separation < 0.0 {
            return Err(RamseyError::NegativePulseSeparation);
        }

        Ok(())
    }
    pub fn solver(self, config: TrotterConfig) -> Result<RamseySolver, RamseyError> {
        self.validate()?;
        config.validate();
        Ok(RamseySolver {
            ramsey: self,
            trotter: config,
        })
    }
}

impl RamseySolver {
    fn final_ground_probability_for(&self, ramsey: &Ramsey) -> f64 {
        let unitary = Unitary::from_system(ramsey, self.trotter);
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

    /// Uses every interval in the configured time grid so that responses can be
    /// evaluated at each boundary; the Trotter reduction tolerance is ignored.
    pub fn detuning_response(&self) -> RamseyResponse {
        let times = Linspace::new(self.trotter.start, self.trotter.stop, self.trotter.nsteps);
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
                .map(|time| time - self.trotter.stop)
                .collect(),
            values,
            time_step: times.step,
        }
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

    fn trotter_config(nsteps: usize) -> TrotterConfig {
        TrotterConfig {
            start: -0.2,
            stop: 2.2,
            nsteps,
            tolerance: 0.0,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance * (1.0 + expected.abs()),
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn solver_uses_the_configured_time_window_for_response() {
        let config = TrotterConfig {
            start: -0.15,
            stop: 2.15,
            nsteps: 80,
            tolerance: 0.0,
        };
        let model = ramsey();
        let response = model.solver(config).unwrap().detuning_response();
        let duration = config.stop - config.start;

        assert_eq!(response.relative_times.len(), config.nsteps + 1);
        assert_eq!(response.values.len(), response.relative_times.len());
        assert_close(response.relative_times[0], -duration, 1.0e-14);
        assert_eq!(*response.relative_times.last().unwrap(), 0.0);
        assert_close(response.time_step, duration / config.nsteps as f64, 1.0e-14);
    }

    #[test]
    fn signal_uses_angular_detunings_without_changing_the_solver_lock_point() {
        let solver = ramsey().solver(trotter_config(501)).unwrap();
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
        let solver = ramsey().solver(trotter_config(4_000)).unwrap();
        let response = solver.detuning_response();
        let dc_response = response.transfer_function(1.0).unwrap().response_values()[0].re;

        let delta = 1.0e-5;
        let finite_difference = (solver.ground_probability_at(1.0 + delta)
            - solver.ground_probability_at(1.0 - delta))
            / (2.0 * delta);

        assert_close(dc_response, finite_difference, 5.0e-4);
    }

    #[test]
    fn solver_rejects_invalid_physics() {
        assert_eq!(
            Ramsey {
                pulse_width: 0.0,
                ..ramsey()
            }
            .solver(trotter_config(501))
            .unwrap_err(),
            RamseyError::NonPositivePulseWidth
        );
    }
}
