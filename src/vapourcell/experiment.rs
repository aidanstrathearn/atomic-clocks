use std::f64::consts::{PI, TAU};

use crate::common::DetuningScanParams;
use crate::maths::demodulation::ModulationParams;
use crate::twolevel::Decay;

use super::hamiltonian::{DrivenAtomParams, Frame, HamiltonianParams};
use super::mts::{MtsParams, MtsSolverParams};

/// Speed of light in vacuum, in metres per second (exact SI value).
const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Planck constant, in joule-seconds (exact SI value).
const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;

/// Saturation intensity in watts per square metre for a closed two-level transition.
/// `angular_linewidth_per_s` is the population-decay linewidth Γ in radians per second.
fn saturation_intensity(angular_linewidth_per_s: f64, wavelength_m: f64) -> f64 {
    PI / 3.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT * angular_linewidth_per_s / wavelength_m.powi(3)
}

#[derive(Clone, Copy, Debug)]
pub struct Laser {
    pub power_milliwatts: f64,
    pub waist_radius_mm: f64,
    /// Used for photon energy, not to derive a laser detuning.
    pub wavelength_nm: f64,
}

impl Laser {
    /// On-axis peak intensity for a Gaussian beam whose waist is its 1/e² radius.
    pub fn intensity_watts_per_m2(&self) -> f64 {
        let power_watts = self.power_milliwatts * 1e-3;
        let waist_m = self.waist_radius_mm * 1e-3;
        2.0 * power_watts / (PI * waist_m.powi(2))
    }

    /// Energy of one photon at the laser wavelength.
    pub fn photon_energy_joules(&self) -> f64 {
        let wavelength_m = self.wavelength_nm * 1e-9;
        PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelength_m
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_as("laser")
    }

    fn validate_as(&self, name: &str) -> Result<(), String> {
        if !self.power_milliwatts.is_finite() || self.power_milliwatts < 0.0 {
            return Err(format!("{name} power must be finite and nonnegative"));
        }
        if !self.waist_radius_mm.is_finite() || self.waist_radius_mm <= 0.0 {
            return Err(format!("{name} waist radius must be positive and finite"));
        }
        if !self.wavelength_nm.is_finite() || self.wavelength_nm <= 0.0 {
            return Err(format!("{name} wavelength must be positive and finite"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Transition {
    pub wavelength_nm: f64,
    /// Natural, lifetime-limited FWHM in ordinary-frequency units.
    pub linewidth_hz: f64,
}

impl Transition {
    /// Resonant single-atom Rabi angular frequency for the laser intensity.
    pub fn rabi_freq_radians_per_s(&self, laser: &Laser) -> f64 {
        let angular_linewidth_per_s = TAU * self.linewidth_hz;
        let wavelength_m = self.wavelength_nm * 1e-9;
        let saturation_intensity = saturation_intensity(angular_linewidth_per_s, wavelength_m);

        angular_linewidth_per_s
            * (laser.intensity_watts_per_m2() / (2.0 * saturation_intensity)).sqrt()
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.wavelength_nm.is_finite() || self.wavelength_nm <= 0.0 {
            return Err("transition wavelength must be positive and finite".to_string());
        }
        if !self.linewidth_hz.is_finite() || self.linewidth_hz <= 0.0 {
            return Err("transition linewidth must be positive and finite".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VapourCell {
    pub transition: Transition,
    pub density_per_m3: f64,
    pub length_m: f64,
}

impl VapourCell {
    pub fn validate(&self) -> Result<(), String> {
        self.transition.validate()?;
        if !self.density_per_m3.is_finite() || self.density_per_m3 < 0.0 {
            return Err("vapour-cell density must be finite and nonnegative".to_string());
        }
        if !self.length_m.is_finite() || self.length_m <= 0.0 {
            return Err("vapour-cell length must be positive and finite".to_string());
        }
        Ok(())
    }
}

/// Experiment-facing MTS parameters expressed in SI-derived laboratory units.
///
/// Frequencies are supplied in hertz and converted to angular rates per second.
/// The laser wavelengths are not used to infer detunings.
#[derive(Clone, Copy, Debug)]
pub struct MtsExperiment {
    pub cell: VapourCell,
    pub pump: Laser,
    pub probe: Laser,
    /// Probe detuning using the same sign convention as [`HamiltonianParams::delta`].
    pub probe_detuning_hz: f64,
    pub modulation_frequency_hz: f64,
    pub modulation_depth_hz: f64,
    /// Pump carrier offset from the probe carrier.
    pub pump_probe_offset_hz: f64,
    /// The pure-dephasing rate `gamma_phi / (2 pi)`.
    pub pure_dephasing_rate_hz: f64,
    pub scan_half_range_hz: f64,
    pub scan_samples: usize,
}

impl MtsExperiment {
    pub fn validate(&self) -> Result<(), String> {
        self.cell.validate()?;
        self.pump.validate_as("pump laser")?;
        self.probe.validate_as("probe laser")?;
        if !self.probe_detuning_hz.is_finite() {
            return Err("probe detuning must be finite".to_string());
        }
        if !self.modulation_frequency_hz.is_finite() || self.modulation_frequency_hz <= 0.0 {
            return Err("modulation frequency must be positive and finite".to_string());
        }
        if !self.modulation_depth_hz.is_finite() || self.modulation_depth_hz < 0.0 {
            return Err("modulation depth must be finite and nonnegative".to_string());
        }
        if !self.pump_probe_offset_hz.is_finite() {
            return Err("pump-probe offset must be finite".to_string());
        }
        if !self.pure_dephasing_rate_hz.is_finite() || self.pure_dephasing_rate_hz < 0.0 {
            return Err("pure-dephasing rate must be finite and nonnegative".to_string());
        }
        if !self.scan_half_range_hz.is_finite() || self.scan_half_range_hz < 0.0 {
            return Err("scan half-range must be finite and nonnegative".to_string());
        }
        if self.scan_samples == 0 {
            return Err("scan sample count must be positive".to_string());
        }
        Ok(())
    }

    /// Converts laboratory inputs to the solver's angular-rate representation.
    ///
    /// The resulting model uses seconds as its time unit and starts in the probe
    /// frame with zero Doppler shift and spatial phase. Those sampling coordinates
    /// may be varied directly on the returned low-level parameters.
    pub fn to_mts_params(&self, solver: MtsSolverParams) -> Result<MtsParams, String> {
        self.validate()?;
        let transition = self.cell.transition;
        let params = MtsParams {
            atom: DrivenAtomParams {
                hamiltonian: HamiltonianParams {
                    modulation: ModulationParams {
                        frequency: TAU * self.modulation_frequency_hz,
                        depth: TAU * self.modulation_depth_hz,
                        shift: TAU * self.pump_probe_offset_hz,
                    },
                    delta: TAU * self.probe_detuning_hz,
                    r_pump: transition.rabi_freq_radians_per_s(&self.pump),
                    r_prbe: transition.rabi_freq_radians_per_s(&self.probe),
                    kv: 0.0,
                    kr: 0.0,
                    frame: Frame::Probe,
                },
                decay: Decay {
                    gamma_up: 0.0,
                    gamma_down: TAU * transition.linewidth_hz,
                    gamma_phi: TAU * self.pure_dephasing_rate_hz,
                },
            },
            solver,
            scan: DetuningScanParams {
                hz_lim: TAU * self.scan_half_range_hz,
                hz_num: self.scan_samples,
            },
        };
        params.validate()?;
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn experiment() -> MtsExperiment {
        MtsExperiment {
            cell: VapourCell {
                transition: Transition {
                    wavelength_nm: 780.24,
                    linewidth_hz: 6.065e6,
                },
                density_per_m3: 2.5e16,
                length_m: 0.075,
            },
            pump: Laser {
                power_milliwatts: 2.0,
                waist_radius_mm: 0.8,
                wavelength_nm: 780.25,
            },
            probe: Laser {
                power_milliwatts: 0.2,
                waist_radius_mm: 0.6,
                wavelength_nm: 780.23,
            },
            probe_detuning_hz: -1.2e6,
            modulation_frequency_hz: 3.4e6,
            modulation_depth_hz: 5.6e6,
            pump_probe_offset_hz: 0.7e6,
            pure_dephasing_rate_hz: 0.15e6,
            scan_half_range_hz: 20.0e6,
            scan_samples: 81,
        }
    }

    #[test]
    fn experiment_converts_to_si_angular_rates() {
        let experiment = experiment();
        let solver = MtsSolverParams {
            kr_n: 7,
            steps_per_period: 120,
            n_periods: 4,
        };
        let params = experiment.to_mts_params(solver).unwrap();
        let hamiltonian = params.atom.hamiltonian;

        assert_eq!(hamiltonian.frame, Frame::Probe);
        assert_eq!(hamiltonian.kv, 0.0);
        assert_eq!(hamiltonian.kr, 0.0);
        assert_eq!(hamiltonian.delta, TAU * experiment.probe_detuning_hz);
        assert_eq!(
            hamiltonian.modulation.frequency,
            TAU * experiment.modulation_frequency_hz
        );
        assert_eq!(
            hamiltonian.modulation.depth,
            TAU * experiment.modulation_depth_hz
        );
        assert_eq!(
            hamiltonian.modulation.shift,
            TAU * experiment.pump_probe_offset_hz
        );
        assert_eq!(
            hamiltonian.r_pump,
            experiment
                .cell
                .transition
                .rabi_freq_radians_per_s(&experiment.pump)
        );
        assert_eq!(
            hamiltonian.r_prbe,
            experiment
                .cell
                .transition
                .rabi_freq_radians_per_s(&experiment.probe)
        );
        assert_eq!(params.atom.decay.gamma_up, 0.0);
        assert_eq!(
            params.atom.decay.gamma_down,
            TAU * experiment.cell.transition.linewidth_hz
        );
        assert_eq!(
            params.atom.decay.gamma_phi,
            TAU * experiment.pure_dephasing_rate_hz
        );
        assert_eq!(params.scan.hz_lim, TAU * experiment.scan_half_range_hz);
        assert_eq!(params.scan.hz_num, experiment.scan_samples);
        assert_eq!(params.solver.kr_n, solver.kr_n);
        assert_eq!(params.solver.steps_per_period, solver.steps_per_period);
        assert_eq!(params.solver.n_periods, solver.n_periods);
    }

    #[test]
    fn laser_wavelength_does_not_set_model_detuning() {
        let experiment = experiment();
        let mut changed_wavelengths = experiment;
        changed_wavelengths.pump.wavelength_nm = 500.0;
        changed_wavelengths.probe.wavelength_nm = 1_000.0;

        let original = experiment
            .to_mts_params(MtsSolverParams::default())
            .unwrap();
        let changed = changed_wavelengths
            .to_mts_params(MtsSolverParams::default())
            .unwrap();
        assert_eq!(
            original.atom.hamiltonian.delta,
            changed.atom.hamiltonian.delta
        );
        assert_eq!(
            original.atom.hamiltonian.modulation.shift,
            changed.atom.hamiltonian.modulation.shift
        );
        assert_eq!(
            original.atom.hamiltonian.r_pump,
            changed.atom.hamiltonian.r_pump
        );
        assert_eq!(
            original.atom.hamiltonian.r_prbe,
            changed.atom.hamiltonian.r_prbe
        );
    }
}
