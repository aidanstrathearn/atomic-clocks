use super::consts::{PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Laser {
    pub power_milliwatts: f64,
    pub waist_radius_mm: f64,
    /// Used for photon energy, not to derive a laser detuning.
    pub wavelength_nm: f64,
    /// Lorentzian optical power-spectrum FWHM in ordinary-frequency units.
    pub linewidth_fwhm_hz: f64,
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
        PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT / wavelength_m
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_as("laser")
    }

    pub(crate) fn validate_as(&self, name: &str) -> Result<(), String> {
        if !self.power_milliwatts.is_finite() || self.power_milliwatts < 0.0 {
            return Err(format!("{name} power must be finite and nonnegative"));
        }
        if !self.waist_radius_mm.is_finite() || self.waist_radius_mm <= 0.0 {
            return Err(format!("{name} waist radius must be positive and finite"));
        }
        if !self.wavelength_nm.is_finite() || self.wavelength_nm <= 0.0 {
            return Err(format!("{name} wavelength must be positive and finite"));
        }
        if !self.linewidth_fwhm_hz.is_finite() || self.linewidth_fwhm_hz < 0.0 {
            return Err(format!("{name} linewidth must be finite and nonnegative"));
        }
        Ok(())
    }
}
