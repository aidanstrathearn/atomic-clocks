use std::f64::consts::{PI, TAU};

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
