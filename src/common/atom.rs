use std::f64::consts::{PI, TAU};

use super::consts::{ATOMIC_MASS_KG, PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT};
use super::laser::Laser;

/// Saturation intensity in watts per square metre for a closed two-level transition.
/// `angular_linewidth_per_s` is the population-decay linewidth Γ in radians per second.
fn saturation_intensity(angular_linewidth_per_s: f64, wavelength_m: f64) -> f64 {
    PI / 3.0 * PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT * angular_linewidth_per_s / wavelength_m.powi(3)
}

#[derive(Clone, Copy, Debug)]
pub struct Transition {
    pub wavelength_nm: f64,
    /// Natural, lifetime-limited FWHM in ordinary-frequency units.
    pub linewidth_hz: f64,
}

impl Transition {
    pub fn wavelength_m(&self) -> f64 {
        self.wavelength_nm * 1e-9
    }

    /// Resonant single-atom Rabi angular frequency for the laser intensity.
    pub fn rabi_freq_radians_per_s(&self, laser: &Laser) -> f64 {
        let angular_linewidth_per_s = TAU * self.linewidth_hz;
        let saturation_intensity =
            saturation_intensity(angular_linewidth_per_s, self.wavelength_m());

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
pub struct TwoLevelAtom {
    /// Isotopic mass in unified atomic mass units.
    pub mass_u: f64,
    pub transition: Transition,
}

impl TwoLevelAtom {
    pub fn mass_kg(&self) -> f64 {
        self.mass_u * ATOMIC_MASS_KG
    }

    pub fn recoil_velocity_m_per_s(&self) -> f64 {
        PLANCK_CONSTANT_J_S / (self.mass_kg() * self.transition.wavelength_m())
    }

    pub fn recoil_frequency_hz(&self) -> f64 {
        PLANCK_CONSTANT_J_S / (2.0 * self.mass_kg() * self.transition.wavelength_m().powi(2))
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.mass_u.is_finite() || self.mass_u <= 0.0 {
            return Err("atomic mass must be positive and finite".to_string());
        }
        self.transition.validate()
    }
}

pub const YB171_CLOCK: TwoLevelAtom = TwoLevelAtom {
    mass_u: 170.936_331_52,
    transition: Transition {
        wavelength_nm: 578.0,
        linewidth_hz: 0.010,
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, relative_tolerance: f64) {
        let scale = actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= relative_tolerance * scale,
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn yb171_clock_has_consistent_wavelength_and_recoil_units() {
        let wavelength_m = YB171_CLOCK.transition.wavelength_m();
        let recoil_velocity_m_per_s = YB171_CLOCK.recoil_velocity_m_per_s();

        assert_close(wavelength_m, 578.0e-9, 1.0e-15);
        assert_close(
            YB171_CLOCK.recoil_frequency_hz(),
            recoil_velocity_m_per_s / (2.0 * wavelength_m),
            1.0e-15,
        );
    }

    #[test]
    fn atom_validation_includes_mass_and_transition() {
        assert!(YB171_CLOCK.validate().is_ok());
        assert!(
            TwoLevelAtom {
                mass_u: 0.0,
                ..YB171_CLOCK
            }
            .validate()
            .is_err()
        );
        assert!(
            TwoLevelAtom {
                transition: Transition {
                    wavelength_nm: f64::NAN,
                    ..YB171_CLOCK.transition
                },
                ..YB171_CLOCK
            }
            .validate()
            .is_err()
        );
    }
}
