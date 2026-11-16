use std::f64::consts::{PI, TAU};

use super::consts::{ATOMIC_MASS_KG, PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT};
use super::laser::{Comb, Laser};

/// Saturation intensity in watts per square metre for a closed two-level transition.
/// `angular_linewidth_per_s` is the population-decay linewidth Γ in radians per second.
fn saturation_intensity(angular_linewidth_per_s: f64, wavelength_m: f64) -> f64 {
    PI / 3.0 * PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT * angular_linewidth_per_s / wavelength_m.powi(3)
}

/// A driven Zeeman component, or an effective two-level component with the same coupling.
///
/// The linewidth normalization assumes that the excited state decays into the
/// selected lower hyperfine manifold. Decay into other manifolds requires an
/// additional branching factor; it is not supplied by a Clebsch–Gordan coefficient.
#[derive(Clone, Copy, Debug)]
pub struct Transition {
    pub wavelength_nm: f64,
    /// Total natural, lifetime-limited FWHM in Hz, before any component factor.
    pub linewidth_hz: f64,
    /// Squared absorption Clebsch–Gordan coefficient
    /// `|<F, mF; 1, q | F', mF'>|^2` for the selected driven component.
    ///
    /// This scales the squared dipole coupling relative to the closed-transition
    /// value inferred from the linewidth. It does not scale spontaneous decay.
    /// The supplied laser intensity belongs to this polarization component.
    /// Only the magnitude is represented; coherent multilevel models also need
    /// signed coefficients and polarization amplitudes.
    pub clebsch_gordan_squared: f64,
}

/// Resonant coupling of one comb tooth to a selected two-level transition.
#[derive(Clone, Copy, Debug)]
pub struct CombRabiMode {
    pub index: i32,
    /// Ordinary-frequency offset from the reference tooth, in Hz; not a detuning.
    pub offset_hz: f64,
    /// Nonnegative resonant Rabi angular-frequency magnitude, in radians per second.
    pub rabi_angular_frequency: f64,
    /// Field phase at `t = 0`, in radians.
    pub phase_rad: f64,
}

impl Transition {
    /// Transition wavelength in metres.
    pub fn wavelength_m(&self) -> f64 {
        self.wavelength_nm * 1e-9
    }

    /// Transition frequency in hertz.
    pub fn frequency_hz(&self) -> f64 {
        SPEED_OF_LIGHT / self.wavelength_m()
    }

    /// Resonant single-atom Rabi angular frequency, including the component's CG factor.
    pub fn rabi_freq_radians_per_s(&self, laser: &Laser) -> f64 {
        let angular_linewidth_per_s = TAU * self.linewidth_hz;
        let saturation_intensity =
            saturation_intensity(angular_linewidth_per_s, self.wavelength_m());

        angular_linewidth_per_s
            * (laser.intensity_watts_per_m2() / (2.0 * saturation_intensity)).sqrt()
            * self.clebsch_gordan_squared.sqrt()
    }

    /// Per-tooth resonant Rabi couplings, including the component's CG factor.
    ///
    /// The total-power Rabi frequency is multiplied by each normalized field
    /// amplitude, equivalently the square root of its power fraction. All modes
    /// share the laser's waist and the selected polarization component. Supplied
    /// mode order, indices, and phases are preserved, including zero-weight teeth.
    ///
    /// The caller supplies the reference detuning separately; wavelength and
    /// optical linewidth do not enter these coupling magnitudes. The broadened
    /// power spectrum is not used to calculate the coherent couplings.
    pub fn comb_rabi_freqs(&self, comb: &Comb) -> Result<Vec<CombRabiMode>, String> {
        self.validate()?;
        let amplitudes = comb.normalized_mode_amplitudes()?;
        let total_rabi = self.rabi_freq_radians_per_s(&comb.laser);
        if !total_rabi.is_finite() {
            return Err("comb Rabi angular frequency must be finite".to_string());
        }
        Ok(comb
            .modes
            .iter()
            .zip(amplitudes)
            .map(|(mode, amplitude)| CombRabiMode {
                index: mode.index,
                offset_hz: f64::from(mode.index) * comb.spacing_hz,
                rabi_angular_frequency: total_rabi * amplitude,
                phase_rad: mode.phase_rad,
            })
            .collect())
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.wavelength_nm.is_finite() || self.wavelength_nm <= 0.0 {
            return Err("transition wavelength must be positive and finite".to_string());
        }
        if !self.linewidth_hz.is_finite() || self.linewidth_hz <= 0.0 {
            return Err("transition linewidth must be positive and finite".to_string());
        }
        if !self.clebsch_gordan_squared.is_finite()
            || !(0.0..=1.0).contains(&self.clebsch_gordan_squared)
        {
            return Err(
                "squared Clebsch–Gordan coefficient must be finite and in [0, 1]".to_string(),
            );
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

/// 171Yb 1S0(F=1/2) -> 3P0(F'=1/2), driven on either mF=+/-1/2 pi component.
pub const YB171_CLOCK: TwoLevelAtom = TwoLevelAtom {
    mass_u: 170.936_331_52,
    transition: Transition {
        wavelength_nm: 578.0,
        linewidth_hz: 0.007,
        clebsch_gordan_squared: 1.0 / 3.0,
    },
};
/// 174Yb 1S0(F=0) -> 3P1(F'=1), 556 nm, with unit CG coupling (e.g. pi light).
///
/// The historical name is retained; this is the green intercombination transition.
pub const YB174_CLOCK: TwoLevelAtom = TwoLevelAtom {
    mass_u: 173.938_867_5,
    transition: Transition {
        wavelength_nm: 556.0,
        linewidth_hz: 1.82e5,
        clebsch_gordan_squared: 1.0,
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
    fn component_couplings_match_independent_physical_baselines() {
        // Pre-refactor values at 1 mW, 1 mm waist. Previously the clock and
        // green pi coefficients were applied by callers, after the shared method.
        let green_pi = Transition {
            clebsch_gordan_squared: 2.0 / 3.0,
            ..YB174_CLOCK.transition
        };
        let dark_green = Transition {
            clebsch_gordan_squared: 0.0,
            ..YB174_CLOCK.transition
        };
        for (transition, expected_omega) in [
            (YB171_CLOCK.transition, 2_081.337_326_748_633),
            (green_pi, 14_160_043.075_592_078),
            (YB174_CLOCK.transition, 17_342_440.135_515_38),
            (dark_green, 0.0),
        ] {
            let laser = Laser {
                power_milliwatts: 1.0,
                waist_radius_mm: 1.0,
                wavelength_nm: transition.wavelength_nm,
                linewidth_fwhm_hz: 0.0,
                white_rin_per_hz: 0.0,
            };
            assert!(transition.validate().is_ok());
            assert_close(
                transition.rabi_freq_radians_per_s(&laser),
                expected_omega,
                2.0e-15,
            );
            assert_eq!(
                transition.rabi_freq_radians_per_s(&Laser {
                    power_milliwatts: 0.0,
                    ..laser
                }),
                0.0
            );
        }
    }

    #[test]
    fn component_validation_rejects_invalid_squared_coefficients() {
        for coefficient in [-0.1, 1.1, f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            let transition = Transition {
                clebsch_gordan_squared: coefficient,
                ..YB171_CLOCK.transition
            };
            assert!(transition.validate().is_err(), "coefficient {coefficient}");
        }
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
