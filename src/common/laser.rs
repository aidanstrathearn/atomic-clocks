use super::consts::{PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT};
use std::collections::HashSet;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Laser {
    pub power_milliwatts: f64,
    pub waist_radius_mm: f64,
    /// Used for photon energy, not to derive a laser detuning.
    pub wavelength_nm: f64,
    /// Lorentzian optical power-spectrum FWHM in ordinary-frequency units.
    pub linewidth_fwhm_hz: f64,
    /// One-sided white relative intensity-noise PSD in linear Hz⁻¹ units.
    pub white_rin_per_hz: f64,
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

    /// Pure-dephasing angular rate implied by the Lorentzian linewidth.
    pub fn pure_dephasing_angular_rate(&self) -> f64 {
        PI * self.linewidth_fwhm_hz
    }

    /// Drive-axis dephasing rate produced by one-sided white relative
    /// intensity noise.
    ///
    /// This uses the weak-noise relation `delta Omega / Omega = delta I / (2 I)`.
    pub fn white_rabi_dephasing_angular_rate(&self, rabi_angular_frequency: f64) -> f64 {
        rabi_angular_frequency.powi(2) * self.white_rin_per_hz / 16.0
    }

    /// Absolute Allan deviation implied by a Lorentzian linewidth produced by
    /// white frequency noise.
    pub fn white_frequency_noise_adev_hz(&self, averaging_time_s: f64) -> f64 {
        if !self.linewidth_fwhm_hz.is_finite()
            || self.linewidth_fwhm_hz < 0.0
            || !averaging_time_s.is_finite()
            || averaging_time_s <= 0.0
        {
            return f64::NAN;
        }

        (self.linewidth_fwhm_hz / (2.0 * PI * averaging_time_s)).sqrt()
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
        if !self.white_rin_per_hz.is_finite() || self.white_rin_per_hz < 0.0 {
            return Err(format!(
                "{name} white RIN PSD must be finite and nonnegative"
            ));
        }
        Ok(())
    }
}

/// A finite set of equally spaced optical modes sharing a Gaussian beam waist.
///
/// The embedded laser's power is the total time-averaged power across the
/// represented modes. Relative field amplitudes are normalized when calculating
/// powers or couplings; their squared magnitudes determine the power fractions.
/// Removing modes and recalculating therefore redistributes the total power.
///
/// Frequencies are offsets from the reference tooth at index zero, which need
/// not be present. As with [`Laser`], wavelength is not used to infer an atomic
/// detuning. All teeth inherit the laser's Lorentzian linewidth; this describes
/// an ideal comb with shared laser phase noise and noiseless spacing, without
/// specifying a stochastic noise realization.
#[derive(Clone, Debug)]
pub struct Comb {
    pub laser: Laser,
    /// Positive ordinary-frequency spacing between adjacent tooth indices, in Hz.
    pub spacing_hz: f64,
    /// Modes in the order used by the power, spectrum, and coupling calculations.
    pub modes: Vec<CombMode>,
}

/// One tooth of a [`Comb`], relative to the same optical reference as every tooth.
#[derive(Clone, Copy, Debug)]
pub struct CombMode {
    /// Signed tooth number; its frequency offset is `index * comb.spacing_hz`.
    pub index: i32,
    /// Nonnegative relative field-amplitude weight, not a power fraction.
    pub amplitude: f64,
    /// Field phase at `t = 0`, in radians. Any finite phase is accepted.
    pub phase_rad: f64,
}

impl Comb {
    /// Creates an equal-amplitude Schroeder comb centered on the carrier.
    ///
    /// `side_modes` is the number of teeth on each side of the carrier, giving
    /// `M = 2 * side_modes + 1` modes in ascending index order from `-side_modes`
    /// through `+side_modes`. Zero side modes produces only the carrier.
    ///
    /// Every tooth has relative amplitude `1.0`, so each receives `1 / M` of the
    /// laser's total power. For the ordinal `n = index + side_modes + 1`, phases
    /// are `-pi * n * (n - 1) / M` radians at `t = 0`. Phases are not wrapped or
    /// shifted to set the carrier phase to zero.
    ///
    /// Returns an error for invalid laser or spacing parameters, unrepresentable
    /// tooth indices, mode counts or offsets, or a failed mode allocation.
    pub fn schroeder(laser: Laser, spacing_hz: f64, side_modes: usize) -> Result<Self, String> {
        laser.validate_as("comb laser")?;
        if !spacing_hz.is_finite() || spacing_hz <= 0.0 {
            return Err("comb spacing must be positive and finite".to_string());
        }
        let side_index = i32::try_from(side_modes)
            .map_err(|_| "Schroeder side mode count must fit in an i32".to_string())?;
        let mode_count = side_modes
            .checked_mul(2)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| "Schroeder mode count is too large".to_string())?;
        if !(f64::from(side_index) * spacing_hz).is_finite() {
            return Err("Schroeder mode frequency offsets must be finite".to_string());
        }

        let mut modes = Vec::new();
        modes
            .try_reserve_exact(mode_count)
            .map_err(|error| format!("cannot allocate Schroeder comb modes: {error}"))?;
        modes.extend(
            (-side_index..=side_index)
                .enumerate()
                .map(|(ordinal, index)| {
                    let n = ordinal as f64 + 1.0;
                    CombMode {
                        index,
                        amplitude: 1.0,
                        phase_rad: -PI * n * (n - 1.0) / mode_count as f64,
                    }
                }),
        );

        let comb = Self {
            laser,
            spacing_hz,
            modes,
        };
        comb.validate()?;
        Ok(comb)
    }

    /// Checks the laser, spacing, unique indices, finite offsets and phases,
    /// and nonnegative finite amplitudes with at least one nonzero weight.
    pub fn validate(&self) -> Result<(), String> {
        self.laser.validate_as("comb laser")?;
        if !self.spacing_hz.is_finite() || self.spacing_hz <= 0.0 {
            return Err("comb spacing must be positive and finite".to_string());
        }
        if self.modes.is_empty() {
            return Err("comb must contain at least one mode".to_string());
        }

        let mut indices = HashSet::with_capacity(self.modes.len());
        let mut has_nonzero_amplitude = false;
        for mode in &self.modes {
            if !indices.insert(mode.index) {
                return Err(format!("duplicate comb mode index {}", mode.index));
            }
            if !(f64::from(mode.index) * self.spacing_hz).is_finite() {
                return Err(format!(
                    "comb mode {} frequency offset must be finite",
                    mode.index
                ));
            }
            if !mode.amplitude.is_finite() || mode.amplitude < 0.0 {
                return Err(format!(
                    "comb mode {} amplitude must be finite and nonnegative",
                    mode.index
                ));
            }
            if !mode.phase_rad.is_finite() {
                return Err(format!("comb mode {} phase must be finite", mode.index));
            }
            has_nonzero_amplitude |= mode.amplitude > 0.0;
        }
        if !has_nonzero_amplitude {
            return Err("comb must contain at least one nonzero amplitude".to_string());
        }
        Ok(())
    }

    /// Ordinary-frequency offsets from the reference tooth, in supplied mode order.
    pub fn mode_offsets_hz(&self) -> Result<Vec<f64>, String> {
        self.validate()?;
        Ok(self
            .modes
            .iter()
            .map(|mode| f64::from(mode.index) * self.spacing_hz)
            .collect())
    }

    /// Normalized squared field amplitudes, in supplied mode order.
    ///
    /// The fractions sum to one up to floating-point rounding. Zero-amplitude
    /// teeth are retained with zero power.
    pub fn mode_power_fractions(&self) -> Result<Vec<f64>, String> {
        Ok(self
            .normalized_mode_amplitudes()?
            .into_iter()
            .map(|amplitude| amplitude * amplitude)
            .collect())
    }

    /// Integrated powers of the discrete teeth, including zero-linewidth teeth.
    pub fn mode_powers_milliwatts(&self) -> Result<Vec<f64>, String> {
        Ok(self
            .mode_power_fractions()?
            .into_iter()
            .map(|fraction| self.laser.power_milliwatts * fraction)
            .collect())
    }

    /// Evaluates the averaged optical power spectrum in W/Hz at offsets in Hz.
    ///
    /// Each tooth is a unit-area Lorentzian with the embedded laser's FWHM,
    /// weighted by its integrated mode power. Phases do not affect this spectrum.
    /// Integrating over all offsets gives the total power in watts; a finite
    /// sampling window is not renormalized. Samples may be negative, repeated,
    /// unsorted, or empty, and are returned in the supplied sampling order.
    ///
    /// Returns an error for invalid inputs, a nonfinite evaluated density, or
    /// zero linewidth: delta-function teeth have no finite pointwise density.
    /// Use [`Self::mode_powers_milliwatts`] for the zero-linewidth case.
    pub fn power_spectrum_watts_per_hz(&self, offsets_hz: &[f64]) -> Result<Vec<f64>, String> {
        let powers = self.mode_powers_milliwatts()?;
        let linewidth = self.laser.linewidth_fwhm_hz;
        if linewidth == 0.0 {
            return Err("a pointwise comb spectrum requires a positive linewidth".to_string());
        }
        if !offsets_hz.iter().all(|offset| offset.is_finite()) {
            return Err("spectrum frequency offsets must be finite".to_string());
        }

        offsets_hz
            .iter()
            .map(|offset| {
                let density: f64 = self
                    .modes
                    .iter()
                    .zip(&powers)
                    .map(|(mode, power)| {
                        let distance = (offset - f64::from(mode.index) * self.spacing_hz).abs();
                        let power_watts = power * 1e-3;
                        // Use dimensionless ratios to avoid squaring very large
                        // or small linewidths and frequency differences.
                        if distance <= linewidth {
                            let ratio = distance / linewidth;
                            (power_watts * (2.0 / PI) / (1.0 + 4.0 * ratio * ratio)) / linewidth
                        } else {
                            let ratio = linewidth / distance;
                            (power_watts * (2.0 / PI) * ratio / (4.0 + ratio * ratio)) / distance
                        }
                    })
                    .sum();
                if !density.is_finite() {
                    return Err(format!(
                        "comb spectrum density at offset {offset} Hz must be finite"
                    ));
                }
                Ok(density)
            })
            .collect()
    }

    pub(crate) fn normalized_mode_amplitudes(&self) -> Result<Vec<f64>, String> {
        self.validate()?;
        let scale = self
            .modes
            .iter()
            .map(|mode| mode.amplitude)
            .fold(0.0, f64::max);
        let mut amplitudes: Vec<_> = self
            .modes
            .iter()
            .map(|mode| mode.amplitude / scale)
            .collect();
        let norm = amplitudes
            .iter()
            .map(|amplitude| amplitude * amplitude)
            .sum::<f64>()
            .sqrt();
        for amplitude in &mut amplitudes {
            *amplitude /= norm;
        }
        Ok(amplitudes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laser_with_linewidth(linewidth_fwhm_hz: f64) -> Laser {
        Laser {
            power_milliwatts: 1.0,
            waist_radius_mm: 1.0,
            wavelength_nm: 578.0,
            linewidth_fwhm_hz,
            white_rin_per_hz: 0.0,
        }
    }

    #[test]
    fn lorentzian_linewidth_sets_dephasing_and_white_frequency_noise_adev() {
        let laser = laser_with_linewidth(200.0);
        let averaging_time_s = 2.5;

        assert_eq!(laser.pure_dephasing_angular_rate(), PI * 200.0);
        assert_eq!(
            laser.white_frequency_noise_adev_hz(averaging_time_s),
            (200.0 / (2.0 * PI * averaging_time_s)).sqrt()
        );
    }

    #[test]
    fn white_frequency_noise_adev_rejects_invalid_inputs() {
        for linewidth_fwhm_hz in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                laser_with_linewidth(linewidth_fwhm_hz)
                    .white_frequency_noise_adev_hz(1.0)
                    .is_nan()
            );
        }
        let laser = laser_with_linewidth(200.0);
        for averaging_time_s in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
            assert!(
                laser
                    .white_frequency_noise_adev_hz(averaging_time_s)
                    .is_nan()
            );
        }
    }

    #[test]
    fn white_rin_sets_rabi_dephasing_from_the_actual_rabi_frequency() {
        let laser = Laser {
            white_rin_per_hz: 2.5e-12,
            ..laser_with_linewidth(0.0)
        };
        let rabi_angular_frequency = 12_000.0;

        assert_eq!(
            laser.white_rabi_dephasing_angular_rate(rabi_angular_frequency),
            rabi_angular_frequency.powi(2) * 2.5e-12 / 16.0
        );
    }

    #[test]
    fn laser_validation_rejects_invalid_white_rin() {
        for white_rin_per_hz in [-1.0, f64::NAN, f64::INFINITY] {
            let laser = Laser {
                white_rin_per_hz,
                ..laser_with_linewidth(0.0)
            };
            assert!(laser.validate().is_err());
        }
    }
}
