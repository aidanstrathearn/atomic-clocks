use std::f64::consts::PI;

use crate::maths::demodulation::ModulationParams;
use crate::twolevel::{Hamiltonian, TimeDependentHamiltonian};

/// Speed of light in vacuum, in metres per second (exact SI value).
const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Planck constant, in joule-seconds (exact SI value).
const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;

/// Saturation intensity in watts per square metre for a closed two-level transition.
/// `angular_linewidth_per_s` is the population-decay linewidth Γ in radians per second.
fn saturation_intensity(angular_linewidth_per_s: f64, wavelength_m: f64) -> f64 {
    PI / 3.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT * angular_linewidth_per_s / wavelength_m.powi(3)
}

pub struct Laser {
    pub power_milliwatts: f64,
    pub waist_radius_mm: f64,
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
}

pub struct Atoms {
    pub density_per_m3: f64,
    pub transition_wavelength_nm: f64,
    pub transition_linewidth_hz: f64,
}

impl Atoms {
    /// Resonant single-atom Rabi angular frequency for the laser intensity.
    pub fn rabi_freq_radians_per_s(&self, laser: &Laser) -> f64 {
        let angular_linewidth_per_s = 2.0 * PI * self.transition_linewidth_hz;
        let transition_wavelength_m = self.transition_wavelength_nm * 1e-9;
        let saturation_intensity =
            saturation_intensity(angular_linewidth_per_s, transition_wavelength_m);

        angular_linewidth_per_s
            * (laser.intensity_watts_per_m2() / (2.0 * saturation_intensity)).sqrt()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Frame {
    Atom,
    Pump,
    Probe,
}

impl Frame {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "atom" => Ok(Self::Atom),
            "pump" => Ok(Self::Pump),
            "probe" => Ok(Self::Probe),
            _ => Err(format!("unknown frame '{value}'")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Atom => "atom",
            Self::Pump => "pump",
            Self::Probe => "probe",
        }
    }
}

/// Hamiltonian parameters for a single atom in the chosen rotating frame.
#[derive(Clone, Copy, Debug)]
pub struct HamiltonianParams {
    pub modulation: ModulationParams,
    pub delta: f64,
    pub r_pump: f64,
    pub r_prbe: f64,
    pub kv: f64,
    pub kr: f64,
    pub frame: Frame,
}

impl Default for HamiltonianParams {
    fn default() -> Self {
        Self {
            modulation: ModulationParams::default(),
            delta: 0.0,
            r_pump: 1.0,
            r_prbe: 0.1,
            kv: 0.0,
            kr: 0.0,
            frame: Frame::Probe,
        }
    }
}

impl HamiltonianParams {
    /// Checks for a positive, finite modulation frequency and finite coefficients.
    pub fn validate(&self) -> Result<(), String> {
        self.modulation.validate()?;
        if ![self.delta, self.r_pump, self.r_prbe, self.kv, self.kr]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("Hamiltonian coefficients must be finite".to_string());
        }
        Ok(())
    }
}

impl TimeDependentHamiltonian for HamiltonianParams {
    fn h(&self, t: f64) -> Hamiltonian {
        let phase = self.modulation.phase(t);
        let freq = self.modulation.freq(t);
        let kvt = t * self.kv;

        let (hz, pump_phase, probe_phase) = match self.frame {
            Frame::Atom => (self.delta, kvt + phase + self.kr, -kvt - self.kr),
            Frame::Pump => (
                self.delta - self.kv - freq,
                0.0,
                -2.0 * kvt - phase - 2.0 * self.kr,
            ),
            Frame::Probe => (self.delta + self.kv, 2.0 * kvt + phase + 2.0 * self.kr, 0.0),
        };

        Hamiltonian::new(
            self.r_pump * pump_phase.cos() + self.r_prbe * probe_phase.cos(),
            self.r_pump * pump_phase.sin() + self.r_prbe * probe_phase.sin(),
            hz,
        )
    }
}
