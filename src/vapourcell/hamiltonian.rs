use crate::maths::demodulation::ModulationParams;
use crate::twolevel::{Hamiltonian, TimeDependentHamiltonian};

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
