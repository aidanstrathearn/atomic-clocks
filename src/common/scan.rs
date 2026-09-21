use std::f64::consts::TAU;

use crate::maths::linspace;

/// Angular-frequency detuning offsets sampled around a central detuning.
#[derive(Clone, Copy, Debug)]
pub struct DetuningScanParams {
    /// Angular-frequency half-range in radians per model time unit.
    pub hz_lim: f64,
    pub hz_num: usize,
}

impl Default for DetuningScanParams {
    fn default() -> Self {
        Self {
            hz_lim: 5.0,
            hz_num: 50,
        }
    }
}

impl DetuningScanParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.hz_num == 0 {
            return Err("hz_num must be positive".to_string());
        }
        if !self.hz_lim.is_finite() || self.hz_lim < 0.0 {
            return Err("hz_lim must be finite and nonnegative".to_string());
        }
        Ok(())
    }

    /// Returns angular-frequency detuning offsets in scan order.
    pub fn angular_offsets(&self) -> Result<Vec<f64>, String> {
        self.validate()?;
        if self.hz_num == 1 {
            Ok(vec![-self.hz_lim])
        } else {
            Ok(linspace(-self.hz_lim, self.hz_lim, self.hz_num - 1))
        }
    }

    /// Returns ordinary-frequency detuning offsets in cycles per model time unit.
    pub fn frequency_offsets(&self) -> Result<Vec<f64>, String> {
        Ok(self
            .angular_offsets()?
            .into_iter()
            .map(|offset| offset / TAU)
            .collect())
    }

    /// Returns the ordinary-frequency half-range in cycles per model time unit.
    pub fn frequency_half_range(&self) -> f64 {
        self.hz_lim / TAU
    }
}
