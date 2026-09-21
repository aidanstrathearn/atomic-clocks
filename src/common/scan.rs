use crate::maths::linspace;

/// Detuning offsets sampled around a central detuning.
#[derive(Clone, Copy, Debug)]
pub struct DetuningScanParams {
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

    /// Returns the detuning offsets in scan order.
    pub fn offsets(&self) -> Result<Vec<f64>, String> {
        self.validate()?;
        if self.hz_num == 1 {
            Ok(vec![-self.hz_lim])
        } else {
            Ok(linspace(-self.hz_lim, self.hz_lim, self.hz_num - 1))
        }
    }
}
