use crate::maths::normalised_gaussian;

/// Gaussian velocity distribution and midpoint integration grid.
#[derive(Clone, Copy, Debug)]
pub struct VelocityParams {
    /// Mean Doppler shift in the coordinates used by the integration grid.
    pub mean: f64,
    /// Standard deviation of the Gaussian Doppler-shift distribution.
    /// Zero selects a single velocity at `mean` with unit weight.
    pub sigma: f64,
    /// Half-width of the integration window centred on zero.
    pub half_width: f64,
    /// Number of midpoint samples across the integration window.
    pub sample_count: usize,
}

impl Default for VelocityParams {
    fn default() -> Self {
        Self {
            mean: 0.0,
            sigma: 10.0,
            half_width: 0.01,
            sample_count: 1,
        }
    }
}

impl VelocityParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.mean.is_finite() {
            return Err("Gaussian velocity mean must be finite".to_string());
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err("Gaussian velocity sigma must be nonnegative and finite".to_string());
        }
        if !self.half_width.is_finite() || self.half_width <= 0.0 {
            return Err("velocity integration half-width must be positive and finite".to_string());
        }
        if self.sample_count == 0 {
            return Err("velocity sample count must be positive".to_string());
        }
        Ok(())
    }

    /// Returns midpoint samples over a window fixed around zero.
    ///
    /// A zero-width distribution returns one sample at `mean` with unit weight;
    /// the integration window and sample count have no effect on that result.
    ///
    /// Weights retain the Gaussian probability mass inside the truncated window
    /// rather than being renormalised to sum to one.
    pub fn samples(&self) -> Result<Vec<VelocitySample>, String> {
        self.validate()?;
        if self.sigma == 0.0 {
            return Ok(vec![VelocitySample {
                kv: self.mean,
                weight: 1.0,
            }]);
        }
        let step = 2.0 * self.half_width / self.sample_count as f64;
        Ok((0..self.sample_count)
            .map(|index| {
                let kv = -self.half_width + (index as f64 + 0.5) * step;
                VelocitySample {
                    kv,
                    weight: step * normalised_gaussian(kv, self.mean, self.sigma),
                }
            })
            .collect())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VelocitySample {
    pub kv: f64,
    pub weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }

    #[test]
    fn samples_use_a_fixed_zero_centered_window() {
        let params = VelocityParams {
            sigma: 2.0,
            half_width: 1.5,
            sample_count: 3,
            ..VelocityParams::default()
        };
        let samples = params.samples().unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].kv, -1.0);
        assert_eq!(samples[1].kv, 0.0);
        assert_eq!(samples[2].kv, 1.0);
        for sample in samples {
            assert_close(
                sample.weight,
                normalised_gaussian(sample.kv, 0.0, 2.0),
                1e-15,
            );
        }
    }

    #[test]
    fn samples_retain_the_gaussian_mass_inside_the_window() {
        let params = VelocityParams {
            sigma: 1.0,
            half_width: 1.0,
            sample_count: 1_000,
            ..VelocityParams::default()
        };
        let mass: f64 = params
            .samples()
            .unwrap()
            .iter()
            .map(|sample| sample.weight)
            .sum();
        assert_close(mass, 0.682_689_492_137, 1e-6);
    }

    #[test]
    fn zero_sigma_returns_one_unit_weight_sample_at_the_mean() {
        let params = VelocityParams {
            mean: 3.7,
            sigma: 0.0,
            half_width: 12.0,
            sample_count: 999,
        };
        let samples = params.samples().unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].kv, params.mean);
        assert_eq!(samples[0].weight, 1.0);
    }

    #[test]
    fn samples_validate_the_distribution_and_grid() {
        for params in [
            VelocityParams {
                mean: f64::NAN,
                ..VelocityParams::default()
            },
            VelocityParams {
                sigma: -1.0,
                ..VelocityParams::default()
            },
            VelocityParams {
                half_width: 0.0,
                ..VelocityParams::default()
            },
            VelocityParams {
                sample_count: 0,
                ..VelocityParams::default()
            },
        ] {
            assert!(params.samples().is_err());
        }
    }
}
