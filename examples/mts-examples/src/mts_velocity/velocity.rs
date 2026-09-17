use atomic_clocks::maths::normalised_gaussian;
use myplotlib::{Slider, SliderGroup};

use super::Params;
use crate::units::frequency_slider;

#[derive(Clone, Copy, Debug)]
pub(super) struct VelocitySample {
    pub(super) kv: f64,
    pub(super) weight: f64,
}

pub(super) fn control_group<'a>(
    kv_sigma: &'a mut f64,
    kv_window_half_width: &'a mut f64,
    kv_samples: &'a mut usize,
) -> SliderGroup<'a> {
    SliderGroup::new(
        "Velocity distribution",
        [
            frequency_slider("Sigma", kv_sigma, 0.1..=3000.0).logarithmic(true),
            frequency_slider("Integration window", kv_window_half_width, 0.1..=40.0),
            Slider::new("Samples", kv_samples, 1..=201),
        ],
    )
}

pub(super) fn velocity_samples(params: &Params) -> Result<Vec<VelocitySample>, String> {
    if params.kv_samples == 0 {
        return Err("velocity sample count must be positive".to_string());
    }
    if !params.kv_sigma.is_finite() || params.kv_sigma <= 0.0 {
        return Err("Gaussian velocity sigma must be positive and finite".to_string());
    }
    if !params.kv_window_half_width.is_finite() || params.kv_window_half_width <= 0.0 {
        return Err("velocity integration half-width must be positive and finite".to_string());
    }
    // In q = kv_physical + shift / 2 coordinates, q = 0 is the selected
    // velocity class and a zero-mean physical distribution is centred at shift / 2.
    let mean = params.mts.hamiltonian.modulation.shift / 2.0;
    if !mean.is_finite() {
        return Err("modulation shift must be finite".to_string());
    }

    // Midpoint quadrature over a window fixed around the resonant class q = 0.
    // Weights retain the Gaussian probability mass inside the truncated window.
    let step = 2.0 * params.kv_window_half_width / params.kv_samples as f64;
    Ok((0..params.kv_samples)
        .map(|index| {
            let kv = -params.kv_window_half_width + (index as f64 + 0.5) * step;
            VelocitySample {
                kv,
                weight: step * normalised_gaussian(kv, mean, params.kv_sigma),
            }
        })
        .collect())
}
