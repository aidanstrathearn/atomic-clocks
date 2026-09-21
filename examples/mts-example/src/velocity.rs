use atomic_clocks::vapourcell::VelocityParams;
use myplotlib::{Slider, SliderGroup};

use crate::units::frequency_slider;

pub(super) fn control_group(params: &mut VelocityParams) -> SliderGroup<'_> {
    SliderGroup::new(
        "Velocity distribution",
        [
            frequency_slider("Sigma", &mut params.sigma, 0.1..=3000.0).logarithmic(true),
            frequency_slider("Integration window", &mut params.half_width, 0.1..=40.0),
            Slider::new("Samples", &mut params.sample_count, 1..=201),
        ],
    )
}
