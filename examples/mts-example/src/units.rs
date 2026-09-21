use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use myplotlib::Slider;

pub(super) fn to_mhz(angular_frequency: f64) -> f64 {
    angular_frequency / TAU
}

fn to_angular_frequency(frequency_mhz: f64) -> f64 {
    frequency_mhz * TAU
}

pub(super) fn frequency_slider<'a>(
    label: &'static str,
    value: &'a mut f64,
    angular_range: RangeInclusive<f64>,
) -> Slider<'a> {
    let frequency_range = to_mhz(*angular_range.start())..=to_mhz(*angular_range.end());
    Slider::from_get_set(label, frequency_range, move |frequency_mhz| {
        if let Some(frequency_mhz) = frequency_mhz {
            *value = to_angular_frequency(frequency_mhz);
        }
        to_mhz(*value)
    })
}

pub(super) fn angular_gradient_to_per_mhz(gradient: f64) -> f64 {
    gradient * TAU
}

#[cfg(test)]
mod tests {
    use super::{angular_gradient_to_per_mhz, to_angular_frequency, to_mhz};

    #[test]
    fn frequency_conversion_round_trips() {
        let angular_frequency = 3.7;
        assert_eq!(
            to_angular_frequency(to_mhz(angular_frequency)),
            angular_frequency
        );
    }

    #[test]
    fn angular_gradient_is_converted_to_per_mhz() {
        let angular_gradient = 2.3;
        assert_eq!(
            angular_gradient_to_per_mhz(angular_gradient),
            to_angular_frequency(angular_gradient)
        );
    }
}
