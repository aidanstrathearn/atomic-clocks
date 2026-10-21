use std::f64::consts::TAU;
use std::ops::RangeInclusive;

use myplotlib::Slider;

const GAUSSIAN_FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;

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

fn sigma_to_fwhm_mhz(angular_sigma: f64) -> f64 {
    to_mhz(angular_sigma * GAUSSIAN_FWHM_PER_SIGMA)
}

fn fwhm_mhz_to_sigma(fwhm_mhz: f64) -> f64 {
    to_angular_frequency(fwhm_mhz) / GAUSSIAN_FWHM_PER_SIGMA
}

pub(super) fn gaussian_fwhm_slider<'a>(
    label: &'static str,
    sigma: &'a mut f64,
    angular_sigma_range: RangeInclusive<f64>,
) -> Slider<'a> {
    let fwhm_range = sigma_to_fwhm_mhz(*angular_sigma_range.start())
        ..=sigma_to_fwhm_mhz(*angular_sigma_range.end());
    Slider::from_get_set(label, fwhm_range, move |fwhm_mhz| {
        if let Some(fwhm_mhz) = fwhm_mhz {
            *sigma = fwhm_mhz_to_sigma(fwhm_mhz);
        }
        sigma_to_fwhm_mhz(*sigma)
    })
}

pub(super) fn angular_gradient_to_per_mhz(gradient: f64) -> f64 {
    gradient * TAU
}

#[cfg(test)]
mod tests {
    use super::{
        GAUSSIAN_FWHM_PER_SIGMA, angular_gradient_to_per_mhz, fwhm_mhz_to_sigma, sigma_to_fwhm_mhz,
        to_angular_frequency, to_mhz,
    };

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

    #[test]
    fn gaussian_width_conversion_round_trips() {
        let angular_sigma = 3.7;
        let fwhm_mhz = sigma_to_fwhm_mhz(angular_sigma);

        assert_eq!(fwhm_mhz, to_mhz(angular_sigma) * GAUSSIAN_FWHM_PER_SIGMA);
        assert!((fwhm_mhz_to_sigma(fwhm_mhz) - angular_sigma).abs() < 1e-15);
    }
}
