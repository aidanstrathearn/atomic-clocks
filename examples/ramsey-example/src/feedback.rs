use crate::params::RamseyParameters;
use crate::ramsey::{angular_frequency_to_khz, measurement_transfer};
use atomic_clocks::maths::linspace;
use atomic_clocks::signal_processing::{
    AdevSamples, AngularFrequencyGrid, Delay, FunctionalPsd, Integrator, LtiFeedback, LtiFilter,
    Psd, PsdSamples, Series, SpectrumError, TransferFunctionSamples,
};
use std::f64::consts::TAU;

const N_AVERAGING_TIMES: usize = 30;

fn psd_per_khz_to_per_angular_frequency(value: f64) -> f64 {
    value / TAU
}

fn psd_per_angular_frequency_to_per_khz(value: f64) -> f64 {
    TAU * value
}

fn density_per_khz(samples: &PsdSamples) -> Vec<f64> {
    samples
        .density_values()
        .iter()
        .map(|&value| psd_per_angular_frequency_to_per_khz(value))
        .collect()
}

pub(crate) struct FeedbackSpectra {
    pub(crate) frequencies_khz: Vec<f64>,
    pub(crate) free_running: Vec<f64>,
    pub(crate) injected_measurement_noise: Vec<f64>,
    pub(crate) total: Vec<f64>,
}

struct FeedbackPsdSamples {
    frequencies_khz: Vec<f64>,
    free_running: PsdSamples,
    #[cfg(test)]
    residual_signal: PsdSamples,
    injected_measurement_noise: PsdSamples,
    total: PsdSamples,
}

pub(crate) fn controller(
    params: &RamseyParameters,
    measurement: &TransferFunctionSamples,
) -> Series<Integrator, Delay> {
    let dc_slope = measurement.response_values()[0].re;
    let signed_gain = params.integrator_gain * dc_slope.signum();
    Integrator::new(signed_gain).then(Delay::new(params.feedback_delay_ms))
}

fn psd_samples(params: &mut RamseyParameters) -> Result<FeedbackPsdSamples, SpectrumError> {
    let measurement = measurement_transfer(params)?;
    let grid = AngularFrequencyGrid::new(measurement.grid().values()[1..].to_vec())?;
    let frequencies_khz = grid
        .values()
        .iter()
        .map(|&omega| angular_frequency_to_khz(omega))
        .collect();

    let oscillator_white_psd = 10.0_f64.powf(params.oscillator_white_psd_log10);
    let oscillator_random_walk = 10.0_f64.powf(params.oscillator_random_walk_log10);
    let measurement_white_psd = 10.0_f64.powf(params.measurement_white_psd_log10);
    let oscillator_noise = FunctionalPsd {
        spectrum: move |omega| {
            let frequency_khz = angular_frequency_to_khz(omega);
            psd_per_khz_to_per_angular_frequency(
                oscillator_white_psd + oscillator_random_walk / frequency_khz.powi(2),
            )
        },
    };
    let measurement_noise = FunctionalPsd {
        spectrum: move |_| psd_per_khz_to_per_angular_frequency(measurement_white_psd),
    };
    let controller = controller(params, &measurement);
    let feedback = LtiFeedback::new(oscillator_noise, measurement_noise, measurement, controller);

    let free_running = feedback.free_running_psd().sample(&grid)?;
    #[cfg(test)]
    let residual_signal = feedback.residual_signal_psd().sample(&grid)?;
    let injected_measurement_noise = feedback.injected_measurement_noise_psd().sample(&grid)?;
    let total = feedback.output_psd().sample(&grid)?;

    Ok(FeedbackPsdSamples {
        frequencies_khz,
        free_running,
        #[cfg(test)]
        residual_signal,
        injected_measurement_noise,
        total,
    })
}

pub(crate) fn spectra(params: &mut RamseyParameters) -> Result<FeedbackSpectra, SpectrumError> {
    let samples = psd_samples(params)?;
    Ok(FeedbackSpectra {
        frequencies_khz: samples.frequencies_khz,
        free_running: density_per_khz(&samples.free_running),
        injected_measurement_noise: density_per_khz(&samples.injected_measurement_noise),
        total: density_per_khz(&samples.total),
    })
}

pub(crate) struct FeedbackAdev {
    pub(crate) free_running: AdevSamples,
    pub(crate) total: AdevSamples,
}

fn logarithmic_grid(start: f64, stop: f64, points: usize) -> Vec<f64> {
    assert!(start.is_finite() && start > 0.0);
    assert!(stop.is_finite() && stop > start);
    assert!(points >= 2);
    linspace(start.ln(), stop.ln(), points - 1)
        .into_iter()
        .map(f64::exp)
        .collect()
}

pub(crate) fn adev(params: &mut RamseyParameters) -> Result<FeedbackAdev, SpectrumError> {
    let samples = psd_samples(params)?;
    let angular_frequencies = samples.total.grid().values();
    let minimum_angular_frequency = angular_frequencies[0];
    let maximum_angular_frequency = *angular_frequencies.last().unwrap();
    // Keep the useful part of the Allan-variance kernel inside the sampled band.
    let averaging_times = logarithmic_grid(
        TAU / maximum_angular_frequency,
        1.0 / minimum_angular_frequency,
        N_AVERAGING_TIMES,
    );
    let free_running = samples.free_running.to_adev(&averaging_times)?;
    let total = samples.total.to_adev(&averaging_times)?;
    Ok(FeedbackAdev {
        free_running,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10 * (1.0 + expected.abs()),
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn psd_frequency_density_conversion_round_trips() {
        let per_khz = 3.25;
        assert_close(
            psd_per_angular_frequency_to_per_khz(psd_per_khz_to_per_angular_frequency(per_khz)),
            per_khz,
        );
    }

    #[test]
    fn default_feedback_spectra_are_finite_and_add_up() {
        let samples = psd_samples(&mut RamseyParameters::default()).unwrap();
        assert!(!samples.frequencies_khz.is_empty());
        assert!(samples.frequencies_khz.iter().all(|value| *value > 0.0));
        for (((free_running, residual), injected), total) in samples
            .free_running
            .density_values()
            .iter()
            .zip(samples.residual_signal.density_values())
            .zip(samples.injected_measurement_noise.density_values())
            .zip(samples.total.density_values())
        {
            assert!(free_running.is_finite() && *free_running >= 0.0);
            assert!(residual.is_finite() && *residual >= 0.0);
            assert!(injected.is_finite() && *injected >= 0.0);
            assert!(total.is_finite() && *total >= 0.0);
            assert_close(*total, residual + injected);
        }
    }

    #[test]
    fn default_feedback_adev_is_finite() {
        let adev = adev(&mut RamseyParameters::default()).unwrap();
        assert_eq!(
            adev.free_running.averaging_times(),
            adev.total.averaging_times()
        );
        assert_eq!(adev.free_running.averaging_times().len(), N_AVERAGING_TIMES);
        assert!(
            adev.free_running
                .deviation_values()
                .iter()
                .chain(adev.total.deviation_values())
                .all(|value| value.is_finite() && *value > 0.0)
        );
    }
}
