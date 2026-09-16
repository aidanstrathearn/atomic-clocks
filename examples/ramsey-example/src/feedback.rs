use crate::params::RamseyParameters;
use crate::ramsey::{angular_frequency_to_khz, measurement_transfer};
use atomic_clocks::maths::logspace;
use atomic_clocks::signal_processing::{
    AdevSamples, AngularFrequencyGrid, Delay, FeedbackPsdSamples, FunctionalPsd, Integrator,
    LtiFeedback, LtiFilter, Series, TransferFunctionSamples,
};
use std::f64::consts::TAU;

const N_AVERAGING_TIMES: usize = 30;

fn psd_per_khz_to_per_angular_frequency(value: f64) -> f64 {
    value / TAU
}

pub(crate) fn controller(
    params: &RamseyParameters,
    measurement: &TransferFunctionSamples,
) -> Series<Integrator, Delay> {
    let dc_slope = measurement.response_values()[0].re;
    let signed_gain = params.integrator_gain * dc_slope.signum();
    Integrator::new(signed_gain).then(Delay::new(params.feedback_delay_ms))
}

pub(crate) fn spectra(
    params: &mut RamseyParameters,
) -> Result<FeedbackPsdSamples, Box<dyn std::error::Error>> {
    let measurement = measurement_transfer(params)?;
    let grid = AngularFrequencyGrid::new(measurement.grid().values()[1..].to_vec())?;

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
    Ok(feedback.sample_psds(&grid)?)
}

pub(crate) struct FeedbackAdev {
    pub(crate) free_running: AdevSamples,
    pub(crate) total: AdevSamples,
}

pub(crate) fn adev(
    params: &mut RamseyParameters,
) -> Result<FeedbackAdev, Box<dyn std::error::Error>> {
    let spectra = spectra(params)?;
    let angular_frequencies = spectra.output().grid().values();
    let minimum_angular_frequency = angular_frequencies[0];
    let maximum_angular_frequency = *angular_frequencies.last().unwrap();
    // Keep the useful part of the Allan-variance kernel inside the sampled band.
    let averaging_times = logspace(
        TAU / maximum_angular_frequency,
        1.0 / minimum_angular_frequency,
        N_AVERAGING_TIMES - 1,
    );
    let free_running = spectra.free_running().to_adev(&averaging_times)?;
    let total = spectra.output().to_adev(&averaging_times)?;
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
    fn default_feedback_spectra_are_finite_and_add_up() {
        let spectra = spectra(&mut RamseyParameters::default()).unwrap();
        assert!(!spectra.output().grid().values().is_empty());
        assert!(
            spectra
                .output()
                .grid()
                .values()
                .iter()
                .all(|value| *value > 0.0)
        );
        for (((free_running, residual), injected), total) in spectra
            .free_running()
            .density_values()
            .iter()
            .zip(spectra.residual_signal().density_values())
            .zip(spectra.injected_measurement_noise().density_values())
            .zip(spectra.output().density_values())
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
