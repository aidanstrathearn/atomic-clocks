use crate::params::RamseyParameters;
use crate::ramsey::{angular_frequency_to_khz, measurement_transfer};
use atomic_clocks::maths::linspace;
use atomic_clocks::signal_processing::{
    AdevSamples, AngularFrequencyGrid, Delay, FunctionalPsd, Integrator, LtiFeedback, LtiFilter,
    Psd, PsdSamples, Series, TransferFunctionSamples,
};
use std::f64::consts::TAU;

const N_AVERAGING_TIMES: usize = 30;

fn psd_per_khz_to_per_angular_frequency(value: f64) -> f64 {
    value / TAU
}

pub(crate) struct FeedbackSpectra {
    pub(crate) free_running: PsdSamples,
    /// Retained as part of the complete feedback decomposition; currently used by tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) residual_signal: PsdSamples,
    pub(crate) injected_measurement_noise: PsdSamples,
    pub(crate) total: PsdSamples,
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
) -> Result<FeedbackSpectra, Box<dyn std::error::Error>> {
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

    let free_running = feedback.free_running_psd().sample(&grid)?;
    let residual_signal = feedback.residual_signal_psd().sample(&grid)?;
    let injected_measurement_noise = feedback.injected_measurement_noise_psd().sample(&grid)?;
    let total = feedback.output_psd().sample(&grid)?;

    Ok(FeedbackSpectra {
        free_running,
        residual_signal,
        injected_measurement_noise,
        total,
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

pub(crate) fn adev(
    params: &mut RamseyParameters,
) -> Result<FeedbackAdev, Box<dyn std::error::Error>> {
    let spectra = spectra(params)?;
    let angular_frequencies = spectra.total.grid().values();
    let minimum_angular_frequency = angular_frequencies[0];
    let maximum_angular_frequency = *angular_frequencies.last().unwrap();
    // Keep the useful part of the Allan-variance kernel inside the sampled band.
    let averaging_times = logarithmic_grid(
        TAU / maximum_angular_frequency,
        1.0 / minimum_angular_frequency,
        N_AVERAGING_TIMES,
    );
    let free_running = spectra.free_running.to_adev(&averaging_times)?;
    let total = spectra.total.to_adev(&averaging_times)?;
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
        assert!(!spectra.total.grid().values().is_empty());
        assert!(
            spectra
                .total
                .grid()
                .values()
                .iter()
                .all(|value| *value > 0.0)
        );
        for (((free_running, residual), injected), total) in spectra
            .free_running
            .density_values()
            .iter()
            .zip(spectra.residual_signal.density_values())
            .zip(spectra.injected_measurement_noise.density_values())
            .zip(spectra.total.density_values())
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
