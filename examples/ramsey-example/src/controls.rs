use crate::params::RamseyParameters;
use atomic_clocks::common::DetuningScanParams;
use myplotlib::{Slider, SliderGrid, SliderGroup};
use std::f64::consts::{PI, TAU};

const DETUNING_STEP_KHZ: f64 = 0.01;

fn ramsey_sliders<'a>(
    pulse_width_ms: &'a mut f64,
    ramsey_time_ms: &'a mut f64,
    pulse_area: &'a mut f64,
    detuning_khz: &'a mut f64,
    time_steps: &'a mut usize,
    detuning_limit_khz: f64,
) -> SliderGroup<'a> {
    SliderGroup::new(
        "Ramsey parameters",
        [
            Slider::new("Pulse width (ms)", pulse_width_ms, 0.02..=0.5),
            Slider::new("Ramsey time (ms)", ramsey_time_ms, 0.02..=4.0),
            Slider::new("Pulse area (rad)", pulse_area, 0.0..=2.0 * PI),
            Slider::new(
                "Detuning (kHz)",
                detuning_khz,
                -detuning_limit_khz..=detuning_limit_khz,
            )
            .step_by(DETUNING_STEP_KHZ),
            Slider::new("Time steps", time_steps, 2..=10_000),
        ],
    )
}

fn prepare_detuning_slider(params: &mut RamseyParameters) -> f64 {
    let limit = params.scan.frequency_half_range();
    params.detuning_khz = params.detuning_khz.clamp(-limit, limit);
    limit
}

fn scan_sliders(scan: &mut DetuningScanParams) -> SliderGroup<'_> {
    let DetuningScanParams { hz_lim, hz_num } = scan;
    SliderGroup::new(
        "Scan",
        [
            Slider::from_get_set(
                "Detuning half-range (kHz)",
                0.01..=65.0,
                move |frequency_khz| {
                    if let Some(frequency_khz) = frequency_khz {
                        *hz_lim = TAU * frequency_khz;
                    }
                    *hz_lim / TAU
                },
            )
            .step_by(DETUNING_STEP_KHZ),
            Slider::new("Detuning samples", hz_num, 2..=1_001),
        ],
    )
}

fn controller_sliders<'a>(
    integrator_gain: &'a mut f64,
    feedback_delay_ms: &'a mut f64,
) -> SliderGroup<'a> {
    SliderGroup::new(
        "Control parameters",
        [
            Slider::new(
                "Integrator gain magnitude (kHz / probability ms)",
                integrator_gain,
                0.0..=5.0,
            )
            .step_by(0.001),
            Slider::new("Feedback delay (ms)", feedback_delay_ms, 0.0..=2.0).step_by(0.001),
        ],
    )
}

fn psd_sliders<'a>(
    oscillator_white_psd_log10: &'a mut f64,
    oscillator_random_walk_log10: &'a mut f64,
    measurement_white_psd_log10: &'a mut f64,
) -> SliderGroup<'a> {
    SliderGroup::new(
        "PSD parameters",
        [
            Slider::new(
                "log10 oscillator white PSD (kHz^2/kHz)",
                oscillator_white_psd_log10,
                -12.0..=2.0,
            ),
            Slider::new(
                "log10 oscillator random walk (kHz^3)",
                oscillator_random_walk_log10,
                -12.0..=2.0,
            ),
            Slider::new(
                "log10 measurement white PSD (probability^2/kHz)",
                measurement_white_psd_log10,
                -12.0..=2.0,
            ),
        ],
    )
}

pub(crate) fn standard(params: &mut RamseyParameters) -> SliderGrid<'_> {
    let detuning_limit_khz = prepare_detuning_slider(params);
    let RamseyParameters {
        pulse_width_ms,
        ramsey_time_ms,
        pulse_area,
        detuning_khz,
        time_steps,
        ..
    } = params;

    SliderGrid::new(
        3,
        [ramsey_sliders(
            pulse_width_ms,
            ramsey_time_ms,
            pulse_area,
            detuning_khz,
            time_steps,
            detuning_limit_khz,
        )],
    )
}

pub(crate) fn signal(params: &mut RamseyParameters) -> SliderGrid<'_> {
    let detuning_limit_khz = prepare_detuning_slider(params);
    let RamseyParameters {
        pulse_width_ms,
        ramsey_time_ms,
        pulse_area,
        detuning_khz,
        time_steps,
        scan,
        ..
    } = params;

    SliderGrid::new(
        3,
        [
            ramsey_sliders(
                pulse_width_ms,
                ramsey_time_ms,
                pulse_area,
                detuning_khz,
                time_steps,
                detuning_limit_khz,
            ),
            scan_sliders(scan),
        ],
    )
}

pub(crate) fn feedback(params: &mut RamseyParameters) -> SliderGrid<'_> {
    let detuning_limit_khz = prepare_detuning_slider(params);
    let RamseyParameters {
        pulse_width_ms,
        ramsey_time_ms,
        pulse_area,
        detuning_khz,
        time_steps,
        oscillator_white_psd_log10,
        oscillator_random_walk_log10,
        measurement_white_psd_log10,
        integrator_gain,
        feedback_delay_ms,
        ..
    } = params;

    SliderGrid::new(
        3,
        [
            ramsey_sliders(
                pulse_width_ms,
                ramsey_time_ms,
                pulse_area,
                detuning_khz,
                time_steps,
                detuning_limit_khz,
            ),
            controller_sliders(integrator_gain, feedback_delay_ms),
            psd_sliders(
                oscillator_white_psd_log10,
                oscillator_random_walk_log10,
                measurement_white_psd_log10,
            ),
        ],
    )
}

pub(crate) fn nyquist(params: &mut RamseyParameters) -> SliderGrid<'_> {
    let detuning_limit_khz = prepare_detuning_slider(params);
    let RamseyParameters {
        pulse_width_ms,
        ramsey_time_ms,
        pulse_area,
        detuning_khz,
        time_steps,
        integrator_gain,
        feedback_delay_ms,
        ..
    } = params;

    SliderGrid::new(
        3,
        [
            ramsey_sliders(
                pulse_width_ms,
                ramsey_time_ms,
                pulse_area,
                detuning_khz,
                time_steps,
                detuning_limit_khz,
            ),
            controller_sliders(integrator_gain, feedback_delay_ms),
        ],
    )
}
