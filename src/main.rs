mod maths;
mod operator;
mod ramsey;
mod state;

use std::f64::consts::PI;

use myplotlib::Figure;

use crate::maths::{Linspace, linspace};
use crate::ramsey::{ModulatedRamsey, Ramsey};
use crate::state::QubitState;

struct Demodulation {
    in_phase: f64,
    quadrature: f64,
}

impl Demodulation {
    fn at_phase(&self, phase_offset: f64) -> f64 {
        self.in_phase * phase_offset.cos() + self.quadrature * phase_offset.sin()
    }
}

fn demodulate_measurements(
    measurement_start_times: &[f64],
    modulation_frequency: f64,
    reference_time_offset: f64,
    mut measure: impl FnMut(f64) -> f64,
) -> Demodulation {
    assert!(
        !measurement_start_times.is_empty(),
        "demodulation requires at least one measurement"
    );

    let (in_phase, quadrature) =
        measurement_start_times
            .iter()
            .fold((0.0, 0.0), |(in_phase, quadrature), &start_time| {
                let phase = modulation_frequency * (start_time + reference_time_offset);
                let (reference_sin, reference_cos) = phase.sin_cos();
                let signal = measure(start_time);

                (
                    in_phase + signal * reference_sin,
                    quadrature + signal * reference_cos,
                )
            });
    let scale = 2.0 / measurement_start_times.len() as f64;

    Demodulation {
        in_phase: scale * in_phase,
        quadrature: scale * quadrature,
    }
}

fn demodulate_measurements_fitted(
    measurement_start_times: &[f64],
    modulation_frequency: f64,
    reference_time_offset: f64,
    mut measure: impl FnMut(f64) -> f64,
) -> Demodulation {
    assert!(
        measurement_start_times.len() >= 3,
        "demodulation requires at least three measurements"
    );

    let samples: Vec<_> = measurement_start_times
        .iter()
        .map(|&start_time| {
            let phase = modulation_frequency * (start_time + reference_time_offset);
            let (reference_sin, reference_cos) = phase.sin_cos();
            (measure(start_time), reference_sin, reference_cos)
        })
        .collect();

    let sample_count = samples.len() as f64;
    let mean_signal = samples.iter().map(|sample| sample.0).sum::<f64>() / sample_count;
    let mean_sin = samples.iter().map(|sample| sample.1).sum::<f64>() / sample_count;
    let mean_cos = samples.iter().map(|sample| sample.2).sum::<f64>() / sample_count;

    let (sin_sin, sin_cos, cos_cos, signal_sin, signal_cos) = samples.iter().fold(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        |(sin_sin, sin_cos, cos_cos, signal_sin, signal_cos), sample| {
            let signal = sample.0 - mean_signal;
            let reference_sin = sample.1 - mean_sin;
            let reference_cos = sample.2 - mean_cos;

            (
                sin_sin + reference_sin * reference_sin,
                sin_cos + reference_sin * reference_cos,
                cos_cos + reference_cos * reference_cos,
                signal_sin + signal * reference_sin,
                signal_cos + signal * reference_cos,
            )
        },
    );

    let determinant = sin_sin * cos_cos - sin_cos * sin_cos;
    assert!(
        determinant.abs() > 1.0e-12,
        "measurement phases do not span both demodulation quadratures"
    );

    Demodulation {
        in_phase: (signal_sin * cos_cos - signal_cos * sin_cos) / determinant,
        quadrature: (signal_cos * sin_sin - signal_sin * sin_cos) / determinant,
    }
}

fn main() -> myplotlib::Result {
    let qubit = QubitState::ground();
    let times = Linspace::new(-1.0, 3.0, 500);
    let pulse_separation = 2.0;
    let pulse_width = 0.2;
    let pulse_area = 0.5 * PI;
    let detuning_lim = 20.0 * 0.2 / pulse_width;
    let detunings = linspace(-detuning_lim, detuning_lim, 1000);

    let ramsey_signal = |detuning, phase_diff| {
        Ramsey {
            pulse_area,
            detuning,
            pulse_width,
            pulse_separation,
            phase_diff,
        }
        .propagate_to_final(&qubit, &times)
        .ground_probability()
    };

    /////////////////////////////////////////////////////////////////////////////////////

    let phase_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| ramsey_signal(detuning, -PI * 0.5) - ramsey_signal(detuning, PI * 0.5))
        .collect();

    //////////////////////////////////////////////////////////////////////////////////////

    let fringe_width = 2.0 * PI / pulse_separation;
    let frequency_shift = fringe_width / 4.0;
    let frequency_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| {
            ramsey_signal(detuning + frequency_shift, 0.0)
                - ramsey_signal(detuning - frequency_shift, 0.0)
        })
        .collect();

    ////////////////////////////////////////////////////////////////////////////////

    let modulation_frequency = frequency_shift;
    let modulation_index = 0.05;
    let modulation_depth = modulation_index * modulation_frequency;
    let modulated_ramsey_signal = |detuning, start_time| {
        ModulatedRamsey {
            pulse_area,
            detuning,
            mod_freq: modulation_frequency,
            mod_depth: modulation_depth,
            start_time,
            pulse_width,
            pulse_separation,
        }
        .propagate_to_final(&qubit, &times)
        .ground_probability()
    };

    let modulation_phase_samples = 8;
    let measurement_start_times: Vec<_> = (0..modulation_phase_samples)
        .map(|index| {
            let phase = 2.0 * PI * index as f64 / modulation_phase_samples as f64;
            phase / modulation_frequency - pulse_separation * 0.5
        })
        .collect();

    let small_signal_gain = modulation_depth / modulation_frequency
        * (modulation_frequency * pulse_separation * 0.5).sin();
    let modulated_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| {
            let demodulation = demodulate_measurements(
                &measurement_start_times,
                modulation_frequency,
                pulse_separation * 0.5,
                |start_time| modulated_ramsey_signal(detuning, start_time),
            );

            demodulation.at_phase(0.0) / small_signal_gain
        })
        .collect();

    let measurement_cycle_time = pulse_separation * 4.3;
    let discrete_measurement_count = 128 * 2;
    let discrete_measurement_start_times: Vec<_> = (0..discrete_measurement_count)
        .map(|index| index as f64 * measurement_cycle_time)
        .collect();
    let sampled_modulated_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| {
            let demodulation = demodulate_measurements(
                &discrete_measurement_start_times,
                modulation_frequency,
                pulse_separation * 0.5,
                |start_time| modulated_ramsey_signal(detuning, start_time),
            );

            demodulation.at_phase(0.0) / small_signal_gain
        })
        .collect();

    ///////////////////////////////////////////////////////////////////

    let phase_freq: Vec<_> = phase_difference
        .iter()
        .zip(frequency_difference.iter())
        .map(|(x, y)| x - y)
        .collect();

    let phase_mod: Vec<_> = phase_difference
        .iter()
        .zip(modulated_difference.iter())
        .map(|(x, y)| x - y)
        .collect();

    let phase_sampled_mod: Vec<_> = phase_difference
        .iter()
        .zip(sampled_modulated_difference.iter())
        .map(|(x, y)| x - y)
        .collect();

    let mut figure = Figure::subplots(1, 2);
    figure.suptitle("Ramsey demodulation comparison");

    let raw_axes = figure.axes_mut(0, 0);
    raw_axes
        .plot(&detunings, &phase_difference)
        .label("Phase shift");
    raw_axes
        .plot(&detunings, &frequency_difference)
        .label("Frequency shift");
    raw_axes
        .plot(&detunings, &modulated_difference)
        .label("Continuous FM demodulation");
    raw_axes
        .plot(&detunings, &sampled_modulated_difference)
        .label(format!("Sampled FM (T_c = {measurement_cycle_time})"));
    raw_axes.title("Demodulated signals");
    raw_axes.xlabel("Detuning (rad / time)");
    raw_axes.ylabel("Normalized differential ground-state probability");

    let difference_axes = figure.axes_mut(0, 1);
    difference_axes
        .plot(&detunings, &phase_freq)
        .label("Phase - freq");
    difference_axes
        .plot(&detunings, &phase_mod)
        .label("Phase - mod");
    difference_axes
        .plot(&detunings, &phase_sampled_mod)
        .label("Phase - sampled mod");
    difference_axes.title("Differences from phase demodulation");
    difference_axes.xlabel("Detuning (rad / time)");
    difference_axes.ylabel("Residual ground-state probability");

    figure.show()?;
    Ok(())
}
