mod maths;
mod operator;
mod ramsey;
mod state;

use std::f64::consts::PI;

use myplotlib::Plotter;

use crate::maths::{Linspace, linspace};
use crate::ramsey::{ModulatedRamsey, Ramsey};
use crate::state::QubitState;

fn main() -> myplotlib::Result {
    let qubit = QubitState::ground();
    let times = Linspace::new(-1.0, 3.0, 1000);
    let pulse_separation = 2.0;
    let pulse_width = 0.1;
    let pulse_area = 0.5 * PI;
    let detuning_lim = 25.0;
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

    let phase_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| ramsey_signal(detuning, -PI * 0.5) - ramsey_signal(detuning, PI * 0.5))
        .collect();

    let fringe_width = 2.0 * PI / pulse_separation;
    let frequency_shift = fringe_width / 4.0;
    let frequency_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| {
            ramsey_signal(detuning + frequency_shift, 0.0)
                - ramsey_signal(detuning - frequency_shift, 0.0)
        })
        .collect();

    let modulation_frequency = frequency_shift;
    let modulation_index = 0.1;
    let modulation_depth = modulation_index * modulation_frequency;
    let modulation_phase_samples = 128;
    let modulation_phases: Vec<_> = (0..modulation_phase_samples)
        .map(|index| 2.0 * PI * index as f64 / modulation_phase_samples as f64)
        .collect();

    let small_signal_gain = modulation_depth / modulation_frequency
        * (modulation_frequency * pulse_separation * 0.5).sin();
    let modulated_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| {
            let in_phase = modulation_phases
                .iter()
                .map(|&phase| {
                    let start_time = phase / modulation_frequency - pulse_separation * 0.5;
                    let probability = ModulatedRamsey {
                        pulse_area,
                        detuning,
                        mod_freq: modulation_frequency,
                        mod_depth: modulation_depth,
                        start_time,
                        pulse_width,
                        pulse_separation,
                    }
                    .propagate_to_final(&qubit, &times)
                    .ground_probability();

                    probability * phase.sin()
                })
                .sum::<f64>()
                * 2.0
                / modulation_phase_samples as f64;

            in_phase / small_signal_gain
        })
        .collect();

    let mut plt = Plotter::new();
    plt.plot(&detunings, &phase_difference).label("Phase shift");
    plt.plot(&detunings, &frequency_difference)
        .label("Frequency shift");
    plt.plot(&detunings, &modulated_difference)
        .label("Continuous FM demodulation");
    plt.xlabel("Detuning (rad / time)");
    plt.ylabel("Normalized differential ground-state probability");
    plt.show()?;
    Ok(())
}
