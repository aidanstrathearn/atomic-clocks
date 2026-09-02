mod maths;
mod operator;
mod ramsey;
mod state;

use std::f64::consts::PI;

use myplotlib::Plotter;

use crate::maths::{Linspace, linspace};
use crate::ramsey::Ramsey;
use crate::state::QubitState;

fn main() -> myplotlib::Result {
    let qubit = QubitState::ground();
    let times = Linspace::new(-1.0, 3.0, 1000);
    let pulse_separation = 2.0;
    let detuning_lim = 25.0;
    let detunings = linspace(-detuning_lim, detuning_lim, 1000);

    let ramsey_signal = |detuning, phase_diff| {
        Ramsey {
            pulse_area: 0.5 * PI,
            detuning,
            pulse_width: 0.1,
            pulse_separation: 2.0,
            phase_diff,
        }
        .propagate_to_final(&qubit, &times)
        .ground_probability()
    };

    let phase_difference: Vec<_> = detunings
        .iter()
        .map(|&detuning| ramsey_signal(detuning, - PI * 0.5) - ramsey_signal(detuning, PI * 0.5))
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

    let mut plt = Plotter::new();
    plt.plot(&detunings, &phase_difference)
        .label("Phase shift");
    plt.plot(&detunings, &frequency_difference)
        .label("Frequency shift");
    plt.xlabel("Detuning (rad / time)");
    plt.ylabel("Differential ground-state probability");
    plt.show()?;
    Ok(())
}
