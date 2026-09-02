mod maths;
mod operator;
mod state;

use std::f64::consts::PI;

use myplotlib::Plotter;

use crate::maths::{Linspace, linspace};
use crate::operator::{GaussianPulse, Operator, Ramsey};
use crate::state::QubitState;

fn main() -> myplotlib::Result {
    let pulse = GaussianPulse {
        pulse_area: 2.0 * PI,
        center: 0.0,
        width: 1.0,
        detuning: 0.0,
    };

    let mut ramsey = Ramsey {
        pulse_area: 0.5 * PI,
        detuning: 0.0,
        pulse_width: 0.1,
        pulse_separation: 2.0,
    };

    let qubit = QubitState::ground();
    let times = Linspace::new(-1.0, 3.0, 1000);

    let mut plt = Plotter::new();
    for j in 0..5 {
        ramsey = Ramsey {
            detuning: 1.0 * j as f64,
            ..ramsey
        };

        let response = qubit.linear_response(&ramsey, &times, Operator::pauli_x());
        plt.plot(&times.array, &response);
    }
    plt.show()?;

    let times = Linspace::new(-1.0, 3.0, 1000);

    let detunings = linspace(-35.0, 35.0, 1000);
    let signal: Vec<_> = detunings
        .iter()
        .map(|&d| {
            ramsey = Ramsey {
                detuning: d,
                ..ramsey
            };
            qubit
                .propagate_to_final(&ramsey, &times)
                .ground_probability()
        })
        .collect();
    let mut plt = Plotter::new();
    plt.plot(&detunings, &signal);
    plt.show()?;
    Ok(())
}
