mod maths;
mod operator;
mod state;

use std::f64::consts::PI;

use myplotlib::Plotter;

use crate::maths::Linspace;
use crate::operator::{GaussianPulse, Operator, Ramsey};
use crate::state::QubitState;

fn main() -> myplotlib::Result {
    let pulse = GaussianPulse {
        pulse_area: 2.0 * PI,
        center: 0.0,
        width: 1.0,
        detuning: 0.0,
    };

    let mut plt = Plotter::new();
    for j in 0..5 {
        let ramsey = Ramsey {
            pulse_area: 0.5 * PI,
            detuning: 1.0 * j as f64,
            pulse_width: 0.1,
            pulse_separation: 2.0,
        };

        let times = Linspace::new(-1.0, 3.0, 1000);
        let qubit = QubitState::ground();

        // let (t, states) = qubit.propagate(&ramsey, times);
        let (t, response) = qubit.linear_response(&ramsey, times, Operator::pauli_x());

        plt.plot(&t, &response);
    }
    plt.show()?;
    Ok(())
}
