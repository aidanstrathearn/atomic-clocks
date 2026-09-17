use atomic_clocks::maths::demodulation::ModulationParams;
use atomic_clocks::vapourcell::{DemodOutput, MtsParams};
use myplotlib::{Plotter, Slider, SliderGroup};

use crate::units::{frequency_slider, to_mhz};

/// Scalar curves derived from one combined demodulation output for plotting.
pub(super) struct MtsCurves {
    pub hz: Vec<f64>,
    pub amp0: Vec<f64>,
    pub proj1: Vec<f64>,
    pub proj2: Vec<f64>,
    pub proj3: Vec<f64>,
}

impl From<DemodOutput> for MtsCurves {
    fn from(output: DemodOutput) -> Self {
        let amp0: Vec<_> = output
            .dc
            .iter()
            .map(|value| value.amplitude() / 2.0)
            .collect();
        Self {
            hz: output.hz,
            amp0,
            proj1: output.harmonic.iter().map(|value| value.in_phase).collect(),
            proj2: output
                .second_harmonic
                .iter()
                .map(|value| value.in_phase)
                .collect(),
            proj3: output
                .third_harmonic
                .iter()
                .map(|value| value.in_phase)
                .collect(),
        }
    }
}

pub(super) fn scan_control_group<'a>(
    hz_lim: &'a mut f64,
    hz_num: &'a mut usize,
) -> SliderGroup<'a> {
    SliderGroup::new(
        "Scan",
        [
            frequency_slider("Detuning half-range", hz_lim, 0.1..=65.0),
            Slider::new("Detuning samples", hz_num, 2..=200),
        ],
    )
}

pub(super) fn atom_control_groups<'a>(
    modulation: &'a mut ModulationParams,
    r_pump: &'a mut f64,
    r_prbe: &'a mut f64,
    gamma_down: &'a mut f64,
    gamma_phi: &'a mut f64,
) -> [SliderGroup<'a>; 2] {
    [
        SliderGroup::new(
            "Modulation",
            [
                frequency_slider("Frequency", &mut modulation.frequency, 0.1..=20.0),
                frequency_slider("Depth", &mut modulation.depth, 0.0..=20.0),
                frequency_slider("Shift", &mut modulation.shift, -40.0..=40.0),
            ],
        ),
        SliderGroup::new(
            "Rates",
            [
                frequency_slider("Pump Rabi", r_pump, 0.0..=20.0),
                frequency_slider("Probe Rabi", r_prbe, 0.0..=20.0),
                frequency_slider("Decay", gamma_down, 0.01..=10.0).logarithmic(true),
                frequency_slider("Dephasing", gamma_phi, 0.0..=10.0),
            ],
        ),
    ]
}

pub(super) fn signal_plot(params: &MtsParams, output: &MtsCurves) -> Plotter {
    let mut plot = Plotter::new();
    let detunings_mhz: Vec<_> = output
        .hz
        .iter()
        .map(|&hz| to_mhz(display_detuning(params, hz)))
        .collect();
    let minus_amp0: Vec<_> = output.amp0.iter().map(|x| -x).collect();
    plot.plot(&detunings_mhz, &minus_amp0).label("DC");
    plot.plot(&detunings_mhz, &output.proj1)
        .label("First harmonic");
    plot.plot(&detunings_mhz, &output.proj2)
        .label("Second harmonic");
    plot.plot(&detunings_mhz, &output.proj3)
        .label("Third harmonic");
    plot.xlabel("Detuning (MHz)");
    plot.ylabel("Demodulated signal");
    plot.xlim(
        to_mhz(display_detuning(params, -params.solver.hz_lim)),
        to_mhz(display_detuning(params, params.solver.hz_lim)),
    );
    plot
}

pub(super) fn display_detuning(params: &MtsParams, detuning: f64) -> f64 {
    detuning + params.hamiltonian.modulation.shift / 2.0
}
