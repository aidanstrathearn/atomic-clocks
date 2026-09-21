use std::f64::consts::FRAC_PI_2;

use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{DemodOutput, HamiltonianParams, MtsParams, compute_demod};
use myplotlib::{AppResult, Plotter, Slider, SliderGrid, SliderGroup};
use rayon::prelude::*;

use crate::app::{Params, common_control_groups};
use crate::units::{frequency_slider, to_mhz};
use crate::velocity;

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

fn scan_control_group<'a>(hz_lim: &'a mut f64, hz_num: &'a mut usize) -> SliderGroup<'a> {
    SliderGroup::new(
        "Scan",
        [
            frequency_slider("Detuning half-range", hz_lim, 0.1..=65.0),
            Slider::new("Detuning samples", hz_num, 2..=200),
        ],
    )
}

fn signal_plot(params: &MtsParams, output: &MtsCurves) -> Plotter {
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

pub(super) fn controls(params: &mut Params) -> SliderGrid<'_> {
    let scan = scan_control_group(&mut params.mts.solver.hz_lim, &mut params.mts.solver.hz_num);
    let common = common_control_groups(
        &mut params.mts.hamiltonian,
        &mut params.mts.decay,
        &mut params.mts.solver.kr_n,
        &mut params.mts.solver.steps_per_period,
        &mut params.mts.solver.n_periods,
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(5, common.into_iter().chain([scan]))
}

fn scale_demod_output(output: &mut DemodOutput, weight: f64) {
    for values in [
        &mut output.dc,
        &mut output.harmonic,
        &mut output.second_harmonic,
        &mut output.third_harmonic,
    ] {
        for value in values {
            value.in_phase *= weight;
            value.quadrature *= weight;
        }
    }
}

fn add_demod_output(sum: &mut DemodOutput, output: DemodOutput, weight: f64) -> Result<(), String> {
    if sum.hz != output.hz {
        return Err("velocity signal grids do not match".to_string());
    }
    for (sums, values) in [
        (&mut sum.dc, output.dc),
        (&mut sum.harmonic, output.harmonic),
        (&mut sum.second_harmonic, output.second_harmonic),
        (&mut sum.third_harmonic, output.third_harmonic),
    ] {
        if sums.len() != values.len() {
            return Err("velocity signal dimensions do not match".to_string());
        }
        for (sum, value) in sums.iter_mut().zip(values) {
            sum.in_phase += weight * value.in_phase;
            sum.quadrature += weight * value.quadrature;
        }
    }
    Ok(())
}

pub(super) fn velocity_average(params: &Params) -> Result<MtsCurves, String> {
    let samples = velocity::velocity_samples(params)?;
    let probe = MtsParams {
        hamiltonian: params.probe_hamiltonian(0.0),
        ..params.mts
    };

    // Collect in sample order so summation is independent of Rayon scheduling.
    let outputs: Result<Vec<_>, String> = samples
        .into_par_iter()
        .map(|sample| {
            compute_demod(
                &MtsParams {
                    hamiltonian: HamiltonianParams {
                        kv: sample.kv,
                        ..probe.hamiltonian
                    },
                    ..probe
                },
                Vec3::from_angles(FRAC_PI_2, FRAC_PI_2),
            )
            .map(|output| (sample.weight, output))
        })
        .collect();

    let mut outputs = outputs?.into_iter();
    let (weight, mut average) = outputs.next().expect("velocity sample count is positive");
    scale_demod_output(&mut average, weight);
    for (weight, output) in outputs {
        add_demod_output(&mut average, output, weight)?;
    }
    Ok(MtsCurves::from(average))
}

pub(super) fn plot(params: &mut Params) -> AppResult {
    let output = velocity_average(params)?;
    Ok(signal_plot(&params.mts, &output))
}
