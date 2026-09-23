use std::f64::consts::{FRAC_PI_2, TAU};

use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{
    DemodOutput, DrivenAtomParams, HamiltonianParams, MtsParams, compute_demod,
};
use myplotlib::{AppResult, Plotter, Slider, SliderGrid, SliderGroup};
use rayon::prelude::*;

use crate::app::{Params, common_control_groups};
use crate::units::to_mhz;

pub(super) const DC_HARMONICS: [usize; 1] = [0];
pub(super) const MODULATION_HARMONICS: [usize; 3] = [1, 2, 3];
pub(super) type DcOutput = DemodOutput<{ DC_HARMONICS.len() }>;
pub(super) type HarmonicOutput = DemodOutput<{ MODULATION_HARMONICS.len() }>;

pub(super) struct DcCurve {
    pub hz: Vec<f64>,
    pub values: Vec<f64>,
}

impl DcCurve {
    /// Builds mean transmission minus one with `gain = -F`.
    pub(super) fn from_demodulated(output: DcOutput, gain: f64) -> Self {
        let DemodOutput {
            hz,
            harmonics,
            values,
        } = output;
        assert_eq!(harmonics, DC_HARMONICS);
        let values = values
            .into_iter()
            .map(|[raw_dc]| {
                // Harmonic zero is twice the mean so need factor of 0.5
                1.0 + 0.5 * gain * raw_dc.in_phase
            })
            .collect();
        Self { hz, values }
    }
}

pub(super) struct HarmonicCurves {
    pub hz: Vec<f64>,
    pub proj1: Vec<f64>,
    pub proj2: Vec<f64>,
    pub proj3: Vec<f64>,
}

impl HarmonicCurves {
    /// Builds normalized-transmission harmonics with `gain = -F`.
    pub(super) fn from_demodulated(output: HarmonicOutput, gain: f64) -> Self {
        let DemodOutput {
            hz,
            harmonics,
            values,
        } = output;
        assert_eq!(harmonics, MODULATION_HARMONICS);
        let mut proj1 = Vec::with_capacity(values.len());
        let mut proj2 = Vec::with_capacity(values.len());
        let mut proj3 = Vec::with_capacity(values.len());
        for [first, second, third] in values {
            proj1.push(gain * first.in_phase);
            proj2.push(gain * second.in_phase);
            proj3.push(gain * third.in_phase);
        }
        Self {
            hz,
            proj1,
            proj2,
            proj3,
        }
    }
}

fn scan_control_group<'a>(half_range: &'a mut f64, sample_count: &'a mut usize) -> SliderGroup<'a> {
    SliderGroup::new(
        "Scan",
        [
            Slider::new(
                "Detuning half-range (MHz)",
                half_range,
                0.1 / TAU..=105.0 / TAU,
            ),
            Slider::new("Detuning samples", sample_count, 2..=200),
        ],
    )
}

fn new_signal_plot(params: &MtsParams) -> Plotter {
    let mut plot = Plotter::new();
    plot.xlabel("Detuning (MHz)");
    plot.xlim(
        to_mhz(display_detuning(params, -params.scan.hz_lim)),
        to_mhz(display_detuning(params, params.scan.hz_lim)),
    );
    plot
}

fn dc_signal_plot(params: &MtsParams, output: &DcCurve) -> Plotter {
    let mut plot = new_signal_plot(params);
    let detunings_mhz: Vec<_> = output
        .hz
        .iter()
        .map(|&hz| to_mhz(display_detuning(params, hz)))
        .collect();
    plot.plot(&detunings_mhz, &output.values);
    plot.ylabel("Transmission");
    plot
}

fn harmonic_signal_plot(params: &MtsParams, output: &HarmonicCurves) -> Plotter {
    let mut plot = new_signal_plot(params);
    let detunings_mhz: Vec<_> = output
        .hz
        .iter()
        .map(|&hz| to_mhz(display_detuning(params, hz)))
        .collect();
    plot.plot(&detunings_mhz, &output.proj1)
        .label("First harmonic");
    plot.plot(&detunings_mhz, &output.proj2)
        .label("Second harmonic");
    plot.plot(&detunings_mhz, &output.proj3)
        .label("Third harmonic");
    plot.ylabel("Normalized transmission change");
    plot
}

pub(super) fn display_detuning(params: &MtsParams, detuning: f64) -> f64 {
    detuning + params.atom.hamiltonian.modulation.shift / 2.0
}

pub(super) fn controls(params: &mut Params) -> SliderGrid<'_> {
    let experiment = &mut params.experiment;
    let scan = scan_control_group(
        &mut experiment.scan_half_range,
        &mut experiment.scan_samples,
    );
    let common = common_control_groups(
        &mut experiment.pump,
        &mut experiment.probe,
        &mut experiment.cell,
        &mut experiment.probe_detuning,
        &mut experiment.modulation_frequency,
        &mut experiment.modulation_depth,
        &mut experiment.pump_probe_offset,
        &mut experiment.pure_dephasing_rate,
        &mut params.solver,
        &mut params.velocity,
    );
    SliderGrid::new(6, common.into_iter().chain([scan]))
}

fn scale_demod_output<const N: usize>(output: &mut DemodOutput<N>, weight: f64) {
    for values in &mut output.values {
        for value in values {
            value.in_phase *= weight;
            value.quadrature *= weight;
        }
    }
}

fn add_demod_output<const N: usize>(
    sum: &mut DemodOutput<N>,
    output: DemodOutput<N>,
    weight: f64,
) -> Result<(), String> {
    if sum.hz != output.hz {
        return Err("velocity signal grids do not match".to_string());
    }
    if sum.harmonics != output.harmonics || sum.values.len() != output.values.len() {
        return Err("velocity signal dimensions do not match".to_string());
    }
    for (sums, values) in sum.values.iter_mut().zip(output.values) {
        for (sum, value) in sums.iter_mut().zip(values) {
            sum.in_phase += weight * value.in_phase;
            sum.quadrature += weight * value.quadrature;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn velocity_average<const N: usize>(
    params: &Params,
    harmonics: [usize; N],
) -> Result<DemodOutput<N>, String> {
    let mts = params.mts_params()?;
    velocity_average_for_model(params, &mts, harmonics)
}

fn velocity_average_for_model<const N: usize>(
    params: &Params,
    mts: &MtsParams,
    harmonics: [usize; N],
) -> Result<DemodOutput<N>, String> {
    let samples = params.velocity_params(mts).samples()?;
    let probe = MtsParams {
        atom: DrivenAtomParams {
            hamiltonian: params.probe_hamiltonian(mts, 0.0),
            ..mts.atom
        },
        ..*mts
    };

    // Collect in sample order so summation is independent of Rayon scheduling.
    let outputs: Result<Vec<_>, String> = samples
        .into_par_iter()
        .map(|sample| {
            compute_demod(
                &MtsParams {
                    atom: DrivenAtomParams {
                        hamiltonian: HamiltonianParams {
                            kv: sample.kv,
                            ..probe.atom.hamiltonian
                        },
                        ..probe.atom
                    },
                    ..probe
                },
                Vec3::from_angles(FRAC_PI_2, FRAC_PI_2),
                harmonics,
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
    Ok(average)
}

pub(super) fn dc_plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let gain = params.transmission_gain()?;
    let output = DcCurve::from_demodulated(
        velocity_average_for_model(params, &mts, DC_HARMONICS)?,
        gain,
    );
    Ok(dc_signal_plot(&mts, &output))
}

pub(super) fn harmonic_plot(params: &mut Params) -> AppResult {
    let mts = params.mts_params()?;
    let gain = params.transmission_gain()?;
    let output = HarmonicCurves::from_demodulated(
        velocity_average_for_model(params, &mts, MODULATION_HARMONICS)?,
        gain,
    );
    Ok(harmonic_signal_plot(&mts, &output))
}
