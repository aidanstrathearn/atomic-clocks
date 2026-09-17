use std::f64::consts::FRAC_PI_2;

use atomic_clocks::twolevel::Vec3;
use atomic_clocks::vapourcell::{DemodOutput, Frame, HamiltonianParams, MtsParams, compute_demod};
use myplotlib::{AppResult, SliderGrid};
use rayon::prelude::*;

use super::{Params, velocity};
use crate::mts::{self, MtsCurves};

pub(super) fn controls(params: &mut Params) -> SliderGrid<'_> {
    let velocity = velocity::control_group(
        &mut params.kv_sigma,
        &mut params.kv_window_half_width,
        &mut params.kv_samples,
    );
    SliderGrid::new(
        5,
        mts::control_groups(&mut params.mts, "Gaussian mean (kv)")
            .into_iter()
            .chain([velocity]),
    )
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
        hamiltonian: HamiltonianParams {
            frame: Frame::Probe,
            ..params.mts.hamiltonian
        },
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
    Ok(mts::signal_plot(
        &params.mts,
        &output,
        format!(
            "Probe frame: Gaussian μ = {:.3}, σ = {:.3}, |kv| ≤ {:.3} ({} samples)",
            params.mts.hamiltonian.kv,
            params.kv_sigma,
            params.kv_window_half_width,
            params.kv_samples,
        ),
    ))
}
