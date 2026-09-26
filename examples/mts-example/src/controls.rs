use std::f64::consts::TAU;

use atomic_clocks::common::Laser;
use atomic_clocks::vapourcell::{MtsExperiment, MtsSolverParams, VapourCell};
use myplotlib::{Slider, SliderGrid, SliderGroup};

use crate::app::{Params, VelocityIntegrationParams};
use crate::units::frequency_slider;

struct CommonControls<'a> {
    pump: &'a mut Laser,
    probe: &'a mut Laser,
    cell: &'a mut VapourCell,
    probe_detuning: &'a mut f64,
    modulation_frequency: &'a mut f64,
    modulation_depth: &'a mut f64,
    pump_probe_offset: &'a mut f64,
    pure_dephasing_rate: &'a mut f64,
    solver: &'a mut MtsSolverParams,
    velocity: &'a mut VelocityIntegrationParams,
}

fn atom_control_groups<'a>(
    pump: &'a mut Laser,
    probe: &'a mut Laser,
    cell: &'a mut VapourCell,
    modulation_frequency: &'a mut f64,
    modulation_depth: &'a mut f64,
    pump_probe_offset: &'a mut f64,
    pure_dephasing_rate: &'a mut f64,
) -> [SliderGroup<'a>; 3] {
    let VapourCell {
        density_per_m3,
        length_m,
        ..
    } = cell;
    let pump_power = &mut pump.power_milliwatts;
    let probe_power = &mut probe.power_milliwatts;
    let pump_waist = &mut pump.waist_radius_mm;
    let probe_waist = &mut probe.waist_radius_mm;
    let common_waist = Slider::from_get_set("Beam waist radius (mm)", 0.1..=5.0, move |waist| {
        if let Some(waist) = waist {
            *pump_waist = waist;
            *probe_waist = waist;
        }
        *pump_waist
    })
    .logarithmic(true);
    let density = Slider::new("Density (m⁻³)", density_per_m3, 1.0e17..=1.0e20)
        .logarithmic(true)
        .custom_formatter(|value, _| format!("{value:.2e}"));
    let length = Slider::from_get_set("Cell length (mm)", 1.0..=100.0, move |length_mm| {
        if let Some(length_mm) = length_mm {
            *length_m = length_mm * 1.0e-3;
        }
        *length_m * 1.0e3
    })
    .logarithmic(true);

    [
        SliderGroup::new(
            "Modulation",
            [
                Slider::new(
                    "Frequency (MHz)",
                    modulation_frequency,
                    0.1 / TAU..=20.0 / TAU,
                ),
                Slider::new("Depth (MHz)", modulation_depth, 0.0..=20.0 / TAU),
                Slider::new("Shift (MHz)", pump_probe_offset, 0.0 / TAU..=600.0 / TAU),
            ],
        ),
        SliderGroup::new(
            "Lasers",
            [
                Slider::new("Pump power (mW)", pump_power, 1e-6..=1.0).logarithmic(true),
                Slider::new("Probe power (mW)", probe_power, 1e-6..=1.0).logarithmic(true),
                common_waist,
            ],
        ),
        SliderGroup::new(
            "Transition and cell",
            [
                Slider::new(
                    "Pure dephasing (MHz)",
                    pure_dephasing_rate,
                    0.0..=10.0 / TAU,
                ),
                density,
                length,
            ],
        ),
    ]
}

fn velocity_control_group(params: &mut VelocityIntegrationParams) -> SliderGroup<'_> {
    SliderGroup::new(
        "Velocity distribution",
        [
            frequency_slider("Sigma", &mut params.sigma, 0.0..=3000.0).logarithmic(true),
            frequency_slider("Integration window", &mut params.half_width, 0.1..=200.0),
            frequency_slider(
                "Maximum velocity step",
                &mut params.max_step,
                0.05 * TAU..=5.0 * TAU,
            )
            .logarithmic(true),
        ],
    )
}

fn common_control_groups(params: CommonControls<'_>) -> [SliderGroup<'_>; 5] {
    let CommonControls {
        pump,
        probe,
        cell,
        probe_detuning,
        modulation_frequency,
        modulation_depth,
        pump_probe_offset,
        pure_dephasing_rate,
        solver,
        velocity,
    } = params;
    *probe_detuning = 0.0;

    let [modulation, lasers, transition] = atom_control_groups(
        pump,
        probe,
        cell,
        modulation_frequency,
        modulation_depth,
        pump_probe_offset,
        pure_dephasing_rate,
    );
    let velocity = velocity_control_group(velocity);
    let MtsSolverParams {
        kr_n,
        steps_per_period,
        n_periods,
    } = solver;
    let solver = SliderGroup::new(
        "Solver",
        [
            Slider::new("Spatial phase samples", kr_n, 1..=21).step_by(2.0),
            Slider::new("Steps per period", steps_per_period, 20..=500),
            Slider::from_get_set("Warmup periods", 0.0..=10.0, move |warmup| {
                if let Some(warmup) = warmup {
                    *n_periods = warmup.round() as usize + 1;
                }
                n_periods.saturating_sub(1) as f64
            })
            .step_by(1.0),
        ],
    );

    [modulation, lasers, transition, velocity, solver]
}

fn common_controls<'a>(
    experiment: &'a mut MtsExperiment,
    solver: &'a mut MtsSolverParams,
    velocity: &'a mut VelocityIntegrationParams,
) -> CommonControls<'a> {
    CommonControls {
        pump: &mut experiment.pump,
        probe: &mut experiment.probe,
        cell: &mut experiment.cell,
        probe_detuning: &mut experiment.probe_detuning,
        modulation_frequency: &mut experiment.modulation_frequency,
        modulation_depth: &mut experiment.modulation_depth,
        pump_probe_offset: &mut experiment.pump_probe_offset,
        pure_dephasing_rate: &mut experiment.pure_dephasing_rate,
        solver,
        velocity,
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

pub(super) fn signal(params: &mut Params) -> SliderGrid<'_> {
    let experiment = &mut params.experiment;
    let scan = scan_control_group(
        &mut experiment.scan_half_range,
        &mut experiment.scan_samples,
    );
    let common = common_control_groups(CommonControls {
        pump: &mut experiment.pump,
        probe: &mut experiment.probe,
        cell: &mut experiment.cell,
        probe_detuning: &mut experiment.probe_detuning,
        modulation_frequency: &mut experiment.modulation_frequency,
        modulation_depth: &mut experiment.modulation_depth,
        pump_probe_offset: &mut experiment.pump_probe_offset,
        pure_dephasing_rate: &mut experiment.pure_dephasing_rate,
        solver: &mut params.solver,
        velocity: &mut params.velocity,
    });
    SliderGrid::new(6, common.into_iter().chain([scan]))
}

pub(super) fn response(params: &mut Params) -> SliderGrid<'_> {
    response_controls(params, true)
}

pub(super) fn demodulated_response(params: &mut Params) -> SliderGrid<'_> {
    response_controls(params, false)
}

fn response_controls(params: &mut Params, show_time: bool) -> SliderGrid<'_> {
    let time_step = 1.0 / params.solver.steps_per_period as f64;
    let time_slider = show_time.then(|| {
        Slider::new(
            "Observation time (t / T)",
            &mut params.response_time_fraction,
            0.0..=1.0,
        )
        .step_by(time_step)
    });
    let response = SliderGroup::new(
        "Linear response",
        time_slider.into_iter().chain([Slider::new(
            "Maximum delay (periods)",
            &mut params.response_delay_periods,
            1..=20,
        )]),
    );
    let common = common_control_groups(common_controls(
        &mut params.experiment,
        &mut params.solver,
        &mut params.velocity,
    ));
    SliderGrid::new(6, common.into_iter().chain([response]))
}

pub(super) fn gradients(params: &mut Params) -> SliderGrid<'_> {
    let gradient = SliderGroup::new(
        "Harmonic gradient",
        [
            frequency_slider(
                "Finite-difference epsilon",
                &mut params.gradient_epsilon,
                1e-6..=0.1,
            )
            .logarithmic(true),
            Slider::new("Maximum harmonic", &mut params.gradient_harmonics, 2..=50),
        ],
    );
    let common = common_control_groups(common_controls(
        &mut params.experiment,
        &mut params.solver,
        &mut params.velocity,
    ));
    SliderGrid::new(6, common.into_iter().chain([gradient]))
}
