use std::f64::consts::PI;

use atomic_clocks::twolevel::{Decay, Vec3};
use atomic_clocks::vapourcell::{DemodOutput, Frame, HamiltonianParams, MtsParams, compute_demod};
use myplotlib::{AppDefinition, AppResult, Plotter, Slider, SliderGrid, SliderGroup, ViewOption};

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

fn controls(params: &mut MtsParams) -> SliderGrid<'_> {
    SliderGrid::new(5, control_groups(params, "Doppler shift (kv)"))
}

pub(super) fn control_groups<'a>(
    params: &'a mut MtsParams,
    kv_label: &'static str,
) -> [SliderGroup<'a>; 4] {
    let [modulation, atom, relaxation] =
        atom_control_groups(&mut params.hamiltonian, &mut params.decay, kv_label);
    [
        modulation,
        atom,
        relaxation,
        SliderGroup::new(
            "Scan and sampling",
            [
                Slider::new("Detuning half-range", &mut params.solver.hz_lim, 0.1..=40.0),
                Slider::new("Detuning samples", &mut params.solver.hz_num, 2..=500),
                Slider::new("Spatial phase samples", &mut params.solver.kr_n, 1..=21).step_by(2.0),
                Slider::new(
                    "Steps per period",
                    &mut params.solver.steps_per_period,
                    20..=1_000,
                ),
                Slider::new("Periods", &mut params.solver.n_periods, 1..=20),
            ],
        ),
    ]
}

pub(super) fn atom_control_groups<'a>(
    hamiltonian: &'a mut HamiltonianParams,
    decay: &'a mut Decay,
    kv_label: &'static str,
) -> [SliderGroup<'a>; 3] {
    [
        SliderGroup::new(
            "Modulation",
            [
                Slider::new(
                    "Frequency",
                    &mut hamiltonian.modulation.frequency,
                    0.1..=10.0,
                )
                .logarithmic(true),
                Slider::new("Depth", &mut hamiltonian.modulation.depth, 0.0..=10.0),
                Slider::new(
                    "Frequency shift",
                    &mut hamiltonian.modulation.shift,
                    -10.0..=10.0,
                ),
            ],
        ),
        SliderGroup::new(
            "Pump, probe and atom",
            [
                Slider::new("Pump Rabi frequency", &mut hamiltonian.r_pump, 0.0..=10.0),
                Slider::new("Probe Rabi frequency", &mut hamiltonian.r_prbe, 0.0..=2.0),
                Slider::new("Detuning offset", &mut hamiltonian.delta, -10.0..=10.0),
                Slider::new(kv_label, &mut hamiltonian.kv, -50.0..=50.0),
                Slider::new("Spatial phase (kr)", &mut hamiltonian.kr, -PI..=PI),
            ],
        ),
        SliderGroup::new(
            "Relaxation",
            [
                Slider::new("Excitation rate", &mut decay.gamma_up, 0.0..=10.0),
                Slider::new("Decay rate", &mut decay.gamma_down, 0.01..=10.0).logarithmic(true),
                Slider::new("Dephasing rate", &mut decay.gamma_phi, 0.0..=10.0),
            ],
        ),
    ]
}

fn plot(params: &MtsParams, frame: Frame) -> AppResult {
    let frame_params = MtsParams {
        hamiltonian: HamiltonianParams {
            frame,
            ..params.hamiltonian
        },
        ..*params
    };
    let output = MtsCurves::from(compute_demod(
        &frame_params,
        Vec3::from_angles(PI / 2.0, PI / 2.0),
    )?);

    Ok(signal_plot(
        params,
        &output,
        format!("MTS: {} frame", frame.as_str()),
    ))
}

pub(super) fn signal_plot(params: &MtsParams, output: &MtsCurves, title: String) -> Plotter {
    let mut plot = Plotter::new();
    let minus_amp0: Vec<_> = output.amp0.iter().map(|x| -x).collect();
    plot.plot(&output.hz, &minus_amp0).label("DC");
    plot.plot(&output.hz, &output.proj1).label("First harmonic");
    plot.plot(&output.hz, &output.proj2)
        .label("Second harmonic");
    plot.plot(&output.hz, &output.proj3).label("Third harmonic");
    plot.title(title);
    plot.xlabel("Detuning relative to offset (rad / time)");
    plot.ylabel("Demodulated signal");
    plot.xlim(-params.solver.hz_lim, params.solver.hz_lim);
    plot
}

fn pump_plot(params: &mut MtsParams) -> AppResult {
    plot(params, Frame::Pump)
}

fn atom_plot(params: &mut MtsParams) -> AppResult {
    plot(params, Frame::Atom)
}

fn probe_plot(params: &mut MtsParams) -> AppResult {
    plot(params, Frame::Probe)
}

pub(crate) fn definition() -> AppDefinition<MtsParams> {
    const VIEWS: &[ViewOption<MtsParams>] = &[
        ViewOption::new("Probe", probe_plot, controls),
        ViewOption::new("Pump", pump_plot, controls),
        ViewOption::new("Atom", atom_plot, controls),
    ];
    AppDefinition::new("MTS", "mts-canvas", VIEWS)
}
