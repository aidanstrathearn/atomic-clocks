use atomic_clocks::maths::demodulation::ModulationParams;
use atomic_clocks::twolevel::Decay;
use atomic_clocks::vapourcell::{
    Frame, HamiltonianParams, LinearResponseSolverParams, MtsParams, VelocityParams,
};
use myplotlib::{AppDefinition, Slider, SliderGroup, ViewOption};

use crate::units::frequency_slider;
use crate::{gradients, probe, response, velocity};

pub(crate) struct Params {
    pub(super) mts: MtsParams,
    pub(super) velocity: VelocityParams,
    pub(super) gradient_epsilon: f64,
    pub(super) gradient_harmonics: usize,
    pub(super) response_time_fraction: f64,
    pub(super) response_delay_periods: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mts: MtsParams::default(),
            velocity: VelocityParams::default(),
            gradient_epsilon: 1e-3,
            gradient_harmonics: 10,
            response_time_fraction: 0.0,
            response_delay_periods: LinearResponseSolverParams::default().delay_periods,
        }
    }
}

impl Params {
    /// Returns the physical velocity distribution in coordinates centred on
    /// the velocity class selected by the carrier shift.
    pub(super) fn velocity_params(&self) -> VelocityParams {
        VelocityParams {
            mean: self.mts.hamiltonian.modulation.shift / 2.0,
            ..self.velocity
        }
    }

    /// Returns the probe-frame Hamiltonian in coordinates centred on the
    /// velocity class selected by the carrier shift.
    pub(super) fn probe_hamiltonian(&self, kv: f64) -> HamiltonianParams {
        let mut modulation = self.mts.hamiltonian.modulation;
        modulation.shift = 0.0;
        HamiltonianParams {
            modulation,
            kv,
            frame: Frame::Probe,
            ..self.mts.hamiltonian
        }
    }
}

fn atom_control_groups<'a>(
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

pub(super) fn common_control_groups<'a>(
    hamiltonian: &'a mut HamiltonianParams,
    decay: &'a mut Decay,
    kr_n: &'a mut usize,
    steps_per_period: &'a mut usize,
    n_periods: &'a mut usize,
    velocity_params: &'a mut VelocityParams,
) -> [SliderGroup<'a>; 4] {
    hamiltonian.delta = 0.0;
    hamiltonian.kv = 0.0;
    hamiltonian.kr = 0.0;
    decay.gamma_up = 0.0;

    let [modulation, rates] = atom_control_groups(
        &mut hamiltonian.modulation,
        &mut hamiltonian.r_pump,
        &mut hamiltonian.r_prbe,
        &mut decay.gamma_down,
        &mut decay.gamma_phi,
    );
    let velocity = velocity::control_group(velocity_params);
    let solver = SliderGroup::new(
        "Solver",
        [
            Slider::new("Spatial phase samples", kr_n, 1..=21).step_by(2.0),
            Slider::new("Steps per period", steps_per_period, 20..=1_000),
            Slider::from_get_set("Warmup periods", 0.0..=100.0, move |warmup| {
                if let Some(warmup) = warmup {
                    *n_periods = warmup.round() as usize + 1;
                }
                n_periods.saturating_sub(1) as f64
            })
            .step_by(1.0),
        ],
    );

    [modulation, rates, velocity, solver]
}

pub(crate) fn definition() -> AppDefinition<Params> {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Signal", probe::plot, probe::controls),
        //ViewOption::new("Linear response", response::plot, response::controls),
        ViewOption::new(
            "Linear response",
            response::demodulated_plot,
            response::demodulated_controls,
        ),
        ViewOption::new(
            "Transfer function",
            response::frequency_plot,
            response::demodulated_controls,
        ),
        //ViewOption::new("Harmonic gradients", gradients::plot, gradients::controls),
        ViewOption::new(
            "Intermod. noise",
            gradients::adjacent_plot,
            gradients::controls,
        ),
    ];
    AppDefinition::new("MTS velocity", VIEWS)
}
