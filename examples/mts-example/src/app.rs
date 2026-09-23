use std::f64::consts::TAU;

use atomic_clocks::vapourcell::{
    Frame, HamiltonianParams, Laser, LinearResponseSolverParams, ModelTimeScale, MtsExperiment,
    MtsParams, MtsSolverParams, Transition, VapourCell, VelocityParams,
};
use myplotlib::{AppDefinition, Slider, SliderGroup, ViewOption};

use crate::{gradients, probe, response, velocity};

pub(crate) struct Params {
    pub(super) experiment: MtsExperiment,
    pub(super) solver: MtsSolverParams,
    pub(super) velocity: VelocityParams,
    pub(super) gradient_epsilon: f64,
    pub(super) gradient_harmonics: usize,
    pub(super) response_time_fraction: f64,
    pub(super) response_delay_periods: usize,
}

impl Default for Params {
    fn default() -> Self {
        let time_scale = ModelTimeScale::MICROSECONDS;
        let transition = Transition {
            wavelength_nm: 556.0,
            // Reproduces the previous gamma_down = 1 rad / microsecond.
            linewidth_hz: 1.82e5 //1.0 / (TAU * time_scale.seconds_per_unit()),
        };
        let laser_for_rabi = |rabi_per_model_time: f64| {
            let mut laser = Laser {
                power_milliwatts: 1.0,
                waist_radius_mm: 1.0,
                wavelength_nm: transition.wavelength_nm,
            };
            let rabi_at_one_milliwatt =
                time_scale.angular_rate(transition.rabi_freq_radians_per_s(&laser));
            laser.power_milliwatts = (rabi_per_model_time / rabi_at_one_milliwatt).powi(2);
            laser
        };

        Self {
            experiment: MtsExperiment {
                time_scale,
                cell: VapourCell {
                    transition,
                    density_per_m3: 1.0e18,
                    length_m: 0.02,
                },
                pump: Laser {
                    power_milliwatts: 0.005,
                    waist_radius_mm: 1.0,
                    wavelength_nm: transition.wavelength_nm,
                },
                probe: Laser {
                    power_milliwatts: 0.002,
                    waist_radius_mm: 1.0,
                    wavelength_nm: transition.wavelength_nm,
                },
                probe_detuning: 0.0,
                modulation_frequency: 1.0 / TAU,
                modulation_depth: 1.0 / TAU,
                pump_probe_offset: 0.0,
                pure_dephasing_rate: 0.5,
                scan_half_range: 20.0 / TAU,
                scan_samples: 40,
            },
            solver: MtsSolverParams::default(),
            velocity: VelocityParams::default(),
            gradient_epsilon: 1e-3,
            gradient_harmonics: 10,
            response_time_fraction: 0.0,
            response_delay_periods: LinearResponseSolverParams::default().delay_periods,
        }
    }
}

impl Params {
    pub(super) fn mts_params(&self) -> Result<MtsParams, String> {
        self.experiment.to_mts_params(self.solver)
    }

    /// Gain from raw probe-frame sigma_y to normalized transmission change.
    pub(super) fn transmission_gain(&self) -> Result<f64, String> {
        Ok(-self
            .experiment
            .probe_readout()?
            .optical_depth_per_coherence())
    }

    /// Returns the physical velocity distribution in coordinates centred on
    /// the velocity class selected by the carrier shift.
    pub(super) fn velocity_params(&self, mts: &MtsParams) -> VelocityParams {
        VelocityParams {
            mean: mts.atom.hamiltonian.modulation.shift / 2.0,
            ..self.velocity
        }
    }

    /// Returns the probe-frame Hamiltonian in coordinates centred on the
    /// velocity class selected by the carrier shift.
    pub(super) fn probe_hamiltonian(&self, mts: &MtsParams, kv: f64) -> HamiltonianParams {
        let mut modulation = mts.atom.hamiltonian.modulation;
        modulation.shift = 0.0;
        HamiltonianParams {
            modulation,
            kv,
            frame: Frame::Probe,
            ..mts.atom.hamiltonian
        }
    }
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
        transition,
        density_per_m3,
        length_m,
    } = cell;
    let pump_power = &mut pump.power_milliwatts;
    let probe_power = &mut probe.power_milliwatts;
    let pump_waist = &mut pump.waist_radius_mm;
    let probe_waist = &mut probe.waist_radius_mm;
    let transition_wavelength = &mut transition.wavelength_nm;
    let pump_wavelength = &mut pump.wavelength_nm;
    let probe_wavelength = &mut probe.wavelength_nm;
    let transition_linewidth = &mut transition.linewidth_hz;

    let common_waist = Slider::from_get_set("Beam waist radius (mm)", 0.1..=5.0, move |waist| {
        if let Some(waist) = waist {
            *pump_waist = waist;
            *probe_waist = waist;
        }
        *pump_waist
    })
    .logarithmic(true);
    let common_wavelength = Slider::from_get_set(
        "Transition wavelength (nm)",
        400.0..=900.0,
        move |wavelength| {
            if let Some(wavelength) = wavelength {
                *transition_wavelength = wavelength;
                *pump_wavelength = wavelength;
                *probe_wavelength = wavelength;
            }
            *transition_wavelength
        },
    );
    let linewidth = Slider::from_get_set(
        "Natural linewidth (MHz)",
        1e-3..=10.0,
        move |linewidth_mhz| {
            if let Some(linewidth_mhz) = linewidth_mhz {
                *transition_linewidth = linewidth_mhz * 1e6;
            }
            *transition_linewidth / 1e6
        },
    )
    .logarithmic(true);
    let density = Slider::new("Density (m⁻³)", density_per_m3, 1.0e8..=1.0e20)
        .logarithmic(true)
        .custom_formatter(|value, _| format!("{value:.2e}"));
    let length = Slider::from_get_set("Cell length (mm)", 1.0..=1_000.0, move |length_mm| {
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
                Slider::new("Shift (MHz)", pump_probe_offset, -600.0 / TAU..=600.0 / TAU),
            ],
        ),
        SliderGroup::new(
            "Lasers",
            [
                Slider::new("Pump power (mW)", pump_power, 1e-6..=2.0).logarithmic(true),
                Slider::new("Probe power (mW)", probe_power, 1e-6..=2.0).logarithmic(true),
                common_waist,
            ],
        ),
        SliderGroup::new(
            "Transition and cell",
            [
                common_wavelength,
                linewidth,
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

pub(super) fn common_control_groups<'a>(
    pump: &'a mut Laser,
    probe: &'a mut Laser,
    cell: &'a mut VapourCell,
    probe_detuning: &'a mut f64,
    modulation_frequency: &'a mut f64,
    modulation_depth: &'a mut f64,
    pump_probe_offset: &'a mut f64,
    pure_dephasing_rate: &'a mut f64,
    solver_params: &'a mut MtsSolverParams,
    velocity_params: &'a mut VelocityParams,
) -> [SliderGroup<'a>; 5] {
    *probe_detuning = 0.0;

    let [modulation, lasers, transition_controls] = atom_control_groups(
        pump,
        probe,
        cell,
        modulation_frequency,
        modulation_depth,
        pump_probe_offset,
        pure_dephasing_rate,
    );
    let velocity = velocity::control_group(velocity_params);
    let MtsSolverParams {
        kr_n,
        steps_per_period,
        n_periods,
    } = solver_params;
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

    [modulation, lasers, transition_controls, velocity, solver]
}

pub(crate) fn definition() -> AppDefinition<Params> {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("DC signal", probe::dc_plot, probe::controls),
        ViewOption::new("Harmonics", probe::harmonic_plot, probe::controls),
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
