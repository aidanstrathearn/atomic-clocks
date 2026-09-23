use std::f64::consts::TAU;

use atomic_clocks::vapourcell::{
    Frame, HamiltonianParams, Laser, LinearResponseSolverParams, ModelTimeScale, MtsExperiment,
    MtsParams, MtsSolverParams, Transition, VapourCell, VelocityParams,
};
use myplotlib::{AppDefinition, ViewOption};

use crate::{controls, gradients, probe, response};

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
        Self {
            experiment: MtsExperiment {
                time_scale,
                cell: VapourCell {
                    transition,
                    density_per_m3: 1.0e18,
                    length_m: 0.02,
                },
                pump: Laser {
                    power_milliwatts: 0.02,
                    waist_radius_mm: 1.0,
                    wavelength_nm: transition.wavelength_nm,
                },
                probe: Laser {
                    power_milliwatts: 0.02,
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

pub(crate) fn definition() -> AppDefinition<Params> {
    const VIEWS: &[ViewOption<Params>] = &[
        ViewOption::new("Harmonics", probe::harmonic_plot, controls::signal),
        ViewOption::new("DC signal", probe::dc_plot, controls::signal),
        //ViewOption::new("Linear response", response::plot, controls::response),
        ViewOption::new(
            "Linear response",
            response::demodulated_plot,
            controls::demodulated_response,
        ),
        ViewOption::new(
            "Transfer function",
            response::frequency_plot,
            controls::demodulated_response,
        ),
        //ViewOption::new("Harmonic gradients", gradients::plot, controls::gradients),
        ViewOption::new(
            "Intermod. noise",
            gradients::adjacent_plot,
            controls::gradients,
        ),
    ];
    AppDefinition::new("MTS velocity", VIEWS)
}
