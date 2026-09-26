use crate::maths::demodulation::Demodulation;

use super::experiment::MtsExperiment;
use super::linear_response::{LinearResponseOutput, demodulate_response};
use super::mts::{DemodOutput, RawSignalOutput};

/// A physical probe quantity derived from the raw probe-frame `sigma_y` signal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeQuantity {
    /// Signed thin-cell optical depth `F C`.
    OpticalDepth,
    /// Normalized thin-cell transmission `1 - F C`.
    Transmission,
}

/// Conversion from raw probe-frame `sigma_y` coherence to a physical readout.
///
/// The conversion assumes a uniform, undepleted probe, so transmission is the
/// thin-cell result `I_out / I_in = 1 - F C`. It applies to raw outputs computed
/// for `observable = (0, 1, 0)` in the probe frame using the same experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeReadout {
    optical_depth_per_coherence: f64,
}

/// A converted periodic probe signal.
#[derive(Clone, Debug)]
pub struct ProbeSignalOutput {
    pub quantity: ProbeQuantity,
    /// Times copied from the raw signal.
    pub times: Vec<f64>,
    /// Optical-depth or normalized-transmission samples.
    pub values: Vec<f64>,
}

/// Converted demodulated values for a detuning scan.
#[derive(Clone, Debug)]
pub struct ProbeDemodOutput<const N: usize> {
    pub quantity: ProbeQuantity,
    /// Detuning offsets copied from the raw output.
    pub hz: Vec<f64>,
    /// Harmonic indices corresponding to each position in `values`.
    pub harmonics: [usize; N],
    /// Converted lock-in results per detuning offset.
    pub values: Vec<[Demodulation; N]>,
}

/// A converted probe linear-response kernel.
#[derive(Clone, Debug)]
pub struct ProbeLinearResponseOutput {
    pub quantity: ProbeQuantity,
    /// Observation times copied from the raw response.
    pub times: Vec<f64>,
    /// Delays copied from the raw response.
    pub delays: Vec<f64>,
    /// Optical-depth or transmission response at each time and delay.
    pub response: Vec<Vec<f64>>,
}

impl ProbeLinearResponseOutput {
    /// Demodulates observation time at each fixed delay.
    ///
    /// The normalization, reference phase, and dimension checks match
    /// [`LinearResponseOutput::demodulate`].
    pub fn demodulate(&self, harmonic: usize) -> Vec<Demodulation> {
        demodulate_response(&self.times, &self.delays, &self.response, harmonic)
    }
}

impl MtsExperiment {
    /// Builds the thin-cell conversion for the experiment's probe and cell.
    ///
    /// The Rabi frequency is evaluated in radians per second, independently of
    /// the experiment's model-time scale. A positive probe intensity is needed
    /// because normalized transmission is undefined without incident light.
    pub fn probe_readout(&self) -> Result<ProbeReadout, String> {
        self.validate()?;
        let probe_intensity = self.probe.intensity_watts_per_m2();
        if probe_intensity <= 0.0 {
            return Err("probe intensity must be positive for normalized readout".to_string());
        }
        let probe_rabi = self.cell.transition.rabi_freq_radians_per_s(&self.probe);
        let optical_depth_per_coherence = 0.5
            * (probe_rabi / probe_intensity)
            * self.cell.density_per_m3
            * self.cell.length_m
            * self.probe.photon_energy_joules();
        if !optical_depth_per_coherence.is_finite() {
            return Err("probe readout factor must be finite".to_string());
        }
        Ok(ProbeReadout {
            optical_depth_per_coherence,
        })
    }
}

impl ProbeReadout {
    /// Multiplicative conversion from raw `sigma_y` coherence to optical depth.
    pub fn optical_depth_per_coherence(self) -> f64 {
        self.optical_depth_per_coherence
    }

    /// Converts one raw coherence sample to signed optical depth.
    pub fn optical_depth(self, coherence: f64) -> f64 {
        self.optical_depth_per_coherence * coherence
    }

    /// Converts one raw coherence sample to normalized transmission.
    pub fn transmission(self, coherence: f64) -> f64 {
        1.0 - self.optical_depth(coherence)
    }

    /// Converts a raw lock-in result to the requested probe quantity.
    ///
    /// Harmonic zero follows the library convention of returning twice the
    /// mean, so normalized transmission has a baseline of two. Positive
    /// harmonics have no contribution from the constant transmission baseline.
    pub fn convert_demodulation(
        self,
        raw: Demodulation,
        harmonic: usize,
        quantity: ProbeQuantity,
    ) -> Demodulation {
        let (offset, gain) = self.affine(quantity);
        Demodulation {
            in_phase: if harmonic == 0 { 2.0 * offset } else { 0.0 } + gain * raw.in_phase,
            quadrature: gain * raw.quadrature,
        }
    }

    /// Converts one raw linear-response value to the requested probe quantity.
    ///
    /// Only the affine gain contributes because a constant readout baseline has
    /// zero response.
    pub fn convert_response(self, raw: f64, quantity: ProbeQuantity) -> f64 {
        let (_, gain) = self.affine(quantity);
        gain * raw
    }

    /// Converts a complete raw periodic signal without modifying it.
    pub fn convert_signal(
        self,
        raw: &RawSignalOutput,
        quantity: ProbeQuantity,
    ) -> ProbeSignalOutput {
        let (offset, gain) = self.affine(quantity);
        ProbeSignalOutput {
            quantity,
            times: raw.times.clone(),
            values: raw
                .values
                .iter()
                .map(|&value| offset + gain * value)
                .collect(),
        }
    }

    /// Converts a complete raw demodulation output without modifying it.
    pub fn convert_demod_output<const N: usize>(
        self,
        raw: &DemodOutput<N>,
        quantity: ProbeQuantity,
    ) -> ProbeDemodOutput<N> {
        let values = raw
            .values
            .iter()
            .map(|values| {
                std::array::from_fn(|i| {
                    self.convert_demodulation(values[i], raw.harmonics[i], quantity)
                })
            })
            .collect();
        ProbeDemodOutput {
            quantity,
            hz: raw.hz.clone(),
            harmonics: raw.harmonics,
            values,
        }
    }

    /// Converts a complete raw linear-response output without modifying it.
    pub fn convert_linear_response(
        self,
        raw: &LinearResponseOutput,
        quantity: ProbeQuantity,
    ) -> ProbeLinearResponseOutput {
        ProbeLinearResponseOutput {
            quantity,
            times: raw.times.clone(),
            delays: raw.delays.clone(),
            response: raw
                .response
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| self.convert_response(value, quantity))
                        .collect()
                })
                .collect(),
        }
    }

    fn affine(self, quantity: ProbeQuantity) -> (f64, f64) {
        match quantity {
            ProbeQuantity::OpticalDepth => (0.0, self.optical_depth_per_coherence),
            ProbeQuantity::Transmission => (1.0, -self.optical_depth_per_coherence),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Laser, Transition};
    use crate::vapourcell::{ModelTimeScale, MtsSolverParams, VapourCell};

    fn experiment() -> MtsExperiment {
        MtsExperiment {
            time_scale: ModelTimeScale::MICROSECONDS,
            cell: VapourCell {
                transition: Transition {
                    wavelength_nm: 780.24,
                    linewidth_hz: 6.065e6,
                },
                density_per_m3: 2.5e16,
                length_m: 0.075,
            },
            pump: Laser {
                power_milliwatts: 2.0,
                waist_radius_mm: 0.8,
                wavelength_nm: 780.25,
                linewidth_fwhm_hz: 0.0,
            },
            probe: Laser {
                power_milliwatts: 0.2,
                waist_radius_mm: 0.6,
                wavelength_nm: 780.23,
                linewidth_fwhm_hz: 0.0,
            },
            probe_detuning: -1.2,
            modulation_frequency: 3.4,
            modulation_depth: 5.6,
            pump_probe_offset: 0.7,
            pure_dephasing_rate: 0.15,
            scan_half_range: 20.0,
            scan_samples: 81,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() < 1e-14 * scale,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn readout_uses_physical_probe_parameters_and_not_model_time() {
        let experiment = experiment();
        let intensity = experiment.probe.intensity_watts_per_m2();
        let expected = 0.5
            * experiment
                .cell
                .transition
                .rabi_freq_radians_per_s(&experiment.probe)
            / intensity
            * experiment.cell.density_per_m3
            * experiment.cell.length_m
            * experiment.probe.photon_energy_joules();
        let readout = experiment.probe_readout().unwrap();
        assert_close(readout.optical_depth_per_coherence(), expected);

        let mut rescaled = experiment;
        rescaled.time_scale = ModelTimeScale::NANOSECONDS;
        assert_eq!(
            rescaled
                .probe_readout()
                .unwrap()
                .optical_depth_per_coherence(),
            readout.optical_depth_per_coherence()
        );
    }

    #[test]
    fn normalized_readout_requires_incident_probe_light() {
        let mut experiment = experiment();
        experiment.probe.power_milliwatts = 0.0;
        assert_eq!(
            experiment.probe_readout().unwrap_err(),
            "probe intensity must be positive for normalized readout"
        );
        assert!(experiment.to_mts_params(MtsSolverParams::default()).is_ok());
    }

    #[test]
    fn affine_conversion_handles_samples_harmonics_and_response() {
        let readout = ProbeReadout {
            optical_depth_per_coherence: 0.25,
        };
        assert_eq!(readout.optical_depth(0.8), 0.2);
        assert_eq!(readout.transmission(0.8), 0.8);

        let raw = Demodulation {
            in_phase: 0.8,
            quadrature: -0.4,
        };
        let depth = readout.convert_demodulation(raw, 0, ProbeQuantity::OpticalDepth);
        assert_eq!(depth.in_phase, 0.2);
        assert_eq!(depth.quadrature, -0.1);
        let transmission = readout.convert_demodulation(raw, 0, ProbeQuantity::Transmission);
        assert_eq!(transmission.in_phase, 1.8);
        assert_eq!(transmission.quadrature, 0.1);
        let harmonic = readout.convert_demodulation(raw, 2, ProbeQuantity::Transmission);
        assert_eq!(harmonic.in_phase, -0.2);
        assert_eq!(harmonic.quadrature, 0.1);
        assert_eq!(
            readout.convert_response(0.8, ProbeQuantity::OpticalDepth),
            0.2
        );
        assert_eq!(
            readout.convert_response(0.8, ProbeQuantity::Transmission),
            -0.2
        );
    }

    #[test]
    fn complete_output_conversions_preserve_raw_data_and_axes() {
        let readout = ProbeReadout {
            optical_depth_per_coherence: 0.5,
        };
        let raw_signal = RawSignalOutput {
            times: vec![0.0, 1.0],
            values: vec![0.2, -0.4],
        };
        let signal = readout.convert_signal(&raw_signal, ProbeQuantity::Transmission);
        assert_eq!(signal.quantity, ProbeQuantity::Transmission);
        assert_eq!(signal.times, raw_signal.times);
        assert_eq!(signal.values, [0.9, 1.2]);
        assert_eq!(raw_signal.values, [0.2, -0.4]);

        let raw_demod = DemodOutput {
            hz: vec![-1.0, 1.0],
            harmonics: [0, 1],
            values: vec![[
                Demodulation {
                    in_phase: 0.4,
                    quadrature: 0.0,
                },
                Demodulation {
                    in_phase: -0.2,
                    quadrature: 0.6,
                },
            ]],
        };
        let demod = readout.convert_demod_output(&raw_demod, ProbeQuantity::Transmission);
        assert_eq!(demod.quantity, ProbeQuantity::Transmission);
        assert_eq!(demod.hz, raw_demod.hz);
        assert_eq!(demod.harmonics, raw_demod.harmonics);
        assert_eq!(demod.values[0][0].in_phase, 1.8);
        assert_eq!(demod.values[0][1].in_phase, 0.1);
        assert_eq!(demod.values[0][1].quadrature, -0.3);
        assert_eq!(raw_demod.values[0][0].in_phase, 0.4);

        let raw_response = LinearResponseOutput {
            times: vec![0.0],
            delays: vec![0.0, 1.0],
            response: vec![vec![0.2, -0.4]],
        };
        let response = readout.convert_linear_response(&raw_response, ProbeQuantity::Transmission);
        assert_eq!(response.quantity, ProbeQuantity::Transmission);
        assert_eq!(response.times, raw_response.times);
        assert_eq!(response.delays, raw_response.delays);
        assert_eq!(response.response, [vec![-0.1, 0.2]]);
        assert_eq!(raw_response.response, [vec![0.2, -0.4]]);
    }

    #[test]
    fn converted_response_retains_demodulation() {
        let readout = ProbeReadout {
            optical_depth_per_coherence: 0.25,
        };
        let times = vec![0.0, std::f64::consts::PI, std::f64::consts::TAU];
        let raw = LinearResponseOutput {
            times,
            delays: vec![0.0],
            response: vec![vec![1.0], vec![-1.0], vec![1.0]],
        };
        let raw_harmonic = raw.demodulate(1)[0];
        let measured = readout.convert_linear_response(&raw, ProbeQuantity::Transmission);
        let measured_harmonic = measured.demodulate(1)[0];
        assert_eq!(measured_harmonic.in_phase, -0.25 * raw_harmonic.in_phase);
        assert_eq!(
            measured_harmonic.quadrature,
            -0.25 * raw_harmonic.quadrature
        );
    }
}
