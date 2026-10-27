#[derive(Clone, Copy)]
pub enum BeamDirection {
    Forward,
    Backward,
}

/// Resolved detunings for the momentum branches addressed by counterpropagating
/// travelling waves.
#[derive(Clone, Copy, Debug)]
pub struct MomentumBranchDetunings {
    pub forward_hz: f64,
    pub backward_lower_hz: f64,
    pub backward_upper_hz: f64,
}

impl MomentumBranchDetunings {
    pub fn from_carrier_doppler_recoil(carrier_hz: f64, doppler_hz: f64, recoil_hz: f64) -> Self {
        Self {
            forward_hz: carrier_hz - doppler_hz - recoil_hz,
            backward_lower_hz: carrier_hz + doppler_hz - recoil_hz,
            backward_upper_hz: carrier_hz + doppler_hz + 3.0 * recoil_hz,
        }
    }
}
