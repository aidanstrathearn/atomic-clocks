use crate::twolevel::{
    BlochVec, Channel, DephasingLiouvillian, DephasingStep, Hamiltonian, PauliDephasingLiouvillian,
    Unitary,
};

/// Resolved parameters for driving one two-level system.
#[derive(Clone, Copy)]
pub struct PulseParameters {
    pub rabi_angular_frequency: f64,
    pub detuning_angular_frequency: f64,
    pub gamma_x: f64,
    pub gamma_phi: f64,
}

/// Evolution for a pulse whose duration and resolved drive parameters are fixed.
#[derive(Clone, Copy)]
#[must_use]
pub enum PreparedPulse {
    Unitary(Unitary),
    Dephasing(DephasingStep),
}

impl PreparedPulse {
    #[inline]
    #[must_use]
    pub fn apply_to(&self, state: BlochVec) -> BlochVec {
        match self {
            Self::Unitary(channel) => channel.apply_to(state),
            Self::Dephasing(channel) => channel.apply_to(state),
        }
    }

    #[inline]
    #[must_use]
    pub fn apply_to_with_phase(&self, state: BlochVec, phase_rad: f64) -> BlochVec {
        if phase_rad == 0.0 {
            return self.apply_to(state);
        }

        self.apply_to(state.rotate_about_z(-phase_rad))
            .rotate_about_z(phase_rad)
    }
}

pub fn prepare_pulse(parameters: PulseParameters, duration_s: f64) -> PreparedPulse {
    let hamiltonian = Hamiltonian::new(
        parameters.rabi_angular_frequency,
        0.0,
        parameters.detuning_angular_frequency,
    );
    if parameters.gamma_x == 0.0 && parameters.gamma_phi == 0.0 {
        return PreparedPulse::Unitary(hamiltonian.for_duration(duration_s));
    }

    if parameters.gamma_x != 0.0 {
        return PreparedPulse::Dephasing(
            PauliDephasingLiouvillian {
                hamiltonian,
                gamma_x: parameters.gamma_x,
                gamma_y: 0.0,
                gamma_z: parameters.gamma_phi,
            }
            .for_duration(duration_s),
        );
    }

    PreparedPulse::Dephasing(
        DephasingLiouvillian {
            hamiltonian,
            gamma_phi: parameters.gamma_phi,
        }
        .for_duration(duration_s),
    )
}

pub(super) fn apply_dark_time(state: BlochVec, phase: f64, coherence: f64) -> BlochVec {
    state.rotate_about_z(phase).dephase(coherence)
}

/// A unital sequence acting on one resolved two-level system.
pub trait ResolvedPulseSequence {
    fn apply_to(&self, parameters: PulseParameters, initial: BlochVec) -> BlochVec;

    /// Returns `M_zz` for the complete sequence.
    fn z_response(&self, parameters: PulseParameters) -> f64 {
        self.apply_to(parameters, BlochVec::excited()).r.z
    }

    /// Returns the final ground-state probability for an initial ground state.
    fn ground_probability(&self, parameters: PulseParameters) -> f64 {
        0.5 * (1.0 + self.z_response(parameters))
    }
}

impl<T> ResolvedPulseSequence for &T
where
    T: ResolvedPulseSequence + ?Sized,
{
    fn apply_to(&self, parameters: PulseParameters, initial: BlochVec) -> BlochVec {
        (*self).apply_to(parameters, initial)
    }
}
