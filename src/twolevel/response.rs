//! Hamiltonian kick response using real states and traceless variations.

use crate::maths::vec3::Vec3;

use super::channel::Channel;
use super::observable::Observable;
use super::process::{ChannelPair, Process};
use super::state::BlochVec;

/// Bloch vector of `-i[V, rho]`, where `V = a I + v.sigma`.
/// The scalar commutes with every state; the variation is `2 v cross r`.
fn kick_variation(perturbation: Observable, state: BlochVec) -> Vec3 {
    (perturbation.vector() * 2.0).cross(state.r)
}

impl<S: IntoIterator> Process<S>
where
    S::Item: Channel,
{
    /// Final observable response to a Hamiltonian kick at every boundary,
    /// including before the first step and after the last.
    ///
    /// `perturbation` is the Hermitian kick generator `V`. Results are
    /// derivatives at zero of the final expectation after inserting
    /// `rho -> exp(-i epsilon V) rho exp(i epsilon V)`. Both arguments use
    /// [`Observable`]'s convention `a I + v.sigma`; their scalar coefficients
    /// do not contribute to response. A parameter-dependent Hamiltonian uses
    /// `V = dH/dlambda` as its perturbation operator.
    ///
    /// Contracts the initial state forward and the observable backward, using
    /// O(N) work and storage without composing channels. With
    /// [`steps`](super::process::steps), responses correspond to all of `times`:
    /// N channels return N + 1 responses. Empty evolution returns the single
    /// immediate kick response of `initial`. These are
    /// kick responses; continuously applied perturbations require time integration.
    ///
    /// ```
    /// use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Observable, Process, steps};
    ///
    /// let times = [0.0, 0.1, 0.3];
    /// let drive = Hamiltonian::new(1.0, 0.0, 0.0);
    /// let channels: Vec<_> = steps(&times, |_t, dt| drive.for_duration(dt)).collect();
    /// let initial = BlochVec::ground();
    /// let observable = Observable::ground_projector();
    /// let kick = Observable::from(Hamiltonian::new(0.4, -0.7, 1.1));
    /// let direct = Process::new(&channels).linear_response(initial, observable, kick);
    /// assert_eq!(direct.len(), times.len());
    /// let pairs = Process::new(channels.iter().copied()).insertion_pairs();
    /// // Insertion pairs exclude the initial boundary.
    /// for (pair, expected) in pairs.iter().zip(&direct[1..]) {
    ///     assert!((pair.linear_response(initial, observable, kick) - expected).abs() < 1e-12);
    /// }
    /// ```
    pub fn linear_response(
        self,
        initial: BlochVec,
        observable: Observable,
        perturbation: Observable,
    ) -> Vec<f64> {
        let mut forward = initial;
        let trajectory: Vec<_> = self
            .channels
            .into_iter()
            .map(|step| {
                forward = step.apply_to(forward);
                (step, forward)
            })
            .collect();

        let mut responses = vec![0.0; trajectory.len() + 1];
        let mut measurement = observable.vector();
        for ((step, forward_state), response) in
            trajectory.into_iter().zip(&mut responses[1..]).rev()
        {
            *response = measurement.dot(kick_variation(perturbation, forward_state));
            measurement = step.pull_back_traceless_observable(measurement);
        }
        responses[0] = measurement.dot(kick_variation(perturbation, initial));
        responses
    }
}

impl<C: Channel> ChannelPair<C> {
    /// Contract a prepared insertion with endpoints and a Hamiltonian kick.
    /// Observable and kick conventions match [`Process::linear_response`].
    pub fn linear_response(
        &self,
        initial: BlochVec,
        observable: Observable,
        perturbation: Observable,
    ) -> f64 {
        let state = self.before.apply_to(initial);
        let variation = kick_variation(perturbation, state);
        observable.pair_traceless(self.after.apply_traceless(variation))
    }
}
