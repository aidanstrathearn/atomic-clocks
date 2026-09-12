//! Sample interval channels, then apply them to a state or compose them.
//!
//! ```
//! use atomic_clocks::twolevel::{BlochVec, Hamiltonian, propagate, steps};
//!
//! let times = [0.0, 0.1, 0.3];
//! let drive = Hamiltonian::new(1.0, 0.0, 0.0);
//! let channels = steps(&times, |_t, dt| drive.for_duration(dt));
//! let trajectory: Vec<_> = propagate(channels, BlochVec::ground()).collect();
//! assert_eq!(trajectory.len(), times.len() - 1);
//! ```

use super::{BlochVec, Channel, ComposableChannel, Hamiltonian, Unitary};

/// Lazily construct one channel per interval using its absolute left endpoint
/// and duration. The final boundary is never sampled.
///
/// Boundaries must be finite and strictly increasing, with finite total
/// duration. A single boundary represents empty evolution. Invalid grids panic
/// immediately, before calling `step_at`, even if the iterator is never used.
pub fn steps<C: Channel>(
    times: &[f64],
    mut step_at: impl FnMut(f64, f64) -> C,
) -> impl ExactSizeIterator<Item = C> {
    assert!(!times.is_empty(), "at least one time boundary is required");
    assert!(
        times.iter().all(|t| t.is_finite()),
        "time boundaries must be finite"
    );
    assert!(
        times.windows(2).all(|pair| pair[1] > pair[0]),
        "time boundaries must be strictly increasing"
    );
    assert!(
        (times[times.len() - 1] - times[0]).is_finite(),
        "evolution duration must be finite"
    );
    times
        .windows(2)
        .map(move |pair| step_at(pair[0], pair[1] - pair[0]))
}

/// Lazily yield the state after each channel, excluding `initial`.
/// With channels from [`steps`], states correspond to `times[1..]`.
/// Collect this iterator to record a trajectory, or map it to observables.
pub fn propagate<C: Channel>(
    channels: impl IntoIterator<Item = C>,
    initial: BlochVec,
) -> impl Iterator<Item = BlochVec> {
    channels.into_iter().scan(initial, |state, channel| {
        *state = channel.apply_to(*state);
        Some(*state)
    })
}

/// Apply all channels without recording a trajectory. Empty evolution returns
/// `initial`. Each channel is applied directly; no composed map is constructed.
pub fn propagate_to_final<C: Channel>(
    channels: impl IntoIterator<Item = C>,
    initial: BlochVec,
) -> BlochVec {
    propagate(channels, initial).last().unwrap_or(initial)
}

/// Compose channels in iteration order, returning identity for an empty input.
/// Convert to affine channels first to compose different representations.
pub fn compose_channels<C: ComposableChannel>(channels: impl IntoIterator<Item = C>) -> C {
    channels
        .into_iter()
        .fold(C::identity(), |total, channel| channel.compose(&total))
}

/// Ground-state probability response to a Hamiltonian kick after each unitary
/// step. `perturbation` is the kick generator, and results are derivatives with
/// respect to its duration at zero. Records the forward history for the backward
/// measurement pass; empty evolution returns an empty vector.
pub fn linear_response(
    channels: impl IntoIterator<Item = Unitary>,
    initial: BlochVec,
    perturbation: Hamiltonian,
) -> Vec<f64> {
    let mut forward = initial;
    let trajectory: Vec<_> = channels
        .into_iter()
        .map(|step| {
            forward = step.apply_to(forward);
            (step, forward)
        })
        .collect();

    let mut responses = vec![0.0; trajectory.len()];
    // A pure state's Bloch vector also specifies its measurement projector.
    let mut measurement = BlochVec::ground();
    for ((step, forward_state), response) in trajectory.into_iter().zip(&mut responses).rev() {
        *response = 0.5 * measurement.r.dot(perturbation.r.cross(forward_state.r));
        measurement = step.inverse().apply_to(measurement);
    }
    responses
}
