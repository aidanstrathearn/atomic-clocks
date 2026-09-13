//! Contract a sequence of channels with states, with each other, or around
//! Hamiltonian insertions. A process represents a Markovian channel chain.
//!
//! ```
//! use atomic_clocks::twolevel::{BlochVec, Hamiltonian, Process, steps};
//!
//! let times = [0.0, 0.1, 0.3];
//! let drive = Hamiltonian::new(1.0, 0.0, 0.0);
//! let channels = steps(&times, |_t, dt| drive.for_duration(dt));
//! let trajectory: Vec<_> = Process::new(channels).propagate(BlochVec::ground()).collect();
//! assert_eq!(trajectory.len(), times.len() - 1);
//! ```

use super::channel::{Channel, ComposableChannel};
use super::state::BlochVec;

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

/// A chronological sequence of channels, independent of the state and
/// measurements with which it will be contracted.
///
/// The source can be a collection, a borrowed collection, or a lazy iterator.
/// Contractions consume the process. To reuse prepared channels, construct a
/// process borrowing them with `Process::new(&channels)`; composition and
/// insertion pairs require owned composable items, such as `channels.iter().copied()`.
pub struct Process<S> {
    pub(super) channels: S,
}

impl<S> Process<S> {
    /// Wrap a channel source without sampling or collecting it.
    pub fn new(channels: S) -> Self {
        Self { channels }
    }
}

impl<S: IntoIterator> Process<S>
where
    S::Item: Channel,
{
    /// Lazily yield the state after each channel, excluding `initial`.
    /// With channels from [`steps`], states correspond to `times[1..]`.
    /// Collect this iterator to record a trajectory, or map it to observables.
    pub fn propagate(self, initial: BlochVec) -> impl Iterator<Item = BlochVec> {
        self.channels.into_iter().scan(initial, |state, channel| {
            *state = channel.apply_to(*state);
            Some(*state)
        })
    }

    /// Apply all channels directly without constructing a composed map or
    /// recording a trajectory. Empty evolution returns `initial`.
    pub fn propagate_to_final(self, initial: BlochVec) -> BlochVec {
        self.propagate(initial).last().unwrap_or(initial)
    }
}

impl<S: IntoIterator> Process<S>
where
    S::Item: ComposableChannel,
{
    /// Compose channels in iteration order, returning identity for an empty
    /// input. Convert steps to affine channels before constructing the process
    /// to compose representations that are not themselves closed under composition.
    pub fn compose(self) -> S::Item {
        self.channels
            .into_iter()
            .fold(S::Item::identity(), |total, channel| {
                channel.compose(&total)
            })
    }

    /// Contract the channels before and after each insertion, independently of
    /// any initial state, observable, or perturbation. Uses O(N) compositions
    /// and storage, without inverses or a requirement that channels be cloneable.
    ///
    /// Pair `i` places the kick after step `i`: `before` includes that step,
    /// while `after` excludes it. The last pair has identity for `after`.
    /// With [`steps`], pairs correspond to `times[1..]`. Empty evolution
    /// returns an empty vector. Pair responses therefore correspond to
    /// `Process::linear_response(...)[1..]`, excluding the initial kick.
    /// Convert steps to affine channels first when
    /// their representation is not closed under composition.
    pub fn insertion_pairs(self) -> Vec<ChannelPair<S::Item>> {
        let mut before = S::Item::identity();
        let mut prepared: Vec<_> = self
            .channels
            .into_iter()
            .map(|step| {
                let next = step.compose(&before);
                let prefix = std::mem::replace(&mut before, next);
                (step, prefix)
            })
            .collect();

        let mut after = S::Item::identity();
        let mut pairs = Vec::with_capacity(prepared.len());
        // `before` is the complete prefix. Each saved prefix becomes the
        // `before` for the next iteration of this backward pass.
        while let Some((step, prefix)) = prepared.pop() {
            let earlier_after = after.compose(&step);
            pairs.push(ChannelPair { before, after });
            before = prefix;
            after = earlier_after;
        }
        pairs.reverse();
        pairs
    }
}

/// Evolution on either side of one open Hamiltonian insertion.
#[derive(Clone, Copy)]
pub struct ChannelPair<C> {
    pub before: C,
    pub after: C,
}
