use crate::maths::mat3::Mat3;

use super::{BlochVec, Vec3};

/// A deterministic, completely positive, trace-preserving qubit map.
///
/// Implementations may use any representation. Applying `to_affine()` must
/// agree with `apply_to()` on every Bloch state, including mixed states.
pub trait Channel {
    fn apply_to(&self, state: BlochVec) -> BlochVec;
    fn to_affine(&self) -> AffineChannel;

    /// Evolve `X = r.sigma / 2`, a traceless Hermitian operator. Unlike a
    /// trace-one state, its vector receives no affine shift.
    fn apply_traceless(&self, r: Vec3) -> Vec3 {
        self.to_affine().linear().apply_to_vec(r)
    }

    /// Pull back the vector coefficient of `O = a I + o.sigma` for pairing
    /// with traceless operators: `o -> M^T o`. The scalar coefficient of the
    /// full adjoint observable can change, but does not enter this pairing.
    fn pull_back_traceless_observable(&self, o: Vec3) -> Vec3 {
        self.to_affine().linear().transpose().apply_to_vec(o)
    }
}

/// A channel representation closed under composition.
pub trait ComposableChannel: Channel + Sized {
    fn identity() -> Self;

    /// Apply `earlier` first, then `self`.
    fn compose(&self, earlier: &Self) -> Self;
}

// Borrowing a prepared channel should not require copying its representation.
impl<C: Channel + ?Sized> Channel for &C {
    fn apply_to(&self, state: BlochVec) -> BlochVec {
        (**self).apply_to(state)
    }

    fn to_affine(&self) -> AffineChannel {
        (**self).to_affine()
    }

    fn apply_traceless(&self, r: Vec3) -> Vec3 {
        (**self).apply_traceless(r)
    }

    fn pull_back_traceless_observable(&self, o: Vec3) -> Vec3 {
        (**self).pull_back_traceless_observable(o)
    }
}

/// A qubit channel acting on column Bloch vectors as `r -> linear * r + shift`.
#[derive(Clone, Copy)]
pub struct AffineChannel {
    linear: Mat3,
    shift: Vec3,
}

impl AffineChannel {
    /// Construct a channel from its Bloch coefficients.
    ///
    /// The caller must supply coefficients describing a completely positive,
    /// trace-preserving map. This constructor does not check physical validity;
    /// mapping the Bloch ball into itself alone is not sufficient.
    pub fn new(linear: Mat3, shift: Vec3) -> Self {
        Self { linear, shift }
    }

    pub fn linear(&self) -> Mat3 {
        self.linear
    }

    pub fn shift(&self) -> Vec3 {
        self.shift
    }
}

impl Channel for AffineChannel {
    fn apply_to(&self, state: BlochVec) -> BlochVec {
        BlochVec {
            r: self.linear.apply_to_vec(state.r) + self.shift,
        }
    }

    fn to_affine(&self) -> Self {
        *self
    }
}

impl ComposableChannel for AffineChannel {
    fn identity() -> Self {
        Self::new(
            Mat3::identity(),
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
    }

    fn compose(&self, earlier: &Self) -> Self {
        Self::new(
            self.linear * earlier.linear,
            self.linear.apply_to_vec(earlier.shift) + self.shift,
        )
    }
}
