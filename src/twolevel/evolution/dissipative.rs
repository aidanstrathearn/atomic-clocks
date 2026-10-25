//! Liouvillian evolution with excitation, decay, and dephasing.

use crate::maths::Mat3;
use crate::maths::complex::Complex;

use crate::twolevel::{AffineChannel, BlochVec, Channel, ComposableChannel, Hamiltonian, Vec3};

#[derive(Copy, Clone, Debug)]
pub struct Decay {
    pub gamma_up: f64,
    pub gamma_down: f64,
    pub gamma_phi: f64,
}

impl Decay {
    /// Checks that excitation, decay, and dephasing rates are finite and nonnegative.
    pub fn validate(self) -> Result<(), String> {
        if ![self.gamma_up, self.gamma_down, self.gamma_phi]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err("decay rates must be nonnegative and finite".to_string());
        }
        Ok(())
    }

    pub fn gamma1(self) -> f64 {
        self.gamma_up + self.gamma_down
    }

    pub fn gamma2(self) -> f64 {
        0.5 * self.gamma1() + self.gamma_phi
    }

    pub fn bloch_diagonal(self) -> Vec3 {
        Vec3 {
            x: -self.gamma2(),
            y: -self.gamma2(),
            z: -self.gamma1(),
        }
    }
}

/// Constant Hamiltonian, population transfer, and independent Pauli dephasing.
///
/// The Hamiltonian vector is in radians per time unit; all rates are in inverse
/// time units. Population transfer is along z: `gamma_down` relaxes toward the
/// ground state and `gamma_up` excites toward the excited state.
///
/// Each Pauli rate leaves that axis unchanged and damps both perpendicular
/// components at that rate. Thus `gamma_z` has the same convention as
/// [`Decay::gamma_phi`]; the Lindblad coefficient multiplying
/// `sigma_i rho sigma_i - rho` is `gamma_i / 2`.
///
/// In Bloch coordinates the generator is `dr/dt = h × r - D r + b`, where
/// `gamma1 = gamma_up + gamma_down` and
///
/// ```text
/// D = diag(gamma1/2 + gamma_y + gamma_z,
///          gamma1/2 + gamma_z + gamma_x,
///          gamma1   + gamma_x + gamma_y)
/// b = (0, 0, gamma_up - gamma_down)
/// ```
///
/// ```
/// use atomic_clocks::twolevel::{BlochLiouvillian, BlochVec, Channel, Hamiltonian};
///
/// let generator = BlochLiouvillian {
///     hamiltonian: Hamiltonian::new(1.0, 0.0, 0.3),
///     gamma_up: 0.0,
///     gamma_down: 0.1,
///     gamma_x: 0.02,
///     gamma_y: 0.0,
///     gamma_z: 0.05,
/// };
/// let channel = generator.for_duration(0.2);
/// let state = channel.apply_to(BlochVec::ground());
/// assert!((0.0..=1.0).contains(&state.excited_probability()));
/// ```
#[derive(Copy, Clone)]
pub struct BlochLiouvillian {
    /// Hamiltonian vector, including the drive and detuning.
    pub hamiltonian: Hamiltonian,
    /// Incoherent excitation rate from ground to excited.
    pub gamma_up: f64,
    /// Population decay rate from excited to ground.
    pub gamma_down: f64,
    /// Dephasing rate about the x axis.
    pub gamma_x: f64,
    /// Dephasing rate about the y axis.
    pub gamma_y: f64,
    /// Dephasing rate about the z axis, including any pure dephasing.
    pub gamma_z: f64,
}

impl BlochLiouvillian {
    /// Checks finite Hamiltonian components and finite, nonnegative rates.
    ///
    /// The combined Bloch damping rates must also be representable as finite
    /// values.
    pub fn validate(self) -> Result<(), String> {
        let h = self.hamiltonian.r;
        if ![h.x, h.y, h.z].into_iter().all(f64::is_finite) {
            return Err("Hamiltonian components must be finite".to_string());
        }
        if ![
            self.gamma_up,
            self.gamma_down,
            self.gamma_x,
            self.gamma_y,
            self.gamma_z,
        ]
        .into_iter()
        .all(|rate| rate.is_finite() && rate >= 0.0)
        {
            return Err(
                "Bloch decay and dephasing rates must be nonnegative and finite".to_string(),
            );
        }
        let d = self.damping();
        if ![d.x, d.y, d.z].into_iter().all(f64::is_finite) {
            return Err("combined Bloch damping rates must be finite".to_string());
        }
        Ok(())
    }

    /// Prepares the affine channel for evolution under this constant generator.
    ///
    /// Uses a Cayley-Hamilton polynomial when the eigenvalues and spectral
    /// projectors are well conditioned. Short source integrals use a vector
    /// series; longer ones use the same spectral projectors as the linear map.
    /// Falls back to the augmented matrix exponential for ill-conditioned cases.
    /// Neither path requires a steady-state inverse; zero rates, singular
    /// generators, and repeated eigenvalues are supported.
    ///
    /// # Panics
    ///
    /// Panics for invalid generator parameters or a nonfinite or negative
    /// duration. A zero duration returns the identity channel.
    pub fn for_duration(&self, dt: f64) -> AffineChannel {
        self.validate()
            .unwrap_or_else(|message| panic!("{message}"));
        assert!(
            dt.is_finite() && dt >= 0.0,
            "evolution duration must be nonnegative and finite"
        );
        if dt == 0.0 {
            return AffineChannel::identity();
        }

        let h = self.hamiltonian.r;
        let d = self.damping();
        if d.x == 0.0 && d.y == 0.0 && d.z == 0.0 && (h * dt).norm_square().is_normal() {
            return self.hamiltonian.for_duration(dt).to_affine();
        }
        let source_z = self.gamma_up - self.gamma_down;
        if let Some(channel) = cayley_affine_step(h, d, source_z, dt) {
            return channel;
        }
        let generator = Mat3::from_rows([[-d.x, -h.z, h.y], [h.z, -d.y, -h.x], [-h.y, h.x, -d.z]]);
        let source = Vec3 {
            x: 0.0,
            y: 0.0,
            z: source_z,
        };
        affine_generator_exp(generator, source, dt)
    }

    fn damping(self) -> Vec3 {
        let gamma1 = self.gamma_up + self.gamma_down;
        Vec3 {
            x: 0.5 * gamma1 + self.gamma_y + self.gamma_z,
            y: 0.5 * gamma1 + self.gamma_z + self.gamma_x,
            z: gamma1 + self.gamma_x + self.gamma_y,
        }
    }
}

/// Evaluate exp(A t) as a degree-two polynomial, with stable source integration.
/// Normalizing the generator keeps the cubic coefficients independent of units.
fn cayley_affine_step(h: Vec3, d: Vec3, source_z: f64, dt: f64) -> Option<AffineChannel> {
    let scale = [h.x.abs(), h.y.abs(), h.z.abs(), d.x, d.y, d.z]
        .into_iter()
        .fold(0.0_f64, f64::max);
    let time = scale * dt;
    if scale == 0.0 || !time.is_finite() {
        return None;
    }
    let h = Vec3 {
        x: h.x / scale,
        y: h.y / scale,
        z: h.z / scale,
    };
    let d = Vec3 {
        x: d.x / scale,
        y: d.y / scale,
        z: d.z / scale,
    };
    let symmetric = (
        d.x + d.y + d.z,
        d.x * d.y + d.x * d.z + d.y * d.z + h.norm_square(),
        d.x * d.y * d.z + d.x * h.x * h.x + d.y * h.y * h.y + d.z * h.z * h.z,
    );
    let (a1, a2, a3) = symmetric;
    let p = a2 - a1 * a1 / 3.0;
    let q = a3 - a1 * a2 / 3.0 + 2.0 * a1 * a1 * a1 / 27.0;
    let q_term = (q * 0.5).powi(2);
    let p_term = (p / 3.0).powi(3);
    if (q_term + p_term).abs() <= 64.0 * f64::EPSILON * (q_term.abs() + p_term.abs()) {
        return None;
    }
    let eigs = if a3 == 0.0 {
        // Factor out the exact zero root. This prevents a spurious decay or
        // growth rate from Cardano roundoff in singular generators.
        let root = Complex::new(a1 * a1 - 4.0 * a2, 0.0).sqrt();
        [
            Complex::new(0.0, 0.0),
            (Complex::new(-a1, 0.0) + root) * 0.5,
            (Complex::new(-a1, 0.0) - root) * 0.5,
        ]
    } else {
        cubic_roots(symmetric)
    };
    if !eigenvalues_are_well_separated(eigs) || eigs.iter().any(|value| value.re > 0.0) {
        return None;
    }

    let mut coefficients = [0.0; 3];
    let mut integral = [0.0; 3];
    let generator_norm = (d.x + h.y.abs() + h.z.abs())
        .max(d.y + h.z.abs() + h.x.abs())
        .max(d.z + h.x.abs() + h.y.abs());
    let use_source_series = generator_norm * time <= 0.5;
    for lambda in eigs {
        let xj = Complex::new(a1, 0.0) + lambda;
        let denominator = lambda * lambda * 3.0 + lambda * (2.0 * a1) + Complex::new(a2, 0.0);
        let inverse = Complex::new(1.0, 0.0) / denominator;
        let weights = [
            (Complex::new(a2, 0.0) + xj * lambda) * inverse,
            xj * inverse,
            inverse,
        ];
        // Separated roots alone do not bound projector cancellation in a
        // nonnormal generator. Limit amplification of roundoff before taking
        // exponentials, especially near defective critical damping.
        let magnitude = |value: Complex| value.re.abs() + value.im.abs();
        let amplification = magnitude(weights[0])
            + generator_norm * magnitude(weights[1])
            + generator_norm.powi(2) * magnitude(weights[2]);
        if amplification > 128.0 {
            return None;
        }
        let exponential = (lambda * time).exp();
        for i in 0..3 {
            coefficients[i] += (weights[i] * exponential).re;
        }
        if source_z != 0.0 && !use_source_series {
            let integrated = integrated_exponential(lambda, time, exponential);
            for i in 0..3 {
                integral[i] += (weights[i] * integrated).re;
            }
        }
    }
    if !coefficients.into_iter().chain(integral).all(f64::is_finite) {
        return None;
    }

    let generator = Mat3::from_rows([[-d.x, -h.z, h.y], [h.z, -d.y, -h.x], [-h.y, h.x, -d.z]]);
    let squared = generator * generator;
    let [c0, c1, c2] = coefficients;
    let linear = c0 * Mat3::identity() + c1 * generator + c2 * squared;
    let source = Vec3 {
        x: 0.0,
        y: 0.0,
        z: source_z / scale,
    };
    let shift = if source_z == 0.0 {
        source
    } else if use_source_series {
        integrated_source_series(generator, source, time)
    } else {
        let [q0, q1, q2] = integral;
        q0 * source + q1 * generator.apply_to_vec(source) + q2 * squared.apply_to_vec(source)
    };
    Some(AffineChannel::new(linear, shift))
}

/// For short intervals, integrating one vector costs less than preparing three
/// spectral integrals. The generator norm times time is at most 0.5 here.
fn integrated_source_series(generator: Mat3, source: Vec3, time: f64) -> Vec3 {
    let mut term = source * time;
    let mut sum = term;
    let norm = |v: Vec3| v.x.abs().max(v.y.abs()).max(v.z.abs());
    for order in 2..=32 {
        term = generator.apply_to_vec(term) * (time / order as f64);
        sum = sum + term;
        if norm(term) <= f64::EPSILON * norm(sum) {
            break;
        }
    }
    sum
}

/// Integral of exp(lambda s) from zero to time, including lambda = 0.
fn integrated_exponential(lambda: Complex, time: f64, exponential: Complex) -> Complex {
    let argument = lambda * time;
    if argument.re.abs() + argument.im.abs() < 0.1 {
        // time * phi_1(lambda * time). The series avoids cancellation in exp-1
        // and division by a zero or very small eigenvalue.
        let mut sum = Complex::new(1.0, 0.0);
        let mut term = sum;
        for order in 1..=16 {
            term = (term * argument) * (1.0 / (order + 1) as f64);
            sum = sum + term;
            if term.re.abs() + term.im.abs() <= f64::EPSILON {
                break;
            }
        }
        sum * time
    } else {
        (exponential - Complex::new(1.0, 0.0)) / lambda
    }
}

/// Exponentiate [A b; 0 0] using its 3x3 and vector blocks. For powers k >= 1,
/// the upper-right block is A^(k-1) b; the bottom row need not be stored.
fn affine_generator_exp(generator: Mat3, source: Vec3, dt: f64) -> AffineChannel {
    let largest = generator
        .rows()
        .iter()
        .flatten()
        .copied()
        .chain([source.x, source.y, source.z])
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if largest == 0.0 {
        return AffineChannel::identity();
    }

    // Compute the norm in log space and scale the duration before multiplying.
    // This avoids overflowing A * dt or the row sums for otherwise finite inputs.
    // Divide entries directly: 1 / largest can overflow for subnormal generators.
    let normalized = Mat3::from_rows(generator.rows().map(|row| row.map(|value| value / largest)));
    let normalized_source = Vec3 {
        x: source.x / largest,
        y: source.y / largest,
        z: source.z / largest,
    };
    let log_norm =
        largest.log2() + dt.log2() + affine_block_norm(normalized, normalized_source).log2();
    let squarings = (log_norm + 1.0).ceil().max(0.0) as u32;
    let mut scaled_dt = dt;
    let mut remaining = squarings;
    while remaining > 0 {
        let chunk = remaining.min(512);
        scaled_dt *= 2.0_f64.powi(-(chunk as i32));
        remaining -= chunk;
    }
    let scaled = generator * scaled_dt;
    let scaled_source = source * scaled_dt;

    let mut linear = Mat3::identity();
    let mut shift = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut term = Mat3::identity();
    for order in 1..=64 {
        let inverse_order = 1.0 / order as f64;
        let source_term = term.apply_to_vec(scaled_source) * inverse_order;
        term = (term * scaled) * inverse_order;
        linear = linear + term;
        shift = shift + source_term;
        if affine_block_norm(term, source_term)
            <= f64::EPSILON * affine_block_norm(linear, shift).max(1.0)
        {
            break;
        }
    }

    for _ in 0..squarings {
        shift = linear.apply_to_vec(shift) + shift;
        linear = linear * linear;
    }
    AffineChannel::new(linear, shift)
}

fn affine_block_norm(linear: Mat3, shift: Vec3) -> f64 {
    linear
        .rows()
        .iter()
        .zip([shift.x, shift.y, shift.z])
        .map(|(row, offset)| row.iter().map(|value| value.abs()).sum::<f64>() + offset.abs())
        .fold(0.0, f64::max)
}

/// Hamiltonian and excitation, decay, and pure-dephasing rates for a two-level system.
#[derive(Copy, Clone)]
pub struct Liouvillian {
    pub hamiltonian: Hamiltonian,
    pub decay: Decay,
}

impl Liouvillian {
    /// Prepare evolution under this constant generator for `dt`.
    /// The duration and decay rates must be finite and nonnegative.
    pub fn for_duration(&self, dt: f64) -> DissipativeStep {
        // TODO: Handle singular steady states (including zero decay) and
        // repeated/near-repeated eigenvalues; the current formula can fail there.
        DissipativeStep {
            liouvillian: *self,
            steady_state: self.steady_state(),
            coefficients: get_cayley_coeffs(*self, dt),
        }
    }

    /// Applies the homogeneous Bloch generator, excluding the population source.
    fn apply_linear(self, v: Vec3) -> Vec3 {
        self.decay.bloch_diagonal().circ(v) + self.hamiltonian.r.cross(v)
    }

    fn steady_state(self) -> BlochVec {
        let Liouvillian { hamiltonian, decay } = self;
        let h = hamiltonian.r;
        let gamma1 = decay.gamma1();
        let gamma2 = decay.gamma2();
        let numerator = decay.gamma_up - decay.gamma_down;
        let denominator = gamma1 * (gamma2 * gamma2 + h.z * h.z) + gamma2 * (h.x * h.x + h.y * h.y);

        let v0 = h.x * h.z + h.y * gamma2;
        let v1 = h.y * h.z - h.x * gamma2;
        let v2 = h.z * h.z + gamma2 * gamma2;

        let scale = numerator / denominator;
        BlochVec {
            r: Vec3 {
                x: scale * v0,
                y: scale * v1,
                z: scale * v2,
            },
        }
    }
}

/// Prepared evolution over one interval of a constant [`Liouvillian`].
/// Applies the analytical Bloch evolution without constructing a matrix.
/// Convert with [`Channel::to_affine`] to compose it with other channels.
#[derive(Copy, Clone)]
pub struct DissipativeStep {
    liouvillian: Liouvillian,
    steady_state: BlochVec,
    coefficients: (f64, f64, f64),
}

impl Channel for DissipativeStep {
    fn apply_to(&self, state: BlochVec) -> BlochVec {
        let w = state.r - self.steady_state.r;
        BlochVec {
            r: self.apply_traceless(w) + self.steady_state.r,
        }
    }

    fn apply_traceless(&self, r: Vec3) -> Vec3 {
        let m_r = self.liouvillian.apply_linear(r);
        let m2_r = self.liouvillian.apply_linear(m_r);
        let (c0, c1, c2) = self.coefficients;
        c0 * r + c1 * m_r + c2 * m2_r
    }

    fn pull_back_traceless_observable(&self, o: Vec3) -> Vec3 {
        // Transposing the Bloch generator reverses its Hamiltonian cross
        // product and preserves the diagonal decay terms.
        let transpose = |v: Vec3| {
            self.liouvillian.decay.bloch_diagonal().circ(v)
                - self.liouvillian.hamiltonian.r.cross(v)
        };
        let m_o = transpose(o);
        let m2_o = transpose(m_o);
        let (c0, c1, c2) = self.coefficients;
        c0 * o + c1 * m_o + c2 * m2_o
    }

    fn to_affine(&self) -> AffineChannel {
        let h = self.liouvillian.hamiltonian.r;
        let d = self.liouvillian.decay.bloch_diagonal();
        let generator = Mat3::from_rows([[d.x, -h.z, h.y], [h.z, d.y, -h.x], [-h.y, h.x, d.z]]);
        let (c0, c1, c2) = self.coefficients;
        let linear = c0 * Mat3::identity() + c1 * generator + c2 * (generator * generator);
        let shift = self.steady_state.r - linear.apply_to_vec(self.steady_state.r);
        AffineChannel::new(linear, shift)
    }
}

/// Hamiltonian and pure-dephasing rate for a two-level system without
/// population decay or excitation.
#[derive(Copy, Clone)]
pub struct DephasingLiouvillian {
    pub hamiltonian: Hamiltonian,
    pub gamma_phi: f64,
}

impl DephasingLiouvillian {
    /// Prepare evolution under this constant generator for `dt`.
    ///
    /// The duration and dephasing rate must be finite and nonnegative.
    pub fn for_duration(&self, dt: f64) -> DephasingStep {
        assert!(
            self.gamma_phi.is_finite() && self.gamma_phi >= 0.0,
            "pure-dephasing rate must be nonnegative and finite"
        );
        assert!(
            dt.is_finite() && dt >= 0.0,
            "evolution duration must be nonnegative and finite"
        );

        let h = self.hamiltonian.r;
        let gamma_phi = self.gamma_phi;
        let generator = Mat3::from_rows([
            [-gamma_phi, -h.z, h.y],
            [h.z, -gamma_phi, -h.x],
            [-h.y, h.x, 0.0],
        ]);
        let symmetric = (
            2.0 * gamma_phi,
            gamma_phi * gamma_phi + h.norm_square(),
            gamma_phi * (h.x * h.x + h.y * h.y),
        );
        let eigs = cubic_roots(symmetric);
        let coefficients = cayley_coeffs(symmetric, eigs, dt);

        let linear = if cayley_result_is_well_conditioned(eigs, coefficients) {
            let (c0, c1, c2) = coefficients;
            c0 * Mat3::identity() + c1 * generator + c2 * (generator * generator)
        } else {
            matrix_exp(generator * dt)
        };

        DephasingStep { linear }
    }
}

/// Hamiltonian and independent pure-dephasing rates about the three Pauli axes
/// for a two-level system without population decay or excitation.
///
/// A rate about one axis leaves that Bloch-vector component unchanged and
/// damps each perpendicular component at that rate.
#[derive(Copy, Clone)]
pub struct PauliDephasingLiouvillian {
    pub hamiltonian: Hamiltonian,
    pub gamma_x: f64,
    pub gamma_y: f64,
    pub gamma_z: f64,
}

impl PauliDephasingLiouvillian {
    /// Prepare evolution under this constant generator for `dt`.
    ///
    /// The duration and dephasing rates must be finite and nonnegative.
    pub fn for_duration(&self, dt: f64) -> DephasingStep {
        assert!(
            [self.gamma_x, self.gamma_y, self.gamma_z]
                .into_iter()
                .all(|rate| rate.is_finite() && rate >= 0.0),
            "pure-dephasing rates must be nonnegative and finite"
        );
        assert!(
            dt.is_finite() && dt >= 0.0,
            "evolution duration must be nonnegative and finite"
        );

        let h = self.hamiltonian.r;
        let damping_x = self.gamma_y + self.gamma_z;
        let damping_y = self.gamma_z + self.gamma_x;
        let damping_z = self.gamma_x + self.gamma_y;
        let generator = Mat3::from_rows([
            [-damping_x, -h.z, h.y],
            [h.z, -damping_y, -h.x],
            [-h.y, h.x, -damping_z],
        ]);
        let symmetric = (
            damping_x + damping_y + damping_z,
            damping_x * damping_y + damping_x * damping_z + damping_y * damping_z + h.norm_square(),
            damping_x * damping_y * damping_z
                + damping_x * h.x * h.x
                + damping_y * h.y * h.y
                + damping_z * h.z * h.z,
        );
        let eigs = cubic_roots(symmetric);
        let coefficients = cayley_coeffs(symmetric, eigs, dt);

        let linear = if cayley_result_is_well_conditioned(eigs, coefficients) {
            let (c0, c1, c2) = coefficients;
            c0 * Mat3::identity() + c1 * generator + c2 * (generator * generator)
        } else {
            matrix_exp(generator * dt)
        };

        DephasingStep { linear }
    }
}

/// Prepared evolution over one interval of a constant
/// [`DephasingLiouvillian`] or [`PauliDephasingLiouvillian`].
///
/// Unlike [`DissipativeStep`], this channel is unital: it has no population
/// source or affine shift.
#[derive(Copy, Clone)]
pub struct DephasingStep {
    linear: Mat3,
}

impl Channel for DephasingStep {
    fn apply_to(&self, state: BlochVec) -> BlochVec {
        BlochVec {
            r: self.linear.apply_to_vec(state.r),
        }
    }

    fn apply_traceless(&self, r: Vec3) -> Vec3 {
        self.linear.apply_to_vec(r)
    }

    fn pull_back_traceless_observable(&self, o: Vec3) -> Vec3 {
        self.linear.transpose().apply_to_vec(o)
    }

    fn to_affine(&self) -> AffineChannel {
        AffineChannel::new(
            self.linear,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
    }
}

fn cayley_result_is_well_conditioned(eigs: [Complex; 3], coefficients: (f64, f64, f64)) -> bool {
    if ![coefficients.0, coefficients.1, coefficients.2]
        .into_iter()
        .all(f64::is_finite)
    {
        return false;
    }

    eigenvalues_are_well_separated(eigs)
}

fn eigenvalues_are_well_separated(eigs: [Complex; 3]) -> bool {
    let scale = eigs
        .iter()
        .map(|value| value.re.hypot(value.im))
        .fold(1.0_f64, f64::max);
    let tolerance = 64.0 * f64::EPSILON.sqrt() * scale;
    [(0, 1), (0, 2), (1, 2)]
        .into_iter()
        .all(|(i, j)| (eigs[i].re - eigs[j].re).hypot(eigs[i].im - eigs[j].im) > tolerance)
}

/// Scaling-and-squaring Taylor fallback for repeated or nearly repeated
/// eigenvalues. This is isolated from the existing dissipative path.
fn matrix_exp(matrix: Mat3) -> Mat3 {
    let norm = matrix_infinity_norm(matrix);
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };
    let scaled = matrix * 2.0_f64.powi(-(squarings as i32));

    let mut sum = Mat3::identity();
    let mut term = Mat3::identity();
    for order in 1..=64 {
        term = (term * scaled) * (1.0 / order as f64);
        sum = sum + term;
        if matrix_infinity_norm(term) <= f64::EPSILON * matrix_infinity_norm(sum).max(1.0) {
            break;
        }
    }

    for _ in 0..squarings {
        sum = sum * sum;
    }
    sum
}

fn matrix_infinity_norm(matrix: Mat3) -> f64 {
    matrix
        .rows()
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn get_cayley_coeffs(liouvillian: Liouvillian, t: f64) -> (f64, f64, f64) {
    let sym = symmetric_polynomials(liouvillian);
    let eigs = cubic_roots(sym);
    cayley_coeffs(sym, eigs, t)
}

fn symmetric_polynomials(liouvillian: Liouvillian) -> (f64, f64, f64) {
    let Liouvillian { hamiltonian, decay } = liouvillian;

    let gamma1 = decay.gamma1();
    let gamma2 = decay.gamma2();
    let h = hamiltonian.r;
    let h2 = h.norm_square();
    let a1 = gamma1 + 2.0 * gamma2;
    let a2 = 2.0 * gamma1 * gamma2 + gamma2 * gamma2 + h2;
    let a3 = h.z * h.z * gamma1 + gamma2 * (h.x * h.x + h.y * h.y + gamma1 * gamma2);
    (a1, a2, a3)
}

fn cubic_roots((a1, a2, a3): (f64, f64, f64)) -> [Complex; 3] {
    let p = a2 - (a1 * a1) / 3.0;
    let q = a3 - a1 * a2 / 3.0 + 2.0 * (a1 * a1 * a1) / 27.0;
    let d = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);

    let sqrt_d = Complex::new(d, 0.0).sqrt();
    let u = cbrt_cardano(Complex::new(-0.5 * q, 0.0) + sqrt_d);
    let v = cbrt_cardano(Complex::new(-0.5 * q, 0.0) - sqrt_d);
    let u_plus_v = u + v;
    let u_minus_v = u - v;
    let sqrt3 = 3.0_f64.sqrt();

    let x1 = u_plus_v;
    let x2 = u_plus_v * -0.5 + u_minus_v * Complex::new(0.0, 0.5 * sqrt3);
    let x3 = u_plus_v * -0.5 - u_minus_v * Complex::new(0.0, 0.5 * sqrt3);
    let shift = Complex::new(a1 / 3.0, 0.0);

    [x1 - shift, x2 - shift, x3 - shift]
}

fn cayley_coeffs((a1, a2, _a3): (f64, f64, f64), eigvals: [Complex; 3], t: f64) -> (f64, f64, f64) {
    let mut c0 = Complex::new(0.0, 0.0);
    let mut c1 = Complex::new(0.0, 0.0);
    let mut c2 = Complex::new(0.0, 0.0);

    for lambda in eigvals {
        let xj = Complex::new(a1, 0.0) + lambda;
        let dj = lambda * lambda * 3.0 + lambda * (2.0 * a1) + Complex::new(a2, 0.0);
        let exp_term = (lambda * t).exp();
        c0 = c0 + ((Complex::new(a2, 0.0) + xj * lambda) / dj) * exp_term;
        c1 = c1 + (xj / dj) * exp_term;
        c2 = c2 + (Complex::new(1.0, 0.0) / dj) * exp_term;
    }

    (c0.re, c1.re, c2.re)
}

fn cbrt_cardano(value: Complex) -> Complex {
    if value.im == 0.0 {
        if value.re >= 0.0 {
            Complex::new(value.re.powf(1.0 / 3.0), 0.0)
        } else {
            Complex::new(-(-value.re).powf(1.0 / 3.0), 0.0)
        }
    } else {
        value.powf(1.0 / 3.0)
    }
}

#[cfg(test)]
mod bloch_cayley_tests {
    use super::*;

    fn reference(h: Vec3, d: Vec3, source_z: f64, dt: f64) -> AffineChannel {
        let generator = Mat3::from_rows([[-d.x, -h.z, h.y], [h.z, -d.y, -h.x], [-h.y, h.x, -d.z]]);
        affine_generator_exp(
            generator,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: source_z,
            },
            dt,
        )
    }

    fn assert_channel(actual: AffineChannel, expected: AffineChannel) {
        for (a, e) in actual
            .linear()
            .rows()
            .iter()
            .flatten()
            .zip(expected.linear().rows().iter().flatten())
        {
            assert!((a - e).abs() < 1e-11, "{a} != {e}");
        }
        let a = actual.shift();
        let e = expected.shift();
        for (a, e) in [a.x, a.y, a.z].into_iter().zip([e.x, e.y, e.z]) {
            assert!((a - e).abs() < 1e-11, "{a} != {e}");
        }
    }

    #[test]
    fn fast_path_matches_augmented_exponential_over_drives_rates_and_durations() {
        let mut accepted = 0;
        for hamiltonian in [
            Hamiltonian::new(0.4, -0.7, 1.1),
            Hamiltonian::new(1.0, 0.0, 0.0),
            Hamiltonian::new(0.0, 0.0, 1.1),
            Hamiltonian::default(),
        ] {
            for gamma_up in [0.0, 0.2] {
                for gamma_down in [0.0, 0.1, 1.0, 4.0] {
                    for gamma_x in [0.0, 0.35] {
                        let model = BlochLiouvillian {
                            hamiltonian,
                            gamma_up,
                            gamma_down,
                            gamma_x,
                            gamma_y: 0.2,
                            gamma_z: 0.3,
                        };
                        let h = hamiltonian.r;
                        let d = model.damping();
                        let source = gamma_up - gamma_down;
                        for dt in [1e-12, 1e-6, 0.025, 0.2, 1.7, 10.0, 100.0] {
                            if let Some(actual) = cayley_affine_step(h, d, source, dt) {
                                accepted += 1;
                                assert_channel(actual, reference(h, d, source, dt));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            accepted > 200,
            "fast path was not exercised: {accepted} cases"
        );
    }

    #[test]
    fn critical_and_nearly_critical_generators_use_the_fallback() {
        for perturbation in [0.0, -1e-10, 1e-10] {
            let model = BlochLiouvillian {
                hamiltonian: Hamiltonian::new(1.0, 0.0, 0.0),
                gamma_up: 0.0,
                gamma_down: 4.0 * (1.0 + perturbation),
                gamma_x: 0.0,
                gamma_y: 0.0,
                gamma_z: 0.0,
            };
            assert!(
                cayley_affine_step(
                    model.hamiltonian.r,
                    model.damping(),
                    -model.gamma_down,
                    0.025
                )
                .is_none()
            );
        }
    }

    #[test]
    fn singular_generator_preserves_its_exact_stationary_component() {
        let h = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.1,
        };
        let d = Vec3 {
            x: 0.7,
            y: 0.7,
            z: 0.0,
        };
        for dt in [1e-12, 0.025, 1.0, 1e6] {
            let channel = cayley_affine_step(h, d, 0.0, dt).expect("separated eigenvalues");
            let result = channel.apply_to(BlochVec::excited());
            assert!((result.r.z - 1.0).abs() < 1e-14);
            assert!(result.r.x.abs() < 1e-14 && result.r.y.abs() < 1e-14);
            if dt <= 1.0 {
                assert_channel(channel, reference(h, d, 0.0, dt));
            }
        }
    }

    #[test]
    fn integrated_exponential_has_a_stable_zero_eigenvalue_limit() {
        for rate in [0.0_f64, 1e-20, 1e-8, 0.5] {
            for time in [1e-12, 0.025, 10.0] {
                let lambda = Complex::new(-rate, 0.0);
                let actual = integrated_exponential(lambda, time, (lambda * time).exp());
                let expected = if rate == 0.0 {
                    time
                } else {
                    -(-rate * time).exp_m1() / rate
                };
                assert!((actual.re - expected).abs() <= 4.0 * f64::EPSILON * expected);
                assert_eq!(actual.im, 0.0);
            }
        }
    }
}
