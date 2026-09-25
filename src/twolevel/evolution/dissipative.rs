//! Liouvillian evolution with excitation, decay, and dephasing.

use crate::maths::Mat3;
use crate::maths::complex::Complex;

use crate::twolevel::{AffineChannel, BlochVec, Channel, Hamiltonian, Vec3};

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

/// Prepared evolution over one interval of a constant
/// [`DephasingLiouvillian`].
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
