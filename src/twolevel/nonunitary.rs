use crate::maths::complex::Complex;

use super::{BlochVec, Hamiltonian, Vec3};

#[derive(Copy, Clone)]
pub struct Decay {
    pub gamma_up: f64,
    pub gamma_down: f64,
    pub gamma_phi: f64,
}

impl Decay {
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

fn r_ss(liouvillian: Liouvillian) -> BlochVec {
    let Liouvillian {
        hamiltonian,
        decay,
    } = liouvillian;
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

fn symmetric_polynomials(liouvillian: Liouvillian) -> (f64, f64, f64) {
    let Liouvillian {
        hamiltonian,
        decay,
    } = liouvillian;
    // let Vec3 {
    //     x: hx,
    //     y: hy,
    //     z: hz,
    // } = hamiltonian.r;
    let gamma1 = decay.gamma1();
    let gamma2 = decay.gamma2();
    //let h2 = hx * hx + hy * hy + hz * hz;
    let h = hamiltonian.r;
    let h2 = h.norm_square();
    let a1 = gamma1 + 2.0 * gamma2;
    let a2 = 2.0 * gamma1 * gamma2 + gamma2 * gamma2 + h2;
    let a3 = h.z * h.z * gamma1 + gamma2 * (h.x * h.x + h.y * h.y + gamma1 * gamma2);
    (a1, a2, a3)
}

pub(crate) fn propagate(liouvillian: Liouvillian, r0: BlochVec, t: f64) -> BlochVec {
    let rss = r_ss(liouvillian).r;
    let r0 = r0.r;
    let w = r0 - rss;

    let Liouvillian {
        hamiltonian,
        decay,
    } = liouvillian;
    // let Vec3 {
    //     x: hx,
    //     y: hy,
    //     z: hz,
    // } = hamiltonian.r;
    let sym = symmetric_polynomials(liouvillian);
    let eigs = cubic_roots(sym);
    let (c0, c1, c2) = cayley_coeffs(sym, eigs, t);

    //let h2 = hx * hx + hy * hy + hz * hz;
    let d = decay.bloch_diagonal();

    // let w_circ = Vec3 {
    //     x: d.x * w.x,
    //     y: d.y * w.y,
    //     z: d.z * w.z,
    // };
    // let w_cross = Vec3 {
    //     x: hy * w.z - hz * w.y,
    //     y: hz * w.x - hx * w.z,
    //     z: hx * w.y - hy * w.x,
    // };
    // let m_w = Vec3 {
    //     x: w_circ.x + w_cross.x,
    //     y: w_circ.y + w_cross.y,
    //     z: w_circ.z + w_cross.z,
    // };

    // let h_dot_w = hx * w.x + hy * w.y + hz * w.z;
    // let h_cross_w_circ = Vec3 {
    //     x: hy * w_circ.z - hz * w_circ.y,
    //     y: hz * w_circ.x - hx * w_circ.z,
    //     z: hx * w_circ.y - hy * w_circ.x,
    // };
    // let d_w_circ = Vec3 {
    //     x: d.x * w_circ.x,
    //     y: d.y * w_circ.y,
    //     z: d.z * w_circ.z,
    // };
    // let d_w_cross = Vec3 {
    //     x: d.x * w_cross.x,
    //     y: d.y * w_cross.y,
    //     z: d.z * w_cross.z,
    // };

    let h = hamiltonian.r;

    let h2 = h.norm_square();

    let w_circ = d.circ(w);

    let w_cross = h.cross(w);

    let m_w = w_circ + w_cross;

    let h_dot_w = h.dot(w);

    let h_cross_w_circ = h.cross(w_circ);

    let d_w_circ = d.circ(w_circ);

    let d_w_cross = d.circ(w_cross);

    // let m2_w = Vec3 {
    //     x: d_w_circ.x + d_w_cross.x + h_cross_w_circ.x + h_dot_w * hx - h2 * w.x,
    //     y: d_w_circ.y + d_w_cross.y + h_cross_w_circ.y + h_dot_w * hy - h2 * w.y,
    //     z: d_w_circ.z + d_w_cross.z + h_cross_w_circ.z + h_dot_w * hz - h2 * w.z,
    // };

    let m2_w = d_w_circ + d_w_cross +  h_cross_w_circ + h_dot_w * h - h2 * w;

    let r = c0 * w + c1 * m_w + c2 * m2_w + rss;
    BlochVec { r }

    // BlochVec {
    //     r: Vec3 {
    //         x: c0 * w.x + c1 * m_w.x + c2 * m2_w.x + rss.x,
    //         y: c0 * w.y + c1 * m_w.y + c2 * m2_w.y + rss.y,
    //         z: c0 * w.z + c1 * m_w.z + c2 * m2_w.z + rss.z,
    //     },
    // }
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
