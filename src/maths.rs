use std::f64::consts::PI;

pub fn normalised_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    f64::exp(-0.5 * ((x - mu) / sigma).powi(2)) / (PI * 2.0).sqrt() / sigma
}

pub struct Linspace {
    pub step: f64,
    pub array: Vec<f64>,
}

impl Linspace {
    pub fn new(start: f64, stop: f64, nsteps: usize) -> Self {
        let step: f64 = (stop - start) / (nsteps as f64);
        Self {
            step,
            array: (0..=nsteps).map(|x| start + (x as f64) * step).collect(),
        }
    }
}

pub fn linspace(start: f64, stop: f64, nsteps: usize) -> Vec<f64> {
    let step: f64 = (stop - start) / (nsteps as f64);
    (0..=nsteps).map(|x| start + (x as f64) * step).collect()
}
