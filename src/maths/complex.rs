#[derive(Clone, Copy, Debug)]
pub(crate) struct Complex {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl Complex {
    pub(crate) fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub(crate) fn sqrt(self) -> Self {
        if self.im == 0.0 {
            if self.re >= 0.0 {
                return Self::new(self.re.sqrt(), 0.0);
            }
            return Self::new(0.0, (-self.re).sqrt());
        }

        let r = self.re.hypot(self.im);
        let re = ((r + self.re) / 2.0).sqrt();
        let im = self.im.signum() * ((r - self.re) / 2.0).sqrt();
        Self::new(re, im)
    }

    pub(crate) fn exp(self) -> Self {
        let scale = self.re.exp();
        Self::new(scale * self.im.cos(), scale * self.im.sin())
    }

    pub(crate) fn powf(self, exponent: f64) -> Self {
        let radius = self.re.hypot(self.im);
        let angle = self.im.atan2(self.re);
        let scaled_radius = radius.powf(exponent);
        let scaled_angle = angle * exponent;
        Self::new(
            scaled_radius * scaled_angle.cos(),
            scaled_radius * scaled_angle.sin(),
        )
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl std::ops::Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}
