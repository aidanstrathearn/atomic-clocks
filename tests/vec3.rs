use std::f64::consts::PI;

use atomic_clocks::maths::vec3::Vec3;

#[test]
fn spherical_angles_use_polar_theta_and_azimuthal_phi() {
    for (theta, phi, expected) in [
        (0.0, 0.7, [0.0, 0.0, 1.0]),
        (PI, 0.7, [0.0, 0.0, -1.0]),
        (PI / 2.0, 0.0, [1.0, 0.0, 0.0]),
        (PI / 2.0, PI / 2.0, [0.0, 1.0, 0.0]),
        (
            PI / 3.0,
            -PI / 4.0,
            [6.0_f64.sqrt() / 4.0, -6.0_f64.sqrt() / 4.0, 0.5],
        ),
    ] {
        let vector = Vec3::from_angles(theta, phi);
        for (actual, expected) in [vector.x, vector.y, vector.z].into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
        assert!((vector.norm() - 1.0).abs() < 1e-12);
    }
}
