use atomic_clocks::maths::{Mat3, vec3::Vec3};

fn assert_vec(actual: Vec3, expected: [f64; 3]) {
    for (a, e) in [actual.x, actual.y, actual.z].into_iter().zip(expected) {
        assert!((a - e).abs() < 1e-12, "{a} != {e}");
    }
}

#[test]
fn row_and_column_constructors_act_on_column_vectors() {
    let rows = [[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [2.0, 0.0, 1.0]];
    let columns = [[1.0, 0.0, 2.0], [2.0, -1.0, 0.0], [3.0, 4.0, 1.0]];
    let matrix = Mat3::from_rows(rows);
    assert_eq!(matrix, Mat3::from_columns(columns));
    assert_eq!(matrix.rows(), &rows);
    assert_eq!(matrix.transpose().rows(), &columns);
    assert_eq!(matrix.transpose().transpose(), matrix);
    assert_vec(
        matrix.apply_to_vec(Vec3 {
            x: 2.0,
            y: -1.0,
            z: 0.5,
        }),
        [1.5, 3.0, 4.5],
    );
}

#[test]
fn multiplication_applies_the_right_matrix_first() {
    let a = Mat3::from_rows([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [2.0, 0.0, 1.0]]);
    let b = Mat3::from_rows([[0.0, 1.0, 2.0], [3.0, -1.0, 0.0], [1.0, 2.0, 1.0]]);
    assert_eq!(
        (a * b).rows(),
        &[[9.0, 5.0, 5.0], [1.0, 9.0, 4.0], [1.0, 4.0, 5.0]]
    );
    assert_ne!(a * b, b * a);
    let v = Vec3 {
        x: 2.0,
        y: -1.0,
        z: 0.5,
    };
    assert_vec((a * b).apply_to_vec(v), [15.5, -5.0, 0.5]);
    assert_vec(a.apply_to_vec(b.apply_to_vec(v)), [15.5, -5.0, 0.5]);
    assert_eq!((a * b).transpose(), b.transpose() * a.transpose());
}

#[test]
fn linear_combinations_and_diagonals() {
    let diagonal = Mat3::diagonal(Vec3 {
        x: 2.0,
        y: -3.0,
        z: 0.5,
    });
    let matrix = 2.0 * diagonal + Mat3::identity() * 3.0;
    assert_vec(
        matrix.apply_to_vec(Vec3 {
            x: 1.0,
            y: 2.0,
            z: -1.0,
        }),
        [7.0, -6.0, -4.0],
    );
    assert_eq!(matrix - diagonal * 2.0, Mat3::identity() * 3.0);
    assert_eq!(matrix * Mat3::identity(), matrix);
    assert_eq!(Mat3::identity() * matrix, matrix);
    assert_eq!(matrix * Mat3::zero(), Mat3::zero());
    assert_eq!(Mat3::zero() + matrix, matrix);
}
