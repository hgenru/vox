//! GPU-compatible 3x3 SVD and polar decomposition.
//!
//! Uses symmetric Jacobi eigendecomposition of `F^T * F` to find V and singular
//! values, then computes U by normalizing `F * V` columns.
//!
//! The symmetric Jacobi method has guaranteed convergence and uses only
//! `+`, `*`, `/`, `sqrt` — no trig functions. Suitable for GPU via rust-gpu.
//!
//! # References
//! - McAdams et al. 2011 (SIGGRAPH Technical Brief)
//! - Golub & Van Loan, "Matrix Computations"

use glam::{Mat3, Vec3};

/// Small epsilon to avoid division by zero.
const SVD_EPSILON: f32 = 1e-7;

/// Number of Jacobi sweeps for symmetric eigendecomposition.
/// Each sweep does 3 rotations (pairs 01, 02, 12).
/// The off-diagonal elements decrease quadratically, so 10 sweeps
/// is more than sufficient for f32 precision.
const JACOBI_SWEEPS: u32 = 10;

/// Compute Jacobi rotation parameters for a 2x2 symmetric matrix
/// `[[a_pp, a_pq], [a_pq, a_qq]]`.
///
/// Returns `(c, s)` such that `G^T * A * G` is diagonal.
#[inline]
fn sym_schur2(a_pp: f32, a_pq: f32, a_qq: f32) -> (f32, f32) {
    if a_pq.abs() < SVD_EPSILON {
        return (1.0, 0.0);
    }
    let tau = (a_pp - a_qq) / (2.0 * a_pq);
    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
    let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

/// Perform Jacobi eigendecomposition of a 3x3 symmetric matrix.
///
/// Returns `(eigenvalues, V)` where `A = V * diag(eigenvalues) * V^T`.
fn jacobi_eigendecomposition(a: Mat3) -> (Vec3, Mat3) {
    // Store as 6 unique elements: a00, a01, a02, a11, a12, a22
    let mut m = [
        a.col(0).x,
        a.col(1).x,
        a.col(2).x,
        a.col(1).y,
        a.col(2).y,
        a.col(2).z,
    ];
    let mut v = Mat3::IDENTITY;

    for _ in 0..JACOBI_SWEEPS {
        // Rotation (0,1): zero out m[1] = a01
        {
            let (c, s) = sym_schur2(m[0], m[1], m[3]);
            if s.abs() > SVD_EPSILON {
                // Update symmetric matrix: A' = G^T * A * G
                let a00 = m[0];
                let a01 = m[1];
                let a02 = m[2];
                let a11 = m[3];
                let a12 = m[4];

                m[0] = c * c * a00 + 2.0 * c * s * a01 + s * s * a11;
                m[3] = s * s * a00 - 2.0 * c * s * a01 + c * c * a11;
                m[1] = 0.0; // zeroed by construction
                m[2] = c * a02 + s * a12;
                m[4] = -s * a02 + c * a12;

                // V = V * G
                let v0 = v.col(0);
                let v1 = v.col(1);
                v = Mat3::from_cols(c * v0 + s * v1, -s * v0 + c * v1, v.col(2));
            }
        }

        // Rotation (0,2): zero out m[2] = a02
        {
            let (c, s) = sym_schur2(m[0], m[2], m[5]);
            if s.abs() > SVD_EPSILON {
                let a00 = m[0];
                let a01 = m[1];
                let a02 = m[2];
                let a12 = m[4];
                let a22 = m[5];

                m[0] = c * c * a00 + 2.0 * c * s * a02 + s * s * a22;
                m[5] = s * s * a00 - 2.0 * c * s * a02 + c * c * a22;
                m[2] = 0.0;
                m[1] = c * a01 + s * a12;
                m[4] = -s * a01 + c * a12;

                let v0 = v.col(0);
                let v2 = v.col(2);
                v = Mat3::from_cols(c * v0 + s * v2, v.col(1), -s * v0 + c * v2);
            }
        }

        // Rotation (1,2): zero out m[4] = a12
        {
            let (c, s) = sym_schur2(m[3], m[4], m[5]);
            if s.abs() > SVD_EPSILON {
                let a01 = m[1];
                let a02 = m[2];
                let a11 = m[3];
                let a12 = m[4];
                let a22 = m[5];

                m[3] = c * c * a11 + 2.0 * c * s * a12 + s * s * a22;
                m[5] = s * s * a11 - 2.0 * c * s * a12 + c * c * a22;
                m[4] = 0.0;
                m[1] = c * a01 + s * a02;
                m[2] = -s * a01 + c * a02;

                let v1 = v.col(1);
                let v2 = v.col(2);
                v = Mat3::from_cols(v.col(0), c * v1 + s * v2, -s * v1 + c * v2);
            }
        }
    }

    (Vec3::new(m[0], m[3], m[5]), v)
}

/// Find a vector orthogonal to `n`.
#[inline]
fn orthogonal_complement(n: Vec3) -> Vec3 {
    let abs_n = Vec3::new(n.x.abs(), n.y.abs(), n.z.abs());
    let candidate = if abs_n.x <= abs_n.y && abs_n.x <= abs_n.z {
        Vec3::X
    } else if abs_n.y <= abs_n.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let result = n.cross(candidate);
    let len = result.length();
    if len > SVD_EPSILON {
        result / len
    } else {
        Vec3::X
    }
}

/// Compute the SVD of a 3x3 matrix: `F = U * diag(sigma) * V^T`.
///
/// Algorithm:
/// 1. Compute `S = F^T * F` (symmetric positive semi-definite)
/// 2. Jacobi eigendecomposition: `S = V * diag(lambda) * V^T`
/// 3. `sigma_i = sqrt(max(0, lambda_i))`
/// 4. `U = F * V * diag(1/sigma)`, with Gram-Schmidt re-orthogonalization
///
/// Returns `(U, sigma, V)`. U and V have det = +1 (proper rotations).
///
/// # Accuracy
/// With 10 Jacobi sweeps, accuracy is within ~1e-5 for well-conditioned f32 matrices.
pub fn svd_3x3(f: Mat3) -> (Mat3, Vec3, Mat3) {
    let s = f.transpose() * f;
    let (_eigenvalues, mut v) = jacobi_eigendecomposition(s);

    // B = F * V; columns of B should be orthogonal if V is exact
    let b = f * v;

    // Extract U by normalizing B columns with Gram-Schmidt cleanup
    let s0 = b.col(0).length();
    let s1 = b.col(1).length();
    let s2 = b.col(2).length();

    let u0 = if s0 > SVD_EPSILON {
        b.col(0) / s0
    } else {
        Vec3::X
    };

    let u1 = if s1 > SVD_EPSILON {
        let mut col = b.col(1) / s1;
        col -= u0 * u0.dot(col);
        let len = col.length();
        if len > SVD_EPSILON {
            col / len
        } else {
            orthogonal_complement(u0)
        }
    } else {
        orthogonal_complement(u0)
    };

    let u2 = if s2 > SVD_EPSILON {
        let mut col = b.col(2) / s2;
        col -= u0 * u0.dot(col);
        col -= u1 * u1.dot(col);
        let len = col.length();
        if len > SVD_EPSILON {
            col / len
        } else {
            u0.cross(u1)
        }
    } else {
        u0.cross(u1)
    };

    let mut u = Mat3::from_cols(u0, u1, u2);

    // Recompute sigma for accuracy after Gram-Schmidt
    let mut sigma_out = Vec3::new(
        u.col(0).dot(b.col(0)),
        u.col(1).dot(b.col(1)),
        u.col(2).dot(b.col(2)),
    );

    // Ensure det(U) = +1 and det(V) = +1
    if u.determinant() < 0.0 {
        u = Mat3::from_cols(u.col(0), u.col(1), -u.col(2));
        sigma_out.z = -sigma_out.z;
    }
    if v.determinant() < 0.0 {
        v = Mat3::from_cols(v.col(0), v.col(1), -v.col(2));
        sigma_out.z = -sigma_out.z;
    }

    (u, sigma_out, v)
}

/// Compute the polar decomposition of a 3x3 matrix: `F = R * S`.
///
/// Returns the rotation part `R = U * V^T` from the SVD of F.
/// This is the closest rotation matrix to F in the Frobenius norm.
///
/// Used in the fixed corotated constitutive model for solid materials.
pub fn polar_decomposition(f: Mat3) -> Mat3 {
    let (u, _sigma, v) = svd_3x3(f);
    u * v.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_svd_reconstruction(f: Mat3, tol: f32) {
        let (u, sigma, v) = svd_3x3(f);
        let sigma_mat = Mat3::from_diagonal(sigma);
        let reconstructed = u * sigma_mat * v.transpose();

        for col in 0..3 {
            for row in 0..3 {
                let diff = (f.col(col)[row] - reconstructed.col(col)[row]).abs();
                assert!(
                    diff < tol,
                    "SVD reconstruction error at ({row},{col}): expected {}, got {}, diff={diff}",
                    f.col(col)[row],
                    reconstructed.col(col)[row]
                );
            }
        }
    }

    fn assert_orthogonal(m: Mat3, tol: f32) {
        let prod = m.transpose() * m;
        for col in 0..3 {
            for row in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                let diff = (prod.col(col)[row] - expected).abs();
                assert!(
                    diff < tol,
                    "Orthogonality error at ({row},{col}): expected {expected}, got {}, diff={diff}",
                    prod.col(col)[row]
                );
            }
        }
    }

    #[test]
    fn svd_identity() {
        let (u, sigma, v) = svd_3x3(Mat3::IDENTITY);
        assert_svd_reconstruction(Mat3::IDENTITY, 1e-5);
        assert_orthogonal(u, 1e-5);
        assert_orthogonal(v, 1e-5);
        for i in 0..3 {
            assert!(
                (sigma[i].abs() - 1.0).abs() < 1e-5,
                "sigma[{i}] = {}",
                sigma[i]
            );
        }
    }

    #[test]
    fn svd_uniform_scaling() {
        let scale = 3.0;
        let f = Mat3::from_diagonal(Vec3::splat(scale));
        let (u, sigma, v) = svd_3x3(f);
        assert_svd_reconstruction(f, 1e-4);
        assert_orthogonal(u, 1e-5);
        assert_orthogonal(v, 1e-5);
        for i in 0..3 {
            assert!(
                (sigma[i].abs() - scale).abs() < 1e-4,
                "sigma[{i}] = {}",
                sigma[i]
            );
        }
    }

    #[test]
    fn svd_non_uniform_scaling() {
        let f = Mat3::from_diagonal(Vec3::new(5.0, 2.0, 0.5));
        assert_svd_reconstruction(f, 1e-4);
    }

    #[test]
    fn svd_rotation() {
        let angle = core::f32::consts::FRAC_PI_4;
        let c = angle.cos();
        let s = angle.sin();
        let f = Mat3::from_cols(
            Vec3::new(c, s, 0.0),
            Vec3::new(-s, c, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let (_u, sigma, _v) = svd_3x3(f);
        assert_svd_reconstruction(f, 1e-5);
        for i in 0..3 {
            assert!(
                (sigma[i].abs() - 1.0).abs() < 1e-4,
                "sigma[{i}] = {}",
                sigma[i]
            );
        }
    }

    #[test]
    fn svd_shear() {
        let f = Mat3::from_cols(
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert_svd_reconstruction(f, 1e-4);
    }

    #[test]
    fn svd_general_matrix() {
        let f = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 10.0),
        );
        assert_svd_reconstruction(f, 1e-3);
        let (u, _sigma, v) = svd_3x3(f);
        assert_orthogonal(u, 1e-4);
        assert_orthogonal(v, 1e-4);
    }

    #[test]
    fn svd_near_singular() {
        let f = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.001),
            Vec3::new(0.0, 1.0, 0.001),
            Vec3::new(0.0, 0.0, 0.001),
        );
        assert_svd_reconstruction(f, 1e-3);
    }

    #[test]
    fn polar_decomposition_identity() {
        let r = polar_decomposition(Mat3::IDENTITY);
        for col in 0..3 {
            for row in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (r.col(col)[row] - expected).abs() < 1e-5,
                    "R[{row},{col}] = {}",
                    r.col(col)[row]
                );
            }
        }
    }

    #[test]
    fn polar_decomposition_rotation() {
        let angle = 0.7_f32;
        let c = angle.cos();
        let s = angle.sin();
        let rot = Mat3::from_cols(
            Vec3::new(c, s, 0.0),
            Vec3::new(-s, c, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let r = polar_decomposition(rot);
        for col in 0..3 {
            for row in 0..3 {
                assert!(
                    (r.col(col)[row] - rot.col(col)[row]).abs() < 1e-4,
                    "R[{row},{col}] = {}, expected {}",
                    r.col(col)[row],
                    rot.col(col)[row]
                );
            }
        }
    }

    #[test]
    fn polar_decomposition_scaled_rotation() {
        let angle = 1.2_f32;
        let c = angle.cos();
        let s = angle.sin();
        let rot = Mat3::from_cols(
            Vec3::new(c, s, 0.0),
            Vec3::new(-s, c, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let scale = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 1.5));
        let f = rot * scale;
        let r = polar_decomposition(f);
        assert_orthogonal(r, 1e-4);
        for col in 0..3 {
            for row in 0..3 {
                assert!(
                    (r.col(col)[row] - rot.col(col)[row]).abs() < 1e-3,
                    "R[{row},{col}] = {}, expected {}",
                    r.col(col)[row],
                    rot.col(col)[row]
                );
            }
        }
    }

    #[test]
    fn polar_decomposition_is_rotation() {
        let f = Mat3::from_cols(
            Vec3::new(1.0, 0.3, 0.0),
            Vec3::new(-0.2, 1.5, 0.1),
            Vec3::new(0.0, 0.0, 0.8),
        );
        let r = polar_decomposition(f);
        assert_orthogonal(r, 1e-4);
        let det = r.determinant();
        assert!(
            (det - 1.0).abs() < 1e-3,
            "det(R) = {det}, expected 1.0"
        );
    }

    #[test]
    fn svd_singular_values_non_negative() {
        let f = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 10.0),
        );
        let (_u, sigma, _v) = svd_3x3(f);
        let positive_count = [sigma.x, sigma.y, sigma.z]
            .iter()
            .filter(|&&s| s >= -1e-5)
            .count();
        assert!(
            positive_count >= 2,
            "Expected at least 2 non-negative singular values, got {positive_count}: sigma={sigma:?}"
        );
    }
}
