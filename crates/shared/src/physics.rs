//! Constitutive stress models for MPM particles.
//!
//! Implements:
//! - **Solid (phase=0):** Fixed corotated elasticity with damage softening
//! - **Liquid (phase=1):** Equation of state + viscosity
//! - **Gas (phase=2):** Weak EOS with very low viscosity

use glam::Mat3;

use crate::{material::MaterialParams, particle::Particle, svd::polar_decomposition};

/// Small epsilon to prevent division by zero.
const STRESS_EPSILON: f32 = 1e-10;

/// Derive Lame parameters (mu, lambda) from Young's modulus and Poisson's ratio.
///
/// - `mu = E / (2 * (1 + nu))`
/// - `lambda = E * nu / ((1 + nu) * (1 - 2*nu))`
///
/// Returns `(mu, lambda)`.
#[inline]
pub fn lame_parameters(youngs_modulus: f32, poissons_ratio: f32) -> (f32, f32) {
    let e = youngs_modulus;
    let nu = poissons_ratio;
    let mu = e / (2.0 * (1.0 + nu));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    (mu, lambda)
}

/// Compute the Cauchy stress tensor for a particle given its material.
///
/// Dispatches to the appropriate constitutive model based on the particle's phase:
/// - Phase 0 (solid): Fixed corotated elasticity
/// - Phase 1 (liquid): EOS pressure + viscous stress
/// - Phase 2 (gas): Weak EOS + minimal viscosity
///
/// Returns the Kirchhoff stress `tau` (= J * sigma, where sigma is Cauchy stress).
/// This is what gets used in the P2G scatter step: `stress_contrib = -dt * Vp * tau`.
pub fn constitutive_stress(particle: &Particle, material: &MaterialParams) -> Mat3 {
    match particle.phase() {
        0 => solid_stress(particle, material),
        1 => liquid_stress(particle, material),
        2 => gas_stress(particle, material),
        _ => Mat3::ZERO,
    }
}

/// Fixed corotated elasticity stress for solid particles.
///
/// `sigma = 2*mu*(F - R) + lambda*(J - 1)*J*F^{-T}`
///
/// Where:
/// - `F` = deformation gradient
/// - `R` = rotation from polar decomposition of F
/// - `J` = det(F) = volume ratio
/// - `mu, lambda` = Lame parameters
/// - `damage` softens mu: `mu *= (1 - damage)`
///
/// Returns the Kirchhoff stress `tau = J * sigma = 2*mu*(F - R)*F^T + lambda*(J-1)*J*I`
/// (simplified using the identity `sigma * F^T` for the Kirchhoff form).
fn solid_stress(particle: &Particle, material: &MaterialParams) -> Mat3 {
    let f = particle.deformation_gradient();
    let j = f.determinant();

    // Lame parameters from material
    let (mut mu, lambda) = lame_parameters(material.elastic.x, material.elastic.y);

    // Damage softening
    let damage = particle.damage();
    mu *= 1.0 - damage.clamp(0.0, 1.0);

    // Polar decomposition: F = R * S
    let r = polar_decomposition(f);

    // Fixed corotated: Piola-Kirchhoff stress P = 2*mu*(F - R) + lambda*(J - 1)*J*F^{-T}
    // Kirchhoff stress tau = P * F^T = 2*mu*(F - R)*F^T + lambda*(J - 1)*J*I
    2.0 * mu * (f - r) * f.transpose() + lambda * (j - 1.0) * j * Mat3::IDENTITY
}

/// Equation-of-state + viscous stress for liquid particles.
///
/// - Pressure: `p = bulk_modulus * (1/J - 1)` (Tait-like EOS)
/// - Viscous stress: `tau_visc = 2 * eta * D` where `D` is the strain rate tensor
///
/// For simplicity in MPM, we compute the strain rate from the APIC C matrix:
/// `D = 0.5 * (C + C^T)` (symmetric part of velocity gradient).
fn liquid_stress(particle: &Particle, material: &MaterialParams) -> Mat3 {
    let f = particle.deformation_gradient();
    let j = f.determinant().max(STRESS_EPSILON);

    // For liquids, Young's modulus is used as bulk modulus
    // (poisson's ratio = 0.0 for incompressible limit doesn't work, so we
    // use the elastic.x field directly as bulk modulus for liquids)
    let bulk_modulus = material.elastic.x;
    let viscosity = material.elastic.w;

    // EOS pressure: compressive when J < 1, expansive when J > 1
    let pressure = bulk_modulus * (1.0 / j - 1.0);

    // Viscous stress from APIC C matrix (velocity gradient approximation)
    let c = particle.affine_momentum();
    let d = 0.5 * (c + c.transpose()); // strain rate tensor
    let deviatoric_d = d - (d.col(0).x + d.col(1).y + d.col(2).z) / 3.0 * Mat3::IDENTITY;

    // Kirchhoff stress: tau = -p*J*I + 2*eta*J*D'
    // (multiply by J to get Kirchhoff from Cauchy)
    -pressure * j * Mat3::IDENTITY + 2.0 * viscosity * j * deviatoric_d
}

/// Gas constitutive model: weak EOS with minimal viscosity.
///
/// Same structure as liquid but with much weaker pressure response
/// and near-zero viscosity.
fn gas_stress(particle: &Particle, material: &MaterialParams) -> Mat3 {
    // Gas uses the same model as liquid but with much smaller coefficients
    // which are already encoded in the material parameters
    liquid_stress(particle, material)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::material::{MAT_STONE, MAT_WATER, default_material_table};

    #[test]
    fn lame_from_youngs_poissons() {
        // Steel-like: E=200GPa, nu=0.3
        let (mu, lambda) = lame_parameters(200e9, 0.3);
        // mu ≈ 76.9 GPa
        assert!((mu - 76.923e9).abs() / 76.923e9 < 0.01);
        // lambda ≈ 115.4 GPa
        assert!((lambda - 115.385e9).abs() / 115.385e9 < 0.01);
    }

    #[test]
    fn solid_stress_at_identity_is_zero() {
        let table = default_material_table();
        let p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, 0);
        let stress = constitutive_stress(&p, &table[MAT_STONE as usize]);

        // At F = I, J = 1, R = I: stress should be zero
        for col in 0..3 {
            for row in 0..3 {
                assert!(
                    stress.col(col)[row].abs() < 1e-3,
                    "Stress at identity should be ~0, got [{row},{col}] = {}",
                    stress.col(col)[row]
                );
            }
        }
    }

    #[test]
    fn solid_stress_compression_positive_pressure() {
        let table = default_material_table();
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, 0);
        // Compress uniformly: F = 0.9 * I → J = 0.729
        p.set_deformation_gradient(Mat3::from_diagonal(Vec3::splat(0.9)));

        let stress = constitutive_stress(&p, &table[MAT_STONE as usize]);

        // Under compression, trace of stress should be negative (compressive)
        // because tau = -p*I in the pressure part, and for J<1, p > 0
        let trace = stress.col(0).x + stress.col(1).y + stress.col(2).z;
        assert!(
            trace < 0.0,
            "Compressed solid should have negative stress trace (compressive), got {trace}"
        );
    }

    #[test]
    fn solid_stress_tension_negative_pressure() {
        let table = default_material_table();
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, 0);
        // Expand: F = 1.1 * I → J > 1
        p.set_deformation_gradient(Mat3::from_diagonal(Vec3::splat(1.1)));

        let stress = constitutive_stress(&p, &table[MAT_STONE as usize]);

        // Under expansion, trace should be positive (tensile)
        let trace = stress.col(0).x + stress.col(1).y + stress.col(2).z;
        assert!(
            trace > 0.0,
            "Expanded solid should have positive stress trace (tensile), got {trace}"
        );
    }

    #[test]
    fn solid_stress_damage_reduces_stress() {
        let table = default_material_table();
        let f = Mat3::from_cols(
            Vec3::new(1.1, 0.05, 0.0),
            Vec3::new(0.0, 0.95, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        // Undamaged
        let mut p1 = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, 0);
        p1.set_deformation_gradient(f);
        let stress1 = constitutive_stress(&p1, &table[MAT_STONE as usize]);

        // 50% damaged
        let mut p2 = Particle::new(Vec3::ZERO, 1.0, MAT_STONE, 0);
        p2.set_deformation_gradient(f);
        p2.set_damage(0.5);
        let stress2 = constitutive_stress(&p2, &table[MAT_STONE as usize]);

        // Damaged stress should have smaller Frobenius norm
        let norm1: f32 = (0..3)
            .flat_map(|c| (0..3).map(move |r| stress1.col(c)[r] * stress1.col(c)[r]))
            .sum();
        let norm2: f32 = (0..3)
            .flat_map(|c| (0..3).map(move |r| stress2.col(c)[r] * stress2.col(c)[r]))
            .sum();

        assert!(
            norm2 < norm1,
            "Damaged stress norm ({norm2}) should be less than undamaged ({norm1})"
        );
    }

    #[test]
    fn liquid_stress_at_identity_is_zero() {
        let table = default_material_table();
        let p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, 1);
        let stress = constitutive_stress(&p, &table[MAT_WATER as usize]);

        // At F = I, J = 1, C = 0: stress should be zero (pressure = bulk*(1-1) = 0)
        for col in 0..3 {
            for row in 0..3 {
                assert!(
                    stress.col(col)[row].abs() < 1e-3,
                    "Liquid stress at identity should be ~0, got [{row},{col}] = {}",
                    stress.col(col)[row]
                );
            }
        }
    }

    #[test]
    fn liquid_stress_compression_isotropic() {
        let table = default_material_table();
        let mut p = Particle::new(Vec3::ZERO, 1.0, MAT_WATER, 1);
        p.set_deformation_gradient(Mat3::from_diagonal(Vec3::splat(0.9)));

        let stress = constitutive_stress(&p, &table[MAT_WATER as usize]);

        // Should be nearly isotropic (diagonal, equal entries)
        let diag = [stress.col(0).x, stress.col(1).y, stress.col(2).z];
        let avg = (diag[0] + diag[1] + diag[2]) / 3.0;

        for i in 0..3 {
            assert!(
                (diag[i] - avg).abs() < avg.abs() * 0.01 + 1e-6,
                "Liquid compression stress should be isotropic, diag={diag:?}"
            );
        }

        // Off-diagonal should be zero
        assert!(stress.col(1).x.abs() < 1e-6);
        assert!(stress.col(2).x.abs() < 1e-6);
        assert!(stress.col(2).y.abs() < 1e-6);
    }

    #[test]
    fn constitutive_stress_unknown_phase_is_zero() {
        let table = default_material_table();
        let mut p = Particle::new(Vec3::ZERO, 1.0, 0, 0);
        // Set to invalid phase
        p.set_phase(99);
        let stress = constitutive_stress(&p, &table[0]);
        assert_eq!(stress, Mat3::ZERO);
    }
}
