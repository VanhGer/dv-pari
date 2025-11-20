//! Wrapper over an existing library for bn254 curve operations

// `unexpected_cfgs` allowed to appease warning thrown by MontConfig macro
#![allow(unexpected_cfgs)]

use std::ops::{Mul, Neg};
use ark_ff::{AdditiveGroup, BigInteger, One, PrimeField, Zero};
use ark_bn254::{Fq, G1Projective};
use num_bigint::BigUint;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use ark_ec::{CurveGroup, PrimeGroup};

/// Represents a scalar field element
pub type Fr = ark_bn254::Fr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// 254-bit Serialized Fr
pub struct FrBits(pub [bool; 254]);

impl Serialize for FrBits {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Pack bits into 32 bytes (254 bits <= 32 * 8)
        let mut bytes = [0u8; 32];
        for (i, bit) in self.0.iter().enumerate() {
            if *bit {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for FrBits {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom("expected 32 bytes"));
        }
        let mut bits = [false; 254];
        for i in 0..254 {
            bits[i] = (bytes[i / 8] >> (i % 8)) & 1 == 1;
        }
        Ok(FrBits(bits))
    }
}

impl FrBits {
    /// serialize fr
    pub fn from_fr(p: Fr) -> Self {
        let n: BigUint = p.into();
        let bytes = n.to_bytes_le();
        let mut bits = [false; 254];
        for i in 0..254 {
            let byte = if i / 8 < bytes.len() { bytes[i / 8] } else { 0 };
            let r = (byte >> (i % 8)) & 1;
            bits[i] = r != 0;
        }
        FrBits(bits)
    }

    /// deserialize to Fr and return is_valid
    pub fn to_fr(&self) -> (Fr, bool) {
        let bits = self.0;
        let mut n = BigUint::zero();
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                n |= BigUint::one() << i;
            }
        }
        let nmod = BigUint::from_str(
            "21888242871839275222246405745257275088548364400416034343698204186575808495617",
        )
        .unwrap();
        if n >= nmod {
            return (nmod.into(), false);
        }
        (n.into(), true)
    }
}

/// Represents a G1 point in curve
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint(pub G1Projective);

impl Serialize for CurvePoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let p = &self.0;

        // serialize 3 projective coordinates
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(p.x.into_bigint().to_bytes_be().as_slice());
        bytes.extend_from_slice(p.y.into_bigint().to_bytes_be().as_slice());
        bytes.extend_from_slice(p.z.into_bigint().to_bytes_be().as_slice());

        serializer.serialize_bytes(&bytes)
    }
}


impl<'de> Deserialize<'de> for CurvePoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;

        if bytes.len() != 96 {
            return Err(serde::de::Error::custom("expected 96 bytes for projective point"));
        }

        let x = Fq::from_be_bytes_mod_order(&bytes[0..32]);
        let y = Fq::from_be_bytes_mod_order(&bytes[32..64]);
        let z = Fq::from_be_bytes_mod_order(&bytes[64..96]);

        Ok(CurvePoint(G1Projective::new_unchecked(x, y, z)))
    }
}

impl PartialEq for CurvePoint {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for CurvePoint {}

impl CurvePoint {
    pub(crate) fn generator() -> Self {
        CurvePoint(G1Projective::generator())
    }

    pub(crate) fn add(a: CurvePoint, b: CurvePoint) -> Self {
        CurvePoint(a.0 + b.0)
    }

    /// Serialize CurvePoint to bytes
    pub fn to_bytes(self) -> [u8; 96] {
        let mut bytes = [0u8; 96];
        let x_bytes: [u8; 32] = self.0.x.into_bigint().to_bytes_le().try_into().unwrap();
        let y_bytes: [u8; 32] = self.0.y.into_bigint().to_bytes_le().try_into().unwrap();
        let z_bytes: [u8; 32] = self.0.z.into_bigint().to_bytes_le().try_into().unwrap();
        bytes[..32].copy_from_slice(&x_bytes);
        bytes[32..64].copy_from_slice(&y_bytes);
        bytes[64..].copy_from_slice(&z_bytes);
        bytes
    }

    /// Deserialize Unchecked CurvePoint from bytes
    pub fn from_bytes(src: &[u8; 96]) -> CurvePoint {
        let x = Fq::from_le_bytes_mod_order(&src[..32]);
        let y = Fq::from_le_bytes_mod_order(&src[32..64]);
        let z = Fq::from_le_bytes_mod_order(&src[64..]);
        let point = G1Projective::new_unchecked(x, y, z);
        CurvePoint(point)
    }

    /// Check if the CurvePoint is valid
    pub fn checked(&self) -> bool {
        let affine_p = self.0.into_affine();
        let checked = affine_p.is_on_curve() && affine_p.is_in_correct_subgroup_assuming_on_curve();
        checked
    }
}

// Calculate point scalar multiplication
pub(crate) fn point_scalar_mul(scalar: Fr, point: CurvePoint) -> CurvePoint {
    let res = point.0.mul(scalar);
    CurvePoint(res)
}

/// Point Scalar Multiplication with [`generator`] as the [`CurvePoint`]
pub(crate) fn point_scalar_mul_gen(scalar: Fr) -> CurvePoint {
    let res = G1Projective::generator().mul(scalar);
    CurvePoint(res)
}

/// Multi Scalar Multiplication
// For now we just compute individual point scalar multiplications and sum up the result
pub(crate) fn multi_scalar_mul(scalars: &[Fr], points: &[CurvePoint]) -> CurvePoint {
    assert_eq!(scalars.len(), points.len());

    let results_par_iter = points
        .par_iter() // Use Rayon's parallel iterator for points
        .zip(scalars.par_iter()) // Use Rayon's parallel iterator for scalars
        .map(|(p, s)| point_scalar_mul(*s, *p))
        .into_par_iter();

    results_par_iter.reduce(
        || CurvePoint(G1Projective::ZERO),
        |p1: CurvePoint, p2: CurvePoint| {
            CurvePoint(p1.0 + p2.0)
        }
    )
}


#[cfg(test)]
mod unit_test {
    use ark_bn254::G1Projective;
    use ark_ec::PrimeGroup;
    use ark_ff::{AdditiveGroup, BigInteger, PrimeField, UniformRand};
    use ark_std::rand::thread_rng;

    use crate::curve::{
        CurvePoint, point_scalar_mul,
    };
    use super::{Fr, multi_scalar_mul};

    #[test]
    // Compares result of msm with one computed directly from point add operation
    fn test_validate_psm_with_point_add() {
        let mut rng = thread_rng();
        let k1 = Fr::rand(&mut rng);
        let k2 = Fr::rand(&mut rng);

        let d = CurvePoint(G1Projective::generator());
        let y1 = point_scalar_mul(k1, d);
        let y2 = point_scalar_mul(k2, d);
        let y3 = point_scalar_mul(k1 + k2, d);

        let y12 = CurvePoint::add(y1, y2);
        let is_iden = y12.eq(&y3);
        assert!(is_iden);
    }

    #[test]
    fn test_msm_2() {
        let mut rng = thread_rng();
        let n = 10_000;
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let points: Vec<CurvePoint> = (0..n).map(|_| CurvePoint(G1Projective::generator())).collect();
        let res = multi_scalar_mul(&scalars, &points);
        let mut total = Fr::ZERO;
        for scalar in scalars {
            total += scalar;
        }
        let total_msm = point_scalar_mul(total, points[0]);
        assert_eq!(total_msm, res);
    }

    #[test]
    // Verifies that a CurvePoint is recovered after serialize-then-deserialize
    fn test_curve_point_to_bytes() {
        let point = CurvePoint(G1Projective::rand(&mut thread_rng()));
        let bytes = point.to_bytes();
        let decoded = CurvePoint::from_bytes(&bytes);
        let checked = decoded.checked();
        assert_eq!(decoded, point);
        assert!(checked);
    }
}
