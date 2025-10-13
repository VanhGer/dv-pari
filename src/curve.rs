//! Wrapper over an existing library for sect233k1 curve operations

// `unexpected_cfgs` allowed to appease warning thrown by MontConfig macro
#![allow(unexpected_cfgs)]
use ark_ff::fields::{Fp256, MontBackend, MontConfig};
use ark_ff::{One, PrimeField, Zero};
use num_bigint::BigUint;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::os::raw::c_void;
use std::str::FromStr;
use xs233_sys::{xsk233_add, xsk233_generator, xsk233_neutral, xsk233_point};
use crate::utils::{msb_bit};

/// FqConfig for Scalar Field of the curve
#[derive(MontConfig, Debug)]
#[modulus = "3450873173395281893717377931138512760570940988862252126328087024741343"]
#[generator = "3"]
pub struct FqConfig;

/// Represents a scalar field element
pub type Fr = Fp256<MontBackend<FqConfig, 4>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// 232-bit Serialized Fr
pub struct FrBits(pub [bool; 232]);

impl Serialize for FrBits {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Pack bits into 29 bytes (232 bits = 29 * 8)
        let mut bytes = [0u8; 29];
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
        if bytes.len() != 29 {
            return Err(serde::de::Error::custom("expected 29 bytes"));
        }
        let mut bits = [false; 232];
        for i in 0..232 {
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
        let mut bits = [false; 232];
        for i in 0..232 {
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
            "3450873173395281893717377931138512760570940988862252126328087024741343",
        )
        .unwrap();
        if n >= nmod {
            return (nmod.into(), false);
        }
        (n.into(), true)
    }
}

/// Represents a point in curve
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint(pub xsk233_point);

/// Lopez–Dahab λ coordinates (x, λ) for a curve point over GF(2^233).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LambdaCurvePoint {
    /// Canonical little-endian encoding of the x-coordinate.
    pub x: [u8; 30],
    /// Canonical little-endian encoding of the λ (= s) coordinate.
    pub lambda: [u8; 30],
}

impl LambdaCurvePoint {
    /// Serialize Lopez–Dahab λ coordinates to a 60 byte array.
    pub fn to_bytes(self) -> [u8; 60] {
        let mut res = [0u8; 60];
        res[..30].copy_from_slice(&self.x);
        res[30..].copy_from_slice(&self.lambda);
        res
    }

    /// Deserialize Lopez–Dahab λ coordinates from a 60 byte array.
    pub fn from_bytes(bytes: &[u8; 60]) -> Self {
        let mut x = [0u8; 30];
        let mut lambda = [0u8; 30];
        x.copy_from_slice(&bytes[..30]);
        lambda.copy_from_slice(&bytes[30..]);
        LambdaCurvePoint { x, lambda }
    }
}

impl PartialEq for CurvePoint {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let r = xs233_sys::xsk233_equals(&self.0, &other.0);
            r != 0
        }
    }
}
impl Eq for CurvePoint {}

impl CurvePoint {
    pub(crate) fn generator() -> Self {
        unsafe { CurvePoint(xsk233_generator) }
    }

    pub(crate) fn add(a: CurvePoint, b: CurvePoint) -> Self {
        unsafe {
            let mut p3 = xsk233_neutral;
            xsk233_add(&mut p3, &a.0, &b.0);
            CurvePoint(p3)
        }
    }

    /// Serialize CurvePoint to Lopez–Dahab λ bytes (x || λ, 60 bytes).
    pub fn to_bytes(self) -> [u8; 60] {
        self.to_lambda().to_bytes()
    }

    /// Deserialize a CurvePoint from Lopez–Dahab λ bytes (x || λ, 60 bytes).
    pub fn from_bytes(src: &[u8; 60]) -> (CurvePoint, bool) {
        let lambda_coords = LambdaCurvePoint::from_bytes(src);
        CurvePoint::from_lambda(&lambda_coords)
    }

    /// Convert an extended point into Lopez–Dahab λ coordinates.
    pub fn to_lambda(&self) -> LambdaCurvePoint {
        let mut x = [0u8; 30];
        let mut lambda = [0u8; 30];
        unsafe {
            xs233_sys::xsk233_to_affine(
                &self.0,
                x.as_mut_ptr() as *mut c_void,
                lambda.as_mut_ptr() as *mut c_void,
            );
        }
        LambdaCurvePoint { x, lambda }
    }

    /// Reconstruct a [`CurvePoint`] from Lopez–Dahab λ coordinates.
    pub fn from_lambda(coords: &LambdaCurvePoint) -> (CurvePoint, bool) {
        unsafe {
            let mut pt = xsk233_neutral;
            let success = xs233_sys::xsk233_from_affine(
                &mut pt,
                coords.x.as_ptr() as *const c_void,
                coords.lambda.as_ptr() as *const c_void,
            );
            (CurvePoint(pt), success != 0)
        }
    }
    
    /// Negate a CurvePoint
    pub fn negate(self) -> CurvePoint {
        unsafe {
            let neg_pt = self.0;
            let mut result = xsk233_neutral;
            xs233_sys::xsk233_neg(&mut result, &neg_pt);
            CurvePoint(result)
        }
    }
}

// Calculate point scalar multiplication
pub(crate) fn point_scalar_mul(scalar: Fr, point: CurvePoint) -> CurvePoint {
    let scalar = fr_to_le_bytes(&scalar);

    unsafe {
        let mut result = xsk233_neutral;
        xs233_sys::xsk233_mul_frob(
            &mut result,
            &point.0,
            scalar.as_ptr() as *const _,
            scalar.len(),
        );
        CurvePoint(result)
    }
}

/// Point Scalar Multiplication with [`generator`] as the [`CurvePoint`]
pub(crate) fn point_scalar_mul_gen(scalar: Fr) -> CurvePoint {
    let scalar = fr_to_le_bytes(&scalar);

    unsafe {
        let mut result = xsk233_neutral;
        xs233_sys::xsk233_mulgen_frob(&mut result, scalar.as_ptr() as *const _, scalar.len());
        CurvePoint(result)
    }
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
        || unsafe { CurvePoint(xsk233_neutral) },
        |p1: CurvePoint, p2: CurvePoint| unsafe {
            let mut p3 = xsk233_neutral;
            xsk233_add(&mut p3, &p1.0, &p2.0);
            CurvePoint(p3)
        },
    )
}

// Optimization with precomputed table T
pub(crate) fn multi_scalar_mul_with_precompute(
    scalars: &[Fr],
    points: &[CurvePoint],
) -> CurvePoint {
    assert_eq!(scalars.len(), points.len());
    // limit to 32 points for now
    assert!(scalars.len() <= 32);
    // Todo: ensure all the scalars are in [0, 2^big_n)

    // Precompute table T
    let t_length = 2_u32.pow(scalars.len() as u32) as usize;
    let t: Vec<CurvePoint> = (0..t_length)
        .into_par_iter()
        .map(|j| unsafe {
            let e_is = (0..scalars.len())
                .map(|i| ((j >> i) & 1) as u8)
                .collect::<Vec<u8>>();
            let mut tmp = xsk233_neutral;
            for (i, &e_i) in e_is.iter().enumerate() {
                // mul with e_i
                let e_i_p = point_scalar_mul(Fr::from(e_i),points[i]);
                xsk233_add(&mut tmp, &tmp, &e_i_p.0);
            }
            CurvePoint(tmp)
        })
        .collect();

    // main loop
    unsafe {
        let mut result = xsk233_neutral;
        for bit_id in 0..256 {
            xsk233_add(&mut result, &result, &result); // double
            let mut t_id: u32 = 0;
            for i in 0..scalars.len() {
                let b_i = msb_bit(&scalars[i], bit_id as usize) as u32;
                t_id += b_i * 2_u32.pow(i as u32);
            }
            let t_id: u32 = scalars
                .par_iter()
                .enumerate()
                .map(|(i, scalar)| {
                    let b_i = msb_bit(&scalars[i], bit_id as usize) as u32;
                    b_i * 2_u32.pow(i as u32)
                })
                .sum();
            if t_id != 0 {
                xsk233_add(&mut result, &result, &t[t_id as usize].0); // add
            }
        }
        CurvePoint(result)
    }

}

/// Convert scalar field element to byte array
/// Pornin's [`xsk233_mulgen_frob`] accepts scalar as a byte array
fn fr_to_le_bytes(fr: &Fr) -> Vec<u8> {
    let big_int = fr.into_bigint();
    let limbs = big_int.0;

    let mut bytes = Vec::with_capacity(32);
    for limb in limbs.iter() {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    bytes.truncate(30); // 30 specified by `xs233_sys`

    // remove trailing zeros
    // helps reduce iteration in double-and-add iterations
    while let Some(&last) = bytes.last() {
        if last == 0 {
            bytes.pop();
        } else {
            break;
        }
    }
    bytes
}

#[cfg(test)]
mod unit_test {
    use ark_ff::{AdditiveGroup, UniformRand};
    use ark_std::rand::thread_rng;
    use xs233_sys::{xsk233_add, xsk233_equals, xsk233_generator, xsk233_neutral};

    use crate::curve::{CurvePoint, LambdaCurvePoint, point_scalar_mul, point_scalar_mul_gen};

    use super::{Fr, multi_scalar_mul};

    #[test]
    // Compares result of msm with one computed directly from point add operation
    fn test_validate_psm_with_point_add() {
        let mut rng = thread_rng();
        let k1 = Fr::rand(&mut rng);
        let k2 = Fr::rand(&mut rng);

        unsafe {
            let d = CurvePoint(xsk233_generator);
            let y1 = point_scalar_mul(k1, d);
            let y2 = point_scalar_mul(k2, d);
            let y3 = point_scalar_mul(k1 + k2, d);

            let mut y12 = xsk233_neutral;
            xsk233_add(&mut y12, &y2.0, &y1.0);

            let is_iden = xsk233_equals(&y12, &y3.0);
            assert!(is_iden != 0);
        }
    }

    #[test]
    fn test_msm() {
        let mut rng = thread_rng();
        let n = 10_000;
        unsafe {
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let points: Vec<CurvePoint> = (0..n).map(|_| CurvePoint(xsk233_generator)).collect();
            let res = multi_scalar_mul(&scalars, &points);
            let mut total = Fr::ZERO;
            for scalar in scalars {
                total += scalar;
            }
            let total_msm = point_scalar_mul(total, points[0]);
            assert_eq!(total_msm, res);
        }
    }

    #[test]
    fn test_msm_with_precompute() {
        let mut rng = thread_rng();
        let n = 10;
        unsafe {
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let points: Vec<CurvePoint> = (0..n).map(|_| CurvePoint(xsk233_generator)).collect();
            let res = super::multi_scalar_mul_with_precompute(&scalars, &points);

            let expected_msm = multi_scalar_mul(&scalars, &points);

            assert_eq!(expected_msm, res);
        }
    }

    #[test]
    // Verifies that a CurvePoint is recovered after serialize-then-deserialize
    fn test_curve_point_to_bytes() {
        let generator = CurvePoint::generator();
        let bytes = generator.to_bytes();
        let (decoded, valid) = CurvePoint::from_bytes(&bytes);
        assert!(valid);
        assert_eq!(decoded, generator);
    }

    #[test]
    fn test_neutral_lambda_coordinates() {
        unsafe {
            let neutral = CurvePoint(xsk233_neutral);
            let coords = neutral.to_lambda();
            assert_eq!(coords.x, [0u8; 30]);
            let mut expected_lambda = [0u8; 30];
            expected_lambda[0] = 1;
            assert_eq!(coords.lambda, expected_lambda);
        }
    }

    #[test]
    fn test_generator_lambda_coordinates_stability() {
        unsafe {
            let generator = CurvePoint(xsk233_generator);
            let coords = generator.to_lambda();
            let expected = LambdaCurvePoint {
                x: [
                    230, 27, 170, 221, 203, 229, 80, 168, 84, 191, 102, 25, 126, 239, 36, 87, 6,
                    185, 133, 101, 71, 236, 61, 251, 208, 118, 39, 185, 236, 1,
                ],
                lambda: [
                    153, 154, 125, 54, 191, 224, 249, 102, 193, 150, 111, 7, 80, 50, 25, 247, 157,
                    102, 243, 253, 252, 71, 170, 91, 78, 77, 59, 255, 237, 0,
                ],
            };
            assert_eq!(coords, expected);
        }
    }

    #[test]
    fn test_lambda_roundtrip_generator() {
        unsafe {
            let generator = CurvePoint(xsk233_generator);
            let coords = generator.to_lambda();
            let (recovered, valid) = CurvePoint::from_lambda(&coords);
            assert!(valid);
            assert_eq!(recovered, generator);
        }
    }

    #[test]
    fn test_lambda_roundtrip_neutral() {
        unsafe {
            let neutral = CurvePoint(xsk233_neutral);
            let coords = neutral.to_lambda();
            let (recovered, valid) = CurvePoint::from_lambda(&coords);
            assert!(valid);
            assert_eq!(recovered, neutral);
        }
    }

    #[test]
    fn test_random_point_lambda_dump() {
        let mut rng = thread_rng();
        let scalar = Fr::rand(&mut rng);
        let point = point_scalar_mul_gen(scalar);
        let coords = point.to_lambda();
        println!("random_point_x={:?}", coords.x);
        println!("random_point_lambda={:?}", coords.lambda);
        assert!(coords.x.iter().any(|&b| b != 0) || coords.lambda.iter().any(|&b| b != 0));
    }
}
