//! Utils
//!
use crate::curve::Fr;
use ark_ff::{BigInt, PrimeField};
use lll_rs::Matrix;
use lll_rs::l2::lll_bignum;
use rug::Integer;
use rug::integer::Order;
use std::ops::Index;

/// Decomp holds the decomposition result (x, is_x_neg)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Decomp(pub Fr, pub bool);

/// Convert BigInt<4> to Integer
pub fn bigint4_to_integer(f: &BigInt<4>) -> Integer {
    let f_i = f.0.iter().map(|x| Integer::from(*x)).collect::<Vec<_>>();

    f_i.into_iter()
        .enumerate()
        .fold(Integer::ZERO, |acc, (i, f_i)| acc + (f_i << (64 * i)))
}

/// Convert Integer to BigInt<4>
pub fn integer_to_bigint4(f: &Integer) -> BigInt<4> {
    // convert to BigInt<4>
    let array: [u64; 4] = {
        let mut padded = f.to_digits::<u64>(Order::LsfLe);
        padded.resize(4, 0);
        padded.try_into().unwrap()
    };
    BigInt::new(array)
}

/// This function computes a hint (x1, x2, z) for k1 = x1/z mod r and k2 = x2/z mod r
/// Adapted from: https://github.com/yelhousni/scalarmul-in-snark/blob/main/sage/decompose.py
pub(crate) fn msm_double_decompose(k1: Fr, k2: Fr) -> (Decomp, Decomp, Decomp) {
    // Convert to Integer
    let r_integer = bigint4_to_integer(&Fr::MODULUS);
    let k1_integer = bigint4_to_integer(&k1.into_bigint());
    let k2_integer = bigint4_to_integer(&k2.into_bigint());

    let basis = vec![
        vec![r_integer.clone(), Integer::from(0), Integer::from(0)],
        vec![Integer::from(0), r_integer.clone(), Integer::from(0)],
        vec![k1_integer.clone(), k2_integer.clone(), Integer::from(1)],
    ];
    let mut matrix = Matrix::from_matrix(basis.clone());

    // LLL
    lll_bignum(&mut matrix, 0.501, 0.99);

    // Check solution
    let bounded = Integer::from(1) << 155; // ~ 1.22r^{2/3}
    let mut sol = 0;
    let (mut x1, mut x2, mut z) = (
        matrix[sol].index(0),
        matrix[sol].index(1),
        matrix[sol].index(2),
    );
    while *z == Integer::ZERO {
        sol += 1;
        (x1, x2, z) = (
            matrix[sol].index(0),
            matrix[sol].index(1),
            matrix[sol].index(2),
        );
    }

    assert!(x1.clone().abs() < bounded);
    assert!(x2.clone().abs() < bounded);
    assert!(z.clone().abs() < bounded);

    let x1_neg = *x1 < Integer::ZERO;
    let x2_neg = *x2 < Integer::ZERO;
    let z_neg = *z < Integer::ZERO;

    // convert to Fr
    let z_fr = Fr::from(integer_to_bigint4(z));
    let x1_fr = Fr::from(integer_to_bigint4(x1));
    let x2_fr = Fr::from(integer_to_bigint4(x2));
    (
        Decomp(x1_fr, x1_neg),
        Decomp(x2_fr, x2_neg),
        Decomp(z_fr, z_neg),
    )
}

/// Get the bit_id-th most significant bit of scalar
pub(crate) fn msb_bit(scalar: &Fr, bit_id: usize) -> u8 {
    let big_int = scalar.into_bigint();
    let u64_contain_bits = big_int.0[3 - (bit_id / 64)];

    let bit_id = 63 - bit_id % 64;
    let bit = (u64_contain_bits >> bit_id) & 1;
    bit as u8
}

#[cfg(test)]
mod tests {
    use crate::curve::Fr;
    use ark_ff::{AdditiveGroup, BigInt};
    use ark_std::UniformRand;
    use ark_std::rand::thread_rng;

    #[test]
    fn test_msm_double_decompose() {
        let mut rng = thread_rng();
        let scalars: Vec<Fr> = (0..2000).map(|_| Fr::rand(&mut rng)).collect();
        for ks in scalars.chunks(2) {
            let k1 = ks[0];
            let k2 = ks[1];
            let (x1, x2, z) = super::msm_double_decompose(k1, k2);

            let x1 = if x1.1 { -x1.0 } else { x1.0 };
            let x2 = if x2.1 { -x2.0 } else { x2.0 };
            let z = if z.1 { -z.0 } else { z.0 };
            let result1 = k1 * z - x1;
            let result2 = k2 * z - x2;
            assert_eq!(result1, Fr::ZERO);
            assert_eq!(result2, Fr::ZERO);
        }
    }

    #[test]
    fn test_msb_bit() {
        let x = 0b1011u64; // 11 in decimal
        let fr = Fr::from(BigInt::new([x, 0, 0, x]));

        // msb
        assert_eq!(super::msb_bit(&fr, 252), 1);
        assert_eq!(super::msb_bit(&fr, 253), 0);
        assert_eq!(super::msb_bit(&fr, 254), 1);
        assert_eq!(super::msb_bit(&fr, 255), 1);
        assert_eq!(super::msb_bit(&fr, 60), 1);
        assert_eq!(super::msb_bit(&fr, 61), 0);
        assert_eq!(super::msb_bit(&fr, 62), 1);
        assert_eq!(super::msb_bit(&fr, 63), 1);
        for i in 64..252 {
            assert_eq!(super::msb_bit(&fr, i), 0);
        }
        for i in 0..60 {
            assert_eq!(super::msb_bit(&fr, i), 0);
        }
    }
}
