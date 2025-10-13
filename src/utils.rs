use std::ops::Index;
use ark_ff::{BigInt, PrimeField};
use lll_rs::l2::lll_bignum;
use lll_rs::Matrix;
use rug::Integer;
use rug::integer::Order;
use crate::curve::Fr;


// Convert BigInt<4> to Integer
pub fn bigint4_to_integer(f: &BigInt<4>) -> Integer {
    let f_i = f.0.iter().map(|x| Integer::from(*x)).collect::<Vec<_>>();

    f_i
        .into_iter()
        .enumerate()
        .fold(Integer::ZERO, |acc, (i, f_i)|
            acc + (f_i << (64 * i))
        )
}

// Convert Integer to BigInt<4>
pub fn integer_to_bigint4(f: &Integer) -> BigInt<4> {
    // convert to BigInt<4>
    let array: [u64; 4] = {
        let mut padded = f.to_digits::<u64>(Order::LsfLe);
        padded.resize(4, 0);
        padded.try_into().unwrap()
    };
    BigInt::new(array)
}

// This function computes a hint (x, z) for a given scalar k such that k = x/z mod r
// Adapted from: https://github.com/yelhousni/scalarmul-in-snark/blob/main/sage/decompose.py
pub(crate) fn msm_simple_hint(k: Fr) -> (Fr, Fr, bool, bool) {
    // Convert to Integer
    let r_integer = bigint4_to_integer(&Fr::MODULUS);
    let k_integer = bigint4_to_integer(&k.into_bigint());

    let mut basis = vec![
        vec![r_integer.clone(), Integer::from(0)],
        vec![Integer::from(0), r_integer.clone()],
        vec![ k_integer.clone(), Integer::from(1)],
    ];
    let mut matrix = Matrix::from_matrix(basis.clone());

    // LLL
    lll_bignum(&mut matrix, 0.501, 0.99);

    // Check solution
    let threshold = Integer::from(1) << 116;
    let mut sol = 0;
    let (mut x, mut z) = (matrix[sol].index(0), matrix[sol].index(1));
    while *z == Integer::ZERO {
        sol += 1;
        (x, z) = (matrix[sol].index(0), matrix[sol].index(1));
    }

    assert!(x.clone().abs() < threshold);
    assert!(z.clone().abs() < threshold);

    let x_neg = x < &Integer::ZERO;
    let z_neg = z < &Integer::ZERO;

    // convert to Fr
    let z_fr = Fr::from(integer_to_bigint4(z));
    let x_fr = Fr::from(integer_to_bigint4(x));
    (x_fr, z_fr, x_neg, z_neg)
}

pub(crate) fn msb_bit(scalar: &Fr, bit_id: usize) -> u8 {
    let big_int = scalar.into_bigint();
    let u64_contain_bits = big_int.0[3 - (bit_id / 64)];

    let bit_id = 63 - bit_id % 64;
    let bit = (u64_contain_bits >> bit_id) & 1;
    bit as u8
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, BigInt};
    use ark_std::rand::thread_rng;
    use ark_std::UniformRand;
    use crate::curve::Fr;

    #[test]
    fn test_msm_simple_hint() {
        let mut rng = thread_rng();
        let scalars: Vec<Fr> = (0..20).map(|_| Fr::rand(&mut rng)).collect();
        for k in scalars {
            let (x, z, x_neg, z_neg) = super::msm_simple_hint(k);
            let x = if x_neg { -x } else { x };
            let z = if z_neg { -z } else { z };
            let result = k * z - x;
            assert_eq!(result, Fr::ZERO);
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