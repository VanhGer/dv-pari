use std::ops::Index;
use ark_ff::{BigInt, PrimeField};
use lll_rs::l2::lll_bignum;
use lll_rs::Matrix;
use rug::Integer;
use rug::integer::Order;
use crate::curve::Fr;


// This function computes a hint (x, z) for a given scalar k such that k = x/z mod r
// Adapted from: https://github.com/yelhousni/scalarmul-in-snark/blob/main/sage/decompose.py
pub fn msm_simple_hint(k: Fr) -> (Fr, Fr) {
    // Convert to Integer
    let r_i = Fr::MODULUS.0.iter().map(|x| Integer::from(*x)).collect::<Vec<_>>();
    let k_i = k.into_bigint().0.iter().map(|x| Integer::from(*x)).collect::<Vec<_>>();

    let r_integer = r_i
        .into_iter()
        .enumerate()
        .fold(Integer::ZERO, |acc, (i, r_is)|
            acc + (r_is << (64 * i))
        );

    let k_integer = k_i
        .into_iter()
        .enumerate()
        .fold(Integer::ZERO, |acc, (i, k)|
            acc + (k << (64 * i))
        );

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

    // convert to Fr
    let z_array: [u64; 4] = {
        let mut padded = z.to_digits::<u64>(Order::LsfLe);
        padded.resize(4, 0);
        padded.try_into().unwrap()
    };

    let x_array: [u64; 4] = {
        let mut padded = x.to_digits::<u64>(Order::LsfLe);
        padded.resize(4, 0);
        padded.try_into().unwrap()
    };

    let z_fr = Fr::from(BigInt::new(z_array));
    let x_fr = Fr::from(BigInt::new(x_array));
    (x_fr, z_fr)
}


#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, BigInt};
    use crate::curve::Fr;

    #[test]
    fn test_msm_simple_hint() {
        let k  = {
            let big_int: Vec<u64> = vec![14367351680299713015, 11652453690653687348, 11898245835999324863, 355572606165];
            Fr::from(BigInt::new(big_int.try_into().unwrap()))
        };

        let (x, z) = super::msm_simple_hint(k);
        let result = k * z - x;
        assert_eq!(result, Fr::ZERO);
    }
}