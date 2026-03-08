#[cfg(feature = "std")]
use std::ops::{Add, AddAssign, Div, Mul, Sub};

#[cfg(not(feature = "std"))]
use core::ops::{Add, AddAssign, Div, Mul, Sub};

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

// GFNI multiplies bytes in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1 (0x11B).
// RFC 6330 uses a different field, so the whole codec needs to move to this field in order for
// encoder and decoder to stay algebraically consistent.
const FIELD_POLYNOMIAL: u16 = 0x11B;
const FIELD_REDUCTION: u8 = (FIELD_POLYNOMIAL & 0xFF) as u8;
// In the GFNI field, 0x02 has multiplicative order 51, so alpha must use a different primitive
// element. 0x03 generates all 255 non-zero elements.
const FIELD_GENERATOR: u8 = 0x03;

const fn field_mul(mut x: u8, mut y: u8) -> u8 {
    let mut result = 0;
    let mut i = 0;
    while i < 8 {
        if y & 1 != 0 {
            result ^= x;
        }

        let carry = x & 0x80;
        x <<= 1;
        if carry != 0 {
            x ^= FIELD_REDUCTION;
        }

        y >>= 1;
        i += 1;
    }

    result
}

const fn calculate_octet_exp_table() -> [u8; 510] {
    let mut result = [0; 510];
    let mut value = 1;
    let mut i = 0;

    while i < 255 {
        result[i] = value;
        value = field_mul(value, FIELD_GENERATOR);
        i += 1;
    }

    while i < result.len() {
        result[i] = result[i - 255];
        i += 1;
    }

    result
}

const fn calculate_octet_log_table() -> [u8; 256] {
    let mut result = [0; 256];
    let mut value = 1;
    let mut i = 0;

    while i < 255 {
        result[value as usize] = i as u8;
        value = field_mul(value, FIELD_GENERATOR);
        i += 1;
    }

    result
}

const OCT_EXP: [u8; 510] = calculate_octet_exp_table();
const OCT_LOG: [u8; 256] = calculate_octet_log_table();

pub static OCTET_MUL: [[u8; 256]; 256] = calculate_octet_mul_table();

// See "Screaming Fast Galois Field Arithmetic Using Intel SIMD Instructions" by Plank et al.
// Further adapted to AVX2.
#[cfg(any(feature = "std", test))]
pub const OCTET_MUL_HI_BITS: [[u8; 32]; 256] = calculate_octet_mul_hi_table();
#[cfg(any(feature = "std", test))]
pub const OCTET_MUL_LOW_BITS: [[u8; 32]; 256] = calculate_octet_mul_low_table();

const fn const_mul(x: usize, y: usize) -> u8 {
    field_mul(x as u8, y as u8)
}

#[cfg(any(feature = "std", test))]
const fn calculate_octet_mul_hi_table() -> [[u8; 32]; 256] {
    let mut result = [[0; 32]; 256];
    let mut i = 1;
    while i < 256 {
        let mut j = 1;
        while j < 16 {
            result[i][j] = const_mul(i, j << 4);
            result[i][j + 16] = const_mul(i, j << 4);
            j += 1;
        }
        i += 1;
    }
    result
}

#[cfg(any(feature = "std", test))]
const fn calculate_octet_mul_low_table() -> [[u8; 32]; 256] {
    let mut result = [[0; 32]; 256];
    let mut i = 1;
    while i < 256 {
        let mut j = 1;
        while j < 16 {
            result[i][j] = const_mul(i, j);
            result[i][j + 16] = const_mul(i, j);
            j += 1;
        }
        i += 1;
    }
    result
}

const fn calculate_octet_mul_table() -> [[u8; 256]; 256] {
    let mut result = [[0; 256]; 256];
    let mut i = 1;
    while i < 256 {
        let mut j = 1;
        while j < 256 {
            result[i][j] = const_mul(i, j);
            j += 1;
        }
        i += 1;
    }
    result
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Octet {
    value: u8,
}

impl Octet {
    pub fn new(value: u8) -> Octet {
        Octet { value }
    }

    pub fn zero() -> Octet {
        Octet { value: 0 }
    }

    pub fn one() -> Octet {
        Octet { value: 1 }
    }

    pub fn alpha(i: usize) -> Octet {
        assert!(i < 256);
        Octet { value: OCT_EXP[i] }
    }

    pub fn byte(&self) -> u8 {
        self.value
    }

    pub fn fma(&mut self, other1: &Octet, other2: &Octet) {
        if other1.value != 0 && other2.value != 0 {
            unsafe {
                // This is safe because the values are u8s and the exp/log tables cover every
                // non-zero field element in the GFNI field.
                let log_u = *OCT_LOG.get_unchecked(other1.value as usize) as usize;
                let log_v = *OCT_LOG.get_unchecked(other2.value as usize) as usize;
                self.value ^= *OCT_EXP.get_unchecked(log_u + log_v)
            }
        }
    }
}

impl Add for Octet {
    type Output = Octet;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: Octet) -> Octet {
        Octet {
            value: self.value ^ other.value,
        }
    }
}

impl<'b> Add<&'b Octet> for &Octet {
    type Output = Octet;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: &'b Octet) -> Octet {
        Octet {
            value: self.value ^ other.value,
        }
    }
}

impl AddAssign for Octet {
    #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, other: Octet) {
        self.value ^= other.value;
    }
}

impl<'a> AddAssign<&'a Octet> for Octet {
    #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, other: &'a Octet) {
        self.value ^= other.value;
    }
}

impl Sub for Octet {
    type Output = Octet;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Octet) -> Octet {
        Octet {
            value: self.value ^ rhs.value,
        }
    }
}

impl Mul for Octet {
    type Output = Octet;

    fn mul(self, other: Octet) -> Octet {
        &self * &other
    }
}

impl<'b> Mul<&'b Octet> for &Octet {
    type Output = Octet;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: &'b Octet) -> Octet {
        if self.value == 0 || other.value == 0 {
            Octet { value: 0 }
        } else {
            unsafe {
                let log_u = *OCT_LOG.get_unchecked(self.value as usize) as usize;
                let log_v = *OCT_LOG.get_unchecked(other.value as usize) as usize;
                Octet {
                    value: *OCT_EXP.get_unchecked(log_u + log_v),
                }
            }
        }
    }
}

impl Div for Octet {
    type Output = Octet;

    fn div(self, rhs: Octet) -> Octet {
        &self / &rhs
    }
}

impl<'b> Div<&'b Octet> for &Octet {
    type Output = Octet;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'b Octet) -> Octet {
        assert_ne!(0, rhs.value);
        if self.value == 0 {
            Octet { value: 0 }
        } else {
            let log_u = OCT_LOG[self.value as usize] as usize;
            let log_v = OCT_LOG[rhs.value as usize] as usize;
            Octet {
                value: OCT_EXP[255 + log_u - log_v],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::octet::FIELD_GENERATOR;
    use crate::octet::OCT_EXP;
    use crate::octet::OCT_LOG;
    use crate::octet::OCTET_MUL_HI_BITS;
    use crate::octet::OCTET_MUL_LOW_BITS;
    use crate::octet::Octet;

    #[test]
    fn multiplication_tables() {
        for i in 0..=255 {
            for j in 0..=255 {
                let expected = Octet::new(i) * Octet::new(j);
                let low = OCTET_MUL_LOW_BITS[i as usize][(j & 0x0F) as usize];
                let hi = OCTET_MUL_HI_BITS[i as usize][((j & 0xF0) >> 4) as usize];
                assert_eq!(low ^ hi, expected.byte());
            }
        }
    }

    #[test]
    fn addition() {
        let octet = Octet {
            value: rand::rng().random(),
        };
        assert_eq!(Octet::zero(), &octet + &octet);
    }

    #[test]
    fn multiplication_identity() {
        let octet = Octet {
            value: rand::rng().random(),
        };
        assert_eq!(octet, &octet * &Octet::one());
    }

    #[test]
    fn multiplicative_inverse() {
        let octet = Octet {
            value: rand::rng().random_range(1..255),
        };
        let one = Octet::one();
        assert_eq!(one, &octet * &(&one / &octet));
    }

    #[test]
    fn division() {
        let octet = Octet {
            value: rand::rng().random_range(1..255),
        };
        assert_eq!(Octet::one(), &octet / &octet);
    }

    #[test]
    fn unsafe_mul_gaurantees() {
        let max_value = *OCT_LOG.iter().max().unwrap() as usize;
        assert!(2 * max_value < OCT_EXP.len());
    }

    #[test]
    fn fma() {
        let mut result = Octet::zero();
        let mut fma_result = Octet::zero();
        for i in 0..255 {
            for j in 0..255 {
                result += Octet::new(i) * Octet::new(j);
                fma_result.fma(&Octet::new(i), &Octet::new(j));
                assert_eq!(result, fma_result);
            }
        }
    }

    #[test]
    fn aes_field_example() {
        assert_eq!(0xFE, (Octet::new(0x57) * Octet::new(0x13)).byte());
    }

    #[test]
    fn alpha_uses_full_order_generator() {
        let mut seen = [false; 256];
        for i in 0..255 {
            let value = Octet::alpha(i).byte();
            assert_ne!(0, value);
            assert!(
                !seen[value as usize],
                "duplicate alpha^{} for generator {FIELD_GENERATOR:#x}",
                i
            );
            seen[value as usize] = true;
        }
        assert_eq!(1, Octet::alpha(255).byte());
    }
}
