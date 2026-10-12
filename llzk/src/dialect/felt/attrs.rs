use super::FeltType;
use llzk_sys::{
    llzkAttributeIsA_Felt_FeltConstAttr, llzkAttributeIsA_Felt_FieldSpecAttr,
    llzkFelt_FeltConstAttrGetFromParts, llzkFelt_FeltConstAttrGetFromPartsUnspecified,
    llzkFelt_FeltConstAttrGetFromString, llzkFelt_FeltConstAttrGetFromStringUnspecified,
    llzkFelt_FeltConstAttrGetType, llzkFelt_FieldSpecAttrGetFromParts,
    llzkFelt_FieldSpecAttrGetFromString,
};
use melior::{
    Context, StringRef,
    ir::{Attribute, AttributeLike, Identifier, TypeLike},
};
use mlir_sys::MlirAttribute;

/// A constant finite field element.
#[derive(Clone, Copy)]
pub struct FeltConstAttribute<'c> {
    inner: Attribute<'c>,
}

impl<'c> FeltConstAttribute<'c> {
    /// # Safety
    /// The MLIR attribute must contain a valid pointer of type `FeltConstAttr`.
    pub unsafe fn from_raw(attr: MlirAttribute) -> Self {
        unsafe {
            Self {
                inner: Attribute::from_raw(attr),
            }
        }
    }

    /// Creates a [`FeltConstAttribute`] from an unsigned integer with an optional field
    /// specification. The full range of `u64` is supported.
    pub fn new(ctx: &'c Context, value: u64, field: Option<&str>) -> Self {
        // The extra zero limb keeps the value positive even when bit 63 is set.
        Self::from_parts(ctx, &[value, 0], field)
    }

    /// Creates a [`FeltConstAttribute`] from a signed base-10 integer of arbitrary precision
    /// with an optional field specification.
    ///
    /// # Panics
    ///
    /// If `value` is not a valid decimal integer.
    pub fn parse(ctx: &'c Context, value: &str, field: Option<&str>) -> Self {
        let value = StringRef::new(value);
        let attr = match field {
            Some(field) => unsafe {
                llzkFelt_FeltConstAttrGetFromString(
                    ctx.to_raw(),
                    value.to_raw(),
                    FeltType::with_field(ctx, field).to_raw(),
                )
            },
            None => unsafe {
                llzkFelt_FeltConstAttrGetFromStringUnspecified(ctx.to_raw(), value.to_raw())
            },
        };
        assert!(!attr.ptr.is_null(), "invalid decimal felt constant");
        unsafe { Self::from_raw(attr) }
    }

    /// Creates a [`FeltConstAttribute`] from signed two's-complement 64-bit parts in LSB
    /// order with an optional field specification. An empty slice represents zero.
    ///
    /// The highest bit of the last part is the sign bit. Append a zero part to represent
    /// an unsigned value whose highest bit is set.
    pub fn from_parts(ctx: &'c Context, parts: &[u64], field: Option<&str>) -> Self {
        match field {
            Some(field) => unsafe {
                Self::from_raw(llzkFelt_FeltConstAttrGetFromParts(
                    ctx.to_raw(),
                    parts.as_ptr(),
                    isize::try_from(parts.len()).expect("part count too large"),
                    FeltType::with_field(ctx, field).to_raw(),
                ))
            },
            None => unsafe {
                Self::from_raw(llzkFelt_FeltConstAttrGetFromPartsUnspecified(
                    ctx.to_raw(),
                    parts.as_ptr(),
                    isize::try_from(parts.len()).expect("part count too large"),
                ))
            },
        }
    }

    /// Creates a [`FeltConstAttribute`] with the optional field specification from a
    /// [`num_bigint::BigUint`].
    #[cfg(feature = "bigint")]
    pub fn from_biguint(
        ctx: &'c Context,
        value: &num_bigint::BigUint,
        field: Option<&str>,
    ) -> Self {
        let mut parts = value.to_u64_digits();
        // The C API interprets the highest bit as the sign bit.
        parts.push(0);
        Self::from_parts(ctx, &parts, field)
    }

    /// Returns the felt type of the attribute.
    pub fn r#type(&self) -> FeltType<'c> {
        unsafe { FeltType::from_raw(llzkFelt_FeltConstAttrGetType(self.to_raw())) }
    }
}

impl<'c> AttributeLike<'c> for FeltConstAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for FeltConstAttribute<'c> {
    type Error = melior::Error;

    fn try_from(t: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkAttributeIsA_Felt_FeltConstAttr(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::AttributeExpected("llzk felt", t.to_string()))
        }
    }
}

impl<'c> std::fmt::Debug for FeltConstAttribute<'c> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FeltConstAttribute(")?;
        std::fmt::Display::fmt(&self.inner, f)?;
        write!(f, ")")
    }
}

impl<'c> std::fmt::Display for FeltConstAttribute<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl<'c> From<FeltConstAttribute<'c>> for Attribute<'c> {
    fn from(attr: FeltConstAttribute<'c>) -> Attribute<'c> {
        attr.inner
    }
}

/// A specification of a prime field for use by felt types.
///
/// These specifications are provided in the `llzk.fields` attribute on the root
/// module as either a single element or a flat array, for example:
///
/// ```text
///    module attributes {llzk.lang, llzk.fields = field<foo, 7> { ... }
///    module attributes {llzk.lang, llzk.fields = [field<>]} { ... }
/// ```
///
/// Specifications should not be provided for built-in fields, which include:
///  - `babybear`
///  - `bn128/bn254`
///  - `goldilocks`
///  - `grumpkin`
///  - `koalabear`
///  - `mersenne31`
#[derive(Clone, Copy)]
pub struct FieldSpecAttribute<'c> {
    inner: Attribute<'c>,
}
impl<'c> FieldSpecAttribute<'c> {
    /// # Safety
    /// The MLIR attribute must contain a valid pointer of type `FieldSpecAttr`.
    pub unsafe fn from_raw(attr: MlirAttribute) -> Self {
        unsafe {
            Self {
                inner: Attribute::from_raw(attr),
            }
        }
    }

    /// Creates a `llzk::felt::FieldSpecAttr` from a base-10 representation of the prime.
    ///
    /// # Panics
    ///
    /// If `prime` is not a valid decimal integer or is less than two.
    pub fn new(ctx: &'c Context, name: &str, prime: &str) -> Self {
        let attr = unsafe {
            llzkFelt_FieldSpecAttrGetFromString(
                ctx.to_raw(),
                Identifier::new(ctx, name).to_raw(),
                StringRef::new(prime).to_raw(),
            )
        };
        assert!(!attr.ptr.is_null(), "invalid field modulus");
        unsafe { Self::from_raw(attr) }
    }

    /// Creates a `llzk::felt::FieldSpecAttr` from signed two's-complement 64-bit parts in
    /// LSB order representing the prime.
    ///
    /// The highest bit of the last part is the sign bit. Append a zero part to represent
    /// an unsigned value whose highest bit is set.
    ///
    /// # Panics
    ///
    /// If the represented modulus is less than two, including an empty slice (zero).
    pub fn from_parts(ctx: &'c Context, name: &str, parts: &[u64]) -> Self {
        let attr = unsafe {
            llzkFelt_FieldSpecAttrGetFromParts(
                ctx.to_raw(),
                Identifier::new(ctx, name).to_raw(),
                parts.as_ptr(),
                isize::try_from(parts.len()).expect("part count too large"),
            )
        };
        assert!(!attr.ptr.is_null(), "invalid field modulus");
        unsafe { Self::from_raw(attr) }
    }

    /// Creates a `llzk::felt::FieldSpecAttr` from a [`num_bigint::BigUint`].
    ///
    /// # Panics
    ///
    /// If the modulus is less than two.
    #[cfg(feature = "bigint")]
    pub fn from_biguint(ctx: &'c Context, name: &str, value: &num_bigint::BigUint) -> Self {
        let mut parts = value.to_u64_digits();
        // The C API interprets the highest bit as the sign bit.
        parts.push(0);
        Self::from_parts(ctx, name, &parts)
    }
}

impl<'c> AttributeLike<'c> for FieldSpecAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.inner.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for FieldSpecAttribute<'c> {
    type Error = melior::Error;

    fn try_from(t: Attribute<'c>) -> Result<Self, Self::Error> {
        if unsafe { llzkAttributeIsA_Felt_FieldSpecAttr(t.to_raw()) } {
            Ok(unsafe { Self::from_raw(t.to_raw()) })
        } else {
            Err(Self::Error::AttributeExpected("llzk felt", t.to_string()))
        }
    }
}

impl<'c> std::fmt::Debug for FieldSpecAttribute<'c> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FieldSpecAttribute(")?;
        std::fmt::Display::fmt(&self.inner, f)?;
        write!(f, ")")
    }
}

impl<'c> std::fmt::Display for FieldSpecAttribute<'c> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl<'c> From<FieldSpecAttribute<'c>> for Attribute<'c> {
    fn from(attr: FieldSpecAttribute<'c>) -> Attribute<'c> {
        attr.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use log::LevelFilter;
    use quickcheck::{Arbitrary, Gen};
    use quickcheck_macros::quickcheck;
    use simplelog::{Config, TestLogger};
    use std::{ops::Deref, ptr::null};

    #[derive(Clone, Debug)]
    struct FieldArg(Option<String>);

    impl Arbitrary for FieldArg {
        fn arbitrary(g: &mut Gen) -> Self {
            if bool::arbitrary(g) {
                Self(None)
            } else {
                Self(Some("mersenne31".to_string()))
            }
        }
    }

    impl Deref for FieldArg {
        type Target = Option<String>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    #[quickcheck]
    fn felt_const_attr_new(value: u64, field: FieldArg) {
        let _ = TestLogger::init(LevelFilter::Debug, Config::default());
        let ctx = LlzkContext::new();
        let f = FeltConstAttribute::new(&ctx, value, field.as_deref());
        assert_ne!(f.to_raw().ptr, null());
        assert_eq!(
            Attribute::from(f),
            Attribute::from(FeltConstAttribute::parse(
                &ctx,
                &value.to_string(),
                field.as_deref()
            ))
        );
    }

    #[quickcheck]
    fn felt_const_attr_conversion(value: u64, field: FieldArg) {
        let _ = TestLogger::init(LevelFilter::Debug, Config::default());
        let ctx = LlzkContext::new();
        let f = FeltConstAttribute::new(&ctx, value, field.as_deref());
        let attr: Attribute = f.into();
        let f: FeltConstAttribute = attr.try_into().unwrap();
        assert_ne!(f.to_raw().ptr, null());
    }

    #[test]
    fn felt_const_attr_fail() {
        let _ = TestLogger::init(LevelFilter::Debug, Config::default());
        let ctx = LlzkContext::new();
        let attrs = [
            Attribute::unit(&ctx),
            StringAttribute::new(&ctx, "string").into(),
            IntegerAttribute::new(IntegerType::new(&ctx, 32).into(), 1).into(),
        ];
        for attr in attrs {
            let f: Result<FeltConstAttribute, _> = attr.try_into();
            assert!(f.is_err());
        }
    }

    #[quickcheck]
    fn felt_const_attr_parse_from_u64(value: u64, field: FieldArg) {
        let _ = TestLogger::init(LevelFilter::Debug, Config::default());
        let ctx = LlzkContext::new();
        let f = FeltConstAttribute::parse(&ctx, &value.to_string(), field.as_deref());
        assert_ne!(f.to_raw().ptr, null());
    }

    #[test]
    fn signed_parts_preserve_values_and_field() {
        let ctx = LlzkContext::new();
        let cases: &[(&[u64], &str)] = &[
            (&[], "0"),
            (&[0, 0], "0"),
            (&[u64::MAX], "-1"),
            (&[1 << 63], "-9223372036854775808"),
            (&[u64::MAX, u64::MAX - 1], "-18446744073709551617"),
            (&[u64::MAX, 0], "18446744073709551615"),
            (
                &[u64::MAX, u64::MAX, 0],
                "340282366920938463463374607431768211455",
            ),
        ];
        for field in [None, Some("mersenne31")] {
            for &(parts, decimal) in cases {
                let attr = FeltConstAttribute::from_parts(&ctx, parts, field);
                let expected = FeltConstAttribute::parse(&ctx, decimal, field);
                assert_eq!(Attribute::from(attr), Attribute::from(expected));
                assert_eq!(attr.r#type(), expected.r#type());
            }
        }
    }

    #[test]
    fn context_helpers_preserve_unsigned_values() {
        let mut ctx = LlzkContext::new();
        ctx.set_field("mersenne31");
        let expected = FeltConstAttribute::parse(&ctx, "18446744073709551615", ctx.field());
        for attr in [
            ctx.felt_attr(u64::MAX),
            ctx.felt_attr_from_str("18446744073709551615"),
            ctx.felt_attr_from_parts(&[u64::MAX, 0]),
        ] {
            assert_eq!(Attribute::from(attr), Attribute::from(expected));
            assert_eq!(attr.r#type(), FeltType::with_field(&ctx, "mersenne31"));
        }
    }

    #[rstest::rstest]
    fn malformed_felt_decimal_panics(#[values("", "12x", "--1")] value: &str) {
        let ctx = LlzkContext::new();
        for field in [None, Some("mersenne31")] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    FeltConstAttribute::parse(&ctx, value, field)
                }))
                .is_err()
            );
        }
    }

    #[test]
    fn field_spec_preserves_large_prime() {
        let ctx = LlzkContext::new();
        // The goldilocks prime has its highest data bit set.
        let expected = FieldSpecAttribute::new(&ctx, "custom", "18446744069414584321");
        let attr = FieldSpecAttribute::from_parts(&ctx, "custom", &[0xffffffff00000001, 0]);
        assert_eq!(Attribute::from(attr), Attribute::from(expected));
    }

    #[rstest::rstest]
    #[should_panic(expected = "invalid field modulus")]
    fn field_spec_rejects_invalid_parts(#[values(&[], &[0], &[1], &[u64::MAX])] parts: &[u64]) {
        let ctx = LlzkContext::new();
        FieldSpecAttribute::from_parts(&ctx, "custom", parts);
    }

    #[rstest::rstest]
    #[should_panic(expected = "invalid field modulus")]
    fn field_spec_rejects_invalid_decimal(#[values("", "12x", "-1", "0", "1")] value: &str) {
        let ctx = LlzkContext::new();
        FieldSpecAttribute::new(&ctx, "custom", value);
    }

    #[cfg(feature = "bigint")]
    mod bigint {
        use crate::{
            context::LlzkContext,
            prelude::{Attribute, FeltConstAttribute, FieldSpecAttribute},
        };
        use num_bigint::BigUint;
        use rstest::rstest;
        use std::str::FromStr as _;

        #[rstest]
        fn felt_const_attr_new_from_bigint(
            #[values(BigUint::from(0u8), BigUint::from(1u8), BigUint::from(u64::MAX), BigUint::from(u128::MAX), BigUint::from_str("21888242871839275222246405745257275088548364400416034343698204186575808495616").unwrap())]
            value: BigUint,
        ) {
            use log::LevelFilter;
            use melior::ir::AttributeLike as _;
            use simplelog::{Config, TestLogger};
            use std::ptr::null;

            let _ = TestLogger::init(LevelFilter::Debug, Config::default());
            let ctx = LlzkContext::new();
            let f = FeltConstAttribute::from_biguint(&ctx, &value, None);
            assert_ne!(f.to_raw().ptr, null());
            assert_eq!(
                Attribute::from(f),
                Attribute::from(FeltConstAttribute::parse(&ctx, &value.to_string(), None))
            );
        }

        #[test]
        fn field_spec_from_biguint_preserves_positive_sign() {
            let ctx = LlzkContext::new();
            let value = BigUint::from(0xffffffff00000001u64);
            let attr = FieldSpecAttribute::from_biguint(&ctx, "custom", &value);
            let expected = FieldSpecAttribute::new(&ctx, "custom", &value.to_string());
            assert_eq!(Attribute::from(attr), Attribute::from(expected));
        }
    }
}
