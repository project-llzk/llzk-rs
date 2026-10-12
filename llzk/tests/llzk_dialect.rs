#![allow(unused_crate_dependencies)]
//! Integration tests for the llzk dialect.

use llzk::{dialect::llzk::*, prelude::*};

mod common;

#[test]
fn create_pub_attr() {
    common::setup();
    let context = LlzkContext::new();
    let a = PublicAttribute::new(&context);

    let ir = format!("{a}");
    let expected = "#llzk.pub";
    assert_eq!(ir, expected);
}

#[test]
fn create_loop_bounds_attr() {
    common::setup();
    let context = LlzkContext::new();
    let a = LoopBoundsAttribute::new(&context, 0, 10, 1);

    let ir = format!("{a}");
    let expected = "#llzk.loopbounds<0 to 10 step 1>";
    assert_eq!(ir, expected);
}

#[test]
fn loop_bounds_preserve_signed_and_large_values() {
    let context = LlzkContext::new();
    let a = LoopBoundsAttribute::new(&context, i64::MAX, i64::MIN, -1);
    assert_eq!(
        a.to_string(),
        "#llzk.loopbounds<9223372036854775807 to -9223372036854775808 step -1>"
    );
    let a = LoopBoundsAttribute::parse(
        &context,
        "-18446744073709551617",
        "340282366920938463463374607431768211455",
        "18446744073709551616",
    );
    assert_eq!(
        a.to_string(),
        "#llzk.loopbounds<-18446744073709551617 to 340282366920938463463374607431768211455 step 18446744073709551616>"
    );
}

#[test]
#[should_panic(expected = "invalid decimal loop bounds")]
fn loop_bounds_reject_malformed_input() {
    let context = LlzkContext::new();
    LoopBoundsAttribute::parse(&context, "0", "12x", "1");
}
