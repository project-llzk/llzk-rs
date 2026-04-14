use llzk::dialect::cast::*;
use llzk::prelude::melior_dialects::arith;
use llzk::prelude::*;

mod common;

#[test]
fn tofelt_unspecified() {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let index_ty = Type::index(&ctx);

    let c = arith::constant(&ctx, IntegerAttribute::new(index_ty, 0).into(), loc);
    let a = tofelt(loc, c.result(0).unwrap().into(), None);

    let ir = format!("{a}");
    let expected = "%0 = cast.tofelt <<UNKNOWN SSA VALUE>> : index, !felt.type\n";
    assert_eq!(ir, expected);
}

#[test]
fn tofelt_specified() {
    common::setup();
    let ctx = LlzkContext::new();
    let loc = Location::unknown(&ctx);
    let index_ty = Type::index(&ctx);
    let felt_ty = FeltType::with_field(&ctx, "babybear");

    let c = arith::constant(&ctx, IntegerAttribute::new(index_ty, 0).into(), loc);
    let a = tofelt(loc, c.result(0).unwrap().into(), Some(felt_ty));

    let ir = format!("{a}");
    let expected = "%0 = cast.tofelt <<UNKNOWN SSA VALUE>> : index, !felt.type<\"babybear\">\n";
    assert_eq!(ir, expected);
}
