use llzk::prelude::*;

mod common;

#[test]
fn append_sym_attr() {
    common::setup();
    let context = LlzkContext::new();
    let base = SymbolRefAttribute::new_from_str(&context, "R", &["A", "B", "C"]);
    let append = "D";

    let mut tail = base.nested();
    tail.push(FlatSymbolRefAttribute::new(&context, append).into());
    let result = SymbolRefAttribute::new(&context, base.root(), &tail);

    let ir = format!("{result}");
    let expected = "@R::@A::@B::@C::@D";
    assert_eq!(ir, expected);
}
