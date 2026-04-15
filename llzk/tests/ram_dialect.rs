use llzk::{
    prelude::melior_dialects::arith,
    prelude::*,
};
use melior::ir::{Location, Type, r#type::FunctionType};

mod common;

#[test]
fn ram_load_store() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = llzk_module(location);
    let index_type = Type::index(&context);
    let element_type: Type = FeltType::new(&context).into();

    let f = dialect::function::def(
        location,
        "ram_test",
        FunctionType::new(&context, &[], &[]),
        &[],
        None,
    )
    .unwrap();
    f.set_allow_witness_attr(true);
    {
        let block = Block::new(&[]);

        // %addr = arith.constant 0 : index
        let addr_op = block.append_operation(arith::constant(
            &context,
            IntegerAttribute::new(index_type, 0).into(),
            location,
        ));
        let addr: Value = addr_op.result(0).unwrap().into();

        // %val = felt.const 42
        let val_op = block.append_operation(
            dialect::felt::constant(location, FeltConstAttribute::new(&context, 42, None))
                .expect("valid felt.const"),
        );
        let val: Value = val_op.result(0).unwrap().into();

        // ram.store %addr, %val : !felt.type
        block.append_operation(dialect::ram::store(location, addr, val));

        // %loaded = ram.load %addr : !felt.type
        let load_op = block.append_operation(dialect::ram::load(location, element_type, addr));
        assert!(dialect::ram::is_ram_load(&load_op));

        block.append_operation(dialect::function::r#return(location, &[]));
        f.region(0)
            .expect("function.def must have at least 1 region")
            .append_block(block);
    }

    let f = module.body().append_operation(f.into());
    assert!(f.verify());
    log::info!("Op passed verification");
}
