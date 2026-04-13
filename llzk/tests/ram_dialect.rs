use llzk::{
    prelude::melior_dialects::arith,
    prelude::*,
};
use melior::ir::{Location, Type, r#type::FunctionType};

mod common;

#[test]
fn ram_alloc_load_store() {
    common::setup();
    let context = LlzkContext::new();
    let location = Location::unknown(&context);
    let module = llzk_module(location);
    let index_type = Type::index(&context);

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
        // %size = arith.constant 1024 : index
        let size_op = block.append_operation(arith::constant(
            &context,
            IntegerAttribute::new(index_type, 1024).into(),
            location,
        ));
        let size: Value = size_op.result(0).unwrap().into();

        // %mem = ram.alloc %size : index
        let alloc_op = block.append_operation(dialect::ram::alloc(location, size));
        let mem: Value = alloc_op.result(0).unwrap().into();

        // %addr = arith.constant 0 : index
        let addr_op = block.append_operation(arith::constant(
            &context,
            IntegerAttribute::new(index_type, 0).into(),
            location,
        ));
        let addr: Value = addr_op.result(0).unwrap().into();

        // %val = arith.constant 42 : index
        let val_op = block.append_operation(arith::constant(
            &context,
            IntegerAttribute::new(index_type, 42).into(),
            location,
        ));
        let val: Value = val_op.result(0).unwrap().into();

        // ram.store %mem[%addr] = %val : index, index
        block.append_operation(dialect::ram::store(location, mem, addr, val));

        // %loaded = ram.load %mem[%addr] : index, index
        let load_op = block.append_operation(dialect::ram::load(location, index_type, mem, addr));
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
