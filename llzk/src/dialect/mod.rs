//! APIs for the different dialects available in LLZK.

use crate::{builder::OpBuilderLike, error::Error};

pub mod array;
pub mod bool;
pub mod cast;
pub mod constrain;
pub mod felt;
pub mod function;
pub mod global;
pub mod llzk;
pub mod pod;
pub mod poly;
pub mod ram;
pub mod r#struct;
pub mod verif;

/// A no-op callback for region-building functions.
///
/// Use this as the `fill` callback when creating an operation with an empty
/// region so body contents can be added later.
pub fn empty_region<'c>(_: &impl OpBuilderLike<'c>) -> Result<(), Error> {
    Ok(())
}

/// Functions for working with `builtin.module` in LLZK.
pub mod module {
    use std::{
        ffi::CStr,
        io::{self, Write},
        os::raw::c_void,
    };

    use llzk_sys::{LANG_ATTR_NAME, LLZK_FIELD_ATTR_NAME, LLZK_LANG_ATTR_NAME, MAIN_ATTR_NAME};
    use melior::{
        Context,
        ir::{
            Location, Module,
            attribute::{Attribute, StringAttribute, TypeAttribute},
            operation::{OperationLike, OperationMutLike as _, OperationRefMut},
        },
    };
    use mlir_sys::{MlirModule, MlirStringRef, mlirModuleGetOperation, mlirOperationWriteBytecode};

    use crate::{
        attributes::array::ArrayAttribute,
        prelude::{FieldSpecAttribute, StructType},
    };

    /// Creates a new `builtin.module` operation preconfigured to meet LLZK's specifications.
    #[deprecated(note = "Use LlzkModuleBuilder::create() instead")]
    pub fn llzk_module<'c>(location: Location<'c>, lang: Option<&str>) -> Module<'c> {
        let mut module = Module::new(location);
        let mut op = module.as_operation_mut();
        let ctx = location.context();
        let attr_name = unsafe { CStr::from_ptr(LLZK_LANG_ATTR_NAME) }
            .to_str()
            .unwrap();
        let attr_value = lang.map_or_else(
            || Attribute::unit(unsafe { ctx.to_ref() }),
            |s| StringAttribute::new(unsafe { ctx.to_ref() }, s).into(),
        );
        op.set_attribute(attr_name, attr_value);
        module
    }

    /// Builder for creating modules that satisfy LLZK's requirements.
    ///
    /// By default the location is set to unknown, the lang property is set to an empty attribute,
    /// and none of the other properties are set.
    ///
    /// You can use the `create` method for quickly creating modules.
    #[derive(Debug)]
    pub struct LlzkModuleBuilder<'c, 'l> {
        location: Location<'c>,
        lang: Option<&'l str>,
        main: Option<StructType<'c>>,
        fields: Vec<FieldSpecAttribute<'c>>,
    }

    impl<'c, 'l> LlzkModuleBuilder<'c, 'l> {
        /// Creates a new builder.
        pub fn new(context: &'c Context) -> Self {
            Self {
                location: Location::unknown(context),
                lang: None,
                main: None,
                fields: Default::default(),
            }
        }

        /// Creates a new `builtin.module` operation preconfigured to meet LLZK's specifications.
        pub fn create(location: Location<'c>, lang: Option<&str>) -> Module<'c> {
            let ctx = location.context();
            let mut builder = Self::new(unsafe { ctx.to_ref() });
            if let Some(lang) = lang {
                builder.with_language(lang);
            }
            builder.with_location(location).build()
        }

        /// Creates the module.
        pub fn build(&mut self) -> Module<'c> {
            let mut module = Module::new(self.location);
            let mut op = module.as_operation_mut();
            let ctx = self.location.context();
            let attr_value = self.lang.map_or_else(
                || Attribute::unit(unsafe { ctx.to_ref() }),
                |s| StringAttribute::new(unsafe { ctx.to_ref() }, s).into(),
            );
            op.set_attribute(LANG_ATTR_NAME.as_ref(), attr_value);
            if let Some(main) = self.main {
                module.add_main(main);
            }
            for field in &self.fields {
                module.add_field_spec(*field);
            }
            module
        }

        /// Sets the location.
        pub fn with_location(&mut self, location: Location<'c>) -> &mut Self {
            self.location = location;
            self
        }

        /// Sets the location to unknown.
        pub fn no_location(&mut self) -> &mut Self {
            let ctx = self.location.context();
            self.location = Location::unknown(unsafe { ctx.to_ref() });
            self
        }

        /// Sets the main struct of the module.
        pub fn with_main(&mut self, main: StructType<'c>) -> &mut Self {
            self.main = Some(main);
            self
        }

        /// Removes the main struct annotation.
        pub fn no_main(&mut self) -> &mut Self {
            self.main = None;
            self
        }

        /// Sets the name of the source language of the module.
        pub fn with_language(&mut self, lang: &'l str) -> &mut Self {
            self.lang = Some(lang);
            self
        }

        /// Removes the source language annotation.
        pub fn no_language(&mut self) -> &mut Self {
            self.lang = None;
            self
        }

        /// Adds a field spec attribute to the module.
        pub fn add_field_spec(&mut self, spec: FieldSpecAttribute<'c>) -> &mut Self {
            self.add_field_specs(std::iter::once(spec))
        }

        /// Adds field spec attributes to the module.
        pub fn add_field_specs(
            &mut self,
            specs: impl IntoIterator<Item = FieldSpecAttribute<'c>>,
        ) -> &mut Self {
            self.fields.extend(specs);
            self
        }

        /// Removes all field specs.
        pub fn clear_field_specs(&mut self) -> &mut Self {
            self.fields.clear();
            self
        }
    }

    /// Extension methods for [`Module`].
    pub trait ModuleExt<'c> {
        /// Return the raw representation of the module.
        fn to_raw(&self) -> MlirModule;

        /// Dump the module's bytecode representation.
        fn write_bytecode(&self, dest: &mut dyn Write) -> std::io::Result<()> {
            struct Wrap<'w>(&'w mut dyn Write, io::Result<()>);

            unsafe extern "C" fn callback(s: MlirStringRef, user_data: *mut c_void) {
                let wrap = unsafe { &mut *(user_data as *mut Wrap) };
                if wrap.1.is_err() {
                    return;
                }
                let buf = unsafe { std::slice::from_raw_parts(s.data as *const u8, s.length) };
                wrap.1 = wrap.0.write_all(buf);
            }

            let mut wrap = Wrap(dest, Ok(()));

            unsafe {
                let op = mlirModuleGetOperation(self.to_raw());
                mlirOperationWriteBytecode(
                    op,
                    Some(callback),
                    &mut wrap as *mut Wrap as *mut c_void,
                );
            }

            wrap.1
        }

        /// Sets the main struct attribute.
        fn add_main(&mut self, main: StructType<'c>) {
            let mut op = unsafe {
                let op = mlirModuleGetOperation(self.to_raw());
                OperationRefMut::from_raw(op)
            };
            op.set_attribute(
                MAIN_ATTR_NAME.as_ref(),
                TypeAttribute::new(main.into()).into(),
            );
        }

        /// Adds the spec attribute to the module, creating the `llzk.fields` attribute if
        /// necessary.
        ///
        /// # Panics
        ///
        /// If the existing `llzk.fields` is not an array attribute.
        fn add_field_spec(&mut self, spec: FieldSpecAttribute<'c>) {
            let mut op = unsafe {
                let op = mlirModuleGetOperation(self.to_raw());
                OperationRefMut::from_raw(op)
            };
            let attr_name = unsafe { CStr::from_ptr(LLZK_FIELD_ATTR_NAME) }
                .to_str()
                .unwrap();
            let elts = if op.has_attribute(attr_name) {
                let array = ArrayAttribute::try_from(op.attribute(attr_name).unwrap()).unwrap();
                array
                    .into_iter()
                    .chain(std::iter::once(spec.into()))
                    .collect::<Vec<_>>()
            } else {
                vec![spec.into()]
            };
            let context = op.context();
            op.set_attribute(
                attr_name,
                ArrayAttribute::new(unsafe { context.to_ref() }, &elts).into(),
            );
        }
    }

    impl<'c> ModuleExt<'c> for Module<'c> {
        fn to_raw(&self) -> MlirModule {
            self.to_raw()
        }
    }
}

/// Extensions for the 'scf' dialect.
pub mod scf_ext {
    crate::macros::isa_fn!(scf, if);
    crate::macros::isa_fn!(scf, yield);
    crate::macros::isa_fn!(scf, condition);
    crate::macros::isa_fn!(scf, for);
    crate::macros::isa_fn!(scf, while);
}
