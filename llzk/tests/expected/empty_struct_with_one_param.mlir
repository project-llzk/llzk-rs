module attributes { llzk.lang } {
  poly.template @tmpl {
    poly.param @T

    struct.def @empty {
      function.def @compute() -> !struct.type<@tmpl::@empty<[@T]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
        %self = struct.new : <@tmpl::@empty<[@T]>>
        function.return %self : !struct.type<@tmpl::@empty<[@T]>>
      }
      function.def @constrain(%arg0: !struct.type<@tmpl::@empty<[@T]>>) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
        function.return
      }
    }
  }
}
