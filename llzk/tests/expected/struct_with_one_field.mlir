module attributes { veridise.lang = "llzk" } {
  struct.def @one_field<[]> {
    struct.field @foo : index
    function.def @compute() -> !struct.type<@one_field<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@one_field<[]>>
      function.return %self : !struct.type<@one_field<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@one_field<[]>>) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      function.return
    }
  }
}
