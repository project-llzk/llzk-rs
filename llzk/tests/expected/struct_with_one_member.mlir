module attributes { llzk.lang } {
  struct.def @one_member<[]> {
    struct.member @foo : index
    function.def @compute() -> !struct.type<@one_member<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@one_member<[]>>
      function.return %self : !struct.type<@one_member<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@one_member<[]>>) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      function.return
    }
  }
}
