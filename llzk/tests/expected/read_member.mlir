module attributes {llzk.lang} {
  struct.def @read_member {
    struct.member @foo : index
    function.def @compute() -> !struct.type<@read_member<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@read_member<[]>>
      function.return %self : !struct.type<@read_member<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@read_member<[]>>) attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      %0 = struct.readm %arg0[@foo] : <@read_member<[]>>, index
      function.return
    }
  }
}
