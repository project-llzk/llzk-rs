!F = !felt.type<"koalabear">
module attributes {llzk.lang} {
  // Input struct is defined in LLZK since we can't write pcl by hand from the rust bindings.
  struct.def @Test {
    struct.member @out : !F
    function.def @compute(%arg0: !F) -> !struct.type<@Test> {
      %self = struct.new : <@Test>
      function.return %self : !struct.type<@Test>
    }
    function.def @constrain(%self: !struct.type<@Test>, %arg0: !F) {
      %0 = struct.readm %self[@out] : <@Test>, !F
      constrain.eq %arg0, %0 : !F, !F
      function.return
    }
  }
}
