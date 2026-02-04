module attributes { llzk.lang } {
  function.def @recursive() -> !felt.type {
    %0 = function.call @recursive() : () -> !felt.type
    function.return %0 : !felt.type
  }
}
