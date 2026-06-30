// Half adder: 1-bit addition with no carry in.
//   sum   = a XOR b
//   carry = a AND b
//
// Truth table:
//   a b | sum carry
//   0 0 |  0    0
//   0 1 |  1    0
//   1 0 |  1    0
//   1 1 |  0    1

module half_adder (
    input  wire a,
    input  wire b,
    output wire sum,
    output wire carry
);
    assign sum   = a ^ b;
    assign carry = a & b;
endmodule
