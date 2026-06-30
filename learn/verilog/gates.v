// Basic combinational gates — structural Verilog (gate-level primitives).
// These are the atoms every digital circuit is built from.
// Simulate with: iverilog -o gates_tb gates_tb.v gates.v && vvp gates_tb

// AND gate: y = a & b
module and_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = a & b;
endmodule

// OR gate: y = a | b
module or_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = a | b;
endmodule

// XOR gate: y = a ^ b  (exclusive or — odd parity of inputs)
module xor_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = a ^ b;
endmodule

// NOT gate: y = ~a
module not_gate (
    input  wire a,
    output wire y
);
    assign y = ~a;
endmodule

// NAND gate: y = ~(a & b)
module nand_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = ~(a & b);
endmodule

// NOR gate: y = ~(a | b)
module nor_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = ~(a | b);
endmodule

// XNOR gate: y = ~(a ^ b)  (equivalence — 1 when inputs match)
module xnor_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = ~(a ^ b);
endmodule

// N-input AND using a generate loop (parameterised width)
module and_n #(parameter N = 4) (
    input  wire [N-1:0] in,
    output wire         y
);
    assign y = &in;   // Verilog reduction AND
endmodule

// N-input OR
module or_n #(parameter N = 4) (
    input  wire [N-1:0] in,
    output wire         y
);
    assign y = |in;   // reduction OR
endmodule

// N-input XOR (odd-parity checker)
module xor_n #(parameter N = 4) (
    input  wire [N-1:0] in,
    output wire         y
);
    assign y = ^in;   // reduction XOR
endmodule
