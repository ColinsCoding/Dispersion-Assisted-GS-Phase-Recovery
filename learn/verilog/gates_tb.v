// Testbench for gates.v — exhaustive truth table for all 2-input gates.
// Compile & run:
//   iverilog -o gates_tb gates_tb.v gates.v && vvp gates_tb

`timescale 1ns/1ps

module gates_tb;

    reg  a, b;
    wire y_and, y_or, y_xor, y_not_a, y_nand, y_nor, y_xnor;

    and_gate  u_and  (.a(a), .b(b), .y(y_and));
    or_gate   u_or   (.a(a), .b(b), .y(y_or));
    xor_gate  u_xor  (.a(a), .b(b), .y(y_xor));
    not_gate  u_not  (.a(a),        .y(y_not_a));
    nand_gate u_nand (.a(a), .b(b), .y(y_nand));
    nor_gate  u_nor  (.a(a), .b(b), .y(y_nor));
    xnor_gate u_xnor (.a(a), .b(b), .y(y_xnor));

    integer i;

    initial begin
        $display("a b | AND OR XOR NOT(a) NAND NOR XNOR");
        $display("----+--------------------------------------");
        for (i = 0; i < 4; i = i + 1) begin
            {a, b} = i[1:0];
            #1;
            $display("%b %b |  %b   %b   %b     %b     %b    %b    %b",
                a, b, y_and, y_or, y_xor, y_not_a, y_nand, y_nor, y_xnor);
        end
        $finish;
    end

endmodule
