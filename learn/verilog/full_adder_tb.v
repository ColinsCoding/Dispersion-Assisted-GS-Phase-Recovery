// Testbench: 4-bit ripple adder — verify against $unsigned arithmetic.
// iverilog -o fa_tb full_adder_tb.v full_adder.v half_adder.v && vvp fa_tb

`timescale 1ns/1ps

module full_adder_tb;

    reg  [3:0] a, b;
    reg        cin;
    wire [3:0] sum;
    wire       cout;

    ripple_adder #(.N(4)) dut (
        .a(a), .b(b), .cin(cin), .sum(sum), .cout(cout)
    );

    integer i, j, errors;
    reg [4:0] expected;

    initial begin
        errors = 0;
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                a = i[3:0]; b = j[3:0]; cin = 0; #1;
                expected = a + b;
                if ({cout, sum} !== expected) begin
                    $display("FAIL: %0d + %0d = got %0d, expected %0d",
                             a, b, {cout,sum}, expected);
                    errors = errors + 1;
                end
            end
        end
        if (errors == 0)
            $display("All 256 additions passed.");
        else
            $display("%0d error(s).", errors);
        $finish;
    end

endmodule
