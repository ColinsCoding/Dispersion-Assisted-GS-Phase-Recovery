// Full adder: 1-bit addition with carry in.
//   sum   = a XOR b XOR cin
//   cout  = (a AND b) OR (cin AND (a XOR b))
//
// Built from two half adders — same structure as digital_logic.py full_adder().

module full_adder (
    input  wire a,
    input  wire b,
    input  wire cin,
    output wire sum,
    output wire cout
);
    wire s1, c1, c2;

    half_adder ha1 (.a(a),  .b(b),   .sum(s1), .carry(c1));
    half_adder ha2 (.a(s1), .b(cin), .sum(sum), .carry(c2));

    assign cout = c1 | c2;
endmodule


// N-bit ripple-carry adder assembled with a generate loop.
// The carry ripples from bit 0 to bit N-1 — latency grows with N.
module ripple_adder #(parameter N = 4) (
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    input  wire         cin,
    output wire [N-1:0] sum,
    output wire         cout
);
    wire [N:0] carry;
    assign carry[0] = cin;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : fa_chain
            full_adder fa (
                .a(a[i]), .b(b[i]), .cin(carry[i]),
                .sum(sum[i]), .cout(carry[i+1])
            );
        end
    endgenerate

    assign cout = carry[N];
endmodule
