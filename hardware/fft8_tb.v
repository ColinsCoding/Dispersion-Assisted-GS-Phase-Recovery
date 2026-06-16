// fft8_tb.v -- testbench for the 8-point fixed-point FFT.
// Feeds an impulse (FFT = flat spectrum) and a 1-cycle cosine (FFT = spikes at
// k=1,7), printing outputs for comparison with numpy. Inputs scaled by 4096.
`timescale 1ns/1ps
module fft8_tb;
    localparam W = 32, S = 12, SCALE = 4096;
    reg  signed [W-1:0] xr [0:7], xi [0:7];
    wire signed [W-1:0] Xr [0:7], Xi [0:7];
    integer k;

    fft8 #(.W(W), .S(S)) dut (.xr(xr), .xi(xi), .Xr(Xr), .Xi(Xi));

    task show; begin
        for (k = 0; k < 8; k = k + 1)
            $display("  X[%0d] = %0d %+0dj  (/%0d)", k, Xr[k], Xi[k], SCALE);
    end endtask

    initial begin
        // test 1: impulse x=[1,0,...] -> FFT all ones (= SCALE)
        for (k = 0; k < 8; k = k + 1) begin xr[k]=0; xi[k]=0; end
        xr[0] = SCALE;
        #1 $display("TEST 1 impulse (expect all %0d + 0j):", SCALE); show;

        // test 2: cosine cos(2*pi*n/8) -> spikes at k=1 and k=7 (= 4*SCALE/2 each)
        for (k = 0; k < 8; k = k + 1) begin
            xr[k] = $rtoi(SCALE * $cos(2.0*3.14159265358979*k/8.0)); xi[k]=0;
        end
        #1 $display("TEST 2 cosine (expect spikes at k=1,7):"); show;
        $finish;
    end
endmodule
