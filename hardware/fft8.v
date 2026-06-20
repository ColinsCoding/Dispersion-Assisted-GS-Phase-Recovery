// fft8.v -- 8-point radix-2 decimation-in-time FFT, fixed-point.
// A hardware extension for the dispersion-GS pipeline: the FFT is the core of
// the dispersion operator H(f)=exp(i pi D f^2) used to propagate I1 -> I2.
// Combinational; twiddles scaled by 2^S (S=12 -> factor 4096). Verify against
// numpy with hardware/fft8_tb.v.  Build: iverilog -g2012 fft8.v fft8_tb.v
`timescale 1ns/1ps
module fft8 #(parameter W = 32, parameter S = 12) (
    input  signed [W-1:0] xr [0:7],
    input  signed [W-1:0] xi [0:7],
    output signed [W-1:0] Xr [0:7],
    output signed [W-1:0] Xi [0:7]
);
    // complex multiply by a scaled twiddle, rescaled by >>> S
    function signed [W-1:0] mr(input signed [W-1:0] ar, ai, wr, wi);
        reg signed [2*W-1:0] p; begin p = ar*wr - ai*wi; mr = p >>> S; end
    endfunction
    function signed [W-1:0] mi(input signed [W-1:0] ar, ai, wr, wi);
        reg signed [2*W-1:0] p; begin p = ar*wi + ai*wr; mi = p >>> S; end
    endfunction

    // W8^k * 4096 :  W0=1, W1=(1-j)/sqrt2, W2=-j, W3=-(1+j)/sqrt2
    localparam signed [W-1:0] W1R = 2896, W1I = -2896,
                              W2R = 0,    W2I = -4096,
                              W3R = -2896, W3I = -2896;

    // bit-reversed input order: 0,4,2,6,1,5,3,7
    wire signed [W-1:0] ar [0:7], ai [0:7];
    assign ar[0]=xr[0]; assign ai[0]=xi[0];
    assign ar[1]=xr[4]; assign ai[1]=xi[4];
    assign ar[2]=xr[2]; assign ai[2]=xi[2];
    assign ar[3]=xr[6]; assign ai[3]=xi[6];
    assign ar[4]=xr[1]; assign ai[4]=xi[1];
    assign ar[5]=xr[5]; assign ai[5]=xi[5];
    assign ar[6]=xr[3]; assign ai[6]=xi[3];
    assign ar[7]=xr[7]; assign ai[7]=xi[7];

    // stage 1: W0 butterflies on adjacent pairs
    wire signed [W-1:0] br [0:7], bi [0:7];
    assign br[0]=ar[0]+ar[1]; assign bi[0]=ai[0]+ai[1];
    assign br[1]=ar[0]-ar[1]; assign bi[1]=ai[0]-ai[1];
    assign br[2]=ar[2]+ar[3]; assign bi[2]=ai[2]+ai[3];
    assign br[3]=ar[2]-ar[3]; assign bi[3]=ai[2]-ai[3];
    assign br[4]=ar[4]+ar[5]; assign bi[4]=ai[4]+ai[5];
    assign br[5]=ar[4]-ar[5]; assign bi[5]=ai[4]-ai[5];
    assign br[6]=ar[6]+ar[7]; assign bi[6]=ai[6]+ai[7];
    assign br[7]=ar[6]-ar[7]; assign bi[7]=ai[6]-ai[7];

    // stage 2: span-2 butterflies, twiddles W0 and W2
    wire signed [W-1:0] cr [0:7], ci [0:7];
    wire signed [W-1:0] t13r = mr(br[3],bi[3],W2R,W2I), t13i = mi(br[3],bi[3],W2R,W2I);
    wire signed [W-1:0] t57r = mr(br[7],bi[7],W2R,W2I), t57i = mi(br[7],bi[7],W2R,W2I);
    assign cr[0]=br[0]+br[2]; assign ci[0]=bi[0]+bi[2];
    assign cr[2]=br[0]-br[2]; assign ci[2]=bi[0]-bi[2];
    assign cr[1]=br[1]+t13r;  assign ci[1]=bi[1]+t13i;
    assign cr[3]=br[1]-t13r;  assign ci[3]=bi[1]-t13i;
    assign cr[4]=br[4]+br[6]; assign ci[4]=bi[4]+bi[6];
    assign cr[6]=br[4]-br[6]; assign ci[6]=bi[4]-bi[6];
    assign cr[5]=br[5]+t57r;  assign ci[5]=bi[5]+t57i;
    assign cr[7]=br[5]-t57r;  assign ci[7]=bi[5]-t57i;

    // stage 3: span-4 butterflies, twiddles W0,W1,W2,W3 -> natural-order output
    wire signed [W-1:0] u5r = mr(cr[5],ci[5],W1R,W1I), u5i = mi(cr[5],ci[5],W1R,W1I);
    wire signed [W-1:0] u6r = mr(cr[6],ci[6],W2R,W2I), u6i = mi(cr[6],ci[6],W2R,W2I);
    wire signed [W-1:0] u7r = mr(cr[7],ci[7],W3R,W3I), u7i = mi(cr[7],ci[7],W3R,W3I);
    assign Xr[0]=cr[0]+cr[4]; assign Xi[0]=ci[0]+ci[4];
    assign Xr[4]=cr[0]-cr[4]; assign Xi[4]=ci[0]-ci[4];
    assign Xr[1]=cr[1]+u5r;   assign Xi[1]=ci[1]+u5i;
    assign Xr[5]=cr[1]-u5r;   assign Xi[5]=ci[1]-u5i;
    assign Xr[2]=cr[2]+u6r;   assign Xi[2]=ci[2]+u6i;
    assign Xr[6]=cr[2]-u6r;   assign Xi[6]=ci[2]-u6i;
    assign Xr[3]=cr[3]+u7r;   assign Xi[3]=ci[3]+u7i;
    assign Xr[7]=cr[3]-u7r;   assign Xi[7]=ci[3]-u7i;
endmodule
