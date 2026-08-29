// tdm_scheduler_tb.v -- drives tdm_scheduler for 16 cycles and checks the
// channel sequence against tdm_scheduler.c's printed golden sequence:
//   0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
`timescale 1ns/1ps

module tdm_scheduler_tb;

    reg clk = 0;
    reg rst_n = 0;
    wire [1:0] channel_sel;
    wire frame_start;

    integer i;
    integer errors = 0;
    integer expected [0:15];

    tdm_scheduler #(.N_CHANNELS(4), .CH_WIDTH(2)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .channel_sel(channel_sel),
        .frame_start(frame_start)
    );

    always #5 clk = ~clk; // 100 MHz-equivalent, 10ns period (matches the C model's 100 MHz demo)

    initial begin
        // golden sequence from tdm_scheduler.c's tdm_active_channel(i, 4)
        expected[0]=0; expected[1]=1; expected[2]=2; expected[3]=3;
        expected[4]=0; expected[5]=1; expected[6]=2; expected[7]=3;
        expected[8]=0; expected[9]=1; expected[10]=2; expected[11]=3;
        expected[12]=0; expected[13]=1; expected[14]=2; expected[15]=3;

        rst_n = 0;
        @(posedge clk);  // edge 1: DUT samples rst_n=0 here (still, see below)
                         // and takes the reset branch: channel_sel <= 0
        rst_n <= 1;      // non-blocking: deassert AFTER this edge's reads
                         // resolve, not racing the DUT's own posedge-triggered
                         // reset check (a blocking `rst_n = 1;` here can lose
                         // that race and let the DUT see the NEW rst_n on the
                         // very edge meant to be its last reset cycle --
                         // channel_sel then increments from X, and X+1 is X
                         // forever)
        @(posedge clk);  // edge 2: by now channel_sel has settled to 0 (from
                         // edge 1's NBA) and rst_n has settled to 1 (from the
                         // non-blocking assign above) -- reading channel_sel
                         // right after this wait, but before edge 2's OWN NBA
                         // applies, correctly samples the post-reset value 0

        $display("cycle: channel_sel  frame_start  (expected)");
        for (i = 0; i < 16; i = i + 1) begin
            $display("%20d:     %0d            %0d            (%0d)",
                      i, channel_sel, frame_start, expected[i]);
            if (channel_sel !== expected[i]) begin
                $display("  MISMATCH at cycle %0d: got %0d, expected %0d",
                          i, channel_sel, expected[i]);
                errors = errors + 1;
            end
            if (i % 4 == 0 && frame_start !== 1'b1) begin
                $display("  MISMATCH: frame_start should be high at cycle %0d", i);
                errors = errors + 1;
            end
            @(posedge clk);
        end

        if (errors == 0)
            $display("\n[PASS] all 16 cycles match the C golden model's round-robin sequence");
        else begin
            $display("\n[FAIL] %0d mismatch(es) against the C golden model", errors);
            $finish(1);
        end
        $finish;
    end

endmodule
