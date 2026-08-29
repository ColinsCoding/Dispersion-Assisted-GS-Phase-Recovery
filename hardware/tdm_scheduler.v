// tdm_scheduler.v -- synthesizable round-robin time-division-multiplexing
// channel scheduler. A free-running mod-N counter clocked at the hardware
// clock: channel_sel advances 0,1,...,N-1,0,1,... every cycle, exactly the
// tdm_active_channel(cycle_index, n_channels) = cycle_index % n_channels
// formula tdm_scheduler.c computes in software. This is the actual
// hardware that would sit in front of one shared ADC, routing each cycle's
// sample to the currently-selected optical channel's buffer.
`timescale 1ns/1ps

module tdm_scheduler #(
    parameter N_CHANNELS = 4,
    parameter CH_WIDTH   = 2   // ceil(log2(N_CHANNELS)) -- must fit N_CHANNELS-1
) (
    input  wire                  clk,
    input  wire                  rst_n,      // active-low synchronous reset
    output reg  [CH_WIDTH-1:0]   channel_sel,
    output reg                   frame_start // pulses high for one cycle when channel_sel wraps to 0
);

    always @(posedge clk) begin
        if (!rst_n) begin
            channel_sel <= 0;
            frame_start <= 1'b1; // channel 0 on the very first valid cycle IS a frame start
        end else if (channel_sel == N_CHANNELS - 1) begin
            channel_sel <= 0;
            frame_start <= 1'b1;
        end else begin
            channel_sel <= channel_sel + 1'b1;
            frame_start <= 1'b0;
        end
    end

endmodule
