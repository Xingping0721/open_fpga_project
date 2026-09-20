// leaky_relu.v
// Combinational Leaky ReLU for Q8.8 fixed-point values.
// y = x            (x >= 0)
// y = (x * ALPHA) >>> 8   (x < 0)
`timescale 1ns / 1ps

module leaky_relu #(
    parameter signed [15:0] ALPHA = 16'sh0003  // 0.01 in Q8.8
) (
    input  wire signed [15:0] x,
    output wire signed [15:0] y
);

    wire signed [31:0] neg_mul = x * ALPHA;  // Q8.8 * Q8.8 = Q16.16

    assign y = (x >= 0) ? x : (neg_mul >>> 8);

endmodule
