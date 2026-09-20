// ae_top.v
// Combinational top module for the 8-4-8 fully-connected autoencoder.
// No clock: encoder and decoder are connected purely combinationally.
`timescale 1ns / 1ps

module ae_top (
    input  wire signed [15:0] x0,
    input  wire signed [15:0] x1,
    input  wire signed [15:0] x2,
    input  wire signed [15:0] x3,
    input  wire signed [15:0] x4,
    input  wire signed [15:0] x5,
    input  wire signed [15:0] x6,
    input  wire signed [15:0] x7,
    output wire signed [15:0] y0,
    output wire signed [15:0] y1,
    output wire signed [15:0] y2,
    output wire signed [15:0] y3,
    output wire signed [15:0] y4,
    output wire signed [15:0] y5,
    output wire signed [15:0] y6,
    output wire signed [15:0] y7
);

    wire signed [15:0] z0, z1, z2, z3;

    encoder u_encoder (
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .x4(x4), .x5(x5), .x6(x6), .x7(x7),
        .z0(z0), .z1(z1), .z2(z2), .z3(z3)
    );

    decoder u_decoder (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y0(y0), .y1(y1), .y2(y2), .y3(y3),
        .y4(y4), .y5(y5), .y6(y6), .y7(y7)
    );

endmodule
