// neuron_enc.v
// Combinational encoder neuron: 8 inputs -> 1 output (Leaky ReLU applied)
`timescale 1ns / 1ps

module neuron_enc #(
    parameter signed [15:0] W0 = 16'sh0000,
    parameter signed [15:0] W1 = 16'sh0000,
    parameter signed [15:0] W2 = 16'sh0000,
    parameter signed [15:0] W3 = 16'sh0000,
    parameter signed [15:0] W4 = 16'sh0000,
    parameter signed [15:0] W5 = 16'sh0000,
    parameter signed [15:0] W6 = 16'sh0000,
    parameter signed [15:0] W7 = 16'sh0000,
    parameter signed [15:0] B  = 16'sh0000
) (
    input  wire signed [15:0] x0,
    input  wire signed [15:0] x1,
    input  wire signed [15:0] x2,
    input  wire signed [15:0] x3,
    input  wire signed [15:0] x4,
    input  wire signed [15:0] x5,
    input  wire signed [15:0] x6,
    input  wire signed [15:0] x7,
    output wire signed [15:0] z
);

    // Each product is Q8.8 * Q8.8 = Q16.16
    wire signed [31:0] p0 = x0 * W0;
    wire signed [31:0] p1 = x1 * W1;
    wire signed [31:0] p2 = x2 * W2;
    wire signed [31:0] p3 = x3 * W3;
    wire signed [31:0] p4 = x4 * W4;
    wire signed [31:0] p5 = x5 * W5;
    wire signed [31:0] p6 = x6 * W6;
    wire signed [31:0] p7 = x7 * W7;

    // Bias is Q8.8; align to Q16.16 before accumulation
    wire signed [31:0] bias_ext = {{16{B[15]}}, B} <<< 8;

    wire signed [31:0] acc = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + bias_ext;

    // Q16.16 -> Q8.8
    wire signed [15:0] z_pre = acc >>> 8;

    leaky_relu u_leaky_relu (
        .x(z_pre),
        .y(z)
    );

endmodule
