// neuron_dec.v
// Combinational decoder neuron: 4 inputs -> 1 output (Leaky ReLU applied)
`timescale 1ns / 1ps

module neuron_dec #(
    parameter signed [15:0] W0 = 16'sh0000,
    parameter signed [15:0] W1 = 16'sh0000,
    parameter signed [15:0] W2 = 16'sh0000,
    parameter signed [15:0] W3 = 16'sh0000,
    parameter signed [15:0] B  = 16'sh0000
) (
    input  wire signed [15:0] z0,
    input  wire signed [15:0] z1,
    input  wire signed [15:0] z2,
    input  wire signed [15:0] z3,
    output wire signed [15:0] y
);

    // Each product is Q8.8 * Q8.8 = Q16.16
    wire signed [31:0] p0 = z0 * W0;
    wire signed [31:0] p1 = z1 * W1;
    wire signed [31:0] p2 = z2 * W2;
    wire signed [31:0] p3 = z3 * W3;

    // Bias is Q8.8; align to Q16.16 before accumulation
    wire signed [31:0] bias_ext = {{16{B[15]}}, B} <<< 8;

    wire signed [31:0] acc = p0 + p1 + p2 + p3 + bias_ext;

    // Q16.16 -> Q8.8
    wire signed [15:0] y_pre = acc >>> 8;

    leaky_relu u_leaky_relu (
        .x(y_pre),
        .y(y)
    );

endmodule
