// decoder.v
// Combinational decoder: 4 latent inputs -> 8 outputs
`timescale 1ns / 1ps

module decoder (
    input  wire signed [15:0] z0,
    input  wire signed [15:0] z1,
    input  wire signed [15:0] z2,
    input  wire signed [15:0] z3,
    output wire signed [15:0] y0,
    output wire signed [15:0] y1,
    output wire signed [15:0] y2,
    output wire signed [15:0] y3,
    output wire signed [15:0] y4,
    output wire signed [15:0] y5,
    output wire signed [15:0] y6,
    output wire signed [15:0] y7
);

    // localparam declarations must live inside the module body,
    // not outside it, or the compiler cannot resolve them here.
    `include "ae_params.vh"

    neuron_dec #(
        .W0(W_DEC_00), .W1(W_DEC_10), .W2(W_DEC_20), .W3(W_DEC_30),
        .B (B_DEC_0)
    ) u_neuron_dec_0 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y0)
    );

    neuron_dec #(
        .W0(W_DEC_01), .W1(W_DEC_11), .W2(W_DEC_21), .W3(W_DEC_31),
        .B (B_DEC_1)
    ) u_neuron_dec_1 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y1)
    );

    neuron_dec #(
        .W0(W_DEC_02), .W1(W_DEC_12), .W2(W_DEC_22), .W3(W_DEC_32),
        .B (B_DEC_2)
    ) u_neuron_dec_2 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y2)
    );

    neuron_dec #(
        .W0(W_DEC_03), .W1(W_DEC_13), .W2(W_DEC_23), .W3(W_DEC_33),
        .B (B_DEC_3)
    ) u_neuron_dec_3 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y3)
    );

    neuron_dec #(
        .W0(W_DEC_04), .W1(W_DEC_14), .W2(W_DEC_24), .W3(W_DEC_34),
        .B (B_DEC_4)
    ) u_neuron_dec_4 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y4)
    );

    neuron_dec #(
        .W0(W_DEC_05), .W1(W_DEC_15), .W2(W_DEC_25), .W3(W_DEC_35),
        .B (B_DEC_5)
    ) u_neuron_dec_5 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y5)
    );

    neuron_dec #(
        .W0(W_DEC_06), .W1(W_DEC_16), .W2(W_DEC_26), .W3(W_DEC_36),
        .B (B_DEC_6)
    ) u_neuron_dec_6 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y6)
    );

    neuron_dec #(
        .W0(W_DEC_07), .W1(W_DEC_17), .W2(W_DEC_27), .W3(W_DEC_37),
        .B (B_DEC_7)
    ) u_neuron_dec_7 (
        .z0(z0), .z1(z1), .z2(z2), .z3(z3),
        .y(y7)
    );

endmodule
