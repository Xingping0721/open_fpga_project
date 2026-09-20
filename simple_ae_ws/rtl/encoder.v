// encoder.v
// Combinational encoder: 8 inputs -> 4 latent outputs
`timescale 1ns / 1ps

module encoder (
    input  wire signed [15:0] x0,
    input  wire signed [15:0] x1,
    input  wire signed [15:0] x2,
    input  wire signed [15:0] x3,
    input  wire signed [15:0] x4,
    input  wire signed [15:0] x5,
    input  wire signed [15:0] x6,
    input  wire signed [15:0] x7,
    output wire signed [15:0] z0,
    output wire signed [15:0] z1,
    output wire signed [15:0] z2,
    output wire signed [15:0] z3
);

    // localparam declarations must live inside the module body,
    // not outside it, or the compiler cannot resolve them here.
    `include "ae_params.vh"

    neuron_enc #(
        .W0(W_ENC_00), .W1(W_ENC_10), .W2(W_ENC_20), .W3(W_ENC_30),
        .W4(W_ENC_40), .W5(W_ENC_50), .W6(W_ENC_60), .W7(W_ENC_70),
        .B (B_ENC_0)
    ) u_neuron_enc_0 (
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .x4(x4), .x5(x5), .x6(x6), .x7(x7),
        .z(z0)
    );

    neuron_enc #(
        .W0(W_ENC_01), .W1(W_ENC_11), .W2(W_ENC_21), .W3(W_ENC_31),
        .W4(W_ENC_41), .W5(W_ENC_51), .W6(W_ENC_61), .W7(W_ENC_71),
        .B (B_ENC_1)
    ) u_neuron_enc_1 (
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .x4(x4), .x5(x5), .x6(x6), .x7(x7),
        .z(z1)
    );

    neuron_enc #(
        .W0(W_ENC_02), .W1(W_ENC_12), .W2(W_ENC_22), .W3(W_ENC_32),
        .W4(W_ENC_42), .W5(W_ENC_52), .W6(W_ENC_62), .W7(W_ENC_72),
        .B (B_ENC_2)
    ) u_neuron_enc_2 (
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .x4(x4), .x5(x5), .x6(x6), .x7(x7),
        .z(z2)
    );

    neuron_enc #(
        .W0(W_ENC_03), .W1(W_ENC_13), .W2(W_ENC_23), .W3(W_ENC_33),
        .W4(W_ENC_43), .W5(W_ENC_53), .W6(W_ENC_63), .W7(W_ENC_73),
        .B (B_ENC_3)
    ) u_neuron_enc_3 (
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .x4(x4), .x5(x5), .x6(x6), .x7(x7),
        .z(z3)
    );

endmodule
