# Technical Design Notes

This document stores technical decisions that should remain useful
across sessions. Update it when a design choice is made or changed;
ordinary task progress belongs in `reports/progress.md`.

## Target Workflow

1.  NumPy floating-point training.
2.  Analyze weights, biases, activations, and intermediate accumulators.
3.  Choose fixed-point formats based on observed ranges and error
    experiments.
4.  Export quantized parameters.
5.  Run a Python integer inference model using explicit hardware-like
    arithmetic.
6.  Implement RTL that matches the integer model.
7.  Verify the RTL, synthesize, and test on the FPGA.

## Initial Models

### Small autoencoder

-   Topology: 8-4-8.
-   Input: samples from a sine-wave signal.
-   Training implementation: NumPy.
-   Purpose: learn the complete train-to-FPGA workflow with a small
    design.

### MNIST autoencoder

-   Initial topology target: 784-64-16-64-784.
-   Dataset and exact training configuration are to be confirmed during
    implementation.
-   The architecture may be revised if resource estimates or
    reconstruction quality justify it; record the rationale here.

## Fixed-Point and Integer Arithmetic

No final fixed-point format is selected yet. Q3.12 may be evaluated as a
candidate, but must not be treated as a settled choice until ranges and
quantization error have been measured.

For each layer, document: - Weight, bias, input, activation, and output
formats. - Signedness and bit widths. - Product and accumulator
widths. - Bias scaling and how it is aligned with the accumulator. -
Rounding rule during requantization. - Overflow behavior: saturation or
wraparound. - Activation function and its exact integer
implementation. - Whether any intermediate truncation occurs.

The Python integer model must implement these rules explicitly. Avoid
relying on implicit NumPy overflow or casting behavior.

## RTL and Verification

-   Keep the integer reference independent enough to serve as a
    meaningful comparison against RTL.
-   Test zero, positive and negative values where applicable, values
    near representable limits, saturation/overflow cases, and
    representative model inputs.
-   Record simulator, commands, test vectors, and results.
-   Compare exact integer outputs where the design is intended to be
    bit-exact.
-   Report synthesis timing/resource results separately from functional
    simulation results.

## Decisions To Confirm

-   Exact sine-wave dataset generation and training configuration.
-   Candidate fixed-point formats after range analysis.
-   Rounding and saturation policy.
-   RTL interface and parameter-loading approach.
-   Clock frequency target and FPGA I/O approach.
-   MNIST model/training details and parameter storage strategy.
