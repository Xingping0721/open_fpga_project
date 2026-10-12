# Project Instructions

## Project Goal

Build an autoencoder accelerator from end to end: 1. Train a small model
in Python using NumPy. 2. Define and validate fixed-point numerical
behavior. 3. Implement a bit-exact integer inference model in Python. 4.
Implement matching RTL and verify it against the Python reference. 5.
Run the design on the DE10-Standard FPGA. 6. Extend the workflow from an
8-4-8 sine-wave autoencoder to an MNIST autoencoder.

The environment is Ubuntu/Python for software work, open-source tools
(Icarus Verilog / Verilator, Yosys) for RTL simulation and logic
synthesis, and Quartus Prime Standard (with Questa FPGA Edition if
needed) for FPGA implementation. The target board is Terasic
DE10-Standard.

## Development Flow

Follow these steps in order for each design (8-4-8 first, then MNIST).
Do not skip ahead: in particular, do not write RTL before the hardware
architecture (step 3) is documented.

1.  Understand the task and specification: inputs/outputs, accuracy
    target, constraints.
2.  Verify the algorithm in Python: floating-point training,
    range analysis and quantization, then the bit-exact integer
    reference model (Project Goal items 1-3).
3.  Design the hardware architecture before coding: arithmetic units,
    registers, FSM (control), and memory (parameter and activation
    storage). Record it in `docs/design.md`.
4.  Write the RTL in Verilog HDL.
5.  RTL simulation with Icarus Verilog or Verilator, compared bit-exactly
    against the Python integer reference.
6.  Logic synthesis with Yosys.
7.  Evaluate circuit size and timing from the synthesis results.
8.  When needed, FPGA implementation and place-and-route evaluation with
    Quartus, and on-board testing on the DE10-Standard.

Report each step's evidence separately (Python tests, RTL simulation,
Yosys synthesis, Quartus place-and-route, on-board results); passing one
step does not imply the next.

## Core Engineering Rules

-   Prefer NumPy and explicit arithmetic for the learning
    implementation. Do not introduce PyTorch unless the user asks for
    it.
-   Keep floating-point training, quantization-aware training (if used),
    integer inference, and RTL verification as distinct stages.
-   The Python integer inference model is the golden reference for RTL.
-   Specify and match integer widths, signedness, scaling, rounding,
    overflow/saturation, bias handling, and activation behavior.
-   Do not assume a fixed-point format is correct before measuring
    relevant value ranges and checking quantization error.
-   Keep the design understandable and educational; explain non-obvious
    arithmetic and hardware trade-offs.
-   Do not claim hardware verification is complete based only on Python
    simulation. Clearly distinguish software tests, RTL simulation,
    synthesis, and on-board results.
-   Prefer small, testable changes. Run relevant tests after changes and
    report their results accurately.
-   Do not overwrite user work or make broad refactors without a clear
    need.

## Project Documentation

-   `docs/roadmap.md`: milestone plan and acceptance criteria.
-   `docs/design.md`: technical decisions and numerical/RTL
    specifications.
-   `reports/progress.md`: current status, latest verification, open
    questions, and next task.

## Reference Book

Use 佐藤竜馬『深層ニューラルネットワークの高速化』(技術評論社, 2024,
ISBN 978-4-297-14309-1, https://gihyo.jp/book/2024/978-4-297-14309-1)
as the main reference for quantization and model acceleration.

-   Primary sections: 3.4 整数量子化 (what to quantize, post-training
    quantization vs. additional training, quantization arithmetic,
    choosing quantization scales, choosing which layers to quantize,
    quantization-aware training, 3.4.10 decision procedure). Also
    relevant: 2.1 量子化と低精度計算, 3.6 why low precision works, and
    for the MNIST stage, 4 枝刈り (including 4.5 ReLU-based sparsity) and
    6 低ランク近似.
-   When proposing or explaining a quantization step, map it to the
    relevant book section and use its terminology where practical.
-   The book targets general hardware and libraries (e.g. PyTorch). This
    project still uses NumPy only, and every scale must become an
    explicit integer/fixed-point rule that RTL can implement (for
    example, power-of-two scales as shifts). Record any deviation from
    the book's approach and the reason in `docs/design.md`.
-   Claude does not have the book's text. Do not invent its content;
    when a decision depends on details of the book, ask the user to
    confirm or quote the relevant part.

Treat this file as stable project-wide instructions. Do not update it
during ordinary development. Only change it when the user explicitly
requests a change to project-wide instructions.

## Session Workflow

Before starting work: 1. Read this file and `reports/progress.md`. 2.
Read the relevant parts of `docs/roadmap.md` and `docs/design.md`. 3.
Inspect the repository, current code, tests, and Git status before
deciding what to do. 4. Continue from the actual repository state; do
not assume previous notes perfectly describe it.

At the end of each work session: 1. Summarize changes made and files
affected. 2. Record tests and verification results, including commands
and outcomes when available. 3. Record unresolved issues and decisions
that need the user's input. 4. Update `reports/progress.md`. 5. State
the next concrete task.

Never mark a milestone complete unless its acceptance criteria have been
verified. If verification could not be run, say so explicitly.
