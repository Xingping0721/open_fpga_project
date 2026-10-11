# Project Instructions

## Project Goal

Build an autoencoder accelerator from end to end: 1. Train a small model
in Python using NumPy. 2. Define and validate fixed-point numerical
behavior. 3. Implement a bit-exact integer inference model in Python. 4.
Implement matching RTL and verify it against the Python reference. 5.
Run the design on the DE10-Standard FPGA. 6. Extend the workflow from an
8-4-8 sine-wave autoencoder to an MNIST autoencoder.

The initial environment is Ubuntu/Python for software work and Quartus
Prime Standard with Questa FPGA Edition for FPGA work. The target board
is Terasic DE10-Standard.

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
