# Project Roadmap

## Goal

Implement and evaluate two autoencoder accelerators, progressing from a
small 8-4-8 sine-wave model to an MNIST model, using a NumPy software
reference, fixed-point integer inference, RTL simulation, and the
DE10-Standard FPGA.

Planning assumption: approximately 8--12 hours per week. This is an
initial plan, not a promise that each milestone will take exactly one
week. Adjust it based on measured progress and blockers.

## Milestones

  ------------------------------------------------------------------------------
  Week              Focus             Main deliverable  Acceptance criteria
  ----------------- ----------------- ----------------- ------------------------
  1                 NumPy             Training and      Training runs
                    floating-point    inference script; reproducibly; loss
                    8-4-8 autoencoder sine-wave         decreases;
                                      reconstruction    reconstruction plot is
                                      plot              saved

  2                 Numerical range   Range report and  Activation/weight/bias
                    analysis and      quantization      ranges are measured;
                    fixed-point       experiment        candidate formats and
                    proposal                            errors are compared

  3                 Integer inference Python integer    Explicit integer
                    reference         model and tests   arithmetic, scaling,
                                                        rounding, and
                                                        saturation; results are
                                                        deterministic

  4                 8-4-8 RTL         RTL modules for   Module-level simulation
                    datapath          layers and        passes basic directed
                                      arithmetic        tests

  5                 RTL integration   Testbench         RTL outputs match the
                    and bit-exact     comparing RTL     Python reference for
                    verification      with integer      defined test vectors
                                      reference         

  6                 FPGA deployment   Quartus project   Design builds; board I/O
                    of 8-4-8          and on-board test test and inference test
                                      record            are recorded where
                                                        practical

  7                 MNIST             Training script   Training runs; loss and
                    floating-point    and               representative
                    autoencoder       reconstruction    reconstructed digits are
                                      examples          saved

  8                 MNIST             Quantized         Quantization error is
                    quantization and  parameters and    measured; integer
                    integer inference integer reference inference is tested on
                                                        held-out samples

  9                 MNIST RTL         Architecture note Layer dimensions, memory
                    architecture      and RTL modules   strategy, and arithmetic
                                                        widths are documented

  10                MNIST RTL         Integrated RTL    RTL is compared with the
                    integration       and simulation    integer reference on
                                      testbench         selected test vectors

  11                MNIST FPGA        Synthesized FPGA  Compilation status,
                    implementation    project           resource use, and timing
                                                        results are recorded

  12                On-board          Demo evidence and Results, limitations,
                    evaluation and    final technical   resource use, timing,
                    report            report            and reconstruction
                                                        quality are documented
  ------------------------------------------------------------------------------

## Planning Rules

-   Treat weeks as planning units, not deadlines that override
    correctness.
-   Do not begin RTL until the integer reference and arithmetic
    conventions are sufficiently stable.
-   If a milestone slips, update this roadmap only when the overall plan
    or milestone definitions meaningfully change. Record day-to-day
    progress in `reports/progress.md`.
-   Prefer a smaller, fully verified design over a larger design that
    has not been validated.
-   FPGA board access, pin assignments, clocking, and I/O constraints
    may affect the schedule; investigate these early enough to avoid
    late surprises.

## Definition of Done

A milestone is complete only when its acceptance criteria are verified
and the evidence is recorded. Evidence may include test output, plots,
simulation logs, synthesis reports, resource/timing reports, or on-board
observations. Mark unavailable evidence as pending rather than assuming
success.
