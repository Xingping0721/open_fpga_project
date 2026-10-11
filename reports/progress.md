# Current Progress

Last updated: 2026-10-11

## Current Stage

Week 1 (NumPy floating-point 8-4-8 autoencoder): acceptance criteria
(reproducible training, decreasing loss, saved reconstruction plot)
verified in software. Artifacts are saved in the working tree but not
yet committed. Week 2 has not started (user decision: do not start yet).

Pre-existing work found in the repository (not yet verified against the
roadmap acceptance criteria): Q8.8 export and integer reference
(`simple_ae_ws/python/export_weights.py`), combinational RTL
(`simple_ae_ws/rtl/`), and a single-vector testbench
(`simple_ae_ws/tb/tb_ae_top.v`). An MNIST float AE also exists in
`mnist_ae_ws/`. Q8.8 was chosen before range analysis and must be
re-evaluated in Week 2.

## Completed (2026-10-11)

-   `simple_ae_ws/python/train.py`: added loss-history recording, loss
    curve and reconstruction plot output, `--epochs` / `--out` /
    `--results-dir` options (defaults unchanged: 300000 epochs,
    `weights/ae_weights.npz`, `results/`), and script-relative paths.
    Training logic, hyperparameters, and seed are unchanged.
-   Adopted the newly trained weights as the baseline (user decision):
    replaced `simple_ae_ws/python/weights/ae_weights.npz`. The previous
    weights remain recoverable from git (commit d99d295).
-   Saved Week 1 artifacts in `simple_ae_ws/python/results/`:
    `loss_curve.png`, `reconstruction.png`, `loss_history.csv`,
    `train_log.txt` (stdout of the full run).

## Verification Evidence (software only)

-   Full training run:
    `python3 simple_ae_ws/python/train.py --out <scratch>/ae_weights.npz --results-dir <scratch>`
    -   Config: seed 0, 90000 train / 10000 val sine samples (8 points,
        random phase, scaled to [0, 1]), full-batch GD, LR 0.001,
        300000 epochs, Leaky ReLU alpha 0.01 on both layers, He init.
    -   Loss: train 0.554 → 1.93e-7, val 0.552 → 1.91e-7. Plateau
        around 0.1 until ~45k epochs, then steady decrease. Train and
        val curves overlap (no overfitting observed).
    -   Wall time ~75 min on this machine (~15 ms/epoch; NumPy BLAS
        multithreading on small matrices).
    -   Loss curve and 4-sample reconstruction plot inspected; outputs
        visually overlap inputs (per-sample MSE 1.5e-9 to 1.0e-7).
    -   Run output went to the scratchpad and was then copied into the
        repository (copied weights verified bit-identical).
-   Determinism: two 3000-epoch runs with the same seed produced
    bit-identical weights.
-   Reproducibility of committed weights: FAILED. Committed
    `simple_ae_ws/python/weights/ae_weights.npz` differs from the new
    run (max |diff| up to 2.08) and also from a run of the original
    commit 5699aa9 `train.py` (900 samples, LR 0.1, 3000 epochs). Its
    training settings are unknown.
-   `rtl/ae_params.vh` and `tb/expected_vectors.vh` were generated from
    the previous (unreproducible) weights; regenerating them from those
    weights in a copy gave identical files. They were NOT regenerated for
    the new baseline weights and are now STALE (do not match
    `ae_weights.npz`). Regenerate after the Week 2 format decision.
-   Not run: RTL simulation (iverilog, Questa, and Quartus were not found
    on PATH or in common install locations), synthesis, on-board tests.

## Observations for Week 2

-   Quick range check with the committed weights (10000 samples): all
    weights, biases, and pre-activations lie within about ±1.4, so Q8.8
    leaves ~6 integer bits unused. Rounding weights/inputs to 8
    fractional bits roughly tripled the reconstruction MSE (4.0e-6 →
    1.2e-5); 12 fractional bits matched float. This was a rough
    float-accumulation experiment, not a formal analysis, and must be
    redone with whichever weights become the baseline.
-   Latent units 0 and 1 were always negative (dead) with the committed
    weights. Consistent with the data being 2-dimensional
    (x = 0.5 + 0.5(sin t·cos p + cos t·sin p)).
-   The current integer path truncates (`>>> 8`), has no saturation when
    narrowing the accumulator to 16 bits, and approximates alpha 0.01 as
    3/256 (+17%).

## Open Questions (need user decision)

-   Commit the Week 1 changes (`train.py`, new weights, `results/`)?
-   Optimizer: keep plain full-batch GD as the default, or add Adam /
    mini-batch as an option to reduce the ~75 min training time?

## Next Concrete Task

When the user is ready for Week 2: write a range analysis script using
the new baseline weights (weights, biases, inputs, pre-activations,
latents, accumulators), then compare candidate fixed-point formats
(e.g. Q8.8, Q4.12, Q2.14) with rounding/saturation options. Afterwards
regenerate `ae_params.vh` / `expected_vectors.vh`.
