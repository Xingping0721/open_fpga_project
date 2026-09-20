# Fully-Connected Autoencoder, HW Inference Implementation

A small autoencoder(AE), trained in python and move the weigth to the hardware.

## This AE's Architechture
```bash
Input (8) --[FC 8x4 + bias]--> Latent z (4) --[FC 4x8 + bias]--> Output (8)
```
- Input : 8-point sine wave sample, random phase, scaled to [0, 1]
- 1 encoder layer and 1 decoder layer. ReLU activation function
- 76 (not 67) params total (64 weight + 12 biases)

