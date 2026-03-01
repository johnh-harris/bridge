# Stim–Syndrilla Bridge

This project generates surface-code circuits with noise using Stim, extracts the
detector-error-model matrices, and converts them into AList files for use with
Syndrilla.

## Overview

1. A noisy Stim circuit is generated with `stim.Circuit.generated(...)`.
2. The detector error model is extracted with `circuit.detector_error_model(...)`.
3. BeliefMatching converts the model into parity-check and logical-check matrices.
4. The matrices are written to `.alist` files (Hx, Hz, Lz).

## Usage

Run the main script:
