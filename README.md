# Blind Spot Theory: Global vs. Product-Aware Anomaly Detection

## Overview
This project tests whether a single, product-agnostic (global) anomaly detector develops blind spots compared to product-aware models. It uses the Extended Tennessee Eastman Process (TEP) dataset for three modes (1, 3, 5) to compare a global autoencoder against per-mode autoencoders and to study thresholding and mode-change behavior.

## Data & Defaults
- **Dataset**: Extended TEP at `/home/admin/gdrive/TEP_Dataset` (HDF5). Modes used: 1, 3, 5.
- **Features**: Process variables (time column dropped) plus appended one-hot mode indicators.
- **Normal data (default)**: Dedicated SpVariation normal runs (`SpVariation/SimulationCompleted/SP1/tRamp_0/SpMagnitude100`) — clean, no faults injected.
- **Fault data (default)**: Stratified sampling from fault runs, excluding IDV6. Samples are taken at timepoints `[70, 75, 80, 85, 90]` from runs `[1..5]`; each sample is the midpoint of a small ±0.1h window.
- **Alternatives**:
  - Set `use_dedicated_normal=False` in `prepare_data` to use pre-fault windows (hours 0–20) from fault runs.
  - Set `stratified_fault_sampling=False` to pull full post-fault windows (hours 70–90) from 10 runs per fault type.

## Pipeline Summary
1. **Data prep (`data_loader.py`)**
   - Extract normal and fault data using the defaults above.
   - Append mode one-hot columns.
   - Split: 2/3 of normal for training; test = remaining normal + all fault (stratified by mode).
   - Fit `StandardScaler` on the training split; apply to train/test; save as `scaler.pkl`.
   - Persist normalized splits to `preprocessed_data.pkl`.
2. **Training & evaluation (`train_eval.py`)**
   - Train global autoencoder on all modes.
   - Train per-mode autoencoders on mode-filtered data.
   - Compare metrics (precision/recall/F1/ROC-AUC), save `results/results_summary.json`.
   - Plot `results/model_comparison.png` and `results/roc_curves.png`.
   - Train per-mode models again with per-mode scalers; save `mode{1,3,5}_autoencoder.pth` and `mode{1,3,5}_scaler.pkl`.
3. **Mode-detection study (`mode_detection.py`)**
   - Load saved models/scalers (fits fallback scalers if missing).
   - Compute thresholds: global = 95th percentile of global errors on all-mode normal data; per-mode thresholds on per-mode normal data.
   - Simulate expected vs. actual mode cases (steady-state or transitions).
   - Plots written to `results/`:
     - `blind_spot_proper_threshold.png` (detection rates)
     - `reconstruction_error_over_time.png` (global + mode-specific errors)
     - `reconstruction_error_over_time_global_only.png` (global-only view for wrong modes)

## Usage
Install dependencies:
```bash
pip install -r requirements.txt
```

Train, evaluate, and produce comparison plots:
```bash
python train_eval.py
```
Outputs: `global_autoencoder.pth`, `mode{1,3,5}_autoencoder.pth`, `mode{1,3,5}_scaler.pkl`, `scaler.pkl`, `results/results_summary.json`, `results/model_comparison.png`, `results/roc_curves.png`, and `preprocessed_data.pkl`.

Run mode-detection visualizations (transitions by default):
```bash
python mode_detection.py
```
Use steady-state instead of transitions:
```bash
python mode_detection.py --steady-state
```

## Notes & Assumptions
- The code is PyTorch-only; no TensorFlow path is implemented.
- Transition trajectories depend on availability in the TEP files; missing transitions fall back to steady-state data.
- Large HDF5 files are assumed local; adjust `DATA_PATH` in `data_loader.py` / `mode_detection.py` if needed.
