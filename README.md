# Blind Spot Theory: Global vs. Product-Aware Anomaly Detection

## The Core Hypothesis: The "Blind Spot" Theory

The project is driven by the hypothesis that product-agnostic (global) anomaly detectors create dangerous blind spots.

### The Logic

In a multi-product plant, different products have different operating conditions (e.g., Product A runs at 120°C, while Product B runs at 130°C).

### The Failure Mode

If a model is trained on all data globally, it learns that 130°C is a "permissible" value. Therefore, if the temperature spikes to 130°C while Product A is running (which should be an anomaly), the global model will likely ignore it because it has seen that value as "normal" elsewhere.

### The Result

This leads to missed detections (False Negatives), allowing product quality to degrade without triggering an alarm.

## The Project Goal

To test this hypothesis, the project aims to conduct a comparative analysis using the Extended Tennessee Eastman dataset (which simulates a modern chemical plant).

### The Experiment

The experiment involves designing and training two distinct model types:

1.  **Global Autoencoder**: A single product-agnostic model trained on data from all products.
2.  **Per-Product Autoencoders**: Specific models trained only on the data relevant to a single product line.

By comparing the performance of these two approaches, the project intends to quantify how many anomalies the global model misses compared to the product-aware models.

## Motivation & Impact

The motivation is strictly operational and safety-oriented. Manufacturing plants often implement product-agnostic models for simplicity, but the project argues this creates:

- **Undetected Deviations**: Subtle shifts in process variables go unnoticed.
- **Compliance Risk**: Products may be produced out-of-spec.
- **Quality Degradation**: Defects may pass through the line undetected.

## Data Setup

The project relies on the **Extended Tennessee Eastman (TEP)** dataset.

**Data Path**: `/home/admin/gdrive/TEP_Dataset`

Please ensure this path is accessible or configure your environment to point to the correct location of the dataset.

### Dataset Overview

The TEP dataset is extremely large (~138 GB total, with 6 mode files of ~23 GB each). To make the analysis manageable, we use a carefully selected subset:

**File Format**: HDF5 (`.h5` files)
**Modes Used**: 1, 3, 5 (3 out of 6 available modes)
**Fault Types**: All 28 fault types (IDV1-IDV28) available in each mode
**Data Access**: Use `h5py` library in Python

### Sampling Strategy

To balance dataset size with statistical validity, we apply the following sampling approach:

**Fault Magnitude**:
- Use only **100% magnitude faults** (simplifies analysis, 200 runs available per fault type)

**Run Sampling per Mode**:
- **30 runs** per fault type for normal data extraction (from hours 0-20)
- **10 runs** per fault type for fault data extraction (from hours 70-90)
- Applied across all 28 fault types

**Total Simulations**:
- Normal data: 3 modes × 28 fault types × 30 runs = **2,520 runs**
- Fault data: 3 modes × 28 fault types × 10 runs = **840 runs**

### Temporal Segmentation

The fault data has specific temporal structure requiring careful handling:

- **Hours 0-30**: Normal operation (pre-fault injection)
- **Hour 30**: Fault activation time
- **Hours 30+**: Fault conditions

**Our Segmentation Strategy**:

1. **Normal Data (Hours 0-20)**: Extract from 30 runs per fault type
   - 20 hours × 20 samples/hour = 400 timesteps per run
   - Per mode: 28 × 30 × 400 = 336,000 timesteps
   - **Total normal: 1,008,000 timesteps across 3 modes**

2. **Fault Data (Hours 70-90)**: Extract from 10 runs per fault type
   - 20 hours × 20 samples/hour = 400 timesteps per run
   - Per mode: 28 × 10 × 400 = 112,000 timesteps
   - **Total fault: 336,000 timesteps across 3 modes**

3. **Excluded Periods**: All other time periods discarded

**Data Ratio**: 3:1 (Normal:Fault) - realistic for anomaly detection applications

### Train/Test Split

Following best practices for anomaly detection, we train only on normal data:

**Training Set** (672,000 timesteps):
- 2/3 of normal data (1,680 randomly selected runs from 2,520 total)
- 0 fault data
- 100% normal operation samples

**Evaluation Set** (672,000 timesteps):
- 1/3 of normal data (840 runs) = 336,000 timesteps
- All fault data (840 runs) = 336,000 timesteps
- 50/50 normal-fault split for balanced performance metrics

**Rationale**: Autoencoders learn to reconstruct normal patterns. High reconstruction error on unseen fault data indicates anomaly detection.

### Dataset Technical Details

**Process Variables**: 54 (including Time as first column)
**Additional Measurements**: 32 variables from revised TEP simulator
**Sampling Rate**: 0.05 hours (3 minutes) = 20 samples per hour
**Timesteps per Simulation**: 2,001 (fault data) or 2,000 (normal data)

**HDF5 File Paths**:
- Normal operation: `Mode{N}/SpVariation/SimulationCompleted/SP1/tRamp_0/SpMagnitude100`
- Fault data: `Mode{N}/SingleFault/SimulationCompleted/IDV{X}/Mode{N}_IDVInfo_{X}_100/Run{Y}`
- Variable labels: `Processdata_Labels` and `Additional_Meas_Labels` (at root level)

**Memory Considerations**:
- Each simulation: ~1.7 MB
- Loading 1,000 runs: ~1.7 GB
- Recommend lazy loading with `h5py` to avoid loading entire 23 GB files into memory

## Implementation Plan

The project implementation is organized into two main Python files for clarity and maintainability.

### File 1: `data_loader.py` (~200-300 lines)

**Responsibilities**:
- Load data from HDF5 files (modes 1, 3, 5)
- Extract temporal windows (hours 0-20 for normal, 70-90 for fault)
- Sample runs (30 normal, 10 fault per fault type)
- Handle all 28 fault types across 3 modes
- Create train/test splits (2/3 normal for train, 1/3 normal + all fault for test)
- Normalization/standardization
- Return PyTorch/TensorFlow datasets or numpy arrays

**Key Functions**:
- `load_tep_data(mode, fault_type, run_number, time_window)` → array
- `extract_normal_data(modes, num_runs=30)` → array
- `extract_fault_data(modes, num_runs=10)` → array
- `create_train_test_split(normal_data, fault_data)` → splits
- `get_dataloaders()` → train_loader, test_loader

### File 2: `train_eval.py` (~250-350 lines)

**Responsibilities**:
- Define autoencoder architecture
- Train global model (all 3 modes combined)
- Train per-mode models (3 separate models for modes 1, 3, 5)
- Evaluate both approaches
- Calculate metrics (reconstruction error, precision, recall, F1, ROC-AUC)
- Compare global vs. product-aware performance
- Visualize results

**Key Components**:
- `class Autoencoder(nn.Module)` → model definition
- `train_autoencoder(model, train_loader)` → trained model
- `evaluate_model(model, test_loader)` → metrics
- `compare_models(global_model, mode_models)` → comparison
- `plot_results()` → visualizations
