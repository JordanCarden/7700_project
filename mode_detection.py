"""
Test blind spot hypothesis with PROPER threshold calculation.

Key difference from previous tests:
- Global model: Uses ONE threshold calculated from ALL modes' normal data
- Mode-specific models: Use mode-specific thresholds

This tests the realistic deployment scenario where the global model
doesn't know which mode is currently being produced.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Tuple

from train_eval import Autoencoder
from data_loader import append_mode_one_hot, MODES

# Configuration
DATA_PATH = "/home/admin/gdrive/TEP_Dataset"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
MODES = [1, 3, 5]

# Load the scaler used during training (global across all modes)
print("Loading global scaler...")
with open("scaler.pkl", "rb") as f:
    GLOBAL_SCALER = pickle.load(f)
print("✓ Global scaler loaded")

# Lazy container for per-mode scalers
PER_MODE_SCALERS: Dict[int, StandardScaler] | None = None


def load_normal_data_with_time(mode: int):
    """Load normal steady-state data for a mode with timestamps."""
    file_path = f"{DATA_PATH}/TEP_Mode{mode}.h5"

    with h5py.File(file_path, 'r') as f:
        path = f'Mode{mode}/SpVariation/SimulationCompleted/SP1/tRamp_0/SpMagnitude100'
        processdata = f[path]['processdata'][:]
        time = processdata[:, 0]
        data = processdata[:, 1:]

    mode_labels = np.full(len(data), mode)
    data = append_mode_one_hot(data, mode_labels, modes=MODES)
    return time, data


def get_per_mode_scalers():
    """Load saved per-mode scalers if available; otherwise fit (with a warning)."""
    global PER_MODE_SCALERS
    if PER_MODE_SCALERS is None:
        PER_MODE_SCALERS = {}
        missing = []

        for mode in MODES:
            path = Path(f"mode{mode}_scaler.pkl")
            if path.exists():
                with open(path, "rb") as f:
                    PER_MODE_SCALERS[mode] = pickle.load(f)
                print(f"✓ Loaded per-mode scaler for Mode {mode}")
            else:
                missing.append(mode)

        if missing:
            print(f"⚠ Per-mode scaler file not found for modes {missing}; fitting from data (retrain and save for consistency).")
            for mode in missing:
                _, data = load_normal_data_with_time(mode)
                scaler = StandardScaler()
                scaler.fit(data)
                PER_MODE_SCALERS[mode] = scaler
                print(f"  Fit fallback scaler for Mode {mode} on {data.shape[0]} samples")

    return PER_MODE_SCALERS


def load_mode_change_data(from_mode: int, to_mode: int):
    """
    Load data to simulate mode change scenario.
    Returns normal data from 'to_mode' as if factory switched modes.
    """
    # Just load normal data from the "new" mode
    return load_normal_data_with_time(to_mode)


def load_transition_data(from_mode: int, to_mode: int, transition_time: int = 20):
    """
    Load the actual transition trajectory from from_mode -> to_mode.
    """
    file_path = f"{DATA_PATH}/TEP_Mode{from_mode}.h5"
    path = (f'Mode{from_mode}/ModeTransition/SimulationCompleted/'
            f'Mode{from_mode}ToMode{to_mode}/TransitionTime{transition_time}')

    with h5py.File(file_path, 'r') as f:
        if path not in f:
            # Fall back to TransitionTime40 if 20 is missing
            path = (f'Mode{from_mode}/ModeTransition/SimulationCompleted/'
                    f'Mode{from_mode}ToMode{to_mode}/TransitionTime40')
            if path not in f:
                raise ValueError(f"Transition path not found for Mode {from_mode}->{to_mode}")

        processdata = f[path]['processdata'][:]
        time = processdata[:, 0]
        data = processdata[:, 1:]

    # Tag with the actual mode being produced (to_mode) so one-hot reflects the data source,
    # not the expected mode.
    mode_labels = np.full(len(data), to_mode)
    data = append_mode_one_hot(data, mode_labels, modes=MODES)
    return time, data


def get_mode_change_timeseries(expected_mode: int, actual_mode: int, use_transitions: bool = True):
    """
    Return (time, data, source_tag) for a mode change scenario.

    If use_transitions and a transition path exists, use it. Otherwise fall back
    to steady-state normal data from the actual_mode (no timestamps available,
    so synthesize a simple time axis).
    """
    if use_transitions and expected_mode != actual_mode:
        try:
            time, data = load_transition_data(expected_mode, actual_mode)
            source = "transition"
        except Exception as e:
            print(f"  Warning: Transition data missing for {expected_mode}->{actual_mode}, "
                  f"falling back to steady-state: {e}")
            time, data = load_mode_change_data(expected_mode, actual_mode)
            source = "steady_state"
    else:
        time, data = load_mode_change_data(expected_mode, actual_mode)
        source = "steady_state"

    # Overwrite the one-hot columns to reflect the EXPECTED mode, not the actual.
    one_hot = np.zeros((len(data), len(MODES)))
    expected_idx = MODES.index(expected_mode)
    one_hot[:, expected_idx] = 1.0
    data[:, -len(MODES):] = one_hot

    return time, data, source


def load_normal_data(mode: int):
    """Load normal steady-state data for a mode (without timestamps)."""
    _, data = load_normal_data_with_time(mode)
    return data


def load_all_modes_normal_data():
    """Load normal data from ALL modes (for global threshold calculation)."""
    all_normal = []
    for mode in MODES:
        data = load_normal_data(mode)
        all_normal.append(data)
    return np.vstack(all_normal)


def load_trained_models():
    """Load all trained models."""
    models = {}

    input_dim = GLOBAL_SCALER.n_features_in_
    encoding_dim = 32
    hidden_dims = [128, 64]

    # Load global model
    global_path = Path('global_autoencoder.pth')
    if global_path.exists():
        model = Autoencoder(input_dim, encoding_dim, hidden_dims)
        model.load_state_dict(torch.load(global_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models['global'] = model
        print(f"✓ Loaded global model")

    # Load mode-specific models
    for mode in MODES:
        mode_path = Path(f'mode{mode}_autoencoder.pth')
        if mode_path.exists():
            model = Autoencoder(input_dim, encoding_dim, hidden_dims)
            model.load_state_dict(torch.load(mode_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models[f'mode{mode}'] = model
            print(f"✓ Loaded mode {mode} model")

    return models


def compute_reconstruction_errors(model: nn.Module, data: np.ndarray, scaler: StandardScaler | None = None):
    """Compute reconstruction errors with a provided scaler."""
    scaler = scaler or GLOBAL_SCALER
    data_normalized = scaler.transform(data)

    X = torch.FloatTensor(data_normalized).to(DEVICE)
    with torch.no_grad():
        X_recon = model(X)
        errors = torch.mean((X - X_recon) ** 2, dim=1)
    return errors.cpu().numpy()


def calculate_thresholds(models: Dict[str, nn.Module], per_mode_scalers: Dict[int, StandardScaler]):
    """
    Calculate thresholds for all models.

    CRITICAL DIFFERENCE:
    - Global model: ONE threshold from ALL modes' normal data
    - Mode-specific models: Separate threshold per mode
    """
    thresholds = {}

    # Global model threshold (from ALL modes)
    if 'global' in models:
        print("\nCalculating GLOBAL model threshold...")
        all_modes_normal = load_all_modes_normal_data()
        print(f"  Using {all_modes_normal.shape[0]} samples from ALL modes")

        errors = compute_reconstruction_errors(models['global'], all_modes_normal, GLOBAL_SCALER)
        threshold = np.percentile(errors, 95)
        thresholds['global'] = threshold

        print(f"  Global threshold: {threshold:.2e}")
        print(f"  (This is used for ALL modes - no mode-specific knowledge)")

    # Mode-specific thresholds
    for mode in MODES:
        model_key = f'mode{mode}'
        if model_key in models:
            print(f"\nCalculating MODE {mode} model threshold...")
            mode_normal = load_normal_data(mode)
            print(f"  Using {mode_normal.shape[0]} samples from Mode {mode} only")

            errors = compute_reconstruction_errors(models[model_key], mode_normal, per_mode_scalers[mode])
            threshold = np.percentile(errors, 95)
            thresholds[model_key] = threshold

            print(f"  Mode {mode} threshold: {threshold:.2e}")

    return thresholds


def test_mode_detection_proper_threshold(models, thresholds, per_mode_scalers, use_transitions: bool = True):
    """Test mode change detection with proper threshold usage."""

    print("\n" + "="*80)
    print("TESTING MODE CHANGE DETECTION WITH PROPER THRESHOLDS")
    print("="*80)

    results = {}

    for expected_mode in MODES:
        print(f"\n{'='*80}")
        print(f"Factory Expected to Produce: MODE {expected_mode}")
        print(f"{'='*80}")

        mode_results = {}
        other_modes = [m for m in MODES if m != expected_mode]

        for actual_mode in other_modes:
            print(f"\n{'-'*80}")
            print(f"Mode Change Detected: Expected Mode {expected_mode}, Actually Producing Mode {actual_mode}")
            print(f"{'-'*80}")

            try:
                time, mode_data, source = get_mode_change_timeseries(expected_mode, actual_mode, use_transitions)

                mode_change_results = {}

                # Test GLOBAL model with GLOBAL threshold
                if 'global' in models:
                    errors = compute_reconstruction_errors(models['global'], mode_data, GLOBAL_SCALER)
                    threshold = thresholds['global']  # ONE global threshold
                    detection_rate = np.mean(errors > threshold)

                    print(f"\n  Global Model ({source}):")
                    print(f"    Threshold: {threshold:.2e} (from ALL modes)")
                    print(f"    Mean error: {errors.mean():.2e}")
                    print(f"    Detection rate: {detection_rate:.1%}")

                    mode_change_results['global'] = {
                        'threshold': float(threshold),
                        'mean_error': float(errors.mean()),
                        'detection_rate': float(detection_rate)
                    }

                # Test MODE-SPECIFIC model with MODE-SPECIFIC threshold
                model_key = f'mode{expected_mode}'
                if model_key in models:
                    errors = compute_reconstruction_errors(models[model_key], mode_data, per_mode_scalers[expected_mode])
                    threshold = thresholds[model_key]  # Mode-specific threshold
                    detection_rate = np.mean(errors > threshold)

                    print(f"\n  Mode {expected_mode} Model ({source}):")
                    print(f"    Threshold: {threshold:.2e} (from Mode {expected_mode} only)")
                    print(f"    Mean error: {errors.mean():.2e}")
                    print(f"    Detection rate: {detection_rate:.1%}")

                    mode_change_results['mode_specific'] = {
                        'threshold': float(threshold),
                        'mean_error': float(errors.mean()),
                        'detection_rate': float(detection_rate)
                    }

                # Compare
                if 'global' in mode_change_results and 'mode_specific' in mode_change_results:
                    global_rate = mode_change_results['global']['detection_rate']
                    specific_rate = mode_change_results['mode_specific']['detection_rate']
                    improvement = specific_rate - global_rate

                    print(f"\n  {'─'*60}")
                    print(f"  COMPARISON:")
                    print(f"    Global:         {global_rate:.1%}")
                    print(f"    Mode-Specific:  {specific_rate:.1%}")
                    print(f"    Improvement:    {improvement:+.1%}")

                    if improvement > 0.05:  # More than 5% improvement
                        print(f"    ✓ Mode-specific WINS!")
                    elif improvement < -0.05:
                        print(f"    ✗ Global WINS")
                    else:
                        print(f"    = Roughly tied")

                mode_results[f'{expected_mode}_to_{actual_mode}'] = mode_change_results

            except Exception as e:
                print(f"  ✗ Error: {e}")

        results[f'mode{expected_mode}'] = mode_results

    return results


def visualize_results(results):
    """Create visualization of results."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Blind Spot Test with Proper Thresholds:\nGlobal vs Mode-Specific Detection',
                 fontsize=16, fontweight='bold')

    for idx, from_mode in enumerate(MODES):
        ax = axes[idx]
        mode_key = f'mode{from_mode}'

        if mode_key not in results:
            continue

        transitions = []
        global_rates = []
        specific_rates = []

        for trans_key, trans_data in results[mode_key].items():
            to_mode = trans_key.split('_to_')[1]
            transitions.append(f"→ Mode {to_mode}")

            if 'global' in trans_data:
                global_rates.append(trans_data['global']['detection_rate'] * 100)
            else:
                global_rates.append(0)

            if 'mode_specific' in trans_data:
                specific_rates.append(trans_data['mode_specific']['detection_rate'] * 100)
            else:
                specific_rates.append(0)

        x = np.arange(len(transitions))
        width = 0.35

        bars1 = ax.bar(x - width/2, global_rates, width,
                      label='Global (ONE threshold)', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x + width/2, specific_rates, width,
                      label=f'Mode {from_mode} (specific threshold)', alpha=0.8, color='#ff7f0e')

        ax.set_ylabel('Detection Rate (%)', fontsize=11)
        ax.set_title(f'Expected: Mode {from_mode}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(transitions, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)

        # Add value labels
        for i, (g, s) in enumerate(zip(global_rates, specific_rates)):
            if g > 2:
                ax.text(i - width/2, g + 2, f'{g:.1f}%', ha='center', va='bottom', fontsize=8)
            if s > 2:
                ax.text(i + width/2, s + 2, f'{s:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = RESULTS_DIR / 'blind_spot_proper_threshold.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")
    plt.close()


def visualize_reconstruction_errors(models, thresholds, per_mode_scalers, use_transitions: bool = True):
    """Visualize reconstruction error distributions with proper thresholds."""

    print("\nCreating reconstruction error distributions...")

    # Test key transitions
    test_cases = [
        (1, 3, "Mode 1→3 (Missed by both)"),
        (1, 5, "Mode 1→5 (Mode-specific WINS)"),
        (3, 1, "Mode 3→1 (Mode-specific WINS)"),
    ]

    fig, axes = plt.subplots(len(test_cases), 2, figsize=(16, 4 * len(test_cases)))
    fig.suptitle('Reconstruction Error Distributions with Proper Thresholds',
                 fontsize=16, fontweight='bold')

    for idx, (from_mode, to_mode, title) in enumerate(test_cases):
        try:
            # Load data
            _, trans_data, source = get_mode_change_timeseries(from_mode, to_mode, use_transitions)
            normal_data = load_normal_data(from_mode)

            # Global model (left)
            ax_global = axes[idx, 0] if len(test_cases) > 1 else axes[0]

            if 'global' in models:
                # Get errors
                normal_errors = compute_reconstruction_errors(models['global'], normal_data, GLOBAL_SCALER)
                trans_errors = compute_reconstruction_errors(models['global'], trans_data, GLOBAL_SCALER)
                threshold = thresholds['global']

                # Plot
                bins = np.linspace(0, max(normal_errors.max(), trans_errors.max()) * 1.1, 50)

                ax_global.hist(normal_errors, bins=bins, alpha=0.6, color='green',
                             label=f'Normal Mode {from_mode}', density=True)
                ax_global.hist(trans_errors, bins=bins, alpha=0.6, color='red',
                             label=f'Transition {from_mode}→{to_mode}', density=True)
                ax_global.axvline(threshold, color='black', linestyle='--', linewidth=2,
                                label=f'Global Threshold')

                ax_global.set_xlabel('Reconstruction Error', fontsize=10)
                ax_global.set_ylabel('Density', fontsize=10)
                ax_global.set_title(f'{title} ({source})\nGlobal Model (threshold: {threshold:.2e})',
                                  fontsize=11, fontweight='bold')
                ax_global.legend(fontsize=9)
                ax_global.grid(alpha=0.3)

                # Add detection rate
                detection_rate = np.mean(trans_errors > threshold)
                ax_global.text(0.98, 0.98, f'Detection: {detection_rate:.1%}',
                             transform=ax_global.transAxes, fontsize=10,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Mode-specific model (right)
            ax_specific = axes[idx, 1] if len(test_cases) > 1 else axes[1]

            model_key = f'mode{from_mode}'
            if model_key in models:
                # Get errors
                normal_errors = compute_reconstruction_errors(models[model_key], normal_data, per_mode_scalers[from_mode])
                trans_errors = compute_reconstruction_errors(models[model_key], trans_data, per_mode_scalers[from_mode])
                threshold = thresholds[model_key]

                # Plot
                bins = np.linspace(0, max(normal_errors.max(), trans_errors.max()) * 1.1, 50)

                ax_specific.hist(normal_errors, bins=bins, alpha=0.6, color='green',
                               label=f'Normal Mode {from_mode}', density=True)
                ax_specific.hist(trans_errors, bins=bins, alpha=0.6, color='red',
                               label=f'Transition {from_mode}→{to_mode}', density=True)
                ax_specific.axvline(threshold, color='black', linestyle='--', linewidth=2,
                                  label=f'Mode {from_mode} Threshold')

                ax_specific.set_xlabel('Reconstruction Error', fontsize=10)
                ax_specific.set_ylabel('Density', fontsize=10)
                ax_specific.set_title(f'{title} ({source})\nMode {from_mode} Model (threshold: {threshold:.2e})',
                                    fontsize=11, fontweight='bold')
                ax_specific.legend(fontsize=9)
                ax_specific.grid(alpha=0.3)

                # Add detection rate
                detection_rate = np.mean(trans_errors > threshold)
                ax_specific.text(0.98, 0.98, f'Detection: {detection_rate:.1%}',
                               transform=ax_specific.transAxes, fontsize=10,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except Exception as e:
            print(f"  Warning: Could not plot {title}: {e}")

    plt.tight_layout()
    save_path = RESULTS_DIR / 'blind_spot_error_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error distributions: {save_path}")
    plt.close()


def visualize_reconstruction_errors_over_time(models, thresholds, per_mode_scalers, use_transitions: bool = True):
    """Visualize reconstruction errors over time when factory switches modes."""

    print("\nCreating reconstruction error over time plots...")

    # Test cases: (Expected mode, Actual mode, Title, Is_correct)
    test_cases = [
        # Mode 1 scenarios
        (1, 1, "Expected: Mode 1, Actual: Mode 1 ✓", True),
        (1, 3, "Expected: Mode 1, Actual: Mode 3 ✗", False),
        (1, 5, "Expected: Mode 1, Actual: Mode 5 ✗", False),
        # Mode 3 scenarios
        (3, 3, "Expected: Mode 3, Actual: Mode 3 ✓", True),
        (3, 1, "Expected: Mode 3, Actual: Mode 1 ✗", False),
        (3, 5, "Expected: Mode 3, Actual: Mode 5 ✗", False),
        # Mode 5 scenarios
        (5, 5, "Expected: Mode 5, Actual: Mode 5 ✓", True),
        (5, 1, "Expected: Mode 5, Actual: Mode 1 ✗", False),
        (5, 3, "Expected: Mode 5, Actual: Mode 3 ✗", False),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    fig.suptitle('Reconstruction Error Over Time: Mode Detection Test\n' +
                 '(Left column: Correct mode, Middle/Right: Wrong modes)',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, (expected_mode, actual_mode, title, is_correct) in enumerate(test_cases):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        try:
            # Load transition trajectory (or steady-state fallback) from expected_mode -> actual_mode
            time, actual_data, source = get_mode_change_timeseries(expected_mode, actual_mode, use_transitions)

            # Compute reconstruction errors for each timepoint
            model_key = f'mode{expected_mode}'

            if 'global' in models and model_key in models:
                # Global model errors
                global_errors = []
                for i in range(len(actual_data)):
                    sample = actual_data[i:i+1]
                    error = compute_reconstruction_errors(models['global'], sample, GLOBAL_SCALER)
                    global_errors.append(error[0])
                global_errors = np.array(global_errors)

                # Mode-specific errors
                specific_errors = []
                for i in range(len(actual_data)):
                    sample = actual_data[i:i+1]
                    error = compute_reconstruction_errors(models[model_key], sample, per_mode_scalers[expected_mode])
                    specific_errors.append(error[0])
                specific_errors = np.array(specific_errors)

                # Plot
                ax.plot(time, global_errors, 'b-', linewidth=2, label='Global Model', alpha=0.7)
                ax.plot(time, specific_errors, 'r-', linewidth=2,
                       label=f'Mode {expected_mode} Model (Expected)', alpha=0.7)

                # Add thresholds
                ax.axhline(thresholds['global'], color='blue', linestyle='--',
                          linewidth=1.5, label=f'Global Threshold', alpha=0.7)
                ax.axhline(thresholds[model_key], color='red', linestyle='--',
                          linewidth=1.5, label=f'Mode {expected_mode} Threshold', alpha=0.7)

                ax.set_xlabel('Time (hours)', fontsize=10)
                ax.set_ylabel('Reconstruction Error', fontsize=10)
                suffix = "transition" if source == "transition" else "steady-state"
                ax.set_title(f'{title} [{suffix}]', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, loc='best')
                ax.grid(alpha=0.3)

                # Add colored border to distinguish correct vs wrong mode
                border_color = 'green' if is_correct else 'red'
                border_width = 3 if is_correct else 2
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_width)

                # Add detection rates in text box
                global_detection = np.mean(global_errors > thresholds['global'])
                specific_detection = np.mean(specific_errors > thresholds[model_key])

                if is_correct:
                    textstr = f'FALSE POSITIVES:\nGlobal: {global_detection:.1%}\nMode {expected_mode}: {specific_detection:.1%}'
                    box_color = 'lightgreen'
                else:
                    textstr = f'Detection Rate:\nGlobal: {global_detection:.1%}\nMode {expected_mode}: {specific_detection:.1%}'
                    box_color = 'wheat'

                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor=box_color, alpha=0.7))

        except Exception as e:
            print(f"  Warning: Could not plot {title}: {e}")
            ax.text(0.5, 0.5, f'Data not available\n{title}',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    save_path = RESULTS_DIR / 'reconstruction_error_over_time.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error over time plot: {save_path}")
    plt.close()


def visualize_global_errors_over_time(models, thresholds, use_transitions: bool = True):
    """Create a zoomed-in view of ONLY the global model errors for wrong-mode cases."""

    print("\nCreating global-only reconstruction error over time plots...")

    # Same test cases as in visualize_reconstruction_errors_over_time,
    # but keep only the wrong-mode scenarios.
    test_cases = [
        # Mode 1 scenarios (wrong actual mode)
        (1, 3, "Expected: Mode 1, Actual: Mode 3 ✗"),
        (1, 5, "Expected: Mode 1, Actual: Mode 5 ✗"),
        # Mode 3 scenarios (wrong actual mode)
        (3, 1, "Expected: Mode 3, Actual: Mode 1 ✗"),
        (3, 5, "Expected: Mode 3, Actual: Mode 5 ✗"),
        # Mode 5 scenarios (wrong actual mode)
        (5, 1, "Expected: Mode 5, Actual: Mode 1 ✗"),
        (5, 3, "Expected: Mode 5, Actual: Mode 3 ✗"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        'Global Model Error Over Time (Wrong Modes Only)\n'
        '(Showing only Global Model and Global Threshold)',
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )

    for idx, (expected_mode, actual_mode, title) in enumerate(test_cases):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        try:
            # Load transition trajectory (or steady-state fallback) from expected_mode -> actual_mode
            time, actual_data, source = get_mode_change_timeseries(expected_mode, actual_mode, use_transitions)

            # Global model errors only
            if 'global' in models:
                global_errors = []
                for i in range(len(actual_data)):
                    sample = actual_data[i:i+1]
                    error = compute_reconstruction_errors(models['global'], sample, GLOBAL_SCALER)
                    global_errors.append(error[0])
                global_errors = np.array(global_errors)

                # Plot global model and its threshold
                ax.plot(time, global_errors, 'b-', linewidth=2, label='Global Model', alpha=0.8)
                ax.axhline(
                    thresholds['global'],
                    color='blue',
                    linestyle='--',
                    linewidth=1.5,
                    label='Global Threshold',
                    alpha=0.8,
                )

                ax.set_xlabel('Time (hours)', fontsize=10)
                ax.set_ylabel('Reconstruction Error', fontsize=10)
                suffix = "transition" if source == "transition" else "steady-state"
                ax.set_title(f'{title} [{suffix}]', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(alpha=0.3)

                # Red border to indicate wrong mode
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)

                # Detection rate for the global model only
                detection_rate = np.mean(global_errors > thresholds['global'])
                textstr = f'Detection Rate (Global):\n{detection_rate:.1%}'

                ax.text(
                    0.02,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                )

        except Exception as e:
            print(f"  Warning: Could not plot {title}: {e}")
            ax.text(
                0.5,
                0.5,
                f'Data not available\n{title}',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )

    plt.tight_layout()
    save_path = RESULTS_DIR / 'reconstruction_error_over_time_global_only.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved global-only error over time plot: {save_path}")
    plt.close()


def visualize_threshold_comparison(per_mode_scalers=None):
    """Create a comparison plot of thresholds."""

    print("\nCreating threshold comparison...")

    per_mode_scalers = per_mode_scalers or get_per_mode_scalers()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Load models and calculate thresholds
    models = load_trained_models()

    # Calculate thresholds
    all_modes_normal = load_all_modes_normal_data()
    global_errors = compute_reconstruction_errors(models['global'], all_modes_normal, GLOBAL_SCALER)
    global_threshold = np.percentile(global_errors, 95)

    mode_thresholds = []
    mode_labels = []
    for mode in MODES:
        normal = load_normal_data(mode)
        errors = compute_reconstruction_errors(models[f'mode{mode}'], normal, per_mode_scalers[mode])
        threshold = np.percentile(errors, 95)
        mode_thresholds.append(threshold)
        mode_labels.append(f'Mode {mode}\nSpecific')

    # Plot
    x_pos = np.arange(len(mode_labels) + 1)
    thresholds = [global_threshold] + mode_thresholds
    colors = ['#1f77b4'] + ['#ff7f0e'] * len(mode_labels)
    labels = ['Global\n(ALL modes)'] + mode_labels

    bars = ax.bar(x_pos, thresholds, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, thresholds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2e}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Threshold (95th percentile)', fontsize=12)
    ax.set_title('Threshold Comparison: Global vs Mode-Specific Models',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    ax.annotate('Global threshold is TOO HIGH\n(wide acceptance range)',
               xy=(0, global_threshold), xytext=(1, global_threshold * 1.15),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    save_path = RESULTS_DIR / 'threshold_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved threshold comparison: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Blind spot test with proper thresholds")
    parser.add_argument(
        "--use-transitions",
        dest="use_transitions",
        action="store_true",
        help="Use actual ModeX->ModeY transition trajectories when available (default: True)"
    )
    parser.add_argument(
        "--steady-state",
        dest="use_transitions",
        action="store_false",
        help="Use steady-state other-mode data instead of transitions"
    )
    parser.set_defaults(use_transitions=True)
    args = parser.parse_args()

    print("="*80)
    print("BLIND SPOT TEST WITH PROPER THRESHOLD CALCULATION")
    print("="*80)
    print("\nKey Difference:")
    print("  • Global model: ONE threshold from ALL modes (realistic deployment)")
    print("  • Mode-specific: Separate threshold per mode")
    print("="*80)
    print(f"\nUsing transitions: {args.use_transitions}")

    # Load models
    print("\n1. Loading models...")
    models = load_trained_models()

    if not models:
        print("\n✗ No models found! Run train_eval.py first.")
        return

    # Build per-mode scalers so each product-aware model sees its own distribution
    print("\n2. Building per-mode scalers...")
    per_mode_scalers = get_per_mode_scalers()

    # Calculate proper thresholds
    print("\n3. Calculating thresholds...")
    thresholds = calculate_thresholds(models, per_mode_scalers)

    # Test mode changes
    print("\n4. Testing mode change detection...")
    results = test_mode_detection_proper_threshold(models, thresholds, per_mode_scalers, use_transitions=args.use_transitions)

    # Visualize
    print("\n5. Creating visualizations...")
    visualize_results(results)
    visualize_reconstruction_errors_over_time(models, thresholds, per_mode_scalers, use_transitions=args.use_transitions)
    visualize_global_errors_over_time(models, thresholds, use_transitions=args.use_transitions)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print("  • blind_spot_proper_threshold.png - Detection rate comparison")
    print("  • reconstruction_error_over_time.png - Error over time for all transitions")
    print("  • reconstruction_error_over_time_global_only.png - Global model only, wrong modes")
    print("="*80)


if __name__ == '__main__':
    main()
