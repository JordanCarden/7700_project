"""
TEP Dataset Loader for Blind Spot Theory Project

This module handles loading and preprocessing of the Extended Tennessee Eastman Process (TEP) dataset.
It extracts normal and fault data according to the project's sampling strategy and prepares
train/test splits for autoencoder training.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional


# Configuration Constants
DATA_PATH = "/home/admin/gdrive/TEP_Dataset"
MODES = [1, 3, 5]  # Modes to use
NUM_FAULT_TYPES = 28  # IDV1 through IDV28
NUM_NORMAL_RUNS = 30  # Runs per fault type for normal data
NUM_FAULT_RUNS = 10  # Runs per fault type for fault data
NORMAL_TIME_WINDOW = (0.0, 20.0)  # Hours 0-20
FAULT_TIME_WINDOW = (70.0, 90.0)  # Hours 70-90
SAMPLING_RATE = 0.05  # Hours (3 minutes)
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_to_indices(time_window: Tuple[float, float], sampling_rate: float = SAMPLING_RATE) -> Tuple[int, int]:
    """
    Convert time window in hours to array indices.

    Args:
        time_window: Tuple of (start_hour, end_hour)
        sampling_rate: Sampling rate in hours

    Returns:
        Tuple of (start_index, end_index)
    """
    start_idx = int(time_window[0] / sampling_rate)
    end_idx = int(time_window[1] / sampling_rate)
    return start_idx, end_idx


def append_mode_one_hot(
    data: np.ndarray,
    mode_labels: np.ndarray,
    modes: List[int] = MODES
) -> np.ndarray:
    """
    Append one-hot encoded mode columns to data.

    Args:
        data: Array of shape (samples, features)
        mode_labels: Array of shape (samples,) with mode ids
        modes: Mode ordering for one-hot encoding

    Returns:
        Array of shape (samples, features + len(modes))
    """
    mode_to_idx = {m: i for i, m in enumerate(modes)}
    one_hot = np.zeros((len(mode_labels), len(modes)))
    for i, m in enumerate(mode_labels):
        one_hot[i, mode_to_idx[m]] = 1.0
    return np.hstack([data, one_hot])


def load_transition_samples(
    modes: List[int] = MODES,
    index_mode: str = "even"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load transition trajectory samples for global-model training.

    Each available ModeXToModeY/TransitionTime{10,20,30,40} trajectory
    is a single run. We downsample by index parity to keep train/eval disjoint.

    Args:
        modes: List of modes to consider (sources and targets)
        index_mode: "even" to take samples [::2], "odd" to take samples [1::2]

    Returns:
        data: (N, features) process variables (time column dropped)
        mode_labels: (N,) destination mode id for stratification/metadata
        fault_labels: (N,) zeros (treated as normal)
    """
    assert index_mode in ("even", "odd"), "index_mode must be 'even' or 'odd'"
    start_idx = 0 if index_mode == "even" else 1

    all_data = []
    all_modes = []
    all_faults = []

    for src_mode in modes:
        file_path = f"{DATA_PATH}/TEP_Mode{src_mode}.h5"
        if not Path(file_path).exists():
            continue

        with h5py.File(file_path, "r") as f:
            base = f.get(f"Mode{src_mode}/ModeTransition/SimulationCompleted", None)
            if base is None:
                continue

            for tgt_mode in modes:
                if tgt_mode == src_mode:
                    continue

                group = f"Mode{src_mode}/ModeTransition/SimulationCompleted/Mode{src_mode}ToMode{tgt_mode}"
                if group not in f:
                    continue

                for trans_time in f[group].keys():
                    path = f"{group}/{trans_time}"
                    if f[path].get("processdata") is None:
                        continue

                    processdata = f[path]["processdata"][:]
                    data = processdata[start_idx::2, 1:]  # drop time, parity slice

                    if data.size == 0:
                        continue

                    all_data.append(data)
                    all_modes.extend([tgt_mode] * data.shape[0])
                    all_faults.extend([0] * data.shape[0])

    if not all_data:
        return np.empty((0, 0)), np.array([]), np.array([])

    data_array = np.vstack(all_data)
    mode_labels = np.array(all_modes, dtype=int)
    fault_labels = np.array(all_faults, dtype=int)

    return data_array, mode_labels, fault_labels


def load_tep_simulation(
    mode: int,
    fault_type: int,
    run_number: int,
    time_window: Tuple[float, float],
    use_additional_meas: bool = False
) -> np.ndarray:
    """
    Load a single TEP simulation run and extract specified time window.

    Args:
        mode: Operating mode (1, 3, or 5)
        fault_type: Fault type (1-28 for IDV1-IDV28)
        run_number: Run number (1-200)
        time_window: Tuple of (start_hour, end_hour) to extract
        use_additional_meas: Whether to include additional measurements

    Returns:
        Array of shape (timesteps, features) containing process data
    """
    file_path = f"{DATA_PATH}/TEP_Mode{mode}.h5"

    with h5py.File(file_path, 'r') as f:
        # Construct path to simulation data
        path = f'Mode{mode}/SingleFault/SimulationCompleted/IDV{fault_type}/Mode{mode}_IDVInfo_{fault_type}_100/Run{run_number}'

        # Load process data
        processdata = f[path]['processdata'][:]

        # Extract time window
        start_idx, end_idx = time_to_indices(time_window)
        data = processdata[start_idx:end_idx, 1:]  # Exclude time column (column 0)

        # Optionally include additional measurements
        if use_additional_meas:
            additional_data = f[path]['additional_meas'][start_idx:end_idx, :]
            data = np.concatenate([data, additional_data], axis=1)

    return data


def load_spvariation_normal_data(
    mode: int,
    setpoint: int = 1,
    magnitude: int = 100,
    use_additional_meas: bool = False
) -> np.ndarray:
    """
    Load dedicated normal operation data from SpVariation path.

    This loads "clean" normal data that never has faults injected,
    addressing the criticism that pre-fault data may be contaminated.

    Args:
        mode: Operating mode (1, 3, or 5)
        setpoint: Setpoint number (1-12, though all are identical at magnitude 100)
        magnitude: Setpoint magnitude (85-115, use 100 for nominal)
        use_additional_meas: Whether to include additional measurements

    Returns:
        Array of shape (timesteps, features) containing normal operation data
    """
    file_path = f"{DATA_PATH}/TEP_Mode{mode}.h5"

    with h5py.File(file_path, 'r') as f:
        # Path to dedicated normal operation
        path = f'Mode{mode}/SpVariation/SimulationCompleted/SP{setpoint}/tRamp_0/SpMagnitude{magnitude}'

        # Load process data
        processdata = f[path]['processdata'][:]
        data = processdata[:, 1:]  # Exclude time column

        # Optionally include additional measurements
        if use_additional_meas:
            additional_data = f[path]['additional_meas'][:]
            data = np.concatenate([data, additional_data], axis=1)

    return data


def extract_dedicated_normal_data(
    modes: List[int] = MODES,
    magnitudes: List[int] = [100],
    use_additional_meas: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract dedicated normal operation data from SpVariation (clean normal data).

    This is the RECOMMENDED approach that addresses the criticism about using
    pre-fault data. All SP datasets at magnitude 100 are identical, so we only
    load SP1 for each mode.

    Args:
        modes: List of modes to extract from
        magnitudes: List of SpMagnitudes to use (e.g., [95, 100, 105] for variety)
        use_additional_meas: Whether to include additional measurements

    Returns:
        Tuple of (data, mode_labels)
        - data: Array of shape (total_samples, features)
        - mode_labels: Array indicating which mode each sample came from
    """
    all_data = []
    all_mode_labels = []

    print(f"Extracting dedicated normal data (SpMagnitude{magnitudes})...")

    for mode in modes:
        print(f"  Processing Mode {mode}...")
        for magnitude in magnitudes:
            # Load SP1 (all SPs are identical at same magnitude)
            data = load_spvariation_normal_data(mode, setpoint=1, magnitude=magnitude,
                                               use_additional_meas=use_additional_meas)
            all_data.append(data)

            # Create labels for each timestep
            num_timesteps = data.shape[0]
            all_mode_labels.extend([mode] * num_timesteps)

            print(f"    Loaded SpMagnitude{magnitude}: {num_timesteps} timesteps")

    # Concatenate all data
    data_array = np.vstack(all_data)
    mode_labels = np.array(all_mode_labels)

    print(f"  Extracted {data_array.shape[0]} normal samples with {data_array.shape[1]} features")

    return data_array, mode_labels


def extract_normal_data_prefault(
    modes: List[int] = MODES,
    num_runs: int = NUM_NORMAL_RUNS,
    time_window: Tuple[float, float] = NORMAL_TIME_WINDOW,
    use_additional_meas: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normal operation data from fault runs (pre-fault period).

    Args:
        modes: List of modes to extract from
        num_runs: Number of runs to extract per fault type
        time_window: Time window to extract (hours 0-20 by default)
        use_additional_meas: Whether to include additional measurements

    Returns:
        Tuple of (data, mode_labels, fault_type_labels)
        - data: Array of shape (total_samples, features)
        - mode_labels: Array indicating which mode each sample came from
        - fault_type_labels: Array indicating which fault type run each sample came from
    """
    all_data = []
    all_mode_labels = []
    all_fault_labels = []

    print(f"Extracting normal data from hours {time_window[0]}-{time_window[1]}...")

    for mode in modes:
        print(f"  Processing Mode {mode}...")
        for fault_type in range(1, NUM_FAULT_TYPES + 1):
            # Skip IDV6 as it's in SimulationStopped (emergency shutdowns)
            if fault_type == 6:
                continue
            for run in range(1, num_runs + 1):
                try:
                    data = load_tep_simulation(mode, fault_type, run, time_window, use_additional_meas)
                    all_data.append(data)

                    # Create labels for each timestep
                    num_timesteps = data.shape[0]
                    all_mode_labels.extend([mode] * num_timesteps)
                    all_fault_labels.extend([fault_type] * num_timesteps)

                except Exception as e:
                    print(f"    Warning: Failed to load Mode{mode}/IDV{fault_type}/Run{run}: {e}")

    # Concatenate all data
    data_array = np.vstack(all_data)
    mode_labels = np.array(all_mode_labels)
    fault_labels = np.array(all_fault_labels)

    print(f"  Extracted {data_array.shape[0]} normal samples with {data_array.shape[1]} features")

    return data_array, mode_labels, fault_labels


def extract_fault_data(
    modes: List[int] = MODES,
    num_runs: int = NUM_FAULT_RUNS,
    time_window: Tuple[float, float] = FAULT_TIME_WINDOW,
    use_additional_meas: bool = False,
    stratified_sampling: bool = False,
    sample_timepoints: Optional[List[float]] = None,
    sample_runs: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fault operation data from fault runs (post-fault period).

    Args:
        modes: List of modes to extract from
        num_runs: Number of runs to extract per fault type (if not using stratified sampling)
        time_window: Time window to extract (hours 70-90 by default)
        use_additional_meas: Whether to include additional measurements
        stratified_sampling: If True, sample specific timepoints from specific runs
        sample_timepoints: Specific time points to sample (e.g., [70, 75, 80, 85, 90])
        sample_runs: Specific runs to sample from (e.g., [1, 2, 3, 4, 5])

    Returns:
        Tuple of (data, mode_labels, fault_type_labels)
        - data: Array of shape (total_samples, features)
        - mode_labels: Array indicating which mode each sample came from
        - fault_type_labels: Array indicating which fault type each sample represents
    """
    all_data = []
    all_mode_labels = []
    all_fault_labels = []

    if stratified_sampling:
        if sample_timepoints is None:
            sample_timepoints = [70.0, 75.0, 80.0, 85.0, 90.0]  # Default 5 timepoints
        if sample_runs is None:
            sample_runs = [1, 2, 3, 4, 5]  # Default first 5 runs

        print(f"Extracting fault data with stratified sampling:")
        print(f"  Timepoints: {sample_timepoints}")
        print(f"  Runs: {sample_runs}")
        print(f"  Samples per (mode, fault_type): {len(sample_timepoints)} × {len(sample_runs)} = {len(sample_timepoints) * len(sample_runs)}")

        for mode in modes:
            print(f"  Processing Mode {mode}...")
            for fault_type in range(1, NUM_FAULT_TYPES + 1):
                # Skip IDV6 as it's in SimulationStopped (emergency shutdowns)
                if fault_type == 6:
                    continue

                # Sample specific timepoints from specific runs
                for run in sample_runs:
                    for timepoint in sample_timepoints:
                        try:
                            # Load single timepoint window (±0.1 hours for tolerance)
                            window = (timepoint - 0.1, timepoint + 0.1)
                            data = load_tep_simulation(mode, fault_type, run, window, use_additional_meas)

                            # Take middle sample if multiple samples in window
                            if data.shape[0] > 0:
                                mid_idx = data.shape[0] // 2
                                sample = data[mid_idx:mid_idx+1]  # Keep 2D shape

                                all_data.append(sample)
                                all_mode_labels.append(mode)
                                all_fault_labels.append(fault_type)

                        except Exception as e:
                            print(f"    Warning: Failed to load Mode{mode}/IDV{fault_type}/Run{run}/Time{timepoint}: {e}")

    else:
        # Original behavior: extract all data from specified runs
        print(f"Extracting fault data from hours {time_window[0]}-{time_window[1]}...")

        for mode in modes:
            print(f"  Processing Mode {mode}...")
            for fault_type in range(1, NUM_FAULT_TYPES + 1):
                # Skip IDV6 as it's in SimulationStopped (emergency shutdowns)
                if fault_type == 6:
                    continue
                for run in range(1, num_runs + 1):
                    try:
                        data = load_tep_simulation(mode, fault_type, run, time_window, use_additional_meas)
                        all_data.append(data)

                        # Create labels for each timestep
                        num_timesteps = data.shape[0]
                        all_mode_labels.extend([mode] * num_timesteps)
                        all_fault_labels.extend([fault_type] * num_timesteps)

                    except Exception as e:
                        print(f"    Warning: Failed to load Mode{mode}/IDV{fault_type}/Run{run}: {e}")

    # Concatenate all data
    if stratified_sampling:
        data_array = np.vstack(all_data)
        mode_labels = np.array(all_mode_labels)
        fault_labels = np.array(all_fault_labels)
    else:
        data_array = np.vstack(all_data)
        mode_labels = np.array(all_mode_labels)
        fault_labels = np.array(all_fault_labels)

    print(f"  Extracted {data_array.shape[0]} fault samples with {data_array.shape[1]} features")

    return data_array, mode_labels, fault_labels


def create_train_test_split(
    normal_data: np.ndarray,
    normal_mode_labels: np.ndarray,
    normal_fault_labels: np.ndarray,
    fault_data: np.ndarray,
    fault_mode_labels: np.ndarray,
    fault_fault_labels: np.ndarray,
    train_ratio: float = 2/3,
    seed: int = RANDOM_SEED
) -> Dict[str, np.ndarray]:
    """
    Create train/test split following anomaly detection best practices.
    Training set: 2/3 of normal data only (stratified by mode)
    Test set: 1/3 of normal data + all fault data

    Uses stratified sampling to ensure balanced mode representation in both
    training and test sets.

    Args:
        normal_data: Normal operation data
        normal_mode_labels: Mode labels for normal data
        normal_fault_labels: Fault type labels for normal data
        fault_data: Fault operation data
        fault_mode_labels: Mode labels for fault data
        fault_fault_labels: Fault type labels for fault data
        train_ratio: Ratio of normal data to use for training
        seed: Random seed for shuffling

    Returns:
        Dictionary containing train/test splits and labels
    """
    # Use stratified train_test_split to ensure balanced mode representation
    # This ensures each mode gets exactly train_ratio in training and (1-train_ratio) in test
    X_train, X_test_normal, mode_train, mode_test_normal, fault_train, fault_test_normal = train_test_split(
        normal_data,
        normal_mode_labels,
        normal_fault_labels,
        train_size=train_ratio,
        stratify=normal_mode_labels,  # Stratify by mode to ensure balanced split
        random_state=seed
    )

    # Training labels (all normal)
    y_train = np.zeros(len(X_train))

    # Test set: normal data (held out) + all fault data
    X_test = np.vstack([X_test_normal, fault_data])

    # Test labels: 0 for normal, 1 for fault
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(fault_data.shape[0])
    ])

    # Combine mode labels for test set
    mode_test = np.concatenate([mode_test_normal, fault_mode_labels])

    # Combine fault labels for test set
    fault_test = np.concatenate([fault_test_normal, fault_fault_labels])

    # Print split summary with mode breakdown
    print(f"\nDataset Split Summary (Stratified by Mode):")
    print(f"  Training:   {X_train.shape[0]:,} samples (100% normal)")

    # Show per-mode distribution in training set
    unique_modes_train, counts_train = np.unique(mode_train, return_counts=True)
    print(f"    Mode distribution:")
    for mode, count in zip(unique_modes_train, counts_train):
        print(f"      Mode {mode}: {count:,} samples ({count/len(mode_train)*100:.1f}%)")

    print(f"  Test:       {X_test.shape[0]:,} samples ({len(X_test_normal):,} normal + {fault_data.shape[0]:,} fault)")

    # Show per-mode distribution in test set (normal portion only)
    unique_modes_test, counts_test = np.unique(mode_test_normal, return_counts=True)
    print(f"    Mode distribution (normal portion):")
    for mode, count in zip(unique_modes_test, counts_test):
        print(f"      Mode {mode}: {count:,} samples ({count/len(mode_test_normal)*100:.1f}%)")

    print(f"  Features:   {X_train.shape[1]}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'mode_train': mode_train,
        'fault_train': fault_train,
        'X_test': X_test,
        'y_test': y_test,
        'mode_test': mode_test,
        'fault_test': fault_test
    }


class TEPDataset(Dataset):
    """PyTorch Dataset for TEP data."""

    def __init__(self, X: np.ndarray, y: np.ndarray, mode_labels: np.ndarray, fault_labels: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mode_labels = torch.LongTensor(mode_labels)
        self.fault_labels = torch.LongTensor(fault_labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx],
            'mode': self.mode_labels[idx],
            'fault': self.fault_labels[idx]
        }


def normalize_data(
    data_splits: Dict[str, np.ndarray],
    save_scaler: bool = True,
    scaler_path: str = "scaler.pkl"
) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """
    Normalize data using StandardScaler fitted on training data only.

    Args:
        data_splits: Dictionary containing train/test splits
        save_scaler: Whether to save the scaler to disk
        scaler_path: Path to save scaler

    Returns:
        Tuple of (normalized_splits, scaler)
    """
    scaler = StandardScaler()

    # Fit on training data only
    scaler.fit(data_splits['X_train'])

    # Transform both train and test
    normalized_splits = data_splits.copy()
    normalized_splits['X_train'] = scaler.transform(data_splits['X_train'])
    normalized_splits['X_test'] = scaler.transform(data_splits['X_test'])

    if save_scaler:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")

    return normalized_splits, scaler


def get_dataloaders(
    data_splits: Dict[str, np.ndarray],
    batch_size: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train and test sets.

    Args:
        data_splits: Dictionary containing normalized train/test splits
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = TEPDataset(
        data_splits['X_train'],
        data_splits['y_train'],
        data_splits['mode_train'],
        data_splits['fault_train']
    )

    test_dataset = TEPDataset(
        data_splits['X_test'],
        data_splits['y_test'],
        data_splits['mode_test'],
        data_splits['fault_test']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader


def prepare_data(
    use_dedicated_normal: bool = True,
    sp_magnitudes: List[int] = [100],
    use_additional_meas: bool = False,
    save_preprocessed: bool = True,
    preprocessed_path: str = "preprocessed_data.pkl",
    stratified_fault_sampling: bool = True,
    fault_sample_timepoints: Optional[List[float]] = None,
    fault_sample_runs: Optional[List[int]] = None,
    use_mode_one_hot: bool = False
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Main function to prepare all data: extract, split, normalize, and create dataloaders.

    Args:
        use_dedicated_normal: If True, use SpVariation normal data (RECOMMENDED).
                             If False, use pre-fault data from fault runs.
        sp_magnitudes: List of SpMagnitude values to use (e.g., [100] or [95,100,105])
                      Only used if use_dedicated_normal=True.
        use_additional_meas: Whether to include additional measurements
        save_preprocessed: Whether to save preprocessed data
        preprocessed_path: Path to save preprocessed data
        stratified_fault_sampling: If True, use stratified sampling for fault data (default: True)
        fault_sample_timepoints: Specific timepoints to sample (default: [70, 75, 80, 85, 90])
        fault_sample_runs: Specific runs to sample from (default: [1, 2, 3, 4, 5])
        use_mode_one_hot: If True, append one-hot mode indicators to feature vectors

    Returns:
        Tuple of (train_loader, test_loader, scaler)
    """
    set_seed()

    # Extract normal and fault data
    if use_dedicated_normal:
        print("Using DEDICATED NORMAL DATA (SpVariation - clean normal)")
        print(f"  SpMagnitudes: {sp_magnitudes}")
        normal_data, normal_modes = extract_dedicated_normal_data(
            magnitudes=sp_magnitudes,
            use_additional_meas=use_additional_meas
        )
        # Create dummy fault labels (all zeros) for compatibility
        normal_faults = np.zeros(len(normal_modes), dtype=int)
    else:
        print("Using PRE-FAULT DATA (hours 0-20 from fault runs)")
        normal_data, normal_modes, normal_faults = extract_normal_data_prefault(
            use_additional_meas=use_additional_meas
        )

    fault_data, fault_modes, fault_faults = extract_fault_data(
        use_additional_meas=use_additional_meas,
        stratified_sampling=stratified_fault_sampling,
        sample_timepoints=fault_sample_timepoints,
        sample_runs=fault_sample_runs
    )

    if use_mode_one_hot:
        normal_data = append_mode_one_hot(normal_data, normal_modes)
        fault_data = append_mode_one_hot(fault_data, fault_modes)

    # Create train/test split
    data_splits = create_train_test_split(
        normal_data, normal_modes, normal_faults,
        fault_data, fault_modes, fault_faults
    )

    # Normalize data
    normalized_splits, scaler = normalize_data(data_splits)

    # Save preprocessed data
    if save_preprocessed:
        with open(preprocessed_path, 'wb') as f:
            pickle.dump(normalized_splits, f)
        print(f"Preprocessed data saved to {preprocessed_path}")

    # Create dataloaders
    train_loader, test_loader = get_dataloaders(normalized_splits)

    return train_loader, test_loader, scaler


if __name__ == "__main__":
    print("=" * 60)
    print("TEP Data Loader - Blind Spot Theory Project")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("MODE 1: Dedicated Normal Data (RECOMMENDED)")
    print("=" * 60)

    # Prepare data with dedicated normal (clean data)
    train_loader, test_loader, scaler = prepare_data(
        use_dedicated_normal=True,
        sp_magnitudes=[100]  # Use only nominal setpoints
    )

    print(f"\nData preparation complete!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Show a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shape: {sample_batch['X'].shape}")
    print(f"Sample labels shape: {sample_batch['y'].shape}")

    print("\n" + "=" * 60)
    print("Alternative: To use pre-fault data instead:")
    print("=" * 60)
    print("train_loader, test_loader, scaler = prepare_data(")
    print("    use_dedicated_normal=False  # Use hours 0-20 from fault runs")
    print(")")
    print("\nOr to use multiple magnitudes for more data:")
    print("train_loader, test_loader, scaler = prepare_data(")
    print("    use_dedicated_normal=True,")
    print("    sp_magnitudes=[95, 100, 105]  # ±5% variation")
    print(")")
