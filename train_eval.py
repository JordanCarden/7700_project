"""
Training and Evaluation for Blind Spot Theory Project

This module implements autoencoder training and evaluation to test the hypothesis that
product-agnostic (global) anomaly detectors create blind spots compared to product-aware models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm
import pickle

from data_loader import (
    prepare_data,
    MODES,
    TEPDataset,
    load_spvariation_normal_data,
    RANDOM_SEED,
    load_transition_samples
)


class Autoencoder(nn.Module):
    """
    Simple feedforward autoencoder for anomaly detection.

    The autoencoder learns to reconstruct normal operation patterns.
    High reconstruction error indicates anomalies.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 32, hidden_dims: List[int] = [128, 64]):
        """
        Initialize autoencoder.

        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the encoded representation
            hidden_dims: List of hidden layer dimensions
        """
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: Optional[str] = None
) -> Tuple[nn.Module, List[float]]:
    """
    Train an autoencoder model.

    Args:
        model: Autoencoder model to train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save trained model

    Returns:
        Tuple of (trained_model, loss_history)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    loss_history = []

    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            X = batch['X'].to(device)

            # Forward pass
            optimizer.zero_grad()
            X_recon = model(X)
            loss = criterion(X_recon, X)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, loss_history


def compute_reconstruction_errors(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reconstruction errors for data.

    Args:
        model: Trained autoencoder model
        data_loader: DataLoader containing data to evaluate
        device: Device to run on

    Returns:
        Tuple of (reconstruction_errors, labels, mode_labels, fault_labels)
    """
    model = model.to(device)
    model.eval()

    all_errors = []
    all_labels = []
    all_modes = []
    all_faults = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing reconstruction errors"):
            X = batch['X'].to(device)
            y = batch['y'].numpy()
            modes = batch['mode'].numpy()
            faults = batch['fault'].numpy()

            # Compute reconstruction
            X_recon = model(X)

            # Compute MSE for each sample
            errors = torch.mean((X - X_recon) ** 2, dim=1).cpu().numpy()

            all_errors.extend(errors)
            all_labels.extend(y)
            all_modes.extend(modes)
            all_faults.extend(faults)

    return (
        np.array(all_errors),
        np.array(all_labels),
        np.array(all_modes),
        np.array(all_faults)
    )


def find_optimal_threshold(
    reconstruction_errors: np.ndarray,
    labels: np.ndarray,
    percentile: float = 95
) -> float:
    """
    Find optimal threshold for anomaly detection.

    Uses percentile of reconstruction errors on normal samples.

    Args:
        reconstruction_errors: Array of reconstruction errors
        labels: Array of labels (0=normal, 1=fault)
        percentile: Percentile to use for threshold

    Returns:
        Threshold value
    """
    normal_errors = reconstruction_errors[labels == 0]
    threshold = np.percentile(normal_errors, percentile)
    return threshold


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: Optional[float] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Evaluate autoencoder model on test data.

    Args:
        model: Trained autoencoder model
        test_loader: DataLoader for test data
        threshold: Threshold for anomaly detection (if None, will be computed)
        device: Device to run on

    Returns:
        Dictionary containing evaluation metrics
    """
    # Compute reconstruction errors
    errors, labels, modes, faults = compute_reconstruction_errors(model, test_loader, device)

    # Find threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(errors, labels)

    # Make predictions
    predictions = (errors > threshold).astype(int)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(labels, errors)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    metrics = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'reconstruction_errors': errors,
        'labels': labels,
        'predictions': predictions,
        'mode_labels': modes,
        'fault_labels': faults
    }

    return metrics


def train_mode_specific_models(
    train_loader: DataLoader,
    input_dim: int,
    modes: List[int] = MODES,
    **train_kwargs
) -> Dict[int, Tuple[nn.Module, List[float]]]:
    """
    Train separate autoencoder models for each mode.

    Args:
        train_loader: DataLoader containing training data from all modes
        input_dim: Input dimension for models
        modes: List of modes to train models for
        **train_kwargs: Additional arguments to pass to train_autoencoder

    Returns:
        Dictionary mapping mode number to (model, loss_history)
    """
    mode_models = {}

    # Get all training data
    all_X = []
    all_modes = []
    for batch in train_loader:
        all_X.append(batch['X'])
        all_modes.append(batch['mode'])

    all_X = torch.cat(all_X)
    all_modes = torch.cat(all_modes)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Training Mode {mode} Autoencoder")
        print(f"{'='*60}")

        # Filter data for this mode
        mode_mask = all_modes == mode
        mode_indices = torch.where(mode_mask)[0].numpy()

        # Create mode-specific dataloader
        mode_dataset = Subset(train_loader.dataset, mode_indices)
        mode_loader = DataLoader(
            mode_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers
        )

        print(f"Training samples for Mode {mode}: {len(mode_dataset):,}")

        # Train model
        model = Autoencoder(input_dim)
        model, loss_history = train_autoencoder(
            model,
            mode_loader,
            save_path=f"mode{mode}_autoencoder.pth",
            **train_kwargs
        )

        mode_models[mode] = (model, loss_history)

    return mode_models


def train_mode_model_with_per_mode_scaler(
    mode: int,
    encoding_dim: int = 32,
    hidden_dims: List[int] = [128, 64],
    batch_size: int = 256,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = "."
) -> Tuple[nn.Module, StandardScaler, List[float]]:
    """
    Train a single mode-specific autoencoder using a scaler fit ONLY on that mode.

    Saves both the model and its scaler to disk:
      - mode{mode}_autoencoder.pth
      - mode{mode}_scaler.pkl
    """
    print(f"\n{'='*60}")
    print(f"Training Mode {mode} Autoencoder with per-mode scaler")
    print(f"{'='*60}")

    # Load clean normal data for this mode (SpVariation magnitude 100)
    data = load_spvariation_normal_data(mode, setpoint=1, magnitude=100, use_additional_meas=False)

    # Split and fit scaler on train split only
    X_train, _ = train_test_split(
        data,
        train_size=2/3,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Normalize all data with the per-mode scaler
    data_norm = scaler.transform(data)

    # Build dataset/loader (labels are dummy; only X is used)
    y = np.zeros(len(data_norm))
    mode_labels = np.full(len(data_norm), mode)
    fault_labels = np.zeros(len(data_norm))

    dataset = TEPDataset(data_norm, y, mode_labels, fault_labels)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Train model
    input_dim = data.shape[1]
    model = Autoencoder(input_dim, encoding_dim=encoding_dim, hidden_dims=hidden_dims)
    save_path = Path(save_dir) / f"mode{mode}_autoencoder.pth"
    model, loss_history = train_autoencoder(
        model,
        train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=str(save_path)
    )

    # Save scaler
    scaler_path = Path(save_dir) / f"mode{mode}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved per-mode scaler to {scaler_path}")

    return model, scaler, loss_history


def train_mode_models_with_per_mode_scalers(
    modes: List[int] = MODES,
    save_dir: str = ".",
    **train_kwargs
) -> Tuple[Dict[int, nn.Module], Dict[int, StandardScaler]]:
    """
    Train all mode-specific models with their own scalers.

    Returns dictionaries of models and scalers keyed by mode.
    """
    mode_models: Dict[int, nn.Module] = {}
    mode_scalers: Dict[int, StandardScaler] = {}

    for mode in modes:
        model, scaler, _ = train_mode_model_with_per_mode_scaler(
            mode,
            save_dir=save_dir,
            **train_kwargs
        )
        mode_models[mode] = model
        mode_scalers[mode] = scaler

    return mode_models, mode_scalers


def build_global_train_loader_with_transitions(
    train_loader: DataLoader,
    scaler: StandardScaler,
    batch_size: int,
    num_workers: int,
    include_transitions: bool = True
) -> DataLoader:
    """
    Build a global train DataLoader, optionally augmented with transition samples.

    Per-mode models should continue to use the original train_loader.
    """
    if not include_transitions:
        return train_loader

    # Base normalized arrays
    base_X = train_loader.dataset.X.cpu().numpy()
    base_y = train_loader.dataset.y.cpu().numpy()
    base_modes = train_loader.dataset.mode_labels.cpu().numpy()
    base_faults = train_loader.dataset.fault_labels.cpu().numpy()

    # Load even-indexed transition samples and scale with the global scaler
    trans_X_raw, trans_modes, trans_faults = load_transition_samples(index_mode="even")
    if trans_X_raw.size == 0:
        return train_loader

    trans_X = scaler.transform(trans_X_raw)
    trans_y = np.zeros(len(trans_X), dtype=float)

    # Concatenate
    X_all = np.vstack([base_X, trans_X])
    y_all = np.concatenate([base_y, trans_y])
    modes_all = np.concatenate([base_modes, trans_modes])
    faults_all = np.concatenate([base_faults, trans_faults])

    dataset = TEPDataset(X_all, y_all, modes_all, faults_all)
    global_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return global_loader


def compare_models(
    global_model: nn.Module,
    mode_models: Dict[int, Tuple[nn.Module, List[float]]],
    test_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Compare performance of global model vs. mode-specific models.

    Args:
        global_model: Global autoencoder trained on all modes
        mode_models: Dictionary of mode-specific autoencoders
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary containing comparison results
    """
    print(f"\n{'='*60}")
    print("COMPARING GLOBAL VS. PRODUCT-AWARE MODELS")
    print(f"{'='*60}")

    # Evaluate global model
    print("\nEvaluating Global Model...")
    global_metrics = evaluate_model(global_model, test_loader, device=device)

    print(f"\nGlobal Model Results:")
    print(f"  Precision: {global_metrics['precision']:.4f}")
    print(f"  Recall:    {global_metrics['recall']:.4f}")
    print(f"  F1 Score:  {global_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {global_metrics['roc_auc']:.4f}")

    # Evaluate mode-specific models
    mode_metrics = {}
    mode_predictions = {}

    for mode, (model, _) in mode_models.items():
        print(f"\nEvaluating Mode {mode} Model...")

        # Filter test data for this mode
        all_X = []
        all_y = []
        all_modes = []
        all_faults = []

        for batch in test_loader:
            mode_mask = batch['mode'] == mode
            if mode_mask.any():
                all_X.append(batch['X'][mode_mask])
                all_y.append(batch['y'][mode_mask])
                all_modes.append(batch['mode'][mode_mask])
                all_faults.append(batch['fault'][mode_mask])

        if not all_X:
            continue

        # Create mode-specific test dataset
        mode_X = torch.cat(all_X)
        mode_y = torch.cat(all_y)
        mode_mode_labels = torch.cat(all_modes)
        mode_fault_labels = torch.cat(all_faults)

        from torch.utils.data import TensorDataset
        mode_test_dataset = TensorDataset(mode_X, mode_y, mode_mode_labels, mode_fault_labels)
        mode_test_loader = DataLoader(mode_test_dataset, batch_size=256, shuffle=False)

        # Need to reformat batches
        class ModeTestLoader:
            def __init__(self, loader):
                self.loader = loader
                self.batch_size = loader.batch_size

            def __iter__(self):
                for X, y, modes, faults in self.loader:
                    yield {
                        'X': X,
                        'y': y,
                        'mode': modes,
                        'fault': faults
                    }

            def __len__(self):
                return len(self.loader)

        mode_test_loader = ModeTestLoader(mode_test_loader)

        metrics = evaluate_model(model, mode_test_loader, device=device)
        mode_metrics[mode] = metrics

        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Aggregate mode-specific results
    avg_precision = np.mean([m['precision'] for m in mode_metrics.values()])
    avg_recall = np.mean([m['recall'] for m in mode_metrics.values()])
    avg_f1 = np.mean([m['f1'] for m in mode_metrics.values()])
    avg_roc_auc = np.mean([m['roc_auc'] for m in mode_metrics.values()])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nGlobal Model:")
    print(f"  F1 Score: {global_metrics['f1']:.4f}")
    print(f"  ROC-AUC:  {global_metrics['roc_auc']:.4f}")
    print(f"\nProduct-Aware (Average across modes):")
    print(f"  F1 Score: {avg_f1:.4f}")
    print(f"  ROC-AUC:  {avg_roc_auc:.4f}")
    print(f"\nImprovement:")
    print(f"  F1 Score: {((avg_f1 - global_metrics['f1']) / global_metrics['f1'] * 100):+.2f}%")
    print(f"  ROC-AUC:  {((avg_roc_auc - global_metrics['roc_auc']) / global_metrics['roc_auc'] * 100):+.2f}%")

    return {
        'global': global_metrics,
        'mode_specific': mode_metrics,
        'summary': {
            'global_f1': global_metrics['f1'],
            'mode_avg_f1': avg_f1,
            'global_roc_auc': global_metrics['roc_auc'],
            'mode_avg_roc_auc': avg_roc_auc,
            'f1_improvement': ((avg_f1 - global_metrics['f1']) / global_metrics['f1'] * 100),
            'roc_auc_improvement': ((avg_roc_auc - global_metrics['roc_auc']) / global_metrics['roc_auc'] * 100)
        }
    }


def plot_results(comparison_results: Dict, save_dir: str = "results"):
    """
    Create visualizations of results.

    Args:
        comparison_results: Results from compare_models
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Comparison bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics_to_plot = ['f1', 'roc_auc']
    metric_names = ['F1 Score', 'ROC-AUC']

    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]

        global_val = comparison_results['global'][metric]
        mode_vals = [m[metric] for m in comparison_results['mode_specific'].values()]
        avg_mode_val = np.mean(mode_vals)

        x = ['Global', 'Product-Aware\n(Average)']
        y = [global_val, avg_mode_val]
        colors = ['#e74c3c', '#2ecc71']

        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(name, fontsize=12)
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11)

        ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_dir}/model_comparison.png")
    plt.close()

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))

    # Global model ROC
    global_metrics = comparison_results['global']
    fpr, tpr, _ = roc_curve(global_metrics['labels'], global_metrics['reconstruction_errors'])
    ax.plot(fpr, tpr, label=f"Global (AUC={global_metrics['roc_auc']:.4f})", linewidth=2)

    # Mode-specific ROCs
    for mode, metrics in comparison_results['mode_specific'].items():
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['reconstruction_errors'])
        ax.plot(fpr, tpr, label=f"Mode {mode} (AUC={metrics['roc_auc']:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Global vs. Product-Aware Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to {save_dir}/roc_curves.png")
    plt.close()

    print(f"\nAll visualizations saved to {save_dir}/")


def main():
    """Main execution function."""
    print("=" * 60)
    print("BLIND SPOT THEORY: ANOMALY DETECTION EXPERIMENT")
    print("=" * 60)

    # Prepare data
    print("\nStep 1: Preparing data...")
    train_loader, test_loader, scaler = prepare_data(
        use_dedicated_normal=True,  # Use clean SpVariation normal data
        sp_magnitudes=[100]  # Use nominal (100%) setpoints only
    )

    # Build global train loader augmented with transition samples (even indices only)
    print("\nStep 1b: Building global train loader with transition exposure (even-indexed samples only)...")
    global_train_loader = build_global_train_loader_with_transitions(
        train_loader,
        scaler,
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        include_transitions=True
    )

    input_dim = next(iter(global_train_loader))['X'].shape[1]
    print(f"Input dimension: {input_dim}")

    # Train global model
    print(f"\n{'='*60}")
    print("Step 2: Training Global Model")
    print(f"{'='*60}")
    global_model = Autoencoder(input_dim)
    global_model, global_loss = train_autoencoder(
        global_model,
        global_train_loader,
        num_epochs=50,
        save_path="global_autoencoder.pth"
    )

    # Train mode-specific models
    print(f"\n{'='*60}")
    print("Step 3: Training Product-Aware Models")
    print(f"{'='*60}")
    mode_models = train_mode_specific_models(
        train_loader,
        input_dim,
        num_epochs=50
    )

    # Compare models
    print(f"\n{'='*60}")
    print("Step 4: Evaluating and Comparing Models")
    print(f"{'='*60}")
    comparison_results = compare_models(global_model, mode_models, test_loader)

    # Save results
    results_summary = {
        'global_f1': float(comparison_results['summary']['global_f1']),
        'mode_avg_f1': float(comparison_results['summary']['mode_avg_f1']),
        'global_roc_auc': float(comparison_results['summary']['global_roc_auc']),
        'mode_avg_roc_auc': float(comparison_results['summary']['mode_avg_roc_auc']),
        'f1_improvement': float(comparison_results['summary']['f1_improvement']),
        'roc_auc_improvement': float(comparison_results['summary']['roc_auc_improvement'])
    }

    with open('results/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults summary saved to results/results_summary.json")

    # Create visualizations
    print(f"\n{'='*60}")
    print("Step 5: Creating Visualizations")
    print(f"{'='*60}")
    plot_results(comparison_results)

    # Train per-mode models with per-mode scalers for downstream evaluation scripts
    print(f"\n{'='*60}")
    print("Step 6: Training Per-Mode Models WITH Per-Mode Scalers")
    print(f"{'='*60}")
    train_mode_models_with_per_mode_scalers(
        num_epochs=50,
        save_dir="."
    )
    print("Per-mode models and scalers saved (modeX_autoencoder.pth, modeX_scaler.pkl)")

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
