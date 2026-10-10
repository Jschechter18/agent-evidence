import torch
import json
from pathlib import Path
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


def load_sae_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> SparseAutoencoder:
    """Restore an SAE for inference, including its sparsity mode and RMS scale.

    Legacy checkpoints require their run's adjacent config.json. Optimizer and
    scheduler restoration is intentionally left to training-resume callers.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("model_config")
    if config is None:
        config_path = checkpoint_path.parent.parent / "config.json"
        if not config_path.is_file():
            raise ValueError("Checkpoint lacks model_config and adjacent run config.json.")
        config = json.loads(config_path.read_text())
    model = SparseAutoencoder(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        sparsity_mode=config.get("sparsity_mode", config.get("sparcity_mode", "l1")),
        top_k=config.get("top_k"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

class CheckpointEvaluatorCallback:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float("inf")
    
    def on_validation_end(self, train_metrics: dict[str, float], val_metrics: dict[str, float], epoch: int,
                          model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau):
        """Evaluate the model checkpoint at the end of a validation epoch and save it if it has the best validation loss so far.

        Parameters
        ----------
        train_metrics : dict[str, float]
            The training metrics for the current epoch.
        val_metrics : dict[str, float]
            The validation metrics for the current epoch.
        epoch : int
            The current epoch number.
        model : torch.nn.Module
            The model being trained.
        optimizer : torch.optim.Optimizer
            The optimizer used for training the model.
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The learning rate scheduler used during training.
        """
        if val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_loss': self.best_loss,
                'scheduler_state_dict': scheduler.state_dict()
            }
            if isinstance(model, SparseAutoencoder):
                checkpoint['model_config'] = {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'latent_dim': model.latent_dim,
                    'sparsity_mode': model.sparsity_mode,
                    'top_k': model.top_k,
                }
            torch.save(checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
