import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE

import torch.optim as optim


class ModelRunner:
    def __init__(
        self, model: SAE, sparsity_coefficient: float, optimizer: optim.Optimizer,
        training_mean: torch.Tensor | None = None,
        inactivity_window_examples: int = 10000,
    ):
        if inactivity_window_examples < 1:
            raise ValueError("inactivity_window_examples must be positive.")
        self.model = model
        self.sparsity_coefficient = sparsity_coefficient
        self.training_mean = training_mean.detach().clone() if training_mean is not None else None
        self.inactivity_window_examples = inactivity_window_examples
        self.training_examples_seen = 0
        self.last_fired_example: torch.Tensor | None = None
        self.last_feature_frequencies: torch.Tensor | None = None
        
        self.optimizer = optimizer

    def _track_training_activity(self, active: torch.Tensor):
        """Record the last training-example position at which each feature fired."""
        if self.last_fired_example is None:
            self.last_fired_example = torch.zeros(self.model.latent_dim, dtype=torch.long)
        positions = torch.arange(1, active.shape[0] + 1, device=active.device)[:, None]
        last_positions = (active * positions).amax(dim=0).cpu()
        fired = last_positions > 0
        self.last_fired_example[fired] = self.training_examples_seen + last_positions[fired]
        self.training_examples_seen += active.shape[0]
    
    
    def _loss_fn(self, reconstructed: torch.Tensor, batch: torch.Tensor, sparse_features: torch.Tensor):
        scale = self.model.input_scale
        rec_loss = F.mse_loss(reconstructed / scale, batch / scale)
        
        if self.model.sparsity_mode == "l1":
            l1_penalty = sparse_features.abs().mean()
            weighted_sparsity_loss = self.sparsity_coefficient * l1_penalty
        else:
            weighted_sparsity_loss = rec_loss.new_zeros(())

        loss = rec_loss + weighted_sparsity_loss
        
        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "weighted_sparsity_loss": weighted_sparsity_loss
        }

    
    def _common(self, batch: torch.Tensor):
        """Run a single forward pass and compute the loss.

        Parameters
        ----------
        batch : torch.Tensor
            Single batch from dataloader of size `batch_size`. The shape is (batch_size, input_size).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Returns tuple of loss, reconstructed activation, sparse_features
        """
        device = next(self.model.parameters()).device
        activations = batch.to(device) # batch.shape = (batch_size, input_size)
        
        sparse_features, reconstructed = self.model(activations)
        
        losses = self._loss_fn(reconstructed, activations, sparse_features)

        return {'loss': losses["loss"],
                'rec_loss': losses["rec_loss"],
                'weighted_sparsity_loss': losses['weighted_sparsity_loss'],
                'reconstructed': reconstructed,
                'sparse_features': sparse_features}
    
    def _run_epoch(self, dataloader: DataLoader, training_mode: bool = False):
        """Run a single epoch for any evaluation mode.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.
        training_mode : bool, optional
            Whether to run the epoch in training mode or eval mode., by default False

        Returns
        -------
        dict
            Average losses, activity statistics, and available reconstruction diagnostics.
        """
        running_loss = 0
        running_rec_loss = 0
        running_weighted_sparsity_loss = 0
        num_examples = 0
        running_baseline_loss = 0.0
        device = next(self.model.parameters()).device
        feature_counts = torch.zeros(self.model.latent_dim, dtype=torch.long, device=device)
        mean = self.training_mean.to(device) if self.training_mean is not None else None
        
        self.model.train() if training_mode else self.model.eval()
        
        description = "Training" if training_mode else "Evaluating"
        progress_bar = tqdm(dataloader, desc=description, leave=False)
        
        for _, batch in enumerate(progress_bar, start=1):
            with torch.set_grad_enabled(training_mode):
                outputs = self._common(batch)
                
                active = outputs["sparse_features"].detach() > 0
                feature_counts += active.sum(dim=0)
                if mean is not None:
                    residual = (batch.to(device) - mean) / self.model.input_scale
                    running_baseline_loss += residual.square().mean().item() * batch.shape[0]
                if training_mode:
                    self._track_training_activity(active)
            
            if training_mode:
                self.optimizer.zero_grad()
                outputs['loss'].backward()
                self.optimizer.step()
                self.model.normalize_decoder_weights()
                
            batch_size = batch.shape[0]
            running_loss += outputs['loss'].item() * batch_size
            running_rec_loss += outputs['rec_loss'].item() * batch_size
            running_weighted_sparsity_loss += outputs['weighted_sparsity_loss'].item() * batch_size
            num_examples += batch_size
            
        if num_examples == 0:
            raise ValueError("Cannot evaluate an empty dataset.")

        frequencies = feature_counts.float().cpu() / num_examples
        self.last_feature_frequencies = frequencies
        percentiles = torch.quantile(frequencies, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
        metrics = {
            "loss": running_loss / num_examples,
            "rec_loss": running_rec_loss / num_examples,
            "weighted_sparsity_loss": running_weighted_sparsity_loss / num_examples,
            "mean_active_features": feature_counts.sum().item() / num_examples,
            "inactive_feature_fraction": (feature_counts == 0).float().mean().item(),
            "firing_frequency_min": percentiles[0].item(),
            "firing_frequency_p50": percentiles[1].item(),
            "firing_frequency_p90": percentiles[2].item(),
            "firing_frequency_p99": percentiles[3].item(),
            "firing_frequency_max": percentiles[4].item(),
        }
        if self.training_mean is not None:
            baseline = running_baseline_loss / num_examples
            metrics["mean_baseline_rec_loss"] = baseline
            metrics["variance_explained"] = 1 - metrics["rec_loss"] / baseline if baseline > 0 else None
        if training_mode:
            last_fired = self.last_fired_example
            if last_fired is None:
                raise RuntimeError("Training did not initialize feature activity counters.")
            metrics["persistent_inactive_feature_fraction"] = (
                (self.training_examples_seen - last_fired >= self.inactivity_window_examples)
                .float().mean().item()
                if self.training_examples_seen >= self.inactivity_window_examples else None
            )
            metrics["training_examples_seen"] = self.training_examples_seen
        return metrics
    
    def train_epoch(self, dataloader: DataLoader):
        """Runs one epoch under full training conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        dict
            Epoch losses and diagnostics, including training inactivity counters.
        """
        return self._run_epoch(dataloader, training_mode=True)
    
    def val_epoch(self, dataloader: DataLoader):
        """Runs one epoch under full validation conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        dict
            Evaluation losses and diagnostics without updating training counters.
        """
        return self._run_epoch(dataloader, training_mode=False)
    
    def test(self, dataloader: DataLoader):
        """Runs one full forward pass under full test conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        dict
            Evaluation losses and diagnostics without updating training counters.
        """
        return self._run_epoch(dataloader, training_mode=False)
