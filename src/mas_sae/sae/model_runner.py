import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE

import torch.optim as optim


class ModelRunner:
    def __init__(self, model: SAE, sparsity_coefficient: float, optimizer: optim.Optimizer):
        self.model = model
        self.sparsity_coefficient = sparsity_coefficient
        
        self.optimizer = optimizer
    
    
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
        float
            The average loss over the epoch.
        """
        running_loss = 0
        running_rec_loss = 0
        running_weighted_sparsity_loss = 0
        running_active_counts = 0
        num_examples = 0
        
        features_seen = torch.zeros(
            self.model.latent_dim,
            dtype=torch.bool,
            device=next(self.model.parameters()).device,
        )
        
        self.model.train() if training_mode else self.model.eval()
        
        description = "Training" if training_mode else "Evaluating"
        progress_bar = tqdm(dataloader, desc=description, leave=False)
        
        for _, batch in enumerate(progress_bar, start=1):
            with torch.set_grad_enabled(training_mode):
                outputs = self._common(batch)
                
                active_in_batch = (outputs["sparse_features"] > 0).any(dim=0)
                features_seen |= active_in_batch # or = operator
            
            if training_mode:
                self.optimizer.zero_grad()
                outputs['loss'].backward()
                self.optimizer.step()
                self.model.normalize_decoder_weights()
                
            batch_size = batch.shape[0]
            running_loss += outputs['loss'].item() * batch_size
            running_rec_loss += outputs['rec_loss'].item() * batch_size
            running_weighted_sparsity_loss += outputs['weighted_sparsity_loss'].item() * batch_size
            running_active_counts += (outputs['sparse_features'] > 0).sum().item()
            num_examples += batch_size
            
        if num_examples == 0:
            raise ValueError("Cannot evaluate an empty dataset.")

        return {
            "loss": running_loss / num_examples,
            "rec_loss": running_rec_loss / num_examples,
            "weighted_sparsity_loss": running_weighted_sparsity_loss / num_examples,
            "mean_active_features": running_active_counts / num_examples,
            "inactive_feature_fraction": (~features_seen).float().mean().item(),
        }
    
    def train_epoch(self, dataloader: DataLoader):
        """Runs one epoch under full training conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        float
            The average loss over the epoch.
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
        float
            The average loss over the epoch.
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
        float
            The average loss over the epoch.
        """
        return self._run_epoch(dataloader, training_mode=False)
