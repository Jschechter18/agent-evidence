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
        rec_loss = F.mse_loss(reconstructed, batch)
        sparsity_loss = sparse_features.abs().mean()
        
        return rec_loss + self.sparsity_coefficient * sparsity_loss
    
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
        
        loss = self._loss_fn(reconstructed, activations, sparse_features)
        
        return loss, reconstructed, sparse_features
    
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
        
        self.model.train() if training_mode else self.model.eval()
        
        description = "Training" if training_mode else "Evaluating"
        progress_bar = tqdm(dataloader, desc=description, leave=False)
        
        # for batch in dataloader:
        for _, batch in enumerate(progress_bar, start=1):
            with torch.set_grad_enabled(training_mode):
                loss, _, _ = self._common(batch)
            
            if training_mode:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.normalize_decoder_weights()
                
            running_loss += loss.item()
            
        final_loss = running_loss / len(dataloader)
        
        return final_loss
    
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
        return self._run_epoch(dataloader)
    
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
        return self._run_epoch(dataloader)
