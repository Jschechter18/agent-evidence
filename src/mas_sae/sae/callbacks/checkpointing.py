import torch
from pathlib import Path

class CheckpointEvaluatorCallback:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float("inf")
    
    def on_validation_end(self, train_loss: float, val_loss: float, epoch: int,
                          model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau):
        """Evaluate the model checkpoint at the end of a validation epoch and save it if it has the best validation loss so far.


        Parameters
        ----------
        train_loss : float
            The training loss for the current epoch.
        val_loss : float
            The validation loss for the current epoch.
        epoch : int
            The current epoch number.
        model : torch.nn.Module
            The model being trained.
        optimizer : torch.optim.Optimizer
            The optimizer used for training the model.
        scheduler : torch.optim.lr_scheduler.StepLR
            The learning rate scheduler used during training.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': self.best_loss,
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
