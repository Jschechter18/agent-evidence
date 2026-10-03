from dataclasses import dataclass

@dataclass
class Hyperparameters:
    epochs: int = 200
    batch_size: int = 256
    
    lr: float = 3e-4
    
    input_dim: int = 768
    hidden_dim: int = 8
    latent_dim: int = 12888
    
    patience: int = 5
    lr_patience: int = 2

    sparsity_coefficient: float = 1e-3
