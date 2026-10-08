from dataclasses import dataclass

@dataclass
class Hyperparameters:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 256
    train_shuffle: bool = True
    
    lr: float = 3e-4
    
    input_dim: int = 768
    hidden_dim: int = 8
    latent_dim: int = 12888
    
    patience: int = 20
    lr_patience: int = 5

    sparsity_coefficient: float = 1e-2
