import torch
from torch import nn, Tensor


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim # added this for clarity
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder_layer = nn.Sequential(
            nn.Linear(latent_dim, self.output_dim)
        )
        self.normalize_decoder_weights()
        
    def encoder(self, activation: Tensor) -> Tensor:
        """Encoder layer of model. Used to convert activation to sparse feature vector.

        Parameters
        ----------
        activation : Tensor
            Activation output is the input of the encoder.

        Returns
        -------
        Tensor
            Sparse feature vector. This is a sparse representation of the activation input.
        """
        return self.encoder_layer(activation)
        
    
    def decoder(self, sparse_features: Tensor) -> Tensor:
        """Decoder layer of model. Used to project sparse feature vector back to original activation vector representation.

        Parameters
        ----------
        sparse_features : Tensor
            Sparse feature vector, output of the encoder.

        Returns
        -------
        Tensor
            Reconstructed activation vector. This is the reconstructed version of the original activation vector from the sparse feature representation.
        """
        return self.decoder_layer(sparse_features)
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize each decoder column to prevent sparsity-penalty scaling."""
        decoder_weight = self.decoder_layer[0].weight
        norms = torch.linalg.vector_norm(decoder_weight, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder_layer[0].weight.data = decoder_weight / norms
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass through model.

        Parameters
        ----------
        x : Tensor
            Input activation vector.

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple containing:
            - Sparse feature vector (output of the encoder)
            - Reconstructed activation vector (output of the decoder)
        """
        sparse_features = self.encoder(x)
        reconstructed_activation = self.decoder(sparse_features)
        
        return sparse_features, reconstructed_activation
