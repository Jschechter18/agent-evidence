"""
python scripts/train_sae.py --run-name natural_4b_layer_scan --layer 33
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mas_sae.experiments.artifacts import (
    append_sae_version,
    create_sae_run_directory,
    update_run_manifest,
    write_run_config,
    write_sae_provenance,
    write_run_history
)
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae.hyperparamters import Hyperparameters as HP
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE
from mas_sae.sae.dataloader import ActivationDataset, create_sae_dataloader
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.callbacks.checkpointing import CheckpointEvaluatorCallback
from mas_sae.sae.callbacks.early_stopping import EarlyStoppingCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default= "natural_4b_layer_scan")
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    ACTIVATION_LOCATION = (
        Path("data")
        / "activations"
        / args.run_name
        / f"layer_{args.layer:02d}"
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    results_root = PROJECT_ROOT / "results" / "sae" / "musique"
    subdirectories = ("checkpoints",)
    
    hp = HP()
    seed_everything(hp.seed)
    
    run_directory = create_sae_run_directory(
        run_name=f"sae-l{hp.latent_dim}",
        layer=args.layer,
        results_root=results_root,
        subdirectories=subdirectories,
    )
    
    try:
        activation_files = {
            split: ACTIVATION_LOCATION / f"{split}.pt"
            for split in ("train", "validation")
        }
        if (ACTIVATION_LOCATION / "test.pt").is_file():
            activation_files["test"] = ACTIVATION_LOCATION / "test.pt"
        write_sae_provenance(run_directory, activation_files)

        train_dataloader = create_sae_dataloader(
            hp.batch_size,
            split="train",
            num_workers=2,
            location=ACTIVATION_LOCATION,
        )
        val_dataloader = create_sae_dataloader(
            hp.batch_size,
            split="validation",
            num_workers=2,
            location=ACTIVATION_LOCATION,
        )

        test_dataloader = None
        if (ACTIVATION_LOCATION / "test.pt").is_file():
            test_dataloader = create_sae_dataloader(
                hp.batch_size,
                split="test",
                num_workers=2,
                location=ACTIVATION_LOCATION,
            )

        hp.input_dim = int(next(iter(train_dataloader)).shape[-1])
        
        config = {
            **asdict(hp),
            "layer": args.layer,
            "activation_run_name": args.run_name,
        }

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        model = SAE(
            input_dim=hp.input_dim,
            hidden_dim=hp.hidden_dim,
            latent_dim=hp.latent_dim,
        ).to(device)
        
        assert isinstance(train_dataloader.dataset, ActivationDataset)
        
        train_activations = train_dataloader.dataset.activations
        activation_rms_scale = train_activations.float().square().mean().sqrt()

        if not torch.isfinite(activation_rms_scale) or activation_rms_scale <= 0:
            raise ValueError("Training activations must have a finite, positive RMS.")

        with torch.no_grad():
            model.input_scale.copy_(activation_rms_scale)

        config.update({
            "activation_rms_scale": activation_rms_scale.item(),
            "input_normalization": "Divide by one RMS over all training activation elements; reuse for every split.",
            "decoder_output": "Multiply reconstruction by the training RMS to restore original units.",
            "decoder_normalization": "Unit L2 norm per column at initialization and after each optimizer step.",
            "reconstruction_loss": "Mean squared error in RMS-normalized units over examples and input dimensions.",
            "sparsity_loss": "Mean absolute latent activation over examples and features.",
            "total_loss": "reconstruction_loss + sparsity_coefficient * sparsity_loss",
            "train_shuffle": False,
        })
        write_run_config(run_directory, config)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=hp.lr_patience,
            factor=0.1,
        )
        
        runner = ModelRunner(model, sparsity_coefficient=hp.sparsity_coefficient, optimizer=optimizer)
        
        checkpoint_evaluator = CheckpointEvaluatorCallback(run_directory / "checkpoints")
        early_stopping = EarlyStoppingCallback(hp.patience)
        
        epoch_history = []
        stop_reason = "max_epochs"
        for epoch in range(hp.epochs):
            train_metrics = runner.train_epoch(train_dataloader)
            val_metrics = runner.val_epoch(val_dataloader)
            print(f"Epoch {epoch+1}/{hp.epochs} - Train Loss: {train_metrics['loss']:.4f} - Val Loss: {val_metrics['loss']:.4f}")
            
            scheduler.step(val_metrics["loss"])
            checkpoint_evaluator.on_validation_end(train_metrics, val_metrics, epoch,
                                                   model, optimizer, scheduler)
            epoch_history.append({
                "epoch": epoch+1,
                "learning_rate": optimizer.param_groups[0]["lr"],  # After scheduler.step(); used for the next epoch.
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_rec_loss": train_metrics["rec_loss"],
                "val_rec_loss": val_metrics["rec_loss"],
                "train_weighted_sparsity_loss": train_metrics["weighted_sparsity_loss"],
                "train_mean_active_features": train_metrics["mean_active_features"],
                "val_mean_active_features": val_metrics["mean_active_features"],
                "val_weighted_sparsity_loss": val_metrics["weighted_sparsity_loss"],
                "train_inactive_feature_fraction": train_metrics["inactive_feature_fraction"],
                "val_inactive_feature_fraction": val_metrics["inactive_feature_fraction"],
                })
            
            write_run_history(run_directory, epoch_history)
            
            if early_stopping.on_validation_end(val_metrics["loss"]):
                stop_reason = "early_stopping"
                print(f"Early stopping after epoch {epoch+1}")
                break
        
        test_metrics = None
        if test_dataloader is not None:
            checkpoint = torch.load(
                run_directory / "checkpoints" / "best_checkpoint.pt",
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            test_metrics = runner.test(test_dataloader)
            print(f"Test Loss: {test_metrics['loss']:.4f}")
        
        best_epoch_metrics = min(epoch_history, key=lambda metrics: metrics["val_loss"])
                            
        summary = {
            "epochs_completed": len(epoch_history),
            "stop_reason": stop_reason,
            "best_epoch": best_epoch_metrics["epoch"],
            "best_val_metrics":
                {
                    "loss": best_epoch_metrics["val_loss"],
                    "rec_loss": best_epoch_metrics["val_rec_loss"],
                    "weighted_sparsity_loss": best_epoch_metrics["val_weighted_sparsity_loss"],
                    "mean_active_features": best_epoch_metrics["val_mean_active_features"],
                    "inactive_feature_fraction": best_epoch_metrics["val_inactive_feature_fraction"],
                }
        }

        write_run_history(
            run_directory,
            epoch_history,
            test_loss=test_metrics["loss"] if test_metrics is not None else None,
            summary=summary,
        )
        
    except (Exception, KeyboardInterrupt) as error:
        update_run_manifest(
            run_directory,
            status="failed",
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
    
    update_run_manifest(run_directory, status="completed")
    append_sae_version(
        run_directory,
        project_root=PROJECT_ROOT,
        best_val_score=checkpoint_evaluator.best_loss,
        test_score=test_metrics["loss"] if test_metrics is not None else None
    )
    
if __name__ == "__main__":
    main()
