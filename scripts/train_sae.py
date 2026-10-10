"""
python scripts/train_sae.py \
  --run-name natural_4b_partitioned \
  --layer 33 \
  --sparsity-mode topk \
  --top-k 512
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mas_sae.experiments.artifacts import (
    append_sae_version,
    create_sae_run_directory,
    get_git_provenance,
    update_run_manifest,
    write_run_config,
    write_sae_provenance,
    write_run_history,
    write_json_atomic,
)
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae.hyperparamters import Hyperparameters as HP
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE
from mas_sae.sae.dataloader import ActivationDataset, create_sae_dataloader
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.callbacks.checkpointing import CheckpointEvaluatorCallback, load_sae_checkpoint
from mas_sae.sae.callbacks.early_stopping import EarlyStoppingCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sparsity-mode", choices=["topk", "l1"], default="l1")
    parser.add_argument("--top-k", type=int, default=None)
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
    hp.sparsity_mode = args.sparsity_mode
    if args.top_k is not None:
        hp.top_k = args.top_k
    if hp.sparsity_mode == "topk" and (
        hp.top_k is None or not 1 <= hp.top_k <= hp.latent_dim
    ):
        parser.error(f"TopK requires 1 <= top_k <= {hp.latent_dim}.")
    seed_everything(hp.seed)
    
    git_provenance = get_git_provenance()
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
        write_sae_provenance(run_directory, activation_files, git_provenance)

        train_dataloader = create_sae_dataloader(
            hp.batch_size,
            split="train",
            num_workers=2,
            location=ACTIVATION_LOCATION,
            shuffle=hp.train_shuffle,
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
            sparsity_mode=hp.sparsity_mode,
            top_k=hp.top_k,
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
            "sparsity_loss": (
                "Mean absolute latent activation over examples and features."
                if hp.sparsity_mode == "l1" else
                "Not applied; weighted_sparsity_loss is recorded as zero for compatibility."
            ),
            "total_loss": (
                "reconstruction_loss + sparsity_coefficient * sparsity_loss"
                if hp.sparsity_mode == "l1" else "reconstruction_loss"
            ),
            "selection_metric": "val_loss",
            "variance_explained": "1 - reconstruction SSE / SSE around the training-mean activation; null if baseline SSE is zero.",
            "persistent_inactivity": "No positive activation in the last inactivity_window_examples training examples; null until that many examples have been observed. Evaluations do not update counters.",
            "firing_frequencies": "Fraction of examples with positive activation per feature; percentiles include unused features.",
            "sparsity_mechanism": (
                "L1 activation penalty" if hp.sparsity_mode == "l1" else
                "Per-example TopK of nonnegative encoder activations."
            ),
        })
        write_run_config(run_directory, config)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=hp.lr_patience,
            factor=0.1,
        )
        
        runner = ModelRunner(
            model, sparsity_coefficient=hp.sparsity_coefficient, optimizer=optimizer,
            training_mean=train_activations.float().mean(dim=0),
            inactivity_window_examples=hp.inactivity_window_examples,
        )
        
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
            epoch_metrics = {
                "epoch": epoch + 1,
                # After scheduler.step(); used for the next epoch.
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
                epoch_metrics.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            epoch_history.append(epoch_metrics)

            write_run_history(run_directory, epoch_history)
            
            if early_stopping.on_validation_end(val_metrics["loss"]):
                stop_reason = "early_stopping"
                print(f"Early stopping after epoch {epoch+1}")
                break
        
        model = load_sae_checkpoint(
            run_directory / "checkpoints" / "best_checkpoint.pt", device=device,
        )
        runner.model = model
        # Use identical frozen weights for every split; no training-state updates.
        frozen_metrics = {}
        feature_frequencies = {}
        evaluation_loaders = {"train": train_dataloader, "validation": val_dataloader}
        if test_dataloader is not None:
            evaluation_loaders["test"] = test_dataloader
        for split, dataloader in evaluation_loaders.items():
            frozen_metrics[split] = runner.val_epoch(dataloader)
            frequencies = runner.last_feature_frequencies
            if frequencies is None:
                raise RuntimeError(f"Evaluation did not produce feature frequencies for {split}.")
            feature_frequencies[split] = frequencies.tolist()
        test_metrics = frozen_metrics.get("test")
        if test_metrics is not None:
            print(f"Test Loss: {test_metrics['loss']:.4f}")

        best_epoch_metrics = min(epoch_history, key=lambda metrics: metrics["val_loss"])
                            
        summary = {
            "sparsity_mode": hp.sparsity_mode,
            "top_k": hp.top_k,
            "total_loss": config["total_loss"],
            "epochs_completed": len(epoch_history),
            "stop_reason": stop_reason,
            "best_epoch": best_epoch_metrics["epoch"],
            "frozen_checkpoint_metrics": frozen_metrics,
            "feature_frequencies_path": "feature_frequencies.json",
            "best_val_metrics": {
                key.removeprefix("val_"): value
                for key, value in best_epoch_metrics.items() if key.startswith("val_")
            },
        }

        write_json_atomic(run_directory / "feature_frequencies.json", {
            "checkpoint": "checkpoints/best_checkpoint.pt",
            "epoch": best_epoch_metrics["epoch"],
            "definition": "Fraction of split examples with a positive activation; array index is latent feature index.",
            "frequencies": feature_frequencies,
        })

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
