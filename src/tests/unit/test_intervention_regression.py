"""Regression test: run_causal_experiment must operate on RAW SAE latents.

Bug it guards against: fit_probe() returns the *standardized* validation matrix, which was being
unpacked into X_val and then fed to the SAE decoder as if it were latents (and re-standardized
for the baseline). With a perfect SAE the reconstruction-only control must flip exactly 0%.
"""

import numpy as np
import torch

from mas_sae.causal.intervention import run_causal_experiment


class _PerfectSAE:
    """encoder = ReLU, decoder = identity: decode(encode(z)) == z for any non-negative z."""

    def encoder(self, x):
        return torch.relu(x)

    def decoder(self, z):
        return z


def _data(seed=0, n=500, d=10):
    rng = np.random.default_rng(seed)
    z = np.abs(rng.normal(size=(n, d))).astype(np.float32)
    y = (z[:, 2] - z[:, 5] + 0.3 * rng.normal(size=n) > 0.2).astype(int)
    return z, y


def test_reconstruction_only_control_is_exactly_zero_for_a_perfect_sae():
    z, y = _data()
    res = run_causal_experiment([2, 5, 7], z, y, _PerfectSAE(), n_random_controls=3, seed=0)
    assert res["reconstruction_only_flip_rate"] == 0.0


def test_informative_features_flip_more_than_noise_features():
    z, y = _data()
    res = run_causal_experiment([2, 5, 7], z, y, _PerfectSAE(), n_random_controls=3, seed=0)
    assert res["features"]["2"]["suppress_flip_rate"] > 0.2          # probe relies on feature 2
    assert res["features"]["5"]["suppress_flip_rate"] > 0.2          # ...and feature 5
    assert res["features"]["7"]["suppress_flip_rate"] < 0.05         # feature 7 is pure noise
    assert res["random_control_mean_flip_rate"] < 0.1
