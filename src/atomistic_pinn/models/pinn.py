"""Network architectures of paper Table 1 (Sprint 3).

Two plain fully connected networks with two hidden layers:

* PINN  ``60 → 15 → 15 → 8``  — maps each atom's 60 descriptors to the eight
  BOP parameters (1283 parameters, Table 1);
* NN    ``60 → 16 → 16 → 1``  — maps each atom's descriptors directly to its
  atomic energy (1265 parameters, Table 1), bypassing the physics layer.

The activation function is not specified in the paper (ledger §7.4); the
labeled decision is ``tanh`` (smooth everywhere, matching the
differentiable-physics design) with ``sigmoid`` kept behind the config flag.
Weight initialization: Xavier-uniform weights and zero biases under a seeded
generator — the paper says only "random states".

All modules run in float64: the BOP physics they feed is float64.
"""

from __future__ import annotations

import torch
from torch import nn

from atomistic_pinn.exceptions import ConfigurationError

SUPPORTED_ACTIVATIONS = ("tanh", "sigmoid")


def _make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ConfigurationError(
        f"unknown activation {name!r}; supported: {SUPPORTED_ACTIVATIONS} (ledger §7.4)"
    )


def _build_sequential(
    layer_sizes: tuple[int, ...],
    activation: str,
    seed: int,
) -> nn.Sequential:
    """Fully connected stack with Xavier-uniform weights and zero biases.

    Initialization is drawn from a private seeded generator so identical
    seeds give bit-identical networks (restart reproducibility).
    """
    generator = torch.Generator().manual_seed(int(seed))
    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1], dtype=torch.float64)
        nn.init.xavier_uniform_(linear.weight, generator=generator)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i < len(layer_sizes) - 2:
            layers.append(_make_activation(activation))
    return nn.Sequential(*layers)


class PINN(nn.Module):
    """Descriptor → BOP-parameter network, ``60 × 15 × 15 × 8``."""

    def __init__(self, seed: int = 0, activation: str = "tanh") -> None:
        super().__init__()
        self.network = _build_sequential((60, 15, 15, 8), activation, seed)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """``(N, 60)`` descriptors → ``(N, 8)`` BOP parameters per atom."""
        return self.network(descriptors.to(torch.float64))


class BaselineNN(nn.Module):
    """Descriptor → atomic-energy network, ``60 × 16 × 16 × 1``."""

    def __init__(self, seed: int = 0, activation: str = "tanh") -> None:
        super().__init__()
        self.network = _build_sequential((60, 16, 16, 1), activation, seed)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """``(N, 60)`` descriptors → ``(N, 1)`` atomic energies (eV/atom)."""
        return self.network(descriptors.to(torch.float64))
