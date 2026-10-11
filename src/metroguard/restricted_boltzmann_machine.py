"""Bernoulli-Bernoulli RBM with Gibbs sampling, CD-k training and free-energy scoring."""

import pandas as pd
import torch
import torch.nn.functional as functional

INIT_STD = 0.01
HISTORY_COLUMNS = (
    "epoch",
    "fit_reconstruction_error",
    "fit_mean_free_energy",
    "validation_reconstruction_error",
    "validation_mean_free_energy",
)


def select_device() -> torch.device:
    """Use the CUDA GPU when torch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RestrictedBoltzmannMachine:
    """Energy E(v,h) = -a^T v - b^T h - v^T W h with binary visible and hidden units."""

    def __init__(
        self, n_visible: int, n_hidden: int, device: torch.device, generator: torch.Generator
    ) -> None:
        self.generator = generator
        self.weights = torch.randn(n_visible, n_hidden, device=device, generator=generator)
        self.weights *= INIT_STD
        self.visible_bias = torch.zeros(n_visible, device=device)
        self.hidden_bias = torch.zeros(n_hidden, device=device)
        self._velocity = [torch.zeros_like(p) for p in self._parameters()]

    def _parameters(self) -> list[torch.Tensor]:
        return [self.weights, self.visible_bias, self.hidden_bias]

    def hidden_probabilities(self, visible: torch.Tensor) -> torch.Tensor:
        """p(h=1 | v) = sigmoid(b + W^T v), one row per sample."""
        return torch.sigmoid(visible @ self.weights + self.hidden_bias)

    def visible_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        """p(v=1 | h) = sigmoid(a + W h), one row per sample."""
        return torch.sigmoid(hidden @ self.weights.T + self.visible_bias)

    def sample(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Draw binary samples from Bernoulli probabilities."""
        return torch.bernoulli(probabilities, generator=self.generator)

    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """F(v) = -a^T v - sum_j softplus(b_j + (v^T W)_j); higher = less typical."""
        hidden_input = visible @ self.weights + self.hidden_bias
        return -(visible @ self.visible_bias) - functional.softplus(hidden_input).sum(dim=1)

    def gibbs_negative_phase(self, visible: torch.Tensor, k: int) -> torch.Tensor:
        """k Gibbs steps starting at the data; returns the last visible probabilities."""
        current = visible
        for step in range(k):
            hidden = self.sample(self.hidden_probabilities(current))
            probabilities = self.visible_probabilities(hidden)
            current = probabilities if step == k - 1 else self.sample(probabilities)
        return current

    def cd_gradients(
        self, visible: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CD-k estimate of <v h^T>_data - <v h^T>_model for W, and the same for both biases."""
        n_samples = visible.shape[0]
        positive_hidden = self.hidden_probabilities(visible)
        negative_visible = self.gibbs_negative_phase(visible, k)
        negative_hidden = self.hidden_probabilities(negative_visible)
        weight_gradient = visible.T @ positive_hidden - negative_visible.T @ negative_hidden
        weight_gradient = weight_gradient / n_samples
        visible_gradient = (visible - negative_visible).mean(dim=0)
        hidden_gradient = (positive_hidden - negative_hidden).mean(dim=0)
        return weight_gradient, visible_gradient, hidden_gradient

    def contrastive_divergence_step(
        self,
        visible: torch.Tensor,
        k: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        """One mini-batch update with momentum and L2 weight decay on W."""
        weight_gradient, visible_gradient, hidden_gradient = self.cd_gradients(visible, k)
        weight_gradient = weight_gradient - weight_decay * self.weights
        gradients = [weight_gradient, visible_gradient, hidden_gradient]
        for parameter, velocity, gradient in zip(
            self._parameters(), self._velocity, gradients, strict=True
        ):
            velocity.mul_(momentum).add_(learning_rate * gradient)
            parameter.add_(velocity)

    def reconstruction_error(self, visible: torch.Tensor) -> torch.Tensor:
        """Mean squared error of v versus p(v | p(h | v)), averaged over units and samples."""
        reconstruction = self.visible_probabilities(self.hidden_probabilities(visible))
        return ((visible - reconstruction) ** 2).mean()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Weights and biases, ready for torch.save."""
        return {
            "weights": self.weights.cpu(),
            "visible_bias": self.visible_bias.cpu(),
            "hidden_bias": self.hidden_bias.cpu(),
        }


@torch.no_grad()
def train_rbm(
    rbm: RestrictedBoltzmannMachine,
    fit_data: torch.Tensor,
    validation_data: torch.Tensor,
    k: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> pd.DataFrame:
    """Train with CD-k on fit data; record reconstruction error and free energy per epoch."""
    rows = []
    n_samples = fit_data.shape[0]
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n_samples, device=fit_data.device, generator=rbm.generator)
        for start in range(0, n_samples, batch_size):
            batch = fit_data[order[start : start + batch_size]]
            rbm.contrastive_divergence_step(batch, k, learning_rate, momentum, weight_decay)
        rows.append(
            {
                "epoch": epoch,
                "fit_reconstruction_error": rbm.reconstruction_error(fit_data).item(),
                "fit_mean_free_energy": rbm.free_energy(fit_data).mean().item(),
                "validation_reconstruction_error": rbm.reconstruction_error(validation_data).item(),
                "validation_mean_free_energy": rbm.free_energy(validation_data).mean().item(),
            }
        )
    return pd.DataFrame(rows, columns=list(HISTORY_COLUMNS))


@torch.no_grad()
def free_energy_scores(rbm: RestrictedBoltzmannMachine, visible: torch.Tensor) -> torch.Tensor:
    """Anomaly score per window: the free energy of its visible vector."""
    return rbm.free_energy(visible)
