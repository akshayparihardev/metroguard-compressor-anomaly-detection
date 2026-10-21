"""Small MLP variational autoencoder: training with early stopping and anomaly scores."""

import copy

import pandas as pd
import torch
from torch import nn

RECONSTRUCTION_ERROR = "reconstruction_error"
NEGATIVE_ELBO = "negative_elbo"
SCORE_NAMES = (RECONSTRUCTION_ERROR, NEGATIVE_ELBO)
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "train_reconstruction",
    "train_kl",
    "validation_loss",
    "validation_reconstruction",
    "validation_kl",
)


def mlp(sizes: list[int]) -> nn.Sequential:
    """Fully connected layers with ReLU between them (no activation after the last layer)."""
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VariationalAutoencoder(nn.Module):
    """MLP encoder to (mu, log sigma^2), reparameterisation, MLP decoder."""

    def __init__(self, n_features: int, hidden_sizes: list[int], latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(mlp([n_features, *hidden_sizes]), nn.ReLU())
        self.mean_head = nn.Linear(hidden_sizes[-1], latent_dim)
        self.log_variance_head = nn.Linear(hidden_sizes[-1], latent_dim)
        self.decoder = mlp([latent_dim, *reversed(hidden_sizes), n_features])

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance of q(z | x)."""
        hidden = self.encoder(features)
        return self.mean_head(hidden), self.log_variance_head(hidden)

    def forward(
        self, features: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruction, mean and log-variance; z is sampled (training) or the mean (scoring)."""
        mean, log_variance = self.encode(features)
        if not sample:
            return self.decoder(mean), mean, log_variance
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return self.decoder(latent), mean, log_variance


def loss_terms(
    features: torch.Tensor,
    reconstruction: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample squared error summed over features, and KL(q(z|x) || N(0, I))."""
    squared_error = ((features - reconstruction) ** 2).sum(dim=1)
    kl = -0.5 * (1 + log_variance - mean**2 - log_variance.exp()).sum(dim=1)
    return squared_error, kl


@torch.no_grad()
def evaluate_loss(
    model: VariationalAutoencoder, data: torch.Tensor, beta: float
) -> tuple[float, float, float]:
    """Mean loss, reconstruction and KL per sample with z = mean of q(z | x)."""
    model.eval()
    reconstruction, mean, log_variance = model(data, sample=False)
    squared_error, kl = loss_terms(data, reconstruction, mean, log_variance)
    return (squared_error + beta * kl).mean().item(), squared_error.mean().item(), kl.mean().item()


def train_vae(
    model: VariationalAutoencoder,
    fit_data: torch.Tensor,
    validation_data: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    patience: int,
) -> pd.DataFrame:
    """Adam training; early stopping on fit_validation loss; best weights are restored."""
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(fit_data.shape[0], device=fit_data.device)
        totals = torch.zeros(3, device=fit_data.device)
        for start in range(0, fit_data.shape[0], batch_size):
            batch = fit_data[order[start : start + batch_size]]
            reconstruction, mean, log_variance = model(batch)
            squared_error, kl = loss_terms(batch, reconstruction, mean, log_variance)
            loss = (squared_error + beta * kl).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            totals += torch.stack([loss, squared_error.mean(), kl.mean()]).detach() * len(batch)
        train_loss, train_reconstruction, train_kl = (totals / fit_data.shape[0]).tolist()
        validation_loss, validation_reconstruction, validation_kl = evaluate_loss(
            model, validation_data, beta
        )
        rows.append(
            dict(
                zip(
                    HISTORY_COLUMNS,
                    [
                        epoch,
                        train_loss,
                        train_reconstruction,
                        train_kl,
                        validation_loss,
                        validation_reconstruction,
                        validation_kl,
                    ],
                    strict=True,
                )
            )
        )
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    model.load_state_dict(best_state)
    return pd.DataFrame(rows, columns=list(HISTORY_COLUMNS))


@torch.no_grad()
def anomaly_scores(model: VariationalAutoencoder, data: torch.Tensor) -> torch.Tensor:
    """Columns: mean squared reconstruction error (primary) and negative ELBO (secondary)."""
    model.eval()
    reconstruction, mean, log_variance = model(data, sample=False)
    squared_error, kl = loss_terms(data, reconstruction, mean, log_variance)
    return torch.stack([squared_error / data.shape[1], squared_error + kl], dim=1)


@torch.no_grad()
def latent_means(model: VariationalAutoencoder, data: torch.Tensor) -> torch.Tensor:
    """Mean of q(z | x) for each row: where the encoder places a window in the latent space."""
    model.eval()
    return model.encode(data)[0]
