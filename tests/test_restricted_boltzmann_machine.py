import itertools

import pytest
import torch

from metroguard.restricted_boltzmann_machine import RestrictedBoltzmannMachine, train_rbm

CPU = torch.device("cpu")


def make_rbm(n_visible: int = 4, n_hidden: int = 3, seed: int = 0) -> RestrictedBoltzmannMachine:
    generator = torch.Generator(device=CPU).manual_seed(seed)
    rbm = RestrictedBoltzmannMachine(n_visible, n_hidden, CPU, generator)
    rbm.weights = torch.randn(n_visible, n_hidden, generator=generator)
    rbm.visible_bias = torch.randn(n_visible, generator=generator)
    rbm.hidden_bias = torch.randn(n_hidden, generator=generator)
    return rbm


def brute_force_free_energy(rbm: RestrictedBoltzmannMachine, visible: torch.Tensor) -> torch.Tensor:
    """-log sum_h exp(-E(v,h)) by enumerating every hidden configuration."""
    n_hidden = rbm.hidden_bias.shape[0]
    result = []
    for v in visible:
        energies = []
        for h_tuple in itertools.product([0.0, 1.0], repeat=n_hidden):
            h = torch.tensor(h_tuple)
            energy = -(v @ rbm.visible_bias) - (h @ rbm.hidden_bias) - (v @ rbm.weights @ h)
            energies.append(-energy)
        result.append(-torch.logsumexp(torch.stack(energies), dim=0))
    return torch.stack(result)


def all_visible_vectors(n_visible: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=n_visible)))


def test_free_energy_matches_brute_force_enumeration():
    rbm = make_rbm()
    visible = all_visible_vectors(4)
    assert torch.allclose(
        rbm.free_energy(visible), brute_force_free_energy(rbm, visible), atol=1e-5
    )


def test_free_energy_is_stable_for_large_inputs():
    rbm = make_rbm()
    rbm.weights = rbm.weights * 1000
    energy = rbm.free_energy(all_visible_vectors(4))
    assert torch.isfinite(energy).all()


def test_conditional_probabilities_have_right_shape_and_range():
    rbm = make_rbm()
    visible = all_visible_vectors(4)
    hidden = rbm.hidden_probabilities(visible)
    assert hidden.shape == (16, 3)
    assert ((hidden > 0) & (hidden < 1)).all()
    back = rbm.visible_probabilities(hidden)
    assert back.shape == (16, 4)
    assert ((back > 0) & (back < 1)).all()


def test_sample_is_binary():
    rbm = make_rbm()
    samples = rbm.sample(rbm.hidden_probabilities(all_visible_vectors(4)))
    assert set(samples.unique().tolist()) <= {0.0, 1.0}


def test_cd_gradient_has_expected_direction_on_zero_rbm():
    generator = torch.Generator(device=CPU).manual_seed(0)
    rbm = RestrictedBoltzmannMachine(4, 3, CPU, generator)
    rbm.weights.zero_()
    data = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 8)
    weight_gradient, visible_gradient, hidden_gradient = rbm.cd_gradients(data, k=1)
    assert torch.allclose(weight_gradient[:2], torch.full((2, 3), 0.25))
    assert torch.allclose(weight_gradient[2:], torch.full((2, 3), -0.25))
    assert torch.allclose(visible_gradient, torch.tensor([0.5, 0.5, -0.5, -0.5]))
    assert torch.allclose(hidden_gradient, torch.zeros(3))


@pytest.mark.parametrize("k", [1, 5])
def test_training_lowers_free_energy_of_the_data(k):
    generator = torch.Generator(device=CPU).manual_seed(1)
    rbm = RestrictedBoltzmannMachine(4, 3, CPU, generator)
    data = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 64 + [[0.0, 0.0, 1.0, 1.0]] * 64)
    before = rbm.free_energy(data).mean().item()
    history = train_rbm(rbm, data, data, k, 30, 32, 0.1, 0.5, 0.0)
    assert rbm.free_energy(data).mean().item() < before
    assert (
        history["fit_reconstruction_error"].iloc[-1] < history["fit_reconstruction_error"].iloc[0]
    )
    unseen = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert rbm.free_energy(unseen).item() > rbm.free_energy(data[:1]).item()
