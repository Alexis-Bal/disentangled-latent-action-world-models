"""
Example: Hierarchical Active Inference for DiLA.

This script demonstrates how to use the ``HierarchicalActiveInference`` planner
to jointly optimise a sub-goal (a_t) and a latent dynamic (z_t) using
Monte-Carlo Dropout for epistemic uncertainty estimation.

Two demos are provided:

1. **Standalone demo** — uses lightweight toy FDM / ActionMLP networks with
   dropout so you can run it immediately without a trained DiLA model.

2. **DiLA integration demo** — shows how to build the planner from a trained
   ``Inverse_World_model`` (skipped if the full model cannot be imported).

Run with::

    python example_active_inference.py
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from models.active_inference import (
    ActionMLP,
    HierarchicalActiveInference,
    enable_mc_dropout,
)


# ---------------------------------------------------------------------------
# Toy networks for the standalone demo
# ---------------------------------------------------------------------------

class ToyFDM(nn.Module):
    """A small forward-dynamics model with dropout.

    Predicts ``delta_s`` from ``(s, z)`` so that ``s_next = s + delta_s``.
    Dropout layers remain active during planning to gauge environmental
    uncertainty.
    """

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """s: [B, T, D], z: [B, T, Dz] -> delta_s: [B, T, D]."""
        # Flatten time for the MLP, then restore it.
        B, T, D = s.shape
        x = torch.cat([s.reshape(B * T, D), z.reshape(B * T, -1)], dim=-1)
        delta = self.net(x)
        return delta.view(B, T, D)


class ToyActionMLP(nn.Module):
    """A small action-mapping network with dropout.

    Maps ``(s, z) -> a``.  Dropout layers remain active during planning to
    gauge motor uncertainty (body curiosity / babbling).
    """

    def __init__(self, state_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """s: [B, D], z: [B, Dz] -> a: [B, A]."""
        x = torch.cat([s, z], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Demo 1: Standalone (no trained model required)
# ---------------------------------------------------------------------------

def demo_standalone():
    print("=" * 72)
    print("DEMO 1: Standalone Hierarchical Active Inference")
    print("=" * 72)

    torch.manual_seed(42)

    # --- configuration --------------------------------------------------
    state_dim = 16
    latent_action_dim = 32
    action_dim = 8
    batch_size = 2

    # --- build toy networks (with dropout) ------------------------------
    fdm = ToyFDM(state_dim=state_dim, latent_dim=latent_action_dim, hidden_dim=128, dropout=0.15)
    action_mlp = ToyActionMLP(
        state_dim=state_dim, latent_dim=latent_action_dim, action_dim=action_dim, hidden_dim=128, dropout=0.15
    )

    # --- build the planner ----------------------------------------------
    planner = HierarchicalActiveInference(
        fdm=fdm,
        action_mlp=action_mlp,
        state_shape=(state_dim,),
        latent_action_dim=latent_action_dim,
        action_dim=action_dim,
        mc_samples=5,          # N = 5 MC Dropout passes
        num_iterations=20,     # K = 20 optimisation iterations
        lr_a=1e-2,
        lr_z=1e-2,
        alpha=1.0,             # pragmatic (goal attainment)
        beta=1.0,              # action (agency alignment)
        gamma=0.5,             # prior (top-down compliance)
        delta1=0.1,            # environmental epistemic (world curiosity)
        delta2=0.1,            # motor epistemic (body curiosity)
        fdm_predicts_delta=True,
    )

    # --- create inputs --------------------------------------------------
    s_t = torch.randn(batch_size, state_dim)                       # current state
    s_goal = torch.randn(batch_size, state_dim)                    # target state
    z_prior = torch.randn(batch_size, latent_action_dim) * 0.1     # top-down prior

    print(f"\nState shape:        {tuple(s_t.shape)}")
    print(f"Goal shape:         {tuple(s_goal.shape)}")
    print(f"Latent prior shape: {tuple(z_prior.shape)}")
    print(f"MC samples (N):     {planner.mc_samples}")
    print(f"Iterations (K):     {planner.num_iterations}")
    print(f"Weights:            alpha={planner.alpha}, beta={planner.beta}, "
          f"gamma={planner.gamma}, delta1={planner.delta1}, delta2={planner.delta2}")

    # --- run single-step planning ---------------------------------------
    print("\n--- Single-step planning ---")
    a_star, z_star, info = planner.plan(
        s_t=s_t,
        s_goal=s_goal,
        z_prior=z_prior,
        return_history=True,
        verbose=True,
    )

    print("\n--- Results ---")
    print(f"Optimised action  a* shape: {tuple(a_star.shape)}")
    print(f"Optimised latent  z* shape: {tuple(z_star.shape)}")
    print("\nFinal energy terms:")
    for name, val in info["final_energies"].items():
        print(f"  {name:20s}: {val:.6f}")

    # Show convergence of the total energy
    history = info["history"]
    print(f"\nTotal energy G: {history['g_total'][0]:.4f} -> {history['g_total'][-1]:.4f}")
    print(f"Pragmatic:      {history['e_pragmatic'][0]:.4f} -> {history['e_pragmatic'][-1]:.4f}")
    print(f"Env epistemic:  {history['e_env_epistemic'][0]:.4f} -> {history['e_env_epistemic'][-1]:.4f}")
    print(f"Motor epistemic:{history['e_motor_epistemic'][0]:.4f} -> {history['e_motor_epistemic'][-1]:.4f}")

    # --- run multi-step trajectory planning -----------------------------
    print("\n--- Multi-step trajectory planning (horizon=3) ---")
    actions, latent_actions, infos = planner.plan_trajectory(
        s_t=s_t,
        s_goal=s_goal,
        horizon=3,
        z_prior=z_prior,
        return_history=False,
        verbose=False,
    )
    print(f"Planned actions shape:         {tuple(actions.shape)}")
    print(f"Planned latent actions shape:  {tuple(latent_actions.shape)}")
    for i, info_i in enumerate(infos):
        print(f"  step {i}: pragmatic={info_i['final_energies']['e_pragmatic']:.4f}, "
              f"env_epi={info_i['final_energies']['e_env_epistemic']:.4f}, "
              f"motor_epi={info_i['final_energies']['e_motor_epistemic']:.4f}")

    print("\nStandalone demo complete.\n")
    return planner


# ---------------------------------------------------------------------------
# Demo 2: Integration with a trained DiLA model
# ---------------------------------------------------------------------------

def demo_dila_integration():
    print("=" * 72)
    print("DEMO 2: Integration with a trained DiLA model")
    print("=" * 72)

    try:
        from models.model import Inverse_World_model
        from models.modules.forward_model import AdaForwardDynamics
    except Exception as exc:  # pragma: no cover
        print(f"Could not import DiLA model components ({exc}). Skipping demo 2.")
        return

    torch.manual_seed(0)

    # --- configuration matching the DiLA defaults -----------------------
    structure_dim = 128
    action_dim = 64
    patch_size = 4            # H = W = 4
    state_shape = (patch_size, patch_size, structure_dim)
    batch_size = 2

    # --- build a (randomly-initialised) FDM as in the real model -------
    fdm = AdaForwardDynamics(
        g_dim=structure_dim,
        z_dim=action_dim,
        hidden_dim=256,
        depth=2,
        dim_head=32,
        heads=4,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    # --- build a default ActionMLP --------------------------------------
    action_mlp = ActionMLP(
        state_shape=state_shape,
        latent_action_dim=action_dim,
        action_dim=action_dim,
        hidden_dim=256,
        hidden_depth=3,
        dropout=0.1,
    )

    # --- build the planner ----------------------------------------------
    planner = HierarchicalActiveInference(
        fdm=fdm,
        action_mlp=action_mlp,
        state_shape=state_shape,
        latent_action_dim=action_dim,
        action_dim=action_dim,
        mc_samples=5,
        num_iterations=15,
        lr_a=1e-2,
        lr_z=1e-2,
        alpha=1.0,
        beta=1.0,
        gamma=0.5,
        delta1=0.1,
        delta2=0.1,
        fdm_predicts_delta=True,
    )

    # --- create inputs --------------------------------------------------
    s_t = torch.randn(batch_size, *state_shape)
    s_goal = torch.randn(batch_size, *state_shape)
    z_prior = torch.randn(batch_size, action_dim) * 0.1

    print(f"\nState shape:        {tuple(s_t.shape)}")
    print(f"Goal shape:         {tuple(s_goal.shape)}")
    print(f"Latent prior shape: {tuple(z_prior.shape)}")

    # --- run planning ---------------------------------------------------
    print("\n--- Planning with DiLA FDM ---")
    a_star, z_star, info = planner.plan(
        s_t=s_t,
        s_goal=s_goal,
        z_prior=z_prior,
        return_history=True,
        verbose=False,
    )

    print("\n--- Results ---")
    print(f"Optimised action  a* shape: {tuple(a_star.shape)}")
    print(f"Optimised latent  z* shape: {tuple(z_star.shape)}")
    print("\nFinal energy terms:")
    for name, val in info["final_energies"].items():
        print(f"  {name:20s}: {val:.6f}")

    history = info["history"]
    print(f"\nTotal energy G: {history['g_total'][0]:.4f} -> {history['g_total'][-1]:.4f}")

    # --- show how to build from a full DiLA model -----------------------
    print("\n--- Building planner from a full DiLA model (from_dila_model) ---")
    print("(Using a mock object with the required attributes for illustration.)")

    class MockDilaModel(nn.Module):
        """Minimal mock that exposes the attributes used by from_dila_model."""

        def __init__(self):
            super().__init__()
            self.forward_dynamics = fdm
            self.action_dim = action_dim
            self.structure_dim = structure_dim
            self.patch_size = patch_size

    mock_model = MockDilaModel()
    planner2 = HierarchicalActiveInference.from_dila_model(
        dila_model=mock_model,
        state_shape=state_shape,
        action_dim=action_dim,
        mc_samples=5,
        num_iterations=10,
    )
    print(f"Planner built successfully. state_shape={planner2.state_shape}, "
          f"latent_action_dim={planner2.latent_action_dim}, "
          f"action_dim={planner2.action_dim}")

    print("\nDiLA integration demo complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    planner = demo_standalone()
    demo_dila_integration()
    print("=" * 72)
    print("All demos finished successfully.")
    print("=" * 72)