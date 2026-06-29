"""
Hierarchical Active Inference for DiLA.

Implements joint optimization of sub-goal (a_t) and latent dynamic (z_t)
using Monte Carlo Dropout for epistemic uncertainty estimation.

Reference: "Hierarchical Active Inference in DiLA: Joint Optimization Synthesis"

The higher-level DiLA agent uses a *flattened joint optimization* approach to
simultaneously infer the agent's optimal sub-goal (a_t) and the world's latent
dynamic response (z_t).  By incorporating Monte Carlo Dropout in both the
forward dynamics (FDM) and action mapping (ActionMLP), the system achieves
true curiosity-driven Active Inference—exploring both environmental physics
and its own motor capabilities—without requiring computationally heavy
ensembles.

Energy terms
------------
1. **Pragmatic Energy** (goal attainment):
       E_pragmatic = || s_mean - s_goal ||_2^2

2. **Environmental Epistemic Energy** (world curiosity):
       E_env_epistemic = (1/N) * sum_i || s_pred^(i) - s_mean ||_2^2

3. **Motor Epistemic Energy** (body curiosity / babbling):
       E_motor_epistemic = (1/N) * sum_i || a_pred^(i) - a_mean ||_2^2

4. **Action Energy** (agency alignment):
       E_action = || a_t - a_mean ||_2^2

5. **Prior Energy** (top-down compliance):
       E_prior = || z_t - z_prior ||_2^2

Expected Free Energy objective (note the *minus* signs for epistemic terms):

    G_total = alpha * E_pragmatic
            + beta  * E_action
            + gamma * E_prior
            - delta1 * E_env_epistemic
            - delta2 * E_motor_epistemic
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Default ActionMLP
# ---------------------------------------------------------------------------

class ActionMLP(nn.Module):
    """Default action-mapping network: (state, latent_action) -> physical_action.

    Maps from the current structure embedding ``s`` and the global latent
    dynamic ``z`` to the agent's required physical action ``a``.

    **Dropout layers remain active** during the planning phase to gauge motor
    uncertainty (body curiosity / babbling).

    Parameters
    ----------
    state_shape : tuple[int, ...] | int
        Shape of the structure embedding *without* the batch dimension,
        e.g. ``(H, W, C)`` for spatial states or ``D`` for flat states.
    latent_action_dim : int
        Dimension ``D`` of the latent action ``z``.
    action_dim : int
        Dimension ``A`` of the physical action ``a``.
    hidden_dim : int
        Hidden layer dimension.
    hidden_depth : int
        Number of hidden residual blocks.
    dropout : float
        Dropout rate — kept active during planning for MC sampling.
    """

    def __init__(
        self,
        state_shape: Union[Tuple[int, ...], int],
        latent_action_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- compute flat state dimension -----------------------------------
        if isinstance(state_shape, (tuple, list)):
            self.state_dim = 1
            for d in state_shape:
                self.state_dim *= d
            self.state_shape: Tuple[int, ...] = tuple(state_shape)
        else:
            self.state_dim = int(state_shape)
            self.state_shape = (int(state_shape),)

        self.latent_action_dim = latent_action_dim
        self.action_dim = action_dim

        # --- state projection (with dropout for MC sampling) ----------------
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # --- latent-action projection (with dropout for MC sampling) --------
        self.action_proj = nn.Sequential(
            nn.Linear(latent_action_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # --- hidden residual blocks (with dropout for MC sampling) ----------
        blocks: List[nn.Module] = []
        for _ in range(hidden_depth):
            blocks.extend([
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*blocks) if blocks else nn.Identity()

        # --- output projection ----------------------------------------------
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            Structure embedding, shape ``[B, *state_shape]``.
        z : torch.Tensor
            Latent action, shape ``[B, D]`` (or ``[B, 1, D]``).

        Returns
        -------
        torch.Tensor
            Physical action, shape ``[B, A]``.
        """
        # Flatten state if it has spatial dims
        if s.dim() > 2:
            s = s.flatten(1)                                  # [B, prod(state_shape)]
        # Squeeze z if it has a time dim
        if z.dim() > 2:
            z = z.squeeze(1)                                  # [B, D]

        s_emb = self.state_proj(s)                            # [B, hidden]
        z_emb = self.action_proj(z)                           # [B, hidden]
        x = torch.cat([s_emb, z_emb], dim=-1)                 # [B, hidden*2]
        x = self.mlp(x)                                       # [B, hidden*2]
        return self.output(x)                                 # [B, A]


# ---------------------------------------------------------------------------
# MC Dropout helper
# ---------------------------------------------------------------------------

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
)


def enable_mc_dropout(model: nn.Module) -> int:
    """Enable dropout layers in *model* for Monte Carlo sampling.

    Sets every ``Dropout`` module to **train** mode while keeping all other
    modules in **eval** mode (important so that BatchNorm / LayerNorm
    statistics are not updated).

    Parameters
    ----------
    model : nn.Module
        The (frozen) model whose dropout layers should be re-activated.

    Returns
    -------
    int
        Number of dropout layers that were enabled.
    """
    count = 0
    model.eval()                          # disable everything first
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()                     # re-enable dropout
            count += 1
    return count


@torch.no_grad()
def suggest_init_from_inverse_model(
    dila_model: nn.Module,
    action_mlp: nn.Module,
    s_t: torch.Tensor,
    s_goal: torch.Tensor,
    z_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warm-start ``(z, a)`` for planning toward *s_goal*.

    The active-inference planner initialises the latent dynamic ``z_t`` to zeros
    by default.  Near ``z = 0`` the (frozen) forward dynamics model is largely
    insensitive, so the pragmatic energy ``‖s_pred − s_goal‖²`` barely moves and
    the optimised action collapses to ~``ActionMLP(s, 0)`` (the agent does
    nothing).  This helper fixes that by initialising ``z_t`` to the latent
    action DiLA's *inverse model* expects to transform ``s_t`` into ``s_goal``
    -- i.e. the latent dynamic that points at the goal -- rescaled to a typical
    in-distribution magnitude so the FDM actually responds.  The matching
    physical action ``a = ActionMLP(s_t, z)`` is returned as the action
    warm-start.

    This is an *initialisation* only; it does **not** add a new energy term, so
    the single-level objective remains ``α·E_pragmatic + β·E_action``.

    Parameters
    ----------
    dila_model : nn.Module
        A trained :class:`Inverse_World_model` (uses ``Inverse_model``).
    action_mlp : nn.Module
        The trained ``(s, z) -> a`` mapping.
    s_t, s_goal : torch.Tensor
        Current and goal structure embeddings ``[B, *state_shape]``.
    z_norm : float | None
        Target L2 norm for ``z`` (e.g. the mean training latent-action norm).
        If ``None``, ``z`` is used as-is.

    Returns
    -------
    z : torch.Tensor  ``[B, D]``
    a : torch.Tensor  ``[B, A]``
    """
    g_prev = s_t.unsqueeze(1)                       # [B, 1, *state_shape]
    g_next = s_goal.unsqueeze(1)
    z = dila_model.Inverse_model(g_prev, g_next)    # [B, 1, D]
    if z.dim() > 2:
        z = z.squeeze(1)                            # [B, D]
    if z_norm is not None:
        n = z.norm(dim=1, keepdim=True).clamp(min=1e-6)
        z = z * (float(z_norm) / n)
    a = action_mlp(s_t, z)                          # [B, A]
    return z, a


# ---------------------------------------------------------------------------
# Hierarchical Active Inference planner
# ---------------------------------------------------------------------------

class HierarchicalActiveInference(nn.Module):
    """Hierarchical Active Inference planner for DiLA.

    Uses *flattened joint optimization* to simultaneously infer:

    * **a_t** — the agent's optimal sub-goal / macro-action.
    * **z_t** — the world's latent dynamic response.

    By incorporating **Monte Carlo Dropout** in both the forward dynamics
    (FDM) and action mapping (ActionMLP), the system achieves curiosity-driven
    Active Inference—exploring both environmental physics and its own motor
    capabilities—without requiring computationally heavy ensembles.

    Parameters
    ----------
    fdm : nn.Module
        Forward Dynamics Model.  Called as ``fdm(s, z)`` where
        ``s`` is ``[B, T, *state_shape]`` and ``z`` is ``[B, T, D]``.
        Returns either the *delta* (``[B, T, *state_shape]``) or the next
        state directly, depending on *fdm_predicts_delta*.
    action_mlp : nn.Module
        Action-mapping network.  Called as ``action_mlp(s, z)`` where
        ``s`` is ``[B, *state_shape]`` and ``z`` is ``[B, D]``.
        Returns ``[B, A]``.
    state_shape : tuple[int, ...]
        Shape of the structure embedding *without* batch/time dims,
        e.g. ``(H, W, C)`` or ``(D,)`` for flat states.
    latent_action_dim : int
        Dimension ``D`` of the latent action ``z``.
    action_dim : int
        Dimension ``A`` of the physical action ``a``.
    mc_samples : int
        Number ``N`` of MC Dropout forward passes (default ``5``).
    num_iterations : int
        Number ``K`` of optimisation iterations (default ``20``).
    lr_a : float
        Learning rate ``eta_a`` for the action ``a_t``.
    lr_z : float
        Learning rate ``eta_z`` for the latent dynamic ``z_t``.
    alpha : float
        Weight for pragmatic energy (goal attainment).
    beta : float
        Weight for action energy (agency alignment).
    gamma : float
        Weight for prior energy (top-down compliance).
    delta1 : float
        Weight for environmental epistemic energy (world curiosity).
    delta2 : float
        Weight for motor epistemic energy (body curiosity).
    fdm_predicts_delta : bool
        If ``True`` (default), the FDM returns ``delta_s`` and
        ``s_next = s + delta_s``.  If ``False``, the FDM returns ``s_next``
        directly.
    max_grad_norm : float | None
        Optional maximum gradient norm for clipping (default ``None``).
    """

    def __init__(
        self,
        fdm: nn.Module,
        action_mlp: nn.Module,
        state_shape: Tuple[int, ...],
        latent_action_dim: int,
        action_dim: int,
        mc_samples: int = 5,
        num_iterations: int = 20,
        lr_a: float = 1e-2,
        lr_z: float = 1e-2,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta1: float = 0.1,
        delta2: float = 0.1,
        fdm_predicts_delta: bool = True,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.fdm = fdm
        self.action_mlp = action_mlp

        # Freeze model parameters (weights are frozen during inference)
        for param in self.fdm.parameters():
            param.requires_grad_(False)
        for param in self.action_mlp.parameters():
            param.requires_grad_(False)

        self.state_shape = tuple(state_shape)
        self.latent_action_dim = latent_action_dim
        self.action_dim = action_dim

        # --- planning hyper-parameters --------------------------------------
        self.mc_samples = mc_samples
        self.num_iterations = num_iterations
        self.lr_a = lr_a
        self.lr_z = lr_z

        # --- energy weights -------------------------------------------------
        self.alpha = alpha      # pragmatic
        self.beta = beta        # action
        self.gamma = gamma      # prior
        self.delta1 = delta1    # env epistemic
        self.delta2 = delta2    # motor epistemic

        self.fdm_predicts_delta = fdm_predicts_delta
        self.max_grad_norm = max_grad_norm

        # --- warn if no dropout layers are present --------------------------
        fdm_dropout = enable_mc_dropout(self.fdm)
        action_dropout = enable_mc_dropout(self.action_mlp)
        if fdm_dropout == 0:
            warnings.warn(
                "FDM has no dropout layers — environmental epistemic energy "
                "will be 0.  Consider adding dropout to the FDM for "
                "curiosity-driven exploration.",
                UserWarning,
                stacklevel=2,
            )
        if action_dropout == 0:
            warnings.warn(
                "ActionMLP has no dropout layers — motor epistemic energy "
                "will be 0.  Consider adding dropout to the ActionMLP for "
                "motor babbling.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # MC Dropout forward pass
    # ------------------------------------------------------------------

    def _mc_forward(
        self,
        s_t: torch.Tensor,
        z_t: torch.Tensor,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run *N* forward passes with MC Dropout active.

        Because dropout is active, the exact same inputs yield slightly
        different predictions, capturing epistemic uncertainty.

        Parameters
        ----------
        s_t : torch.Tensor
            Current state ``[B, 1, *state_shape]`` (with time dim).
        z_t : torch.Tensor
            Candidate latent action ``[B, 1, D]`` (with time dim).
        n_samples : int
            Number of MC samples ``N``.

        Returns
        -------
        s_preds : torch.Tensor
            Predicted next states ``[N, B, *state_shape]``.
        a_preds : torch.Tensor
            Predicted actions ``[N, B, A]``.
        """
        s_preds: List[torch.Tensor] = []
        a_preds: List[torch.Tensor] = []

        for _ in range(n_samples):
            # --- FDM forward pass (dropout active) --------------------------
            fdm_out = self.fdm(s_t, z_t)                       # [B, 1, *state_shape]

            if self.fdm_predicts_delta:
                s_pred = s_t + fdm_out
            else:
                s_pred = fdm_out

            # --- ActionMLP forward pass (dropout active) --------------------
            # Remove the time dimension for the ActionMLP
            s_flat = s_t.squeeze(1)                            # [B, *state_shape]
            z_flat = z_t.squeeze(1)                            # [B, D]
            a_pred = self.action_mlp(s_flat, z_flat)           # [B, A]

            s_preds.append(s_pred.squeeze(1))                  # [B, *state_shape]
            a_preds.append(a_pred)                             # [B, A]

        return torch.stack(s_preds, dim=0), torch.stack(a_preds, dim=0)

    # ------------------------------------------------------------------
    # Free-energy computation
    # ------------------------------------------------------------------

    def _compute_energies(
        self,
        s_t: torch.Tensor,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        s_goal: torch.Tensor,
        z_prior: Optional[torch.Tensor],
        s_preds: torch.Tensor,
        a_preds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all free-energy terms.

        Parameters
        ----------
        s_t : torch.Tensor
            Current state ``[B, *state_shape]``.
        z_t : torch.Tensor
            Candidate latent action ``[B, D]``.
        a_t : torch.Tensor
            Candidate action ``[B, A]``.
        s_goal : torch.Tensor
            Goal state ``[B, *state_shape]``.
        z_prior : torch.Tensor | None
            Prior on latent action ``[B, D]`` or ``None``.
        s_preds : torch.Tensor
            MC predictions of next state ``[N, B, *state_shape]``.
        a_preds : torch.Tensor
            MC predictions of action ``[N, B, A]``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys ``e_pragmatic``, ``e_env_epistemic``,
            ``e_motor_epistemic``, ``e_action``, ``e_prior``,
            ``s_mean``, ``a_mean``.
        """
        # Mean predictions
        s_mean = s_preds.mean(dim=0)                           # [B, *state_shape]
        a_mean = a_preds.mean(dim=0)                           # [B, A]

        # Flatten all dims except batch for distance computation
        s_mean_flat = s_mean.flatten(1)                        # [B, prod]
        s_goal_flat = s_goal.flatten(1)                        # [B, prod]
        s_preds_flat = s_preds.flatten(2)                      # [N, B, prod]

        # 1. Pragmatic Energy (Goal Attainment)
        #    E_pragmatic = || s_mean - s_goal ||_2^2
        e_pragmatic = ((s_mean_flat - s_goal_flat) ** 2).sum(dim=1).mean()

        # 2. Environmental Epistemic Energy (World Curiosity)
        #    E_env_epistemic = (1/N) * sum_i || s_pred^(i) - s_mean ||_2^2
        e_env_epistemic = (
            ((s_preds_flat - s_mean_flat.unsqueeze(0)) ** 2)
            .sum(dim=2)
            .mean()
        )

        # 3. Motor Epistemic Energy (Body Curiosity / Babbling)
        #    E_motor_epistemic = (1/N) * sum_i || a_pred^(i) - a_mean ||_2^2
        e_motor_epistemic = (
            ((a_preds - a_mean.unsqueeze(0)) ** 2)
            .sum(dim=2)
            .mean()
        )

        # 4. Action Energy (Agency Alignment)
        #    E_action = || a_t - a_mean ||_2^2
        e_action = ((a_t - a_mean) ** 2).sum(dim=1).mean()

        # 5. Prior Energy (Top-Down Compliance)
        #    E_prior = || z_t - z_prior ||_2^2
        if z_prior is not None:
            e_prior = ((z_t - z_prior) ** 2).sum(dim=1).mean()
        else:
            e_prior = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)

        return {
            "e_pragmatic": e_pragmatic,
            "e_env_epistemic": e_env_epistemic,
            "e_motor_epistemic": e_motor_epistemic,
            "e_action": e_action,
            "e_prior": e_prior,
            "s_mean": s_mean,
            "a_mean": a_mean,
        }

    def _compute_total_energy(
        self,
        energies: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the total Expected Free Energy.

        .. math::
            G_{total} = \\alpha \\, E_{pragmatic}
                      + \\beta  \\, E_{action}
                      + \\gamma \\, E_{prior}
                      - \\delta_1 \\, E_{env\\_epistemic}
                      - \\delta_2 \\, E_{motor\\_epistemic}

        Note the **minus signs** for the epistemic energies: the model wants
        to *minimise* prediction error while *maximising* information gain
        about both the world and itself.
        """
        g_total = (
            self.alpha * energies["e_pragmatic"]
            + self.beta * energies["e_action"]
            + self.gamma * energies["e_prior"]
            - self.delta1 * energies["e_env_epistemic"]
            - self.delta2 * energies["e_motor_epistemic"]
        )
        return g_total

    # ------------------------------------------------------------------
    # Single-step planning
    # ------------------------------------------------------------------

    def plan(
        self,
        s_t: torch.Tensor,
        s_goal: torch.Tensor,
        z_prior: Optional[torch.Tensor] = None,
        a_init: Optional[torch.Tensor] = None,
        return_history: bool = True,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Run the active-inference planning loop.

        For each planning step, performs *K* iterations of joint optimisation
        to settle on the optimal ``(a_t, z_t)`` pair.

        Parameters
        ----------
        s_t : torch.Tensor
            Current structure embedding ``[B, *state_shape]``.
        s_goal : torch.Tensor
            Target structure embedding ``[B, *state_shape]``.
        z_prior : torch.Tensor | None
            Top-down expectation of the global transition ``[B, D]``.
            If ``None``, ``z_t`` is initialised to zeros.
        a_init : torch.Tensor | None
            Warm start from the previous plan ``[B, A]``.
            If ``None``, ``a_t`` is initialised randomly.
        return_history : bool
            If ``True``, return the history of energy terms.
        verbose : bool
            If ``True``, print energy terms at each iteration.

        Returns
        -------
        a_star : torch.Tensor
            Optimised sub-goal ``[B, A]``.
        z_star : torch.Tensor
            Optimised latent dynamic ``[B, D]``.
        info : dict
            Dictionary with final energy terms and (optionally) history.
        """
        device = s_t.device
        dtype = s_t.dtype
        B = s_t.shape[0]

        # --- shape checks ---------------------------------------------------
        assert s_t.shape[1:] == self.state_shape, (
            f"Expected s_t shape {self.state_shape}, got {tuple(s_t.shape[1:])}"
        )
        assert s_goal.shape[1:] == self.state_shape, (
            f"Expected s_goal shape {self.state_shape}, "
            f"got {tuple(s_goal.shape[1:])}"
        )

        # --- Step 1: Initialise free parameters -----------------------------
        # a_t <- Random or a_{t-1} (warm start)
        if a_init is not None:
            a_t = a_init.clone().detach().to(device=device, dtype=dtype)
        else:
            a_t = torch.randn(B, self.action_dim, device=device, dtype=dtype) * 0.1
        a_t.requires_grad_(True)

        # z_t <- z_prior (start with top-down expectation)
        if z_prior is not None:
            z_t = z_prior.clone().detach().to(device=device, dtype=dtype)
        else:
            z_t = torch.zeros(B, self.latent_action_dim, device=device, dtype=dtype)
        z_t.requires_grad_(True)

        # --- Enable MC Dropout in frozen models -----------------------------
        enable_mc_dropout(self.fdm)
        enable_mc_dropout(self.action_mlp)

        # --- Optimiser for the free parameters ------------------------------
        optimizer = torch.optim.Adam(
            [
                {"params": [a_t], "lr": self.lr_a},
                {"params": [z_t], "lr": self.lr_z},
            ]
        )

        # Prepare input with time dimension for the FDM
        s_t_expanded = s_t.unsqueeze(1)                        # [B, 1, *state_shape]

        # --- History tracking -----------------------------------------------
        history: Dict[str, List[float]] = {
            "g_total": [],
            "e_pragmatic": [],
            "e_env_epistemic": [],
            "e_motor_epistemic": [],
            "e_action": [],
            "e_prior": [],
            "grad_a_norm": [],
            "grad_z_norm": [],
        }

        # --- Steps 2–5: Optimisation loop -----------------------------------
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            # Expand z_t with time dimension for the FDM
            z_t_expanded = z_t.unsqueeze(1)                    # [B, 1, D]

            # Step 2: MC Dropout forward pass
            s_preds, a_preds = self._mc_forward(
                s_t_expanded, z_t_expanded, self.mc_samples
            )
            # s_preds: [N, B, *state_shape], a_preds: [N, B, A]

            # Step 3: Compute free-energy terms
            energies = self._compute_energies(
                s_t, z_t, a_t, s_goal, z_prior, s_preds, a_preds
            )

            # Step 4: Compute total Expected Free Energy
            g_total = self._compute_total_energy(energies)

            # Step 5: Backward + simultaneous gradient update
            g_total.backward()

            # Optional gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([a_t, z_t], self.max_grad_norm)

            # Track gradient norms (before step)
            grad_a_norm = a_t.grad.norm().item() if a_t.grad is not None else 0.0
            grad_z_norm = z_t.grad.norm().item() if z_t.grad is not None else 0.0

            optimizer.step()

            # --- record history ---------------------------------------------
            if return_history or verbose:
                h = {
                    "g_total": g_total.item(),
                    "e_pragmatic": energies["e_pragmatic"].item(),
                    "e_env_epistemic": energies["e_env_epistemic"].item(),
                    "e_motor_epistemic": energies["e_motor_epistemic"].item(),
                    "e_action": energies["e_action"].item(),
                    "e_prior": energies["e_prior"].item(),
                    "grad_a_norm": grad_a_norm,
                    "grad_z_norm": grad_z_norm,
                }
                if return_history:
                    for k, v in h.items():
                        history[k].append(v)
                if verbose:
                    print(
                        f"  iter {iteration:3d}/{self.num_iterations} | "
                        f"G={h['g_total']:.4f}  "
                        f"prag={h['e_pragmatic']:.4f}  "
                        f"env_epi={h['e_env_epistemic']:.4f}  "
                        f"motor_epi={h['e_motor_epistemic']:.4f}  "
                        f"act={h['e_action']:.4f}  "
                        f"prior={h['e_prior']:.4f}  "
                        f"|g_a|={h['grad_a_norm']:.4f}  "
                        f"|g_z|={h['grad_z_norm']:.4f}"
                    )

        # --- Extract final results ------------------------------------------
        a_star = a_t.detach().clone()
        z_star = z_t.detach().clone()

        # --- Final energy computation (no grad) -----------------------------
        with torch.no_grad():
            z_t_expanded = z_star.unsqueeze(1)
            s_preds, a_preds = self._mc_forward(
                s_t_expanded, z_t_expanded, self.mc_samples
            )
            final_energies = self._compute_energies(
                s_t, z_star, a_star, s_goal, z_prior, s_preds, a_preds
            )

        info: Dict[str, Any] = {
            "final_energies": {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in final_energies.items()
                if k not in ("s_mean", "a_mean")
            },
            "s_pred_mean": final_energies["s_mean"],
            "a_pred_mean": final_energies["a_mean"],
        }
        if return_history:
            info["history"] = history

        return a_star, z_star, info

    # ------------------------------------------------------------------
    # Multi-step trajectory planning
    # ------------------------------------------------------------------

    def plan_trajectory(
        self,
        s_t: torch.Tensor,
        s_goal: torch.Tensor,
        horizon: int,
        z_prior: Optional[torch.Tensor] = None,
        a_init: Optional[torch.Tensor] = None,
        return_history: bool = True,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Plan a trajectory of actions over multiple steps.

        At each step the planner optimises ``(a_t, z_t)``, then uses the FDM
        to roll the state forward.  The previous ``a*`` is used as a warm
        start and the previous ``z*`` as the prior for the next step.

        Parameters
        ----------
        s_t : torch.Tensor
            Current state ``[B, *state_shape]``.
        s_goal : torch.Tensor
            Goal state ``[B, *state_shape]``.
        horizon : int
            Number of steps to plan.
        z_prior : torch.Tensor | None
            Initial prior on latent action ``[B, D]``.
        a_init : torch.Tensor | None
            Initial action warm-start ``[B, A]``.
        return_history : bool
            Whether to include per-step history in the info dicts.
        verbose : bool
            If ``True``, print progress for each step.

        Returns
        -------
        actions : torch.Tensor
            Planned actions ``[B, horizon, A]``.
        latent_actions : torch.Tensor
            Planned latent actions ``[B, horizon, D]``.
        infos : list[dict]
            List of info dictionaries, one per step.
        """
        actions: List[torch.Tensor] = []
        latent_actions: List[torch.Tensor] = []
        infos: List[Dict[str, Any]] = []

        current_s = s_t
        prev_a = a_init
        current_z_prior = z_prior

        for step in range(horizon):
            if verbose:
                print(f"\n=== Trajectory step {step + 1}/{horizon} ===")

            a_star, z_star, info = self.plan(
                current_s,
                s_goal,
                z_prior=current_z_prior,
                a_init=prev_a,
                return_history=return_history,
                verbose=verbose,
            )
            actions.append(a_star)
            latent_actions.append(z_star)
            infos.append(info)

            # Roll the state forward using the FDM (no grad)
            with torch.no_grad():
                s_expanded = current_s.unsqueeze(1)            # [B, 1, *state_shape]
                z_expanded = z_star.unsqueeze(1)               # [B, 1, D]
                fdm_out = self.fdm(s_expanded, z_expanded)
                if self.fdm_predicts_delta:
                    current_s = (s_expanded + fdm_out).squeeze(1)
                else:
                    current_s = fdm_out.squeeze(1)

            # Warm-start / prior for the next step
            prev_a = a_star
            current_z_prior = z_star

        actions_tensor = torch.stack(actions, dim=1)           # [B, horizon, A]
        latent_tensor = torch.stack(latent_actions, dim=1)     # [B, horizon, D]
        return actions_tensor, latent_tensor, infos

    # ------------------------------------------------------------------
    # Convenience: build from a trained DiLA model
    # ------------------------------------------------------------------

    @classmethod
    def from_dila_model(
        cls,
        dila_model: nn.Module,
        action_mlp: Optional[nn.Module] = None,
        state_shape: Optional[Tuple[int, ...]] = None,
        action_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> "HierarchicalActiveInference":
        """Create a planner from a trained ``Inverse_World_model``.

        Parameters
        ----------
        dila_model : nn.Module
            A trained :class:`models.model.Inverse_World_model` instance.
        action_mlp : nn.Module | None
            Custom action-mapping network.  If ``None``, a default
            :class:`ActionMLP` is created.
        state_shape : tuple[int, ...] | None
            Shape of the structure embedding ``(H, W, C)``.  If ``None``,
            an attempt is made to infer it from the model attributes
            (``patch_size`` and ``structure_dim``).
        action_dim : int | None
            Dimension of the physical action.  If ``None``, defaults to the
            model's ``action_dim``.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`HierarchicalActiveInference`.

        Returns
        -------
        HierarchicalActiveInference
            A ready-to-use planner.
        """
        fdm = dila_model.forward_dynamics
        latent_action_dim = dila_model.action_dim

        if action_dim is None:
            action_dim = dila_model.action_dim

        if state_shape is None:
            # Heuristic inference — override if incorrect for your architecture
            structure_dim = getattr(dila_model, "structure_dim", 128)
            patch_size = getattr(dila_model, "patch_size", 4)
            # Common convention: H = W = patch_size, C = structure_dim
            state_shape = (patch_size, patch_size, structure_dim)
            warnings.warn(
                f"state_shape inferred as {state_shape}.  If this is "
                "incorrect, pass state_shape explicitly.",
                UserWarning,
                stacklevel=2,
            )

        if action_mlp is None:
            action_mlp = ActionMLP(
                state_shape=state_shape,
                latent_action_dim=latent_action_dim,
                action_dim=action_dim,
            )

        return cls(
            fdm=fdm,
            action_mlp=action_mlp,
            state_shape=state_shape,
            latent_action_dim=latent_action_dim,
            action_dim=action_dim,
            **kwargs,
        )