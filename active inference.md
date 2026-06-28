# Hierarchical Active Inference in DiLA: Joint Optimization

This document outlines the design for the higher-level DiLA agent. It uses a **flattened joint optimization** approach to simultaneously infer the agent's optimal sub-goal ($a_t$) and the world's latent dynamic response ($z_t$). By incorporating **Monte Carlo Dropout in both the forward dynamics and action mapping**, the system achieves true curiosity-driven Active Inference—exploring both environmental physics and its own motor capabilities—without requiring computationally heavy ensembles.

## 1. System Components & Variables

**Given State (Constants during step** $t$**):**

- $s_t$: Current structure embedding.
    
- $s_{goal}$: Target structure embedding (long-term goal).
    
- $\hat{z}_t^{prior}$: Top-down expectation of the global transition (from the level above, if any).
    

**Neural Networks (Weights are Frozen during inference):**

- `ActionMLP(s, z)`: Mapping from state and global dynamics to the agent's required action. **Dropout layers remain active** during this planning phase to gauge motor uncertainty.
    
- `FDM(s, z)`: Forward Dynamics Model. **Dropout layers remain active** during this planning phase to gauge environmental uncertainty.
    

**Free Parameters (Optimized during step** $t$**):**

- $a_t$: The candidate sub-goal or macro-action (`requires_grad=True`).
    
- $z_t$: The candidate global latent action (`requires_grad=True`).
    

## 2. The Optimization Loop

For each planning step, the agent performs the following iterative process to settle on the optimal $(a_t, z_t)$ pair.

### Step 1: Initialization

Initialize the free parameters.

- $a_t \leftarrow \text{Random or } a_{t-1}$ (Warm start from the previous plan).
    
- $z_t \leftarrow \hat{z}_t^{prior}$ (Start with the top-down expectation).
    

### Step 2: The Forward Pass (with MC Dropout)

To capture epistemic uncertainty (exploration potential) for both the environment and the agent's own body, we pass the current $(s_t, z_t)$ through the FDM and ActionMLP $N$ times (e.g., $N=5$). Because dropout is active, the exact same inputs yield slightly different predictions.

For $i \in \{1 \dots N\}$:

$$\hat{s}_{t+1}^{(i)} = \text{FDM}_{dropout}(s_t, z_t)$$$$\hat{a}_{t}^{(i)} = \text{ActionMLP}_{dropout}(s_t, z_t)$$

Calculate the mean predictions:

$$\bar{s}_{t+1} = \frac{1}{N} \sum_{i=1}^{N} \hat{s}_{t+1}^{(i)}$$$$\bar{a}_{t} = \frac{1}{N} \sum_{i=1}^{N} \hat{a}_{t}^{(i)}$$

### Step 3: Compute the Free Energy Terms

We construct the computational graph by calculating the pillars of Active Inference:

1. **Pragmatic Energy (Goal Attainment):**
    
    Measures the distance from the average predicted future to the goal.
    
    $$E_{pragmatic} = || \bar{s}_{t+1} - s_{goal} ||_2^2$$
2. **Environmental Epistemic Energy (World Curiosity):**
    
    Measures the variance across the MC Dropout FDM predictions. High variance means the model is uncertain about what the world will do.
    
    $$E_{env\_epistemic} = \frac{1}{N} \sum_{i=1}^{N} || \hat{s}_{t+1}^{(i)} - \bar{s}_{t+1} ||_2^2$$
3. **Motor Epistemic Energy (Body Curiosity / Babbling):** Measures the variance across the MC Dropout ActionMLP predictions. High variance means the model doesn't know how to physically execute this dynamic yet.
    
    $$E_{motor\_epistemic} = \frac{1}{N} \sum_{i=1}^{N} || \hat{a}_{t}^{(i)} - \bar{a}_{t} ||_2^2$$
4. **Action Energy (Agency Alignment):**
    
    Forces the candidate action to logically trigger the candidate latent dynamic.
    
    $$E_{action} = || a_t - \bar{a}_{t} ||_2^2$$
5. **Prior Energy (Top-Down Compliance):**
    
    Constrains the global transition to remain close to the higher level's expectation.
    
    $$E_{prior} = || z_t - \hat{z}_t^{prior} ||_2^2$$

### Step 4: The Expected Free Energy Objective

Combine the terms into a single scalar loss. Note the **minus signs** for the epistemic energies: the model wants to minimize prediction error while _maximizing_ information gain about both the world and itself.

$$G_{total}(a_t, z_t) = \alpha E_{pragmatic} + \beta E_{action} + \gamma E_{prior} - \delta_1 E_{env\_epistemic} - \delta_2 E_{motor\_epistemic}$$

_(Hyperparameters_ $\alpha, \beta, \gamma, \delta_1, \delta_2$ _control the behavioral profile of the agent: greedy vs. curious, rigid vs. compliant, environment-focused vs. motor-focused)._

### Step 5: Simultaneous Gradient Update

Call `.backward()` on $G_{total}$. The automatic differentiator calculates how tweaking $a_t$ and $z_t$ will affect the total energy. Update both parameters simultaneously using gradient descent (e.g., via an Adam optimizer loop):

$$a_t \leftarrow a_t - \eta_a \nabla_{a_t} G_{total}$$$$z_t \leftarrow z_t - \eta_z \nabla_{z_t} G_{total}$$

## 3. The Push-Pull Dynamics (Why it Works)

As this loop runs for $K$ iterations (e.g., $K=20$), the architecture naturally balances several competing forces:

1. **The Pull to the Goal:** $\nabla_{z_t} E_{pragmatic}$ drags the latent dynamic toward the target state.
    
2. **The Rubber Band of Agency:** As $z_t$ moves toward the goal, $\nabla_{a_t} E_{action}$ violently yanks the candidate action $a_t$ to follow it, ensuring the agent actually plans a physical command that causes the world to move toward the goal.
    
3. **The Call of the Unknown (Environmental):** If $E_{pragmatic}$ encounters a flat gradient, $-\nabla_{z_t} E_{env\_epistemic}$ pulls the planned dynamics toward regions of the state space where the FDM's dropout nodes disagree the most (i.e., unexplored physical interactions).
    
4. **Motor Babbling (System Identification):** If the agent knows the environment but not its own body, $-\nabla_{z_t} E_{motor\_epistemic}$ pulls the optimization toward dynamics that the ActionMLP is uncertain about. $E_{action}$ then forces the agent to actually output and test that unknown physical action ($a_t$).
    
5. **The Bounds of Reality:** Impossible $z_t$ transitions naturally cause the FDM predictions to collapse or explode, which spikes $E_{pragmatic}$ and $E_{action}$, forcing the gradient back into the realm of physically plausible structural changes.
    

Once the gradients converge, the agent extracts the final, optimized $a_t^*$ as its output sub-goal, and passes it down to the next level of the hierarchy.