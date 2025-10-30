# Added by RST: FiLM-conditioned two-head policy for Phase 4
"""Custom SB3 policy with FiLM conditioning and two action heads.

Architecture:
  1. CNN(RGB) → h_visual (256,)
  2. Concat [h_visual, READY_TO_SHOOT, TIMESTEP] → h_raw (258,)
  3. Linear projection → h (256,)
  4. [Optional] LSTM(h) → h_lstm (256,)
  5. Global FiLM: h̃ = γ(C) ⊙ h + β(C)
  6. Game head: h̃ → MLP → g (10 logits for non-zap actions)
  7. Sanction head (Local FiLM): h̃ → FiLM_s → SiLU → s (1 logit for zap)
  8. Compose L[11]: L[7]=s, L[non_zap]=g
  9. Value head (shared): h̃ → MLP → V(s)

Key features:
- FiLM initialized to identity (no effect at start)
- Treatment uses C_onehot from env; Control uses zeros (no institutional signal)
- Architectural parity across arms (FiLM modules exist in both)
- Head-wise entropy for separate annealing

Usage:
    from agents.train.film_policy import FiLMTwoHeadPolicy

    # Feed-forward variant
    policy_kwargs = {
        'features_extractor_class': DictFeaturesExtractor,
        'features_extractor_kwargs': {'recurrent': False},
    }
    model = PPO(FiLMTwoHeadPolicy, env, policy_kwargs=policy_kwargs, ...)

    # Recurrent variant (LSTM)
    policy_kwargs = {
        'features_extractor_class': DictFeaturesExtractor,
        'features_extractor_kwargs': {'recurrent': True, 'lstm_hidden_size': 256},
    }
    model = PPO(FiLMTwoHeadPolicy, env, policy_kwargs=policy_kwargs, ...)
"""

from typing import Dict, List, Tuple, Type, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import CategoricalDistribution


# Action space constants
K = 11  # Total actions
K_ZAP = 7  # Index of ZAP action
NON_ZAP_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]  # Non-zap action indices (length 10)


class DictFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observation space with RGB + vectors.

    Processing:
        1. CNN(RGB) → h_visual (256,)
        2. Concat [h_visual, READY_TO_SHOOT, [TIMESTEP]] → h_raw (257 or 258,)
        3. Linear projection → h (256,)
        4. [Optional] LSTM(h) → h_lstm (256,)

    Note: TIMESTEP is optional. If not present in observations, it's padded with zeros
    to maintain fixed input dimension for the combine_projection layer.

    Args:
        observation_space: Dict space with keys 'rgb', 'ready_to_shoot', ['timestep'], ['permitted_color']
        recurrent: Whether to use LSTM (default False)
        lstm_hidden_size: LSTM hidden size (default 256)
        trunk_dim: Output feature dimension (default 256)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        recurrent: bool = False,
        lstm_hidden_size: int = 256,
        trunk_dim: int = 256,
    ):
        # Output features_dim is trunk_dim
        super().__init__(observation_space, features_dim=trunk_dim)

        self.recurrent = recurrent
        self.trunk_dim = trunk_dim

        # CNN for RGB (88, 88, 3) → (256,)
        # Added by RST: Simple CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),  # (88,88,3) → (21,21,32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # (21,21,32) → (9,9,64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # (9,9,64) → (7,7,64)
            nn.ReLU(),
            nn.Flatten(),  # (7,7,64) → 3136
        )

        # Compute CNN output size
        with torch.no_grad():
            sample_rgb = torch.zeros(1, 3, 88, 88)
            cnn_out_size = self.cnn(sample_rgb).shape[1]

        # Linear projection from CNN output to trunk_dim
        self.cnn_projection = nn.Linear(cnn_out_size, trunk_dim)

        # Vector features: READY_TO_SHOOT (1) + [TIMESTEP (1) if included]
        # Note: We'll determine actual vector_dim dynamically in forward()
        # Max vector_dim = 2 (ready_to_shoot + timestep)
        max_vector_dim = 2

        # Linear projection: [h_visual; vectors] → trunk_dim
        # Added by RST: Combine visual and vector features
        self.combine_projection = nn.Linear(trunk_dim + max_vector_dim, trunk_dim)

        # Optional LSTM
        if self.recurrent:
            self.lstm = nn.LSTM(trunk_dim, lstm_hidden_size, batch_first=True)
            self.lstm_hidden_size = lstm_hidden_size

            # Store LSTM states (will be managed externally in recurrent policies)
            self.lstm_states = None
        else:
            self.lstm = None

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations.

        Args:
            observations: Dict with keys 'rgb', 'ready_to_shoot', ['timestep'], ['permitted_color']

        Returns:
            Features tensor of shape (batch_size, trunk_dim)
        """
        # Extract RGB and process through CNN
        rgb = observations['rgb']  # (batch, 88, 88, 3) uint8

        # Permute to (batch, 3, 88, 88) and normalize to [0, 1]
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0

        h_visual = self.cnn(rgb)  # (batch, cnn_out_size)
        h_visual = self.cnn_projection(h_visual)  # (batch, trunk_dim)
        h_visual = torch.relu(h_visual)

        # Extract vector features
        ready_to_shoot = observations['ready_to_shoot']  # (batch, 1)

        # Optionally include timestep if present (otherwise pad with zeros)
        if 'timestep' in observations:
            timestep = observations['timestep']  # (batch, 1)
        else:
            # Pad with zeros to maintain fixed input dimension for combine_projection
            timestep = torch.zeros_like(ready_to_shoot)

        # Concatenate visual and vector features
        h_raw = torch.cat([h_visual, ready_to_shoot, timestep], dim=1)  # (batch, trunk_dim + 2)

        # Project to trunk_dim
        h = self.combine_projection(h_raw)  # (batch, trunk_dim)
        h = torch.relu(h)

        # Optional LSTM
        if self.recurrent and self.lstm is not None:
            # Add sequence dimension: (batch, 1, trunk_dim)
            h = h.unsqueeze(1)

            # LSTM forward (using stored states if available)
            if self.lstm_states is not None:
                h, self.lstm_states = self.lstm(h, self.lstm_states)
            else:
                h, self.lstm_states = self.lstm(h)

            # Remove sequence dimension: (batch, trunk_dim)
            h = h.squeeze(1)

        return h


class FiLMModule(nn.Module):
    """FiLM (Feature-wise Linear Modulation) module.

    Applies affine transformation: h̃ = γ(C) ⊙ h + β(C)

    Initialized to identity (γ=1, β=0) so FiLM has no effect at start.

    Args:
        input_dim: Dimension of conditioning input C (e.g., 3 for one-hot)
        feature_dim: Dimension of features h to modulate
    """

    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()

        # Separate linear layers for γ and β
        # Added by RST: Initialize to identity (γ=1, β=0)
        self.gamma_layer = nn.Linear(input_dim, feature_dim)
        self.beta_layer = nn.Linear(input_dim, feature_dim)

        # Initialize gamma to output 1 (weights=0, bias=1)
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.ones_(self.gamma_layer.bias)

        # Initialize beta to output 0 (weights=0, bias=0)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)

    def forward(self, conditioning: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            conditioning: Conditioning input C (batch, input_dim)
            features: Features to modulate h (batch, feature_dim)

        Returns:
            Modulated features h̃ = γ(C) ⊙ h + β(C)
        """
        gamma = self.gamma_layer(conditioning)  # (batch, feature_dim)
        beta = self.beta_layer(conditioning)  # (batch, feature_dim)

        return gamma * features + beta


class FiLMTwoHeadPolicy(ActorCriticPolicy):
    """Custom ActorCriticPolicy with FiLM conditioning and two action heads.

    Architecture:
        - Trunk: CNN + vector concat + optional LSTM → h (256,)
        - Global FiLM: h̃ = γ(C) ⊙ h + β(C)
        - Game head: h̃ → MLP → 10 logits (non-zap actions)
        - Sanction head: h̃ → Local FiLM → SiLU → scalar logit (zap)
        - Value head: h̃ → MLP → scalar

    Args:
        observation_space: Gymnasium Dict observation space
        action_space: Gymnasium Discrete(11) action space
        lr_schedule: Learning rate schedule
        trunk_dim: Trunk feature dimension (default 256)
        sanction_hidden_dim: Sanction head hidden dimension (default 128)
        ent_coef_game: Entropy coefficient for game head (default 0.01)
        ent_coef_sanction: Entropy coefficient for sanction head (default 0.02)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        trunk_dim: int = 256,
        sanction_hidden_dim: int = 128,
        ent_coef_game: float = 0.01,
        ent_coef_sanction: float = 0.02,
        *args,
        **kwargs,
    ):
        self.trunk_dim = trunk_dim
        self.sanction_hidden_dim = sanction_hidden_dim
        self.ent_coef_game = ent_coef_game
        self.ent_coef_sanction = ent_coef_sanction

        # Initialize parent (will call _build_mlp_extractor)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # Override action distribution to use single Categorical
        self.action_dist = CategoricalDistribution(K)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy network (called by parent __init__).

        Added by RST: Custom build for FiLM two-head architecture
        """
        # Build features extractor (CNN + LSTM)
        self._build_features_extractor()

        # Get conditioning dimension (3 for one-hot, or 0 if control uses internal null)
        # We always allocate space for 3 since FiLM modules exist in both arms
        conditioning_dim = 3

        # Global FiLM module (3 → trunk_dim)
        self.global_film = FiLMModule(conditioning_dim, self.trunk_dim)

        # Game head: h̃ → MLP → 10 logits
        # Added by RST: MLP for non-zap actions
        self.game_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(NON_ZAP_INDICES)),  # 10 logits
        )

        # Sanction head with Local FiLM
        # Added by RST: Base path + residual path with FiLM gating

        # Base path: h̃ → scalar
        self.sanction_base = nn.Linear(self.trunk_dim, 1)

        # Residual path: h̃ → projection → Local FiLM → SiLU → scalar
        self.sanction_projection = nn.Linear(self.trunk_dim, self.sanction_hidden_dim)
        self.local_film = FiLMModule(conditioning_dim, self.sanction_hidden_dim)
        self.sanction_gate_weight = nn.Parameter(torch.zeros(self.sanction_hidden_dim))  # u vector

        # Value head (shared): h̃ → MLP → scalar
        # Added by RST: Shared value function
        self.value_net = nn.Sequential(
            nn.Linear(self.trunk_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Setup optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_conditioning(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract conditioning signal from observations.

        Treatment: uses permitted_color from env
        Control: uses zeros (fixed null token, not learnable)

        Args:
            observations: Dict with keys 'rgb', 'ready_to_shoot', 'timestep', ['permitted_color']

        Returns:
            Conditioning tensor (batch, 3)
        """
        if 'permitted_color' in observations:
            # Treatment arm: use permitted_color from env
            return observations['permitted_color']  # (batch, 3)
        else:
            # Control arm: use fixed zeros (null token)
            batch_size = observations['rgb'].shape[0]
            device = observations['rgb'].device
            return torch.zeros(batch_size, 3, device=device)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through policy network.

        Args:
            obs: Observations dict
            deterministic: Whether to sample deterministically

        Returns:
            actions: Sampled actions (batch,)
            values: Value estimates (batch,)
            log_probs: Log probabilities of actions (batch,)
        """
        # Extract trunk features
        h = self.extract_features(obs)  # (batch, trunk_dim)

        # Get conditioning signal
        conditioning = self._get_conditioning(obs)  # (batch, 3)

        # Apply Global FiLM
        h_tilde = self.global_film(conditioning, h)  # (batch, trunk_dim)

        # === Game head ===
        game_logits = self.game_head(h_tilde)  # (batch, 10)

        # === Sanction head ===
        # Base path
        sanction_base_logit = self.sanction_base(h_tilde)  # (batch, 1)

        # Residual path with Local FiLM
        h_proj = self.sanction_projection(h_tilde)  # (batch, sanction_hidden_dim)
        h_s = self.local_film(conditioning, h_proj)  # (batch, sanction_hidden_dim)
        gated = torch.nn.functional.silu(h_s)  # SiLU activation
        scalar_residual = torch.sum(self.sanction_gate_weight * gated, dim=1, keepdim=True)  # (batch, 1)

        sanction_logit = sanction_base_logit + scalar_residual  # (batch, 1)

        # === Compose full logits ===
        # Create tensor of shape (batch, 11)
        full_logits = torch.zeros(h.shape[0], K, device=h.device)

        # Fill game head logits
        for i, idx in enumerate(NON_ZAP_INDICES):
            full_logits[:, idx] = game_logits[:, i]

        # Fill sanction head logit
        full_logits[:, K_ZAP] = sanction_logit.squeeze(1)

        # Sample action
        distribution = CategoricalDistribution(K)
        distribution.distribution = torch.distributions.Categorical(logits=full_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        # Value estimate
        values = self.value_net(h_tilde).squeeze(1)  # (batch,)

        return actions, values, log_probs

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions (used in PPO loss computation).

        Args:
            obs: Observations dict
            actions: Actions taken (batch,)

        Returns:
            values: Value estimates (batch,)
            log_probs: Log probabilities of actions (batch,)
            entropy: Entropy of action distribution (batch,)
        """
        # Extract trunk features
        h = self.extract_features(obs)  # (batch, trunk_dim)

        # Get conditioning signal
        conditioning = self._get_conditioning(obs)  # (batch, 3)

        # Apply Global FiLM
        h_tilde = self.global_film(conditioning, h)  # (batch, trunk_dim)

        # === Game head ===
        game_logits = self.game_head(h_tilde)  # (batch, 10)

        # === Sanction head ===
        sanction_base_logit = self.sanction_base(h_tilde)  # (batch, 1)
        h_proj = self.sanction_projection(h_tilde)  # (batch, sanction_hidden_dim)
        h_s = self.local_film(conditioning, h_proj)  # (batch, sanction_hidden_dim)
        gated = torch.nn.functional.silu(h_s)
        scalar_residual = torch.sum(self.sanction_gate_weight * gated, dim=1, keepdim=True)
        sanction_logit = sanction_base_logit + scalar_residual  # (batch, 1)

        # === Compose full logits ===
        full_logits = torch.zeros(h.shape[0], K, device=h.device)
        for i, idx in enumerate(NON_ZAP_INDICES):
            full_logits[:, idx] = game_logits[:, i]
        full_logits[:, K_ZAP] = sanction_logit.squeeze(1)

        # Compute log probs and entropy
        distribution = CategoricalDistribution(K)
        distribution.distribution = torch.distributions.Categorical(logits=full_logits)
        log_probs = distribution.log_prob(actions)

        # Head-wise entropy
        entropy = self._compute_head_wise_entropy(full_logits)

        # Value estimate
        values = self.value_net(h_tilde).squeeze(1)  # (batch,)

        return values, log_probs, entropy

    def _compute_head_wise_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute head-wise entropy with separate coefficients.

        Args:
            logits: Full action logits (batch, 11)

        Returns:
            Total entropy (batch,)
        """
        eps = 1e-8

        # Full distribution
        probs = torch.softmax(logits, dim=1)  # (batch, 11)

        # Sanction head entropy (binary: zap vs non-zap)
        p_zap = probs[:, K_ZAP]  # (batch,)
        p_non_zap = 1.0 - p_zap
        H_sanction = -(p_zap * torch.log(p_zap + eps) + p_non_zap * torch.log(p_non_zap + eps))

        # Game head entropy (renormalized over non-zap actions)
        p_game_unnorm = probs[:, NON_ZAP_INDICES]  # (batch, 10)
        p_game = p_game_unnorm / (p_non_zap.unsqueeze(1) + eps)  # Renormalize
        H_game = -torch.sum(p_game * torch.log(p_game + eps), dim=1)

        # Weighted sum
        total_entropy = self.ent_coef_game * H_game + self.ent_coef_sanction * H_sanction

        return total_entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict values for observations.

        Args:
            obs: Observations dict

        Returns:
            Value estimates (batch,)
        """
        h = self.extract_features(obs)
        conditioning = self._get_conditioning(obs)
        h_tilde = self.global_film(conditioning, h)
        values = self.value_net(h_tilde).squeeze(1)
        return values
