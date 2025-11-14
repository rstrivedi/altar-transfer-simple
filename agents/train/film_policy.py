# Added by RST: FiLM-conditioned two-head policy for Phase 4
"""Custom SB3 policy with FiLM conditioning and two action heads.

Architecture:
  1. CNN(RGB) → h_visual (256,)
  2. Concat [h_visual, READY_TO_SHOOT, [TIMESTEP]] → h_raw (257 or 258,)
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
- Dynamic architecture: NO zero padding (timestep only included if in obs_space)

Usage:
    from agents.train.film_policy import FiLMTwoHeadPolicy, DictFeaturesExtractor
    from agents.train.film_policy import RecurrentFiLMTwoHeadPolicy, RecurrentDictFeaturesExtractor
    from sb3_contrib import RecurrentPPO

    # Non-recurrent variant (feedforward, no LSTM)
    policy_kwargs = {
        'features_extractor_class': DictFeaturesExtractor,
        'features_extractor_kwargs': {'trunk_dim': 256},
    }
    model = PPO(FiLMTwoHeadPolicy, env, policy_kwargs=policy_kwargs, ...)

    # Recurrent variant (proper LSTM with state management)
    policy_kwargs = {
        'features_extractor_class': RecurrentDictFeaturesExtractor,
        'features_extractor_kwargs': {'lstm_hidden_size': 256, 'trunk_dim': 256},
    }
    model = RecurrentPPO(RecurrentFiLMTwoHeadPolicy, env, policy_kwargs=policy_kwargs, ...)
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

# Import RecurrentPPO support from sb3-contrib
try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    HAS_SB3_CONTRIB = True
except ImportError:
    HAS_SB3_CONTRIB = False
    RecurrentActorCriticPolicy = None
    RNNStates = None


# Action space constants
K = 11  # Total actions
K_ZAP = 7  # Index of ZAP action
NON_ZAP_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]  # Non-zap action indices (length 10)


class DictFeaturesExtractor(BaseFeaturesExtractor):
    """Feedforward feature extractor for Dict observation space with RGB + vectors.

    Processing:
        1. CNN(RGB) → h_visual (256,)
        2. Concat [h_visual, READY_TO_SHOOT, [TIMESTEP]] → h_raw (257 or 258,)
        3. Linear projection → h (256,)

    Architecture adapts dynamically to observation space:
    - If timestep in obs_space: combine_projection input = 258 (trunk_dim + 2)
    - If timestep NOT in obs_space: combine_projection input = 257 (trunk_dim + 1)
    - NO zero padding - strict hypothesis testing requires clean architecture

    Args:
        observation_space: Dict space with keys 'rgb', 'ready_to_shoot', ['timestep'], ['permitted_color']
        trunk_dim: Output feature dimension (default 256)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        trunk_dim: int = 256,
    ):
        # Output features_dim is trunk_dim
        super().__init__(observation_space, features_dim=trunk_dim)

        self.trunk_dim = trunk_dim

        # Check if timestep is included in observation space
        self.include_timestep = 'timestep' in observation_space.spaces

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

        # Vector features: READY_TO_SHOOT (1) + [TIMESTEP (1) if include_timestep=True]
        # Dynamically set input dimension based on observation space
        vector_dim = 1  # ready_to_shoot
        if self.include_timestep:
            vector_dim += 1  # + timestep

        # Linear projection: [h_visual; vectors] → trunk_dim
        # Input size matches actual observation space (no padding)
        combine_input_dim = trunk_dim + vector_dim
        self.combine_projection = nn.Linear(combine_input_dim, trunk_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations.

        Args:
            observations: Dict with keys 'rgb', 'ready_to_shoot', ['timestep'], ['permitted_color']

        Returns:
            Features tensor of shape (batch_size, trunk_dim)
        """
        # Extract RGB and process through CNN
        # Note: VecTransposeImage already transposes to (batch, 3, 88, 88)
        rgb = observations['rgb']  # (batch, 3, 88, 88) uint8 after VecTransposeImage

        # Normalize to [0, 1]
        rgb = rgb.float() / 255.0

        h_visual = self.cnn(rgb)  # (batch, cnn_out_size)
        h_visual = self.cnn_projection(h_visual)  # (batch, trunk_dim)
        h_visual = torch.relu(h_visual)

        # Extract vector features and concatenate
        ready_to_shoot = observations['ready_to_shoot']  # (batch, 1)

        # Concatenate visual and vector features (no padding)
        if self.include_timestep:
            timestep = observations['timestep']  # (batch, 1)
            h_raw = torch.cat([h_visual, ready_to_shoot, timestep], dim=1)  # (batch, trunk_dim + 2)
        else:
            h_raw = torch.cat([h_visual, ready_to_shoot], dim=1)  # (batch, trunk_dim + 1)

        # Project to trunk_dim
        h = self.combine_projection(h_raw)  # (batch, trunk_dim)
        h = torch.relu(h)

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
        # Features extractor already built by parent __init__ via make_features_extractor()
        # It's accessible as self.features_extractor

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


# ============================================================================
# RECURRENT VERSIONS (for RecurrentPPO)
# ============================================================================

class RecurrentDictFeaturesExtractor(BaseFeaturesExtractor):
    """Recurrent feature extractor with LSTM for sequence processing.

    Processing:
        1. CNN(RGB) → h_visual (256,)
        2. Concat [h_visual, READY_TO_SHOOT, [TIMESTEP]] → h_raw (257 or 258,)
        3. Linear projection → h (256,)
        4. LSTM(h_sequence) → h_lstm (256,) [proper sequence processing]

    Args:
        observation_space: Dict space with keys 'rgb', 'ready_to_shoot', ['timestep'], ['permitted_color']
        lstm_hidden_size: LSTM hidden size (default 256)
        trunk_dim: Output feature dimension before LSTM (default 256)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        lstm_hidden_size: int = 256,
        trunk_dim: int = 256,
    ):
        # Output features_dim is lstm_hidden_size (after LSTM)
        super().__init__(observation_space, features_dim=lstm_hidden_size)

        self.trunk_dim = trunk_dim
        self.lstm_hidden_size = lstm_hidden_size

        # Check if timestep is included in observation space
        self.include_timestep = 'timestep' in observation_space.spaces

        # CNN for RGB (88, 88, 3) → (256,)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            sample_rgb = torch.zeros(1, 3, 88, 88)
            cnn_out_size = self.cnn(sample_rgb).shape[1]

        # Linear projection from CNN output to trunk_dim
        self.cnn_projection = nn.Linear(cnn_out_size, trunk_dim)

        # Vector features dimension
        vector_dim = 1  # ready_to_shoot
        if self.include_timestep:
            vector_dim += 1  # + timestep

        # Linear projection: [h_visual; vectors] → trunk_dim
        combine_input_dim = trunk_dim + vector_dim
        self.combine_projection = nn.Linear(combine_input_dim, trunk_dim)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(trunk_dim, lstm_hidden_size, batch_first=True)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations (WITHOUT LSTM processing).

        Note: For recurrent policies, this extracts per-timestep features.
        LSTM processing happens in the policy's _process_sequence() method.

        Args:
            observations: Dict with keys 'rgb', 'ready_to_shoot', ['timestep'], ['permitted_color']

        Returns:
            Features tensor of shape (batch, trunk_dim) - BEFORE LSTM
        """
        # Extract RGB and process through CNN
        rgb = observations['rgb']  # (batch, 3, 88, 88)
        rgb = rgb.float() / 255.0

        h_visual = self.cnn(rgb)
        h_visual = self.cnn_projection(h_visual)
        h_visual = torch.relu(h_visual)

        # Extract vector features and concatenate
        ready_to_shoot = observations['ready_to_shoot']

        if self.include_timestep:
            timestep = observations['timestep']
            h_raw = torch.cat([h_visual, ready_to_shoot, timestep], dim=1)
        else:
            h_raw = torch.cat([h_visual, ready_to_shoot], dim=1)

        # Project to trunk_dim
        h = self.combine_projection(h_raw)
        h = torch.relu(h)

        return h  # Return BEFORE LSTM (LSTM processed in _process_sequence)
class RecurrentFiLMTwoHeadPolicy(RecurrentActorCriticPolicy):
    """Recurrent FiLM two-head policy with proper LSTM state management.

    Inherits from RecurrentActorCriticPolicy to get proper sequence processing.
    Architecture identical to FiLMTwoHeadPolicy but with recurrent processing.

    Args:
        observation_space: Gymnasium Dict observation space
        action_space: Gymnasium Discrete(11) action space
        lr_schedule: Learning rate schedule
        trunk_dim: Trunk feature dimension before LSTM (default 256)
        lstm_hidden_size: LSTM hidden size (default 256)
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
        lstm_hidden_size: int = 256,
        sanction_hidden_dim: int = 128,
        ent_coef_game: float = 0.01,
        ent_coef_sanction: float = 0.02,
        *args,
        **kwargs,
    ):
        if not HAS_SB3_CONTRIB:
            raise ImportError(
                "sb3-contrib is required for RecurrentFiLMTwoHeadPolicy. "
                "Install with: pip install sb3-contrib"
            )

        self.trunk_dim = trunk_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.sanction_hidden_dim = sanction_hidden_dim
        self.ent_coef_game = ent_coef_game
        self.ent_coef_sanction = ent_coef_sanction

        # Initialize parent (will call _build)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # Override action distribution
        self.action_dist = CategoricalDistribution(K)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build the recurrent policy network.

        Note: features_extractor (RecurrentDictFeaturesExtractor) already built by parent.
        """
        # Get conditioning dimension
        conditioning_dim = 3

        # Global FiLM module
        self.global_film = FiLMModule(conditioning_dim, self.lstm_hidden_size)

        # Game head: h_lstm → MLP → 10 logits
        self.game_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, len(NON_ZAP_INDICES)),
        )

        # Sanction head with Local FiLM
        self.sanction_base = nn.Linear(self.lstm_hidden_size, 1)
        self.sanction_projection = nn.Linear(self.lstm_hidden_size, self.sanction_hidden_dim)
        self.local_film = FiLMModule(conditioning_dim, self.sanction_hidden_dim)
        self.sanction_gate_weight = nn.Parameter(torch.zeros(self.sanction_hidden_dim))

        # Value head (shared)
        self.value_net = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Setup optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _has_permitted_color(self, observations: Dict[str, torch.Tensor]) -> bool:
        """Check if permitted_color is in observations."""
        return 'permitted_color' in observations

    def _process_sequence(
        self,
        features: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        obs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, RNNStates]:
        """Process sequence through LSTM.

        This is called by RecurrentPPO during rollout collection and training.

        Args:
            features: Features before LSTM (batch, seq_len, trunk_dim)
            lstm_states: Previous LSTM states (hidden, cell)
            episode_starts: Episode start flags (batch, seq_len)
            obs: Observations dict (for extracting conditioning)

        Returns:
            lstm_output: Features after LSTM (batch, seq_len, lstm_hidden_size)
            new_lstm_states: Updated LSTM states
        """
        # Get LSTM from features_extractor
        lstm = self.features_extractor.lstm

        # Unpack LSTM states
        # RNNStates from sb3-contrib has .pi and .vf attributes, each containing (hidden, cell)
        # We have one shared LSTM, so we just use .pi states (they should be identical to .vf)
        if hasattr(lstm_states, 'pi'):
            # It's a RNNStates object
            hidden, cell = lstm_states.pi
        else:
            # Fallback for plain tuple (shouldn't happen in normal use)
            hidden, cell = lstm_states

        # Reset LSTM states at episode boundaries
        # episode_starts: (batch, seq_len) with 1.0 at episode start
        # We need to reset hidden/cell states where episode_starts == 1.0

        batch_size, seq_len, _ = features.shape
        lstm_output = []

        # Get device from features
        device = features.device

        for step_idx in range(seq_len):
            # Get features for this timestep
            step_features = features[:, step_idx:step_idx+1, :]  # (batch, 1, trunk_dim)

            # Check if episode starts at this step
            # episode_starts[:, step_idx] is (batch,) with values 0 or 1
            # LSTM hidden/cell are (num_layers, batch, hidden_size)
            # Need to reshape starts to (1, batch, 1) for broadcasting
            starts_step = episode_starts[:, step_idx]  # (batch,)

            # Convert to float and reshape, ensure on correct device
            starts = starts_step.to(dtype=torch.float32, device=device).view(1, -1, 1)  # (1, batch, 1)

            # Reset LSTM states where episode starts
            # Broadcasting: (1, batch, 1) * (num_layers, batch, hidden_size) = (num_layers, batch, hidden_size)
            hidden = (1.0 - starts) * hidden
            cell = (1.0 - starts) * cell

            # Process through LSTM
            step_output, (hidden, cell) = lstm(step_features, (hidden, cell))
            lstm_output.append(step_output)

        # Concatenate outputs across sequence
        lstm_output = torch.cat(lstm_output, dim=1)  # (batch, seq_len, lstm_hidden_size)

        # Return states in RNNStates format
        # We have one shared LSTM, so return same states for both pi (actor) and vf (critic)
        new_lstm_states = RNNStates(pi=(hidden, cell), vf=(hidden, cell))

        return lstm_output, new_lstm_states

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        """Forward pass for recurrent policy.

        Args:
            obs: Observations dict
            lstm_states: LSTM states (hidden, cell)
            episode_starts: Episode start flags
            deterministic: Whether to sample deterministically

        Returns:
            actions: Sampled actions
            values: Value estimates
            log_probs: Log probabilities
            lstm_states: Updated LSTM states
        """
        # Extract features (before LSTM)
        features = self.extract_features(obs)  # (batch, trunk_dim)

        # Add sequence dimension for LSTM
        features = features.unsqueeze(1)  # (batch, 1, trunk_dim)
        episode_starts = episode_starts.unsqueeze(1)  # (batch, 1)

        # Process through LSTM
        h_lstm, lstm_states = self._process_sequence(features, lstm_states, episode_starts, obs)
        h_lstm = h_lstm.squeeze(1)  # (batch, lstm_hidden_size)

        # Get conditioning signal
        if self._has_permitted_color(obs):
            conditioning = obs['permitted_color']
        else:
            batch_size = obs['rgb'].shape[0]
            device = obs['rgb'].device
            conditioning = torch.zeros(batch_size, 3, device=device)

        # Apply Global FiLM
        h_tilde = self.global_film(conditioning, h_lstm)

        # Game head
        game_logits = self.game_head(h_tilde)

        # Sanction head
        sanction_base_logit = self.sanction_base(h_tilde)
        h_proj = self.sanction_projection(h_tilde)
        h_s = self.local_film(conditioning, h_proj)
        gated = torch.nn.functional.silu(h_s)
        scalar_residual = torch.sum(self.sanction_gate_weight * gated, dim=1, keepdim=True)
        sanction_logit = sanction_base_logit + scalar_residual

        # Compose full logits
        full_logits = torch.zeros(h_lstm.shape[0], K, device=h_lstm.device)
        for i, idx in enumerate(NON_ZAP_INDICES):
            full_logits[:, idx] = game_logits[:, i]
        full_logits[:, K_ZAP] = sanction_logit.squeeze(1)

        # Sample action
        distribution = CategoricalDistribution(K)
        distribution.distribution = torch.distributions.Categorical(logits=full_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        # Value estimate
        values = self.value_net(h_tilde).squeeze(1)

        return actions, values, log_probs, lstm_states

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for recurrent policy (used in PPO loss).

        Args:
            obs: Observations dict
            actions: Actions taken
            lstm_states: LSTM states (managed by RecurrentPPO)
            episode_starts: Episode start flags

        Returns:
            values: Value estimates
            log_probs: Log probabilities
            entropy: Entropy
        """
        # Extract features
        features = self.extract_features(obs)

        # Process through LSTM
        # During training, RecurrentPPO batches data into sequences
        # features has shape (num_sequences*seq_len, trunk_dim) after extract_features
        # Need to reshape for sequence processing

        # Get number of sequences from LSTM states (NOT number of environments)
        # LSTM hidden state has shape (num_layers, num_sequences, hidden_size)
        if hasattr(lstm_states, 'pi'):
            hidden, _ = lstm_states.pi
        else:
            hidden, _ = lstm_states
        num_sequences = hidden.shape[1]  # Number of sequences in this batch

        # Calculate sequence length
        total_steps = features.shape[0]
        seq_len = total_steps // num_sequences

        # Reshape features and episode_starts to (num_sequences, seq_len, ...)
        features = features.reshape(num_sequences, seq_len, -1)
        if episode_starts.dim() == 1:
            episode_starts = episode_starts.reshape(num_sequences, seq_len)
        elif episode_starts.shape[0] != num_sequences:
            episode_starts = episode_starts.reshape(num_sequences, seq_len)

        h_lstm, lstm_states = self._process_sequence(features, lstm_states, episode_starts, obs)
        h_lstm = h_lstm.reshape(-1, self.lstm_hidden_size)  # Flatten back

        # Get conditioning
        if self._has_permitted_color(obs):
            conditioning = obs['permitted_color']
            if conditioning.dim() == 3:  # (batch, seq_len, 3)
                conditioning = conditioning.reshape(-1, 3)
        else:
            batch_total = h_lstm.shape[0]
            device = h_lstm.device
            conditioning = torch.zeros(batch_total, 3, device=device)

        # Apply Global FiLM
        h_tilde = self.global_film(conditioning, h_lstm)

        # Game head
        game_logits = self.game_head(h_tilde)

        # Sanction head
        sanction_base_logit = self.sanction_base(h_tilde)
        h_proj = self.sanction_projection(h_tilde)
        h_s = self.local_film(conditioning, h_proj)
        gated = torch.nn.functional.silu(h_s)
        scalar_residual = torch.sum(self.sanction_gate_weight * gated, dim=1, keepdim=True)
        sanction_logit = sanction_base_logit + scalar_residual

        # Compose full logits
        full_logits = torch.zeros(h_lstm.shape[0], K, device=h_lstm.device)
        for i, idx in enumerate(NON_ZAP_INDICES):
            full_logits[:, idx] = game_logits[:, i]
        full_logits[:, K_ZAP] = sanction_logit.squeeze(1)

        # Compute log probs and entropy
        distribution = CategoricalDistribution(K)
        distribution.distribution = torch.distributions.Categorical(logits=full_logits)
        log_probs = distribution.log_prob(actions.flatten())

        # Head-wise entropy
        entropy = self._compute_head_wise_entropy(full_logits)

        # Value estimate
        values = self.value_net(h_tilde).squeeze(1)

        return values, log_probs, entropy

    def _compute_head_wise_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute head-wise entropy (same as non-recurrent version)."""
        eps = 1e-8
        probs = torch.softmax(logits, dim=1)

        # Sanction head entropy
        p_zap = probs[:, K_ZAP]
        p_non_zap = 1.0 - p_zap
        H_sanction = -(p_zap * torch.log(p_zap + eps) + p_non_zap * torch.log(p_non_zap + eps))

        # Game head entropy
        p_game_unnorm = probs[:, NON_ZAP_INDICES]
        p_game = p_game_unnorm / (p_non_zap.unsqueeze(1) + eps)
        H_game = -torch.sum(p_game * torch.log(p_game + eps), dim=1)

        # Weighted sum
        total_entropy = self.ent_coef_game * H_game + self.ent_coef_sanction * H_sanction

        return total_entropy

    def predict_values(
        self,
        obs: Dict[str, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Predict values for recurrent policy."""
        features = self.extract_features(obs)
        features = features.unsqueeze(1)
        episode_starts = episode_starts.unsqueeze(1) if episode_starts.dim() == 1 else episode_starts

        h_lstm, _ = self._process_sequence(features, lstm_states, episode_starts, obs)
        h_lstm = h_lstm.squeeze(1)

        if self._has_permitted_color(obs):
            conditioning = obs['permitted_color']
        else:
            batch_size = obs['rgb'].shape[0]
            device = obs['rgb'].device
            conditioning = torch.zeros(batch_size, 3, device=device)

        h_tilde = self.global_film(conditioning, h_lstm)
        values = self.value_net(h_tilde).squeeze(1)
        return values
