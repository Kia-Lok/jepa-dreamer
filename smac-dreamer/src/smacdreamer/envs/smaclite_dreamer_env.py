"""Gymnasium-compatible centralised-control SMAClite environment.

One R2-Dreamer agent drives all allied SMAClite units (centralised single controller).

Phase 1A migration note
-----------------------
This file was migrated from the JAX-DreamerV3 ``embodied.Env`` interface to the PyTorch
R2-Dreamer ``gymnasium.Env`` interface. The *framework-facing* layer changed:

    embodied.Env                -> gymnasium.Env
    elements.Space              -> gymnasium.spaces
    obs_space / act_space props -> observation_space / action_space
    step(action_dict w/ reset)  -> reset(*, seed, options) + step(action)
    obs returns reward + log/*  -> step returns (obs, reward, terminated, truncated, info)

The SMAClite-facing behaviour is preserved exactly: flattened ``state``, ``avail_actions``,
``agent_mask`` / ``real_agent_action_mask`` (Phase 3), reward shaping (legacy + v2),
original-reward tracking, action sanitisation, noop rescue, timing-lag vs masking-failure
classification, invalid-action metrics, per-map metadata, map sampling, and padding.

Diagnostic / logging metrics that used to live under ``obs["log/..."]`` now live in the
``info`` dict returned from ``reset``/``step``, with the ``log/`` prefix renamed to
``log_`` (Hydra/PyTorch-friendly, and the convention R2-Dreamer's trainer aggregates).
Logging-only values are **not** model observation fields. See METRIC_NAME_MAP below for
the old→new mapping.

Observation fields kept in the model observation (fixed-shape within a run):
    state, avail_actions, agent_mask*, real_agent_action_mask*, is_first, is_last,
    is_terminal      (* Phase 3 padding only)

``reward`` is returned as the second element of ``step`` (no longer an obs field).

Action interface
----------------
The environment accepts the centralised factorised action and converts it to a list of
per-unit integer actions before calling SMAClite. Accepted action forms (see ``_to_int_actions``):
    - flat one-hot / logits vector of length A*C  (decoded via FactorisedActionCodec)
    - sequence of A (or n_real_agents) integer action indices
    - legacy dict {"action_0":..., ... , optional "reset"} for backward-compat callers
Only actions for real agents are sent to SMAClite; padded-agent slots are dropped.
Environment-side sanitisation remains the final safety net (not the primary mask).

Invalid-action metric categories (unchanged definitions)
--------------------------------------------------------
post_mask_invalid  — action invalid at step time even after policy masking. Split into:
  timing_lag       — valid in the last obs mask given to the policy, invalid now (unit
                     state changed between observation and step).
  masking_failure  — already invalid in the last obs mask (policy should have suppressed
                     it). Should be 0 with correct policy-side masking.
"""

import pathlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from smacdreamer.envs.action_codec import FactorisedActionCodec, NOOP_ACTION


def _unit_ehp(unit) -> float:
    """Effective HP of a unit: current HP plus current shield (0 when the unit has no shield).

    Used for the allied/enemy effective-HP fractions consumed by the ally_ehp_v4 reward and
    the EHP diagnostics logged for every reward. Shields count toward survivability, so a
    shield-only hit must register as EHP loss.
    """
    return float(unit.hp) + float(getattr(unit, "shield", 0.0))


# Old obs "log/..." key  ->  new info "log_..." key. Definitions and aggregation level are
# unchanged; only the framework location (obs -> info) and the prefix (log/ -> log_) change.
METRIC_NAME_MAP = {
    "log/battle_won":                          "log_battle_won",
    "log/post_mask_invalid_action_count":      "log_post_mask_invalid_action_count",
    "log/post_mask_invalid_action_rate":       "log_post_mask_invalid_action_rate",
    "log/total_action_count":                  "log_total_action_count",
    "log/timing_lag_invalid_action_count":     "log_timing_lag_invalid_action_count",
    "log/timing_lag_invalid_action_rate":      "log_timing_lag_invalid_action_rate",
    "log/masking_failure_count":               "log_masking_failure_count",
    "log/masking_failure_rate":                "log_masking_failure_rate",
    "log/episode_invalid_action_count":        "log_episode_invalid_action_count",
    "log/episode_total_action_count":          "log_episode_total_action_count",
    "log/episode_invalid_action_rate":         "log_episode_invalid_action_rate",
    "log/step_post_mask_invalid_count":        "log_step_post_mask_invalid_count",
    "log/step_timing_lag_invalid_count":       "log_step_timing_lag_invalid_count",
    "log/step_masking_failure_count":          "log_step_masking_failure_count",
    "log/step_avail_mask_mismatch_count":      "log_step_avail_mask_mismatch_count",
    "log/step_kill_bonus":                     "log_step_kill_bonus",
    "log/step_step_penalty":                   "log_step_step_penalty",
    "log/step_v2_win_bonus":                   "log_step_v2_win_bonus",
    "log/step_v2_loss_penalty":                "log_step_v2_loss_penalty",
    "log/step_v2_enemy_kill_bonus":            "log_step_v2_enemy_kill_bonus",
    "log/step_v2_ally_death_penalty":          "log_step_v2_ally_death_penalty",
    "log/step_v2_ally_survival_bonus":         "log_step_v2_ally_survival_bonus",
    "log/step_v2_step_penalty":                "log_step_v2_step_penalty",
    "log/step_v2_damage_progress":             "log_step_v2_damage_progress",
    "log/enemy_kills_this_step":               "log_enemy_kills_this_step",
    "log/ally_deaths_this_step":               "log_ally_deaths_this_step",
    "log/enemies_alive":                       "log_enemies_alive",
    "log/allies_alive":                        "log_allies_alive",
    "log/original_env_reward":                 "log_original_env_reward",
    "log/shaped_reward":                       "log_shaped_reward",
    "log/reward_shaping_bonus":                "log_reward_shaping_bonus",
    "log/reward_shaping_enabled":              "log_reward_shaping_enabled",
    "log/episode_original_env_return":         "log_episode_original_env_return",
    "log/episode_shaped_return":               "log_episode_shaped_return",
    "log/episode_reward_shaping_bonus":        "log_episode_reward_shaping_bonus",
    "log/enemy_hp_total":                      "log_enemy_hp_total",
    "log/enemy_hp_damage_this_step":           "log_enemy_hp_damage_this_step",
    "log/episode_enemy_hp_damage":             "log_episode_enemy_hp_damage",
    "log/final_enemy_hp_total":                "log_final_enemy_hp_total",
    "log/first_ally_death_step":               "log_first_ally_death_step",
    "log/episode_attack_action_rate":          "log_episode_attack_action_rate",
    "log/episode_move_action_rate":            "log_episode_move_action_rate",
    "log/episode_noop_rate":                   "log_episode_noop_rate",
    "log/step_invalid_count":                  "log_step_invalid_count",
    "log/step_invalid_was_prev_valid_count":   "log_step_invalid_was_prev_valid_count",
    "log/step_invalid_was_prev_invalid_count": "log_step_invalid_was_prev_invalid_count",
    "log/map_id":                              "log_map_id",
    "log/num_real_agents":                     "log_num_real_agents",
    "log/num_real_enemies":                    "log_num_real_enemies",
    "log/padded_agent_count":                  "log_padded_agent_count",
    "log/extra_real_agent_action_slot_count":  "log_extra_real_agent_action_slot_count",
    "log/padded_agent_action_slot_count":      "log_padded_agent_action_slot_count",
    "log/ignored_padded_agent_action_count":   "log_ignored_padded_agent_action_count",
    "log/agent_mask_sum":                      "log_agent_mask_sum",
    "log/sampling_cycle":                      "log_sampling_cycle",
    "log/maps_seen_this_cycle":                "log_maps_seen_this_cycle",
    "log/total_unique_maps_seen":              "log_total_unique_maps_seen",
    "log/dataset_coverage_fraction":           "log_dataset_coverage_fraction",
    "log/total_train_maps":                    "log_total_train_maps",
}


class SMACliteDreamerEnv(gym.Env):
    """Centralised-control adapter: one R2-Dreamer agent drives all allied SMAClite units.

    Phase 1: single fixed scenario.
    Phase 2: pass a MapSampler to rotate across same-shape maps each episode.
    Phase 3: pass pad_dims to enable variable-size map padding.

    Gymnasium API
    -------------
    reset(*, seed=None, options=None) -> (observation, info)
    step(action)                      -> (observation, reward, terminated, truncated, info)

        terminated  = natural battle termination (is_terminal)
        truncated   = time-limit / wrapper truncation
        is_last     = terminated or truncated
        is_terminal = terminated
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str,
        max_episode_steps: int = 200,
        seed: int = 0,
        map_sampler=None,
        pad_dims=None,                    # Optional[PaddingDims] — enables Phase 3 padding
        kill_reward_bonus: float = 0.0,   # legacy flat-param shaping (used when v2 disabled)
        step_penalty: float = 0.0,        # legacy flat-param shaping (used when v2 disabled)
        reward_shaping_config=None,       # Optional[RewardShapingConfig] — v2 shaping system
        reward_fn=None,                   # Optional[callable(RewardContext)->(reward,terms)]
        gamma: float = 0.997,             # shaping discount; MUST equal the agent's discount
        obs_mode: str = "flat",           # "flat" (legacy right-padded) | "structured" (P0.3)
        strict_actions: bool = False,     # raise on an invalid requested action (vs sanitise)
        include_jepa_obs: bool = False,
        jepa_visibility_config=None,
    ):
        super().__init__()
        from smacdreamer.envs.reward_shaping import RewardShapingConfig as _RSC
        # Optional swappable reward callable (from reward_registry.resolve). When set, it
        # OVERRIDES the legacy/v2 shaping branch. When None, behaviour is unchanged.
        self._reward_fn = reward_fn
        self._gamma = float(gamma)
        # Observation representation. "flat" = the legacy whole-vector right-padding;
        # "structured" = the canonical per-entity layout (smacdreamer.envs.structured_obs).
        self._obs_mode = str(obs_mode)
        self._include_jepa_obs = bool(include_jepa_obs)
        self._jepa_visibility_config = jepa_visibility_config
        if self._obs_mode not in ("flat", "structured"):
            raise ValueError(
                f"unsupported obs_mode {self._obs_mode!r}; expected 'flat' or 'structured' "
                "(no silent fallback)"
            )
        self._local_to_global = np.zeros(0, dtype=np.int64)  # map-local -> global type ids
        # Action sanitiser is a DEBUG FALLBACK. In strict mode an invalid requested action
        # raises immediately (it should never happen once the policy is masked). The last
        # requested vs executed action lists are kept for inspection / tests.
        self._strict_actions = bool(strict_actions)
        self._last_requested_action: list = []
        self._last_executed_action: list = []
        # Initial totals captured at reset, used to normalise potentials to [0,1] fractions.
        self._init_enemy_hp_total: float = 0.0
        self._init_n_allies: int = 0
        self._prev_enemy_hp_frac: float = 1.0
        self._prev_ally_alive_frac: float = 1.0
        # Effective-HP (HP+shields) tracking for the ally_ehp_v4 ablation + EHP diagnostics.
        self._init_ally_ehp_total: float = 0.0
        self._init_enemy_ehp_total: float = 0.0
        self._prev_ally_ehp_frac: float = 1.0
        self._prev_enemy_ehp_frac: float = 1.0
        self._cur_ally_ehp_frac: float = 1.0
        self._cur_enemy_ehp_frac: float = 1.0
        # Per-episode shaping-component sums (read at episode end for log_episode_* keys).
        self._ep_ally_ehp_shaping: float = 0.0
        self._ep_terminal_anchor: float = 0.0
        self._ep_timeout_penalty: float = 0.0
        self._scenario = scenario
        self._max_episode_steps = max_episode_steps
        self._seed = seed
        self._map_sampler = map_sampler
        self._pad_dims = pad_dims
        self._kill_reward_bonus = float(kill_reward_bonus)
        self._step_penalty = float(step_penalty)
        self._prev_n_enemies: int = 0
        self._rs: _RSC = reward_shaping_config if reward_shaping_config is not None else _RSC()
        self._use_new_shaping: bool = (
            reward_shaping_config is not None and reward_shaping_config.enabled
        )
        self._prev_n_allies: int = 0
        self._prev_enemy_hp_total: float = 0.0
        self._ep_original_return: float = 0.0
        self._ep_shaped_return: float = 0.0
        self._ep_shaping_bonus: float = 0.0
        self._ep_enemy_hp_damage: float = 0.0
        self._first_ally_death_step: int = -1
        self._ep_attack_actions: int = 0
        self._ep_move_actions: int = 0
        self._ep_noop_actions: int = 0

        if map_sampler is not None:
            init_entry = map_sampler.peek()
            self._env = self._open_env(init_entry)
            self._current_map_name = init_entry.name
            # Phase 4: entries carry stable map_id from manifest (all distinct).
            # Phase 2/3: entries all have map_id=0, so use sequential index.
            _entry_ids = [e.map_id for e in map_sampler.maps]
            if len(set(_entry_ids)) == len(_entry_ids):
                self._map_id_map = {e.name: e.map_id for e in map_sampler.maps}
            else:
                self._map_id_map = {e.name: i for i, e in enumerate(map_sampler.maps)}
        else:
            import smaclite  # noqa: registers smaclite/* gymnasium IDs
            self._env = gym.make(f"smaclite/{scenario}-v0")
            self._current_map_name = scenario
            self._map_id_map = {scenario: 0}

        uw = getattr(self._env, 'unwrapped', self._env)

        self.n_agents: int = uw.n_agents
        self.n_enemies: int = uw.n_enemies
        self.n_actions: int = uw.n_actions
        self.obs_size: int = uw.obs_size
        self._refresh_type_vocab(uw)   # structured-mode map-local -> global type map

        # Phase 1/2 shape key for exact-match validation on map switch.
        # Phase 3 uses fit-within-pad-dims check instead.
        self._shape_key = (self.n_agents, self.n_enemies, self.n_actions, self.obs_size)

        obs_tuple, _ = self._env.reset(seed=seed)
        avail = uw.get_avail_actions()
        avail_lens = [len(a) for a in avail]
        if len(set(avail_lens)) != 1 or avail_lens[0] != self.n_actions:
            raise ValueError(
                f"Non-uniform avail_actions lengths {avail_lens} for scenario "
                f"'{self._current_map_name}'. "
                "Phase 1/2 requires all allied units to share the same n_actions."
            )

        self._last_obs_tuple = obs_tuple
        self._last_avail = avail
        self._done = True   # require an explicit reset() before stepping (Gym semantics)
        self._last_avail_returned: list = avail

        # Per-episode accumulators (reset on each reset() call).
        self._ep_step = 0
        self._ep_invalid_count = 0        # total post-mask invalids
        self._ep_total_count = 0          # total unit-action slots checked
        self._ep_timing_lag_count = 0     # invalids due to state change between obs and step
        self._ep_masking_failure_count = 0  # invalids that were already invalid in prev mask

        self._lifetime_invalid = 0
        self._lifetime_total = 0

        self._ep_metrics = self._zero_ep_metrics()
        self._ep_reward_terms: dict = {}
        self._current_map_id: int = self._map_id_map.get(self._current_map_name, 0)

        # Factorised action codec. Dimensions are the padded dims under Phase 3, otherwise
        # the real map dims. A = agent slots, C = actions per agent.
        A = self._pad_dims.max_agents if self._pad_dims else self.n_agents
        C = self._pad_dims.max_actions if self._pad_dims else self.n_actions
        self.codec = FactorisedActionCodec(num_agents=A, num_actions=C)

        # ---- Gymnasium spaces -------------------------------------------------
        self.observation_space = self._build_observation_space()
        # One categorical action per agent slot. R2-Dreamer's MultiOneHotAction wrapper
        # converts this MultiDiscrete into a flat one-hot Box tagged multi_discrete=True.
        self.action_space = self.codec.action_space()

    # ------------------------------------------------------------------
    # Gymnasium spaces
    # ------------------------------------------------------------------

    def _refresh_type_vocab(self, uw) -> None:
        """Recompute the map-local -> global unit-type map (structured mode only)."""
        if self._obs_mode != "structured":
            return
        from smacdreamer.envs.structured_obs import build_local_to_global
        type_ids = getattr(getattr(uw, "map_info", None), "unit_type_ids", None)
        self._local_to_global = build_local_to_global(type_ids)

    def _structured_pad_dims(self):
        """PaddingDims for structured obs — the configured caps, or real dims when unpadded."""
        if self._pad_dims is not None:
            return self._pad_dims
        from smacdreamer.envs.padding import PaddingDims
        return PaddingDims(max_agents=self.n_agents, max_enemies=self.n_enemies,
                           max_actions=self.n_actions, max_obs_size=self.obs_size)

    def _obs_dims(self):
        """Return (A, C, O) for the current observation layout (padded or real)."""
        if self._pad_dims is not None:
            return (self._pad_dims.max_agents, self._pad_dims.max_actions, self._pad_dims.max_obs_size)
        return (self.n_agents, self.n_actions, self.obs_size)

    def _build_observation_space(self) -> spaces.Dict:
        """Model observation space. Logging-only values are NOT included (they go in info).

        Fixed-shape within a run. ``agent_mask`` / ``real_agent_action_mask`` are present
        only under Phase 3 padding, matching the previous behaviour.
        """
        if self._obs_mode == "structured":
            from smacdreamer.envs import structured_obs as _so
            d = dict(_so.observation_space(self._structured_pad_dims()).spaces)
            if self._include_jepa_obs:
                from smacdreamer.jepa.online_tokens import spec_from_env
                spec = spec_from_env(getattr(self._env, "unwrapped", self._env), self._structured_pad_dims())
                d["jepa_entity"] = spaces.Box(
                    -np.inf, np.inf, shape=(spec.entities, spec.token_dim), dtype=np.float32)
                d["jepa_entity_mask"] = spaces.Box(0.0, 1.0, shape=(spec.entities,), dtype=np.float32)
                d["jepa_entity_slot_mask"] = spaces.Box(0.0, 1.0, shape=(spec.entities,), dtype=np.float32)
                d["jepa_static_condition"] = spaces.Box(
                    -np.inf, np.inf, shape=(spec.static_dim,), dtype=np.float32)
            return spaces.Dict(d)
        A, C, O = self._obs_dims()
        d = {
            "state":         spaces.Box(-np.inf, np.inf, shape=(A * O,), dtype=np.float32),
            "avail_actions": spaces.Box(0.0, 1.0, shape=(A * C,), dtype=np.float32),
            "is_first":      spaces.Box(0, 1, shape=(), dtype=bool),
            "is_last":       spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal":   spaces.Box(0, 1, shape=(), dtype=bool),
        }
        if self._pad_dims is not None:
            d["agent_mask"]             = spaces.Box(0.0, 1.0, shape=(A,), dtype=np.float32)
            d["real_agent_action_mask"] = spaces.Box(0.0, 1.0, shape=(A * C,), dtype=np.float32)
        return spaces.Dict(d)

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _to_int_actions(self, action) -> list:
        """Convert any accepted action representation into a list of per-agent ints.

        Accepted forms:
          * flat one-hot / logits vector length A*C (decoded by the codec; argmax/group)
          * sequence of A or n_agents integer action indices
          * legacy dict {"action_i": int, optional "reset"}
        Only the first ``n_agents`` (real) actions are returned; padded slots are dropped.
        """
        # Legacy dict form (backward-compat with old callers / smoke tests).
        if isinstance(action, dict):
            return [int(action[f"action_{i}"]) for i in range(self.n_agents)]

        arr = np.asarray(action)
        # Flat factorised one-hot / logits of length A*C -> decode to ints.
        if arr.ndim >= 1 and arr.size == self.codec.flat_dim and self.codec.flat_dim != self.codec.num_agents:
            # decode with validate=False so logits (not strictly one-hot) are also accepted;
            # argmax per group recovers the chosen action. Real-agent slice only.
            return self.codec.decode(arr, num_real_agents=self.n_agents, validate=False)
        # A vector of per-agent integer indices.
        flat = arr.reshape(-1)
        if flat.shape[0] in (self.n_agents, self.codec.num_agents):
            return [int(x) for x in flat[: self.n_agents]]
        raise ValueError(
            f"Unrecognised action of shape {arr.shape}; expected flat one-hot of length "
            f"{self.codec.flat_dim}, or {self.n_agents}/{self.codec.num_agents} integer actions."
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Standard Gymnasium reset. Returns (observation, info)."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        if self._map_sampler is not None:
            entry = self._map_sampler.next()
            if entry.name != self._current_map_name:
                self._env.close()
                self._env = self._open_env(entry)
                uw = getattr(self._env, 'unwrapped', self._env)
                new_shape = (uw.n_agents, uw.n_enemies, uw.n_actions, uw.obs_size)

                if self._pad_dims is not None:
                    # Phase 3: verify new map fits within padding dims.
                    if (uw.n_agents > self._pad_dims.max_agents or
                            uw.n_actions > self._pad_dims.max_actions or
                            uw.obs_size > self._pad_dims.max_obs_size):
                        raise ValueError(
                            f"Map '{entry.name}' shape {new_shape} exceeds padding dims "
                            f"(max_agents={self._pad_dims.max_agents}, "
                            f"max_actions={self._pad_dims.max_actions}, "
                            f"max_obs_size={self._pad_dims.max_obs_size}). "
                            "Update the 'padding' block in the Phase 3 manifest."
                        )
                    # Update real dims; padded dims stay fixed via _pad_dims.
                    self.n_agents = uw.n_agents
                    self.n_enemies = uw.n_enemies
                    self.n_actions = uw.n_actions
                    self.obs_size = uw.obs_size
                else:
                    # Phase 1/2: shape must match exactly.
                    if new_shape != self._shape_key:
                        raise ValueError(
                            f"Map '{entry.name}' shape {new_shape} != "
                            f"expected {self._shape_key}. "
                            "All Phase 2 maps must share the same "
                            "n_agents/n_enemies/n_actions/obs_size. "
                            f"Expected state shape {(self.n_agents * self.obs_size,)}, "
                            f"avail_actions shape {(self.n_agents * self.n_actions,)}, "
                            f"action keys action_0..action_{self.n_agents - 1}."
                        )
                self._current_map_name = entry.name
                self._current_map_id = self._map_id_map.get(entry.name, 0)
            # Propagate an EXPLICITLY-provided seed to the underlying SMAClite env so
            # evaluation is deterministic per (map, seed). During training reset() is called
            # with seed=None, so this stays an unseeded reset (SMAClite RNG keeps advancing for
            # episode diversity) — identical to the previous behaviour.
            obs_tuple, _ = self._env.reset(seed=seed)
        else:
            obs_tuple, _ = self._env.reset(seed=self._seed)

        uw = getattr(self._env, 'unwrapped', self._env)
        self._refresh_type_vocab(uw)   # map may have changed (multimap) -> refresh type map
        avail = uw.get_avail_actions()
        self._prev_n_enemies = len(uw.enemies)
        self._prev_n_allies  = len(uw.agents)
        self._prev_enemy_hp_total = float(sum(u.hp for u in uw.enemies.values()))

        # Capture initial totals for potential-based reward normalisation (fractions in [0,1]).
        self._init_enemy_hp_total = self._prev_enemy_hp_total
        self._init_n_allies = self._prev_n_allies
        self._prev_enemy_hp_frac = 1.0
        self._prev_ally_alive_frac = 1.0

        # Effective-HP (HP+shields) totals captured immediately after reset from all units.
        self._init_ally_ehp_total = float(sum(_unit_ehp(u) for u in uw.agents.values()))
        self._init_enemy_ehp_total = float(sum(_unit_ehp(u) for u in uw.enemies.values()))
        self._prev_ally_ehp_frac = 1.0
        self._prev_enemy_ehp_frac = 1.0
        self._cur_ally_ehp_frac = 1.0
        self._cur_enemy_ehp_frac = 1.0
        self._ep_ally_ehp_shaping = 0.0
        self._ep_terminal_anchor = 0.0
        self._ep_timeout_penalty = 0.0

        self._last_obs_tuple = obs_tuple
        self._last_avail = avail
        self._done = False
        self._ep_step = 0
        self._ep_invalid_count = 0
        self._ep_total_count = 0
        self._ep_timing_lag_count = 0
        self._ep_masking_failure_count = 0
        self._ep_original_return = 0.0
        self._ep_shaped_return   = 0.0
        self._ep_shaping_bonus   = 0.0
        self._ep_enemy_hp_damage = 0.0
        self._first_ally_death_step = -1
        self._ep_attack_actions = 0
        self._ep_move_actions   = 0
        self._ep_noop_actions   = 0
        self._ep_reward_terms: dict = {}     # running per-term sums for the episode
        self._ep_metrics = self._zero_ep_metrics()

        obs, info = self._build_obs(
            obs_tuple=obs_tuple,
            avail=avail,
            is_first=True,
            is_last=False,
            is_terminal=False,
            step_invalid=0,
            step_timing_lag=0,
            step_masking_failure=0,
            step_mask_mismatch=0,
            step_kill_bonus=0.0,
            step_step_penalty=0.0,
        )
        return obs, info

    def step(self, action):
        """Standard Gymnasium step. Returns (observation, reward, terminated, truncated, info)."""
        if self._done:
            raise RuntimeError(
                "step() called on a finished or unreset environment. Call reset() first."
            )

        # Only extract actions for real agents; padded agent action slots are ignored.
        requested_acts = self._to_int_actions(action)

        avail = getattr(self._env, 'unwrapped', self._env).get_avail_actions()
        acts, n_invalid, n_was_prev_valid, n_was_prev_invalid = (
            self._sanitise_actions(requested_acts, avail, strict=self._strict_actions)
        )
        # Expose requested vs EXECUTED action (what SMAClite actually runs) for tests/logging.
        self._last_requested_action = list(requested_acts)
        self._last_executed_action = list(acts)

        # Verify every sanitised action is valid under the current step-time mask.
        for _i, (_act, _mask) in enumerate(zip(acts, avail)):
            _valid = [j for j, v in enumerate(_mask) if v]
            if _valid and not (_act < len(_mask) and _mask[_act]):
                raise AssertionError(
                    f"Sanitised action is invalid: agent={_i}, action={_act}, "
                    f"current_avail={list(_mask)}, "
                    f"prev_avail={list(self._last_avail_returned[_i]) if _i < len(self._last_avail_returned) else 'N/A'}, "
                    f"map={self._current_map_name}"
                )

        n_mask_mismatch = sum(
            int(bool(c) != bool(p))
            for curr_mask, prev_mask in zip(avail, self._last_avail_returned)
            for c, p in zip(curr_mask, prev_mask)
        )

        self._ep_total_count += self.n_agents
        self._ep_invalid_count += n_invalid
        self._ep_timing_lag_count += n_was_prev_valid
        self._ep_masking_failure_count += n_was_prev_invalid
        self._lifetime_total += self.n_agents
        self._lifetime_invalid += n_invalid

        obs_tuple, reward, terminated, truncated, info = self._env.step(acts)
        self._ep_step += 1

        # --- Reward shaping ---
        # Dead units are removed from uw_post.agents / uw_post.enemies immediately
        # by SMACliteEnv.__update_deaths() after each inner step (verified).
        uw_post = getattr(self._env, 'unwrapped', self._env)
        cur_n_enemies = len(uw_post.enemies)
        cur_n_allies  = len(uw_post.agents)
        kill_delta    = max(0, self._prev_n_enemies - cur_n_enemies)
        ally_deaths   = max(0, self._prev_n_allies  - cur_n_allies)
        self._prev_n_enemies = cur_n_enemies
        self._prev_n_allies  = cur_n_allies

        # Enemy HP tracking (alive enemies only; dead units are already removed).
        cur_enemy_hp_total = float(sum(u.hp for u in uw_post.enemies.values()))
        enemy_hp_damage_this_step = max(0.0, self._prev_enemy_hp_total - cur_enemy_hp_total)
        self._prev_enemy_hp_total = cur_enemy_hp_total
        self._ep_enemy_hp_damage += enemy_hp_damage_this_step

        # Effective-HP (HP+shields) fractions, clamped to [0,1] (guards numerical noise /
        # healing; units cannot exceed their initial EHP here). Computed for EVERY reward so
        # the EHP diagnostics are available regardless of which reward is active.
        cur_ally_ehp_total = float(sum(_unit_ehp(u) for u in uw_post.agents.values()))
        cur_enemy_ehp_total = float(sum(_unit_ehp(u) for u in uw_post.enemies.values()))
        ally_ehp_frac = min(1.0, max(0.0, cur_ally_ehp_total / max(self._init_ally_ehp_total, 1e-8)))
        enemy_ehp_frac = min(1.0, max(0.0, cur_enemy_ehp_total / max(self._init_enemy_ehp_total, 1e-8)))
        self._cur_ally_ehp_frac = ally_ehp_frac
        self._cur_enemy_ehp_frac = enemy_ehp_frac

        # First ally death step.
        if ally_deaths > 0 and self._first_ally_death_step < 0:
            self._first_ally_death_step = self._ep_step

        # Action-type breakdown for the real agents this step.
        for _act_idx in acts:
            if _act_idx <= 1:
                self._ep_noop_actions += 1
            elif _act_idx <= 5:
                self._ep_move_actions += 1
            else:
                self._ep_attack_actions += 1

        # Enforce our own step-limit truncation. terminated reflects natural battle
        # termination; truncated reflects time-limit / wrapper truncation.
        if self._ep_step >= self._max_episode_steps:
            truncated = True
        is_last     = bool(terminated or truncated)
        is_terminal = bool(terminated)

        base_reward = np.float32(reward)  # raw normalised SMAClite reward

        if self._use_new_shaping:
            rs = self._rs
            # Terminal signals: applied exactly once on the episode's last step.
            # loss_penalty fires on ANY non-winning episode end, including time-limit truncation.
            terminal_win  = is_last and     info.get("battle_won", False)
            terminal_loss = is_last and not info.get("battle_won", False)
            v2_win_bonus    = rs.win_bonus    if terminal_win  else 0.0
            v2_loss_penalty = rs.loss_penalty if terminal_loss else 0.0  # negative in config

            v2_enemy_kill    = float(kill_delta)   * rs.enemy_kill_bonus
            v2_ally_death    = float(ally_deaths)  * rs.ally_death_penalty  # negative in config
            v2_ally_survival = float(cur_n_allies) * rs.ally_survival_bonus
            v2_step_penalty  = rs.step_penalty
            v2_damage_progress = enemy_hp_damage_this_step * rs.damage_delta_scale

            shaping_bonus = (
                v2_win_bonus + v2_loss_penalty
                + v2_enemy_kill + v2_ally_death
                + v2_ally_survival - v2_step_penalty
                + v2_damage_progress
            )
            shaped_reward = base_reward + np.float32(shaping_bonus)

            # Legacy flat-param diagnostics are zero when v2 is active.
            step_kill_bonus   = 0.0
            step_step_penalty = 0.0
        else:
            # Legacy flat-param shaping: kill_reward_bonus / step_penalty.
            # Behaviour identical to pre-v2 code.
            step_kill_bonus   = float(kill_delta) * self._kill_reward_bonus
            step_step_penalty = self._step_penalty
            shaping_bonus     = step_kill_bonus - step_step_penalty
            shaped_reward     = base_reward + np.float32(shaping_bonus)

            v2_win_bonus = v2_loss_penalty = v2_enemy_kill = 0.0
            v2_ally_death = v2_ally_survival = v2_step_penalty = 0.0
            v2_damage_progress = 0.0

        # --- Swappable reward callable (overrides the v2/legacy result when set) ---
        # Computes potentials from normalised fractions of the initial totals. The callable
        # returns (reward, terms); terms are logged per-step + per-episode as log_reward_*.
        cur_enemy_hp_frac = (
            cur_enemy_hp_total / self._init_enemy_hp_total
            if self._init_enemy_hp_total > 0 else 0.0
        )
        cur_ally_alive_frac = (
            cur_n_allies / self._init_n_allies if self._init_n_allies > 0 else 0.0
        )
        if self._reward_fn is not None:
            from smacdreamer.envs.reward_registry import RewardContext
            ctx = RewardContext(
                base_reward=float(base_reward),
                kill_delta=int(kill_delta),
                ally_deaths=int(ally_deaths),
                enemy_hp_damage=float(enemy_hp_damage_this_step),
                enemy_hp_frac=float(cur_enemy_hp_frac),
                prev_enemy_hp_frac=float(self._prev_enemy_hp_frac),
                ally_alive_frac=float(cur_ally_alive_frac),
                prev_ally_alive_frac=float(self._prev_ally_alive_frac),
                allies_alive=int(cur_n_allies),
                enemies_alive=int(cur_n_enemies),
                is_last=bool(is_last),
                battle_won=bool(info.get("battle_won", False)),
                step_idx=int(self._ep_step),
                max_episode_steps=int(self._max_episode_steps),
                gamma=self._gamma,
                ally_ehp_frac=float(ally_ehp_frac),
                prev_ally_ehp_frac=float(self._prev_ally_ehp_frac),
                enemy_ehp_frac=float(enemy_ehp_frac),
                prev_enemy_ehp_frac=float(self._prev_enemy_ehp_frac),
                # terminated = true battle end; truncated = timeout WITHOUT a natural terminal
                # (mutually exclusive, so terminal vs timeout anchors each fire at most once).
                terminated=bool(is_terminal),
                truncated=bool(truncated and not is_terminal),
            )
            reward_val, reward_terms = self._reward_fn(ctx)
            shaped_reward = np.float32(reward_val)
            shaping_bonus = float(reward_val) - float(base_reward)
        else:
            # No swappable reward: synthesise the term breakdown from the v2/legacy result
            # so the log_reward_* keys are always present and consistent.
            reward_terms = {
                "original": float(base_reward),
                "shaping_total": float(shaping_bonus),
            }

        # Accumulate per-episode shaping components (for log_episode_* keys), read at the
        # episode end in _compute_ep_metrics (which runs BEFORE _build_obs's term accumulation,
        # so we track these here rather than from self._ep_reward_terms). Keys are absent for
        # rewards that do not emit them (smaclite_default/dense_v3) -> contribute 0.
        self._ep_ally_ehp_shaping += float(reward_terms.get("ally_ehp", 0.0))
        self._ep_terminal_anchor += float(reward_terms.get("terminal", 0.0))
        self._ep_timeout_penalty += float(reward_terms.get("timeout", 0.0))

        # Advance potential bookkeeping for the next step.
        self._prev_enemy_hp_frac = float(cur_enemy_hp_frac)
        self._prev_ally_alive_frac = float(cur_ally_alive_frac)
        self._prev_ally_ehp_frac = float(ally_ehp_frac)
        self._prev_enemy_ehp_frac = float(enemy_ehp_frac)

        # Episode-level return accumulation.
        self._ep_original_return += float(base_reward)
        self._ep_shaped_return   += float(shaped_reward)
        self._ep_shaping_bonus   += float(shaping_bonus)

        self._done = is_last

        if is_last:
            self._ep_metrics = self._compute_ep_metrics(info)

        obs, out_info = self._build_obs(
            obs_tuple=obs_tuple,
            avail=avail,
            is_first=False,
            is_last=is_last,
            is_terminal=is_terminal,
            step_invalid=n_invalid,
            step_timing_lag=n_was_prev_valid,
            step_masking_failure=n_was_prev_invalid,
            step_mask_mismatch=n_mask_mismatch,
            step_kill_bonus=step_kill_bonus,
            step_step_penalty=step_step_penalty,
            step_v2_win_bonus=v2_win_bonus,
            step_v2_loss_penalty=v2_loss_penalty,
            step_v2_enemy_kill_bonus=v2_enemy_kill,
            step_v2_ally_death_penalty=v2_ally_death,
            step_v2_ally_survival_bonus=v2_ally_survival,
            step_v2_step_penalty=v2_step_penalty,
            step_v2_damage_progress=v2_damage_progress,
            enemy_kills_this_step=kill_delta,
            ally_deaths_this_step=ally_deaths,
            enemies_alive=cur_n_enemies,
            allies_alive=cur_n_allies,
            original_env_reward=float(base_reward),
            shaped_reward_val=float(shaped_reward),
            reward_shaping_bonus=float(shaping_bonus),
            enemy_hp_total=cur_enemy_hp_total,
            enemy_hp_damage_this_step=enemy_hp_damage_this_step,
            reward_terms=reward_terms,
        )
        # Surface battle_won at the top level of info too (SMAClite convention preserved).
        if "battle_won" in info:
            out_info["battle_won"] = info["battle_won"]
        return obs, np.float32(shaped_reward), is_terminal, truncated, out_info

    def close(self):
        self._env.close()

    def get_debug_context(self) -> dict:
        return {
            "current_map": getattr(self, "_current_map_name", None),
            "episode_step": getattr(self, "_ep_step", None),
            "last_requested_action": list(getattr(self, "_last_requested_action", [])),
            "last_executed_action": list(getattr(self, "_last_executed_action", [])),
            "episode_invalid_action_count": getattr(self, "_ep_invalid_count", None),
            "episode_total_action_count": getattr(self, "_ep_total_count", None),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_env(self, entry):
        if entry.type == 'builtin':
            import smaclite  # noqa
            return gym.make(f'smaclite/{entry.name}-v0')
        root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
        abs_path = root / entry.path
        from smaclite.env.smaclite import SMACliteEnv as _SMACliteEnv
        return _SMACliteEnv(map_file=str(abs_path))

    def _sanitise_actions(
        self, acts: list, avail: list, strict: bool = False
    ) -> tuple[list, int, int, int]:
        """Replace each invalid action with the first valid fallback (DEBUG FALLBACK).

        With ``strict=True`` an invalid requested action raises immediately instead of being
        replaced — used to prove the masked policy never emits an invalid action.

        Returns (sanitised, n_post_mask_invalid, n_timing_lag, n_masking_failure).

        Action categories
        -----------------
        valid            — action is valid in the current step-time mask; passed through.
        post_mask_invalid — action is invalid at step time despite the policy having
                            applied policy-side masking. Split into:
          timing_lag     — action was valid in the last obs mask returned to the policy
                           but invalid now because unit state changed between obs
                           generation and this step. Expected source of all invalids.
          masking_failure — action was already invalid in the last obs mask. Indicates
                            that policy-side masking failed to suppress it. Should be
                            exactly 0 with correct masking.
        """
        n_invalid = 0
        n_timing_lag = 0
        n_masking_failure = 0
        sanitised = []
        for i, (act, mask) in enumerate(zip(acts, avail)):
            if act < len(mask) and mask[act]:
                sanitised.append(act)
            else:
                if strict:
                    raise ValueError(
                        f"strict_actions: invalid requested action {act} for agent {i} "
                        f"(avail={list(mask)}, map={self._current_map_name}). The masked policy "
                        "must never request an invalid action."
                    )
                n_invalid += 1
                prev_mask = (
                    self._last_avail_returned[i]
                    if i < len(self._last_avail_returned)
                    else mask
                )
                if act < len(prev_mask) and prev_mask[act]:
                    n_timing_lag += 1
                else:
                    n_masking_failure += 1
                valid_indices = [j for j, v in enumerate(mask) if v]
                sanitised.append(valid_indices[0] if valid_indices else NOOP_ACTION)
        return sanitised, n_invalid, n_timing_lag, n_masking_failure

    def _build_obs(
        self,
        obs_tuple,
        avail,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
        step_invalid: int = 0,
        step_timing_lag: int = 0,
        step_masking_failure: int = 0,
        step_mask_mismatch: int = 0,
        step_kill_bonus: float = 0.0,
        step_step_penalty: float = 0.0,
        step_v2_win_bonus: float = 0.0,
        step_v2_loss_penalty: float = 0.0,
        step_v2_enemy_kill_bonus: float = 0.0,
        step_v2_ally_death_penalty: float = 0.0,
        step_v2_ally_survival_bonus: float = 0.0,
        step_v2_step_penalty: float = 0.0,
        step_v2_damage_progress: float = 0.0,
        enemy_kills_this_step: int = 0,
        ally_deaths_this_step: int = 0,
        enemies_alive: int = 0,
        allies_alive: int = 0,
        original_env_reward: float = 0.0,
        shaped_reward_val: float = 0.0,
        reward_shaping_bonus: float = 0.0,
        enemy_hp_total: float = 0.0,
        enemy_hp_damage_this_step: float = 0.0,
        reward_terms: dict = None,
    ) -> tuple[dict, dict]:
        """Build the (observation, info) pair.

        observation: model-relevant fixed-shape fields only.
        info: all diagnostic / logging metrics, keyed with the ``log_`` prefix.
        """
        self._last_avail_returned = avail

        # Padding diagnostics (independent of the obs representation; logged below).
        if self._pad_dims is not None:
            MA = self._pad_dims.max_agents
            MC = self._pad_dims.max_actions
            n_padded            = MA - self.n_agents
            extra_real_slots    = self.n_agents * (MC - self.n_actions)
            padded_agent_slots  = n_padded * MC
            ignored_pad_actions = n_padded
        else:
            n_padded = extra_real_slots = padded_agent_slots = ignored_pad_actions = 0
        agent_mask_sum = float(self.n_agents)

        _is = dict(
            is_first=np.array(is_first, dtype=bool),
            is_last=np.array(is_last, dtype=bool),
            is_terminal=np.array(is_terminal, dtype=bool),
        )
        if self._obs_mode == "structured":
            from smacdreamer.envs import structured_obs as _so
            uw = getattr(self._env, 'unwrapped', self._env)
            mi = uw.map_info
            # Global unit-type index per allied/enemy slot from the UNIT OBJECTS (consistent
            # global vocab, independent of SMAClite's per-map local one-hot which is dropped
            # for single-type maps). -1 for dead/absent slots; raises on an unknown type.
            def _gtype(unit):
                nm = getattr(unit.type, "name", str(unit.type)).upper()
                if nm not in _so.UNIT_TYPE_TO_GLOBAL:
                    raise ValueError(f"unit type {nm!r} not in global vocabulary")
                return _so.UNIT_TYPE_TO_GLOBAL[nm]
            agent_type_g = np.full(self.n_agents, -1, dtype=np.int64)
            for _u in uw.agents.values():
                _j = int(_u.id_in_faction)   # = canonical agent slot (matches the obs layout)
                if 0 <= _j < self.n_agents:
                    agent_type_g[_j] = _gtype(_u)
            enemy_type_g = np.full(self.n_enemies, -1, dtype=np.int64)
            for _u in uw.enemies.values():
                _j = int(_u.id_in_faction)
                if 0 <= _j < self.n_enemies:
                    enemy_type_g[_j] = _gtype(_u)
            blocks = _so.build_structured_obs(
                obs_tuple, avail,
                n_agents=self.n_agents, n_enemies=self.n_enemies,
                enemy_feat_size=int(uw.enemy_feat_size), ally_feat_size=int(uw.ally_feat_size),
                enemy_has_shields=bool(mi.enemy_has_shields),
                ally_has_shields=bool(mi.ally_has_shields),
                num_unit_types=int(mi.num_unit_types), n_actions=self.n_actions,
                alive_ids=list(uw.agents.keys()), local_to_global=self._local_to_global,
                pad_dims=self._structured_pad_dims(),
                agent_type_g=agent_type_g, enemy_type_g=enemy_type_g,
            )
            obs = {**_so.flatten_for_model(blocks), **_is}
            if self._include_jepa_obs:
                from smacdreamer.jepa.online_tokens import build_jepa_observation
                jepa_obs, _ = build_jepa_observation(
                    uw,
                    self._structured_pad_dims(),
                    visibility=self._jepa_visibility_config,
                )
                obs.update(jepa_obs)
        elif self._pad_dims is not None:
            from smacdreamer.envs.padding import (
                pad_state, pad_avail, make_agent_mask, make_real_agent_action_mask,
            )
            MA = self._pad_dims.max_agents
            MC = self._pad_dims.max_actions
            obs = {
                "state":                  pad_state(obs_tuple, self.obs_size, MA, self._pad_dims.max_obs_size),
                "avail_actions":          pad_avail(avail, self.n_actions, MA, MC),
                **_is,
                "agent_mask":             make_agent_mask(self.n_agents, MA),
                "real_agent_action_mask": make_real_agent_action_mask(make_agent_mask(self.n_agents, MA), MC),
            }
        else:
            obs = {
                "state":         np.concatenate(obs_tuple).astype(np.float32),
                "avail_actions": np.concatenate(avail).astype(np.float32),
                **_is,
            }

        _f = lambda x: np.array(float(x), dtype=np.float32)
        info = {
            # Step-level metrics (new canonical names)
            "log_step_post_mask_invalid_count":        _f(step_invalid),
            "log_step_sanitisation_occurred":          _f(1.0 if step_invalid > 0 else 0.0),
            "log_step_timing_lag_invalid_count":       _f(step_timing_lag),
            "log_step_masking_failure_count":          _f(step_masking_failure),
            "log_step_avail_mask_mismatch_count":      _f(step_mask_mismatch),
            # Legacy flat-param shaping diagnostics (0.0 when v2 active)
            "log_step_kill_bonus":                     _f(step_kill_bonus),
            "log_step_step_penalty":                   _f(step_step_penalty),
            # V2 shaping components (distinct namespace; 0.0 when v2 disabled)
            "log_step_v2_win_bonus":                   _f(step_v2_win_bonus),
            "log_step_v2_loss_penalty":                _f(step_v2_loss_penalty),
            "log_step_v2_enemy_kill_bonus":            _f(step_v2_enemy_kill_bonus),
            "log_step_v2_ally_death_penalty":          _f(step_v2_ally_death_penalty),
            "log_step_v2_ally_survival_bonus":         _f(step_v2_ally_survival_bonus),
            "log_step_v2_step_penalty":                _f(step_v2_step_penalty),
            "log_step_v2_damage_progress":             _f(step_v2_damage_progress),
            # Combat state (always present)
            "log_enemy_kills_this_step":               _f(enemy_kills_this_step),
            "log_ally_deaths_this_step":               _f(ally_deaths_this_step),
            "log_enemies_alive":                       _f(enemies_alive),
            "log_allies_alive":                        _f(allies_alive),
            "log_enemy_hp_total":                      _f(enemy_hp_total),
            "log_enemy_hp_damage_this_step":           _f(enemy_hp_damage_this_step),
            # Reward breakdown (always present)
            "log_original_env_reward":                 _f(original_env_reward),
            "log_shaped_reward":                       _f(shaped_reward_val),
            "log_reward_shaping_bonus":                _f(reward_shaping_bonus),
            "log_reward_shaping_enabled":              _f(1.0 if self._use_new_shaping else 0.0),
            # Step-level metrics (old aliases)
            "log_step_invalid_count":                  _f(step_invalid),
            "log_step_invalid_was_prev_valid_count":   _f(step_timing_lag),
            "log_step_invalid_was_prev_invalid_count": _f(step_masking_failure),
            # Map and padding metrics
            "log_map_id":                              _f(self._current_map_id),
            "log_num_real_agents":                     _f(self.n_agents),
            "log_num_real_enemies":                    _f(self.n_enemies),
            "log_padded_agent_count":                  _f(n_padded),
            "log_extra_real_agent_action_slot_count":  _f(extra_real_slots),
            "log_padded_agent_action_slot_count":      _f(padded_agent_slots),
            "log_ignored_padded_agent_action_count":   _f(ignored_pad_actions),
            "log_agent_mask_sum":                      _f(agent_mask_sum),
            # Episode metrics (carried forward; overwritten at episode end)
            **self._ep_metrics,
        }

        # --- Per-term reward breakdown (log_reward_term_*) ---
        # Per-step contribution of each reward term + the running per-episode sum, plus the
        # unshaped original and the shaping total. These are log_* so the encoder excludes
        # them and the trainer aggregates them. Keys are stable across maps/steps (the union
        # of canonical term names) so ParallelEnv can stack them across workers.
        canonical_terms = ("win", "hp", "ally", "positioning", "kill", "death",
                           "survival", "step_penalty", "damage")
        rt = reward_terms or {}
        # Accumulate episode sums.
        for k, v in rt.items():
            self._ep_reward_terms[k] = self._ep_reward_terms.get(k, 0.0) + float(v)
        for name in canonical_terms:
            info[f"log_reward_term_{name}"] = _f(rt.get(name, 0.0))
            info[f"log_reward_term_{name}_ep_sum"] = _f(self._ep_reward_terms.get(name, 0.0))
        info["log_reward_original"]      = _f(rt.get("original", original_env_reward))
        info["log_reward_shaping_total"] = _f(rt.get("shaping_total", reward_shaping_bonus))

        if self._map_sampler is not None:
            cm = self._map_sampler.coverage_metrics()
            info["log_sampling_cycle"]            = _f(cm["sampling_cycle"])
            info["log_maps_seen_this_cycle"]      = _f(cm["maps_seen_this_cycle"])
            info["log_total_unique_maps_seen"]    = _f(cm["total_unique_maps_seen"])
            info["log_dataset_coverage_fraction"] = _f(cm["dataset_coverage_fraction"])
            info["log_total_train_maps"]          = _f(cm["total_train_maps"])
        return obs, info

    def _zero_ep_metrics(self) -> dict:
        _f = lambda: np.array(0.0, dtype=np.float32)
        return {
            # New canonical names
            "log_battle_won":                          _f(),
            "log_post_mask_invalid_action_count":      _f(),
            "log_post_mask_invalid_action_rate":       _f(),
            "log_total_action_count":                  _f(),
            "log_timing_lag_invalid_action_count":     _f(),
            "log_timing_lag_invalid_action_rate":      _f(),
            "log_masking_failure_count":               _f(),
            "log_masking_failure_rate":                _f(),
            # Old aliases
            "log_episode_invalid_action_count":        _f(),
            "log_episode_total_action_count":          _f(),
            "log_episode_invalid_action_rate":         _f(),
            # Episode-level return totals
            "log_episode_original_env_return":         _f(),
            "log_episode_shaped_return":               _f(),
            "log_episode_reward_shaping_bonus":        _f(),
            # Effective-HP shaping breakdown + diagnostics (ally_ehp_v4 ablation; present for
            # every reward so the schema is stable across runs and held-out eval).
            "log_episode_total_shaping":               _f(),
            "log_episode_ally_ehp_shaping":            _f(),
            "log_episode_terminal_anchor":             _f(),
            "log_episode_timeout_penalty":             _f(),
            "log_final_ally_ehp_frac":                 _f(),
            "log_final_enemy_ehp_frac":                _f(),
            "log_episode_ally_ehp_lost":               _f(),
            "log_episode_enemy_ehp_lost":              _f(),
            "log_shaping_to_original_ratio":           _f(),
            # Damage-progress diagnostics
            "log_episode_enemy_hp_damage":             _f(),
            "log_final_enemy_hp_total":                _f(),
            "log_first_ally_death_step":               _f(),
            "log_episode_attack_action_rate":          _f(),
            "log_episode_move_action_rate":            _f(),
            "log_episode_noop_rate":                   _f(),
        }

    def _compute_ep_metrics(self, info: dict) -> dict:
        total   = self._ep_total_count
        invalid = self._ep_invalid_count
        lag     = self._ep_timing_lag_count
        failure = self._ep_masking_failure_count
        inv_rate  = invalid / total if total > 0 else 0.0
        lag_rate  = lag     / total if total > 0 else 0.0
        fail_rate = failure / total if total > 0 else 0.0
        _f = lambda x: np.array(float(x), dtype=np.float32)
        return {
            # New canonical names
            "log_battle_won":                          _f(info.get("battle_won", False)),
            "log_post_mask_invalid_action_count":      _f(invalid),
            "log_post_mask_invalid_action_rate":       _f(inv_rate),
            "log_total_action_count":                  _f(total),
            "log_timing_lag_invalid_action_count":     _f(lag),
            "log_timing_lag_invalid_action_rate":      _f(lag_rate),
            "log_masking_failure_count":               _f(failure),
            "log_masking_failure_rate":                _f(fail_rate),
            # Old aliases
            "log_episode_invalid_action_count":        _f(invalid),
            "log_episode_total_action_count":          _f(total),
            "log_episode_invalid_action_rate":         _f(inv_rate),
            # Episode-level return totals
            "log_episode_original_env_return":         _f(self._ep_original_return),
            "log_episode_shaped_return":               _f(self._ep_shaped_return),
            "log_episode_reward_shaping_bonus":        _f(self._ep_shaping_bonus),
            # Effective-HP shaping breakdown + diagnostics (ally_ehp_v4 ablation).
            "log_episode_total_shaping":               _f(self._ep_shaping_bonus),
            "log_episode_ally_ehp_shaping":            _f(self._ep_ally_ehp_shaping),
            "log_episode_terminal_anchor":             _f(self._ep_terminal_anchor),
            "log_episode_timeout_penalty":             _f(self._ep_timeout_penalty),
            "log_final_ally_ehp_frac":                 _f(self._cur_ally_ehp_frac),
            "log_final_enemy_ehp_frac":                _f(self._cur_enemy_ehp_frac),
            "log_episode_ally_ehp_lost":               _f(1.0 - self._cur_ally_ehp_frac),
            "log_episode_enemy_ehp_lost":              _f(1.0 - self._cur_enemy_ehp_frac),
            "log_shaping_to_original_ratio":           _f(
                self._ep_shaping_bonus / max(abs(self._ep_original_return), 1e-8)
            ),
            # Damage-progress diagnostics
            "log_episode_enemy_hp_damage":             _f(self._ep_enemy_hp_damage),
            "log_final_enemy_hp_total":                _f(self._prev_enemy_hp_total),
            "log_first_ally_death_step":               _f(self._first_ally_death_step),
            "log_episode_attack_action_rate":          _f(
                self._ep_attack_actions / self._ep_total_count if self._ep_total_count > 0 else 0.0
            ),
            "log_episode_move_action_rate":            _f(
                self._ep_move_actions / self._ep_total_count if self._ep_total_count > 0 else 0.0
            ),
            "log_episode_noop_rate":                   _f(
                self._ep_noop_actions / self._ep_total_count if self._ep_total_count > 0 else 0.0
            ),
        }
