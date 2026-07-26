# TACTICAL_MIXTURE_V1_2_CENTERED_TRUST_REGION
import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32
from tactical_policy import TacticalMixturePolicy
from hierarchical_dreamer import (
    apply_hierarchy_gradient_guards,
    build_hierarchical_modules, clone_and_freeze_hierarchy,
    hierarchical_act_logits, hierarchical_auxiliary_loss,
    hierarchy_state_dict_fields, hierarchy_training_state,
    load_hierarchy_training_state, load_hierarchical_compatible_state,
    update_slow_option_critic,
)

# OPTION_CRITIC_HIERARCHY_V2
# OPTION_CRITIC_P0P1_HOTFIX_V3

# TACTICAL_MIXTURE_V1
# TACTICAL_MIXTURE_HARDENING_V1_1
try:
    from smacdreamer.jepa.state import unpack_state as _jepa_unpack_state
except Exception:  # smacdreamer may be absent for non-SMAClite RSSM users.
    _jepa_unpack_state = None


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)
        wm_cfg = getattr(config, "world_model", None)
        self.world_model_backend = str(getattr(wm_cfg, "backend", "rssm") if wm_cfg is not None else "rssm").lower()
        if self.world_model_backend not in ("rssm", "jepa"):
            raise ValueError(f"Unsupported world_model.backend {self.world_model_backend!r}")

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        if self.world_model_backend == "rssm":
            self.encoder = networks.MultiEncoder(config.encoder, shapes)
            self.embed_size = self.encoder.out_dim
            self.rssm = rssm.RSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
            self.feat_size = self.rssm.feat_size
            self.jepa_world_model = None
        else:
            from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint
            from smacdreamer.jepa.world_model import FrozenJEPAWorldModel

            jepa_cfg = wm_cfg.jepa
            if bool(getattr(jepa_cfg, "freeze_core", True)) is False:
                raise NotImplementedError("JEPA fine-tuning is not implemented; set freeze_core=true")
            if "jepa_entity" not in shapes:
                raise ValueError("JEPA backend requires jepa_entity observation fields; enable include_jepa_obs")
            e, token_dim = shapes["jepa_entity"]
            static_dim = shapes["jepa_static_condition"][0]
            actor_nvec = getattr(act_space, "nvec", None)
            if actor_nvec is not None:
                max_agents = int(len(actor_nvec))
                max_actions = int(actor_nvec[0])
            else:
                max_agents = int(shapes.get("agent_slot_mask", (1,))[0])
                max_actions = int(self.act_dim // max(max_agents, 1))
            max_enemies = int(e) - max_agents
            live_metadata = {
                "mode": "entity",
                "max_agents": max_agents,
                "max_enemies": max_enemies,
                "max_actions": max_actions,
                "token_dim": int(token_dim),
                "static_dim": int(static_dim),
                "n_actions": max_actions,
            }
            configured_live = getattr(jepa_cfg, "live_metadata", None)
            if configured_live is not None:
                configured = dict(configured_live)
                configured.update(live_metadata)
                live_metadata = configured
            core, memory, info = load_frozen_jepa_checkpoint(
                str(jepa_cfg.checkpoint),
                map_location=self.device,
                live_metadata=live_metadata,
                strict=bool(getattr(jepa_cfg, "strict_checkpoint", True)),
            )
            feature_dim = getattr(jepa_cfg, "feature_dim", None)
            if feature_dim is None:
                feature_dim = int(config.rssm.deter) + int(config.rssm.stoch) * int(config.rssm.discrete)
            self.jepa_world_model = FrozenJEPAWorldModel(
                core=core,
                memory_module=memory,
                info=info,
                feature_dim=int(feature_dim),
                presence_threshold=float(getattr(jepa_cfg, "presence_threshold", 0.5)),
            )
            self.encoder = None
            self.rssm = None
            self.embed_size = 0
            self.feat_size = self.jepa_world_model.feat_size
            self.rep_loss = "jepa"
            frozen = sum(p.numel() for p in self.jepa_world_model.parameters_frozen())
            trainable_adapter = sum(p.numel() for p in self.jepa_world_model.feature_adapter.parameters())
            print(f"{frozen:>14,}: frozen_jepa_core")
            print(f"{trainable_adapter:>14,}: jepa_feature_adapter")
        self.reward = networks.MLPHead(config.reward, self.feat_size)
        self.cont = networks.MLPHead(config.cont, self.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.feat_size)
        self.value = networks.MLPHead(config.critic, self.feat_size)

        # --- Action masking (P0.1 real / P0.2 imagined). Gated on config.action_masking; the
        # actor shape is (C,)*A so A/C come straight from it. unimix is spread over VALID
        # actions only inside MaskedMultiOneHotDist. ---------------------------------------
        self.action_masking = bool(getattr(config, "action_masking", False))
        self._actor_shape = tuple(map(int, config.actor.shape))
        self._mask_A = len(self._actor_shape)
        self._mask_C = int(self._actor_shape[0]) if self._mask_A else 0
        self._actor_unimix = (
            float(getattr(config.actor.dist, "unimix_ratio", 0.0)) if self.act_discrete else 0.0
        )
        _tactical_cfg = getattr(config, 'tactical_mixture', None)
        self.tactical_enabled = bool(
            getattr(_tactical_cfg, 'enabled', False)
            if _tactical_cfg is not None else False
        )
        self.tactical_policy = None
        if self.tactical_enabled:
            if self.world_model_backend != 'jepa':
                raise NotImplementedError(
                    'Tactical Mixture v1 is validated only for JEPA'
                )
            if not self.action_masking or not self.act_discrete:
                raise ValueError(
                    'Tactical Mixture v1 requires masked discrete actions'
                )
            self.tactical_policy = TacticalMixturePolicy(
                self.feat_size, sum(self._actor_shape), _tactical_cfg
            )
            self.tactical_policy.assert_legacy_equivalence_ready()

        build_hierarchical_modules(self, config)

        # P0.2: prob threshold for the predicted availability mask -> logit cut.
        _p = min(max(float(getattr(config, "mask_threshold", 0.7)), 1e-4), 1 - 1e-4)
        self._mask_threshold = _p
        self._mask_threshold_logit = float(math.log(_p / (1.0 - _p)))
        # P0.2 latent heads: predict next available-action mask (A*C) and next alive mask (A).
        # Binary (BCE) heads built from the continuation head's config (same trunk/dist). Only
        # created when masking is on so non-masked runs and clone_and_freeze are unaffected.
        if self.action_masking:
            avail_cfg = copy.deepcopy(config.cont)
            avail_cfg.shape = [self._mask_A * self._mask_C]
            avail_cfg.name = "avail_head"
            alive_cfg = copy.deepcopy(config.cont)
            alive_cfg.shape = [self._mask_A]
            alive_cfg.name = "alive_head"
            self.avail_head = networks.MLPHead(avail_cfg, self.feat_size)
            self.alive_head = networks.MLPHead(alive_cfg, self.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
        }
        if self.tactical_enabled:
            modules["tactical_policy"] = self.tactical_policy
        if self.world_model_backend == "rssm":
            modules["rssm"] = self.rssm
            modules["encoder"] = self.encoder
        else:
            modules["jepa_feature_adapter"] = self.jepa_world_model.feature_adapter
        if self.action_masking:
            modules["avail_head"] = self.avail_head
            modules["alive_head"] = self.alive_head

        if self.world_model_backend == "jepa":
            pass
        elif self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        if self.tactical_enabled:
            _tactical_settings = self.tactical_policy.settings
            if _tactical_settings.freeze_base_actor:
                for _param in self.actor.parameters():
                    _param.requires_grad_(False)
                modules.pop('actor', None)
                print(' tactical safety: inherited base actor frozen')
            if _tactical_settings.freeze_feature_adapter:
                if self.world_model_backend != 'jepa':
                    raise RuntimeError(
                        'freeze_feature_adapter requires JEPA backend'
                    )
                for _param in (
                    self.jepa_world_model.feature_adapter.parameters()
                ):
                    _param.requires_grad_(False)
                modules.pop('jepa_feature_adapter', None)
                print(' tactical safety: inherited JEPA adapter frozen')
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        if self.hierarchical_enabled:
            modules["hierarchical_options"] = self.hierarchical_options
            modules["option_critic"] = self.option_critic
            if self.hierarchical_options.settings.freeze_base_actor:
                modules.pop("actor", None)
            if self.hierarchical_options.settings.freeze_feature_adapter:
                modules.pop("jepa_feature_adapter", None)
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")
        _optimizer_param_ids = [
            id(param) for param in self._named_params.values()
        ]
        if len(_optimizer_param_ids) != len(set(_optimizer_param_ids)):
            raise RuntimeError(
                'optimizer parameter registry contains duplicates'
            )
        if self.tactical_enabled:
            _tactical_param_ids = {
                id(param) for param in self.tactical_policy.parameters()
            }
            _registered_tactical_ids = {
                id(param)
                for name, param in self._named_params.items()
                if name.startswith('tactical_policy.')
            }
            if _registered_tactical_ids != _tactical_param_ids:
                raise RuntimeError(
                    'tactical parameters are not registered exactly once'
                )

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        # AMP dtype is explicit. bf16 avoids fp16 overflow on large structured observations;
        # float32 disables autocast and GradScaler. Unknown values fail instead of silently
        # becoming fp16.
        _amp = str(getattr(config, "amp_dtype", "float16")).lower()
        if _amp in ("bfloat16", "bf16"):
            self._amp_dtype = torch.bfloat16
        elif _amp in ("float16", "fp16"):
            self._amp_dtype = torch.float16
        elif _amp in ("float32", "fp32", "none", "off"):
            self._amp_dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported amp_dtype {getattr(config, 'amp_dtype', None)!r}; "
                "expected bfloat16, float16, or float32."
            )
        self._scaler = GradScaler(
            enabled=(self.device.type == "cuda" and self._amp_dtype == torch.float16))

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def training_state_dict(self):
        return {
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": self._scaler.state_dict(),
            "slow_value": self._slow_value.state_dict(),
            "slow_value_updates": self._slow_value_updates,
            "return_ema": self.return_ema.state_dict() if hasattr(self.return_ema, "state_dict") else None,
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "hierarchical_options": hierarchy_training_state(self),
        }

    def load_training_state_dict(self, state):
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self._scaler.load_state_dict(state["scaler"])
        self._slow_value.load_state_dict(state["slow_value"])
        self._slow_value_updates = int(state.get("slow_value_updates", self._slow_value_updates))
        if state.get("return_ema") is not None and hasattr(self.return_ema, "load_state_dict"):
            self.return_ema.load_state_dict(state["return_ema"])
        load_hierarchy_training_state(self, state)

    def tactical_metadata(self):
        if not self.tactical_enabled:
            return {
                "schema_version": 2,
                "architecture": "legacy",
                "enabled": False,
            }
        metadata = self.tactical_policy.metadata()
        metadata["enabled"] = True
        return metadata

    def load_tactical_compatible_state_dict(
        self,
        state_dict,
        checkpoint_metadata=None,
    ):
        """Strict tactical resume or allowlisted migration from legacy.

        Metadata-less tactical best checkpoints from v1 are accepted only when
        their live tactical keys load shape-strictly. This repairs the original
        best-checkpoint metadata omission without relaxing legacy migration.
        """
        if not self.tactical_enabled:
            self.load_state_dict(state_dict, strict=True)
            self.clone_and_freeze()
            return {"migrated_legacy": False, "strict": True}

        state_keys = tuple(state_dict.keys())
        has_live_tactical = any(
            key.startswith("tactical_policy.") for key in state_keys
        )

        metadata_is_legacy = bool(
            checkpoint_metadata is not None
            and (
                checkpoint_metadata.get("enabled") is False
                or checkpoint_metadata.get("architecture") == "legacy"
            )
        )
        if metadata_is_legacy and has_live_tactical:
            raise RuntimeError(
                "checkpoint metadata declares a legacy policy but tactical "
                "parameter keys are present"
            )

        if checkpoint_metadata is not None and not metadata_is_legacy:
            architecture = checkpoint_metadata.get("architecture")
            if architecture not in (
                "tactical_mixture_v1",
                "tactical_mixture_v1_1",
                "tactical_mixture_v1_2",
            ):
                raise RuntimeError(
                    f"unsupported tactical checkpoint architecture: {architecture!r}"
                )
            expected = self.tactical_metadata()
            for key in (
                "num_tactics",
                "embedding_dim",
                "hidden_dim",
                "duration",
                "feature_dim",
                "action_logit_dim",
                "eval_confidence_threshold",
                "freeze_base_actor",
                "freeze_feature_adapter",
                "max_residual_to_base",
                "max_abs_residual_logit",
                "selector_symmetry_break_std",
                "residual_scale",
                "min_selector_mi_normalized",
                "base_kl_target",
                "base_kl_scale",
            ):
                if key in checkpoint_metadata and (
                    checkpoint_metadata.get(key) != expected.get(key)
                ):
                    raise RuntimeError(
                        f"tactical metadata mismatch for {key}: "
                        f"{checkpoint_metadata.get(key)!r} "
                        f"!= {expected.get(key)!r}"
                    )
            self.load_state_dict(state_dict, strict=True)
            self.clone_and_freeze()
            return {
                "migrated_legacy": False,
                "strict": True,
                "checkpoint_architecture": architecture,
            }

        if has_live_tactical:
            incompatible = self.load_state_dict(state_dict, strict=False)
            illegal_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("_frozen_tactical_policy.")
            ]
            if illegal_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "metadata-less tactical checkpoint is incompatible: "
                    f"illegal_missing={illegal_missing}, "
                    f"unexpected={list(incompatible.unexpected_keys)}"
                )
            self.clone_and_freeze()
            return {
                "migrated_legacy": False,
                "strict": not bool(incompatible.missing_keys),
                "metadata_inferred": True,
            }

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_prefixes = (
            "tactical_policy.",
            "_frozen_tactical_policy.",
        )
        illegal_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "legacy tactical migration found incompatible keys: "
                f"illegal_missing={illegal_missing}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
        if not incompatible.missing_keys:
            raise RuntimeError(
                "checkpoint has no tactical metadata and no missing tactical keys"
            )
        self.tactical_policy.assert_legacy_equivalence_ready()
        self.clone_and_freeze()
        return {
            "migrated_legacy": True,
            "strict": False,
            "missing_keys": list(incompatible.missing_keys),
        }

    def set_hierarchy_training_step(self, step):
        if not self.hierarchical_enabled:
            return
        self.hierarchical_options.set_training_step(step)
        if getattr(self, "_frozen_hierarchical_options", None) is not None:
            self._frozen_hierarchical_options.set_training_step(step)

    def hierarchical_metadata(self):
        if not self.hierarchical_enabled:
            return {
                "schema_version": 1,
                "architecture": "legacy",
                "enabled": False,
            }
        metadata = self.hierarchical_options.metadata()
        metadata["enabled"] = True
        return metadata

    def load_hierarchical_compatible_state_dict(
        self,
        state_dict,
        checkpoint_metadata=None,
        tactical_metadata=None,
    ):
        return load_hierarchical_compatible_state(
            self,
            state_dict,
            checkpoint_metadata=checkpoint_metadata,
            tactical_metadata=tactical_metadata,
        )

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1
        update_slow_option_critic(self)

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        if self.world_model_backend == "jepa":
            self.jepa_world_model.train(False)
        if self.tactical_enabled and hasattr(
            self, '_frozen_tactical_policy'
        ):
            self._frozen_tactical_policy.train(False)
        if self.hierarchical_enabled and hasattr(
            self, "_frozen_hierarchical_options"
        ):
            self._frozen_hierarchical_options.train(False)
            self._slow_option_critic.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        if self.world_model_backend == "rssm":
            self._frozen_encoder = copy.deepcopy(self.encoder)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

            self._frozen_rssm = copy.deepcopy(self.rssm)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)
        else:
            self._frozen_encoder = None
            self._frozen_rssm = None
            self._frozen_jepa_world_model = self.jepa_world_model
            self._frozen_jepa_world_model.train(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        if self.tactical_enabled:
            self._frozen_tactical_policy = copy.deepcopy(
                self.tactical_policy
            )
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.tactical_policy.named_parameters(),
                self._frozen_tactical_policy.named_parameters(),
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)
            self._frozen_tactical_policy.train(False)
        else:
            self._frozen_tactical_policy = None
        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        # P0.2 predicted-mask heads: frozen copies (shared storage -> track trained weights),
        # used to derive the imagination action mask without grad flowing into them.
        if self.action_masking:
            for _src, _name in ((self.avail_head, "_frozen_avail_head"),
                                (self.alive_head, "_frozen_alive_head")):
                _frozen = copy.deepcopy(_src)
                for (no, po), (nn_, pn) in zip(_src.named_parameters(), _frozen.named_parameters()):
                    assert no == nn_
                    pn.data = po.data
                    pn.requires_grad_(False)
                setattr(self, _name, _frozen)
        clone_and_freeze_hierarchy(self)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        if self.world_model_backend == "jepa":
            encoded = self._frozen_jepa_world_model.encode_obs(p_obs)
            stoch, deter = self._frozen_jepa_world_model.obs_step(
                prev_stoch, prev_deter, prev_action, encoded, obs["is_first"])
            feat = self._frozen_jepa_world_model.get_feat(stoch, deter)
        else:
            # (B, E)
            embed = self._frozen_encoder(p_obs)
            # (B, S, K), (B, D)
            stoch, deter, _ = self._frozen_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
        option_fields = {}
        if self.action_masking:
            # Real masking: invalid actions can never be requested; padded/dead -> NOOP. Uses
            # the RAW (un-preprocessed) avail + agent masks from the structured obs.
            from smacdreamer.masked_actions import MaskedMultiOneHotDist, build_action_mask
            raw_logits = self._frozen_actor.last(
                self._frozen_actor.mlp(feat)
            )
            if self.tactical_enabled:
                if eval:
                    (
                        raw_logits,
                        tactic,
                        tactic_confidence,
                        tactic_applied,
                    ) = self._frozen_tactical_policy.eval_combined_logits(
                        raw_logits, feat
                    )
                else:
                    tactic = self._frozen_tactical_policy.select_tactic(
                        feat, deterministic=False
                    )
                    raw_logits = (
                        self._frozen_tactical_policy.combine_logits(
                            raw_logits, feat, tactic
                        )
                    )
            if self.hierarchical_enabled:
                raw_logits, option_fields = hierarchical_act_logits(
                    self, feat, raw_logits, state, obs,
                    deterministic=eval,
                )
            agent_active = obs["agent_slot_mask"] * obs["agent_alive_mask"]
            amask, aactive = build_action_mask(
                obs["avail_actions"], agent_active, self._mask_A, self._mask_C)
            action_dist = MaskedMultiOneHotDist(
                raw_logits, amask, aactive, self._actor_shape, self._actor_unimix)
        else:
            action_dist = self._frozen_actor(feat)
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action, **option_fields},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        if self.world_model_backend == "jepa":
            stoch, deter = self.jepa_world_model.initial(B, device=self.device)
        else:
            stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        initial_state = TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))
        if self.hierarchical_enabled:
            for key, value in hierarchy_state_dict_fields(
                B, self.device
            ).items():
                initial_state[key] = value
        return initial_state

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, sample_info, initial = replay_buffer.sample()
        importance_weights = getattr(sample_info, 'importance_weights', None)
        self._priority_sequence_weights = (
            importance_weights.to(self.device) if importance_weights is not None else None
        )
        self._priority_sequence_priorities = None
        self._priority_map_feedback = None
        # UNIFIED_PRIORITY_V1
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        # CUDA autocast is used only for real low-precision AMP. float32 intentionally runs
        # without autocast and with GradScaler disabled.
        amp_enabled = self.device.type == "cuda" and self._amp_dtype in (torch.float16, torch.bfloat16)
        with autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=amp_enabled):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
            if self.hierarchical_enabled:
                hierarchy_loss, hierarchy_metrics = (
                    hierarchical_auxiliary_loss(
                        self, data, stoch, deter
                    )
                )
                self._scaler.scale(hierarchy_loss).backward()
                mets.update(hierarchy_metrics)
                mets["option/total_loss"] = hierarchy_loss.detach()
        if self.hierarchical_enabled:
            apply_hierarchy_gradient_guards(self)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_last_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        transition_indices = getattr(sample_info, 'transition_indices', sample_info)
        replay_buffer.update(transition_indices, stoch.detach(), deter.detach())
        if (
            hasattr(sample_info, 'sequence_uids')
            and self._priority_sequence_priorities is not None
            and hasattr(replay_buffer, 'update_priorities')
        ):
            replay_buffer.update_priorities(
                sample_info.sequence_uids,
                self._priority_sequence_priorities,
            )
        if self._priority_map_feedback is not None:
            controller = getattr(replay_buffer, 'priority_controller', None)
            if controller is not None:
                map_ids, errors, valid = self._priority_map_feedback
                controller.record_critic_feedback(
                    map_ids, errors, valid,
                    env_step=(replay_buffer.current_env_step()
                              if hasattr(replay_buffer, 'current_env_step') else None),
                )
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        if self.world_model_backend == "jepa":
            return self._cal_grad_jepa(data, initial)
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        if self.action_masking:
            # P0.2 auxiliary heads (BCE) on REAL data: predict this step's availability + alive
            # mask, so imagination can reconstruct them. Low loss weight (auxiliary, not reward).
            losses["avail"] = torch.mean(
                -self.avail_head(feat).log_prob(to_f32(data["avail_actions"])))
            losses["alive"] = torch.mean(
                -self.alive_head(feat).log_prob(to_f32(data["agent_alive_mask"])))
            from smacdreamer.masked_actions import mask_quality_metrics
            _avail_logits = self.avail_head.last(self.avail_head.mlp(feat)).detach()
            _mq = mask_quality_metrics(
                _avail_logits, to_f32(data["avail_actions"]), self._mask_threshold_logit)
            metrics["mask_precision"] = _mq["precision"]
            metrics["mask_recall"] = _mq["recall"]
            metrics["mask_fpr"] = _mq["fpr"]
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B, T, ...) -> (B*T, ...)
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        if self.action_masking:
            # P0.2: exclude reset/terminal states from imagination START states —
            # imag_start_mask = valid_time & ~is_first & ~is_last. Trajectories launched from
            # those states get zero actor-critic loss weight.
            start_valid = (
                (1.0 - to_f32(data["is_first"])) * (1.0 - to_f32(data["is_last"]))
            ).reshape(-1, 1, 1)   # (B*T, 1, 1)
            weight = weight * start_valid
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        if self.action_masking:
            # Masked policy over imagined states, using the SAME detached predicted mask the
            # imagination sampler used (frozen heads on imag_feat). log_prob/entropy normalise
            # over living agents and exclude invalid actions.
            from smacdreamer.masked_actions import (
                MaskedMultiOneHotDist, invalid_mass_and_greedy_rate, empty_mask_rate,
                hard_mask_from_logits)
            policy_logits = self.actor.last(self.actor.mlp(imag_feat))
            _amask, _aactive = self._predicted_action_mask(imag_feat)
            policy = MaskedMultiOneHotDist(
                policy_logits, _amask, _aactive, self._actor_shape, self._actor_unimix)
            # Pre-mask (unmasked greedy/mass), post-mask (must be 0), and empty-mask diagnostics.
            _mass, _grate = invalid_mass_and_greedy_rate(policy_logits.detach(), _amask, _aactive)
            metrics["imag_pre_mask_invalid_mass"] = _mass
            metrics["imag_pre_mask_invalid_sample_rate"] = _grate
            metrics["imag_invalid_rate"] = _grate   # back-compat alias (= pre-mask greedy rate)
            metrics["imag_post_mask_invalid_sample_rate"] = policy.post_mask_invalid_sample_rate()
            _pred_avail = hard_mask_from_logits(
                self._frozen_avail_head.last(self._frozen_avail_head.mlp(imag_feat)),
                self._mask_threshold_logit,
            ).reshape(*imag_feat.shape[:-1], self._mask_A, self._mask_C)
            metrics["imag_empty_mask_rate"] = empty_mask_rate(_pred_avail, _aactive)
        else:
            policy = self.actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        if self.action_masking:
            # Real-data (posterior) masking diagnostics + per-horizon mask quality.
            from smacdreamer.masked_actions import (
                MaskedMultiOneHotDist, build_action_mask, invalid_mass_and_greedy_rate)
            real_logits = self.actor.last(self.actor.mlp(feat)).detach()
            _ract = to_f32(data["agent_slot_mask"]) * to_f32(data["agent_alive_mask"])
            _rmask, _ractive = build_action_mask(
                to_f32(data["avail_actions"]), _ract, self._mask_A, self._mask_C)
            _rmass, _ = invalid_mass_and_greedy_rate(real_logits, _rmask, _ractive)
            metrics["real_pre_mask_invalid_mass"] = _rmass
            _rpolicy = MaskedMultiOneHotDist(
                real_logits, _rmask, _ractive, self._actor_shape, self._actor_unimix)
            metrics["real_post_mask_invalid_sample_rate"] = _rpolicy.post_mask_invalid_sample_rate()
            metrics.update(self._horizon_mask_diagnostics(data, post_stoch, post_deter))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    def _cal_grad_jepa(self, data, initial):
        losses = {}
        metrics = {}
        B, T = data.shape
        _seq_is = self._priority_sequence_weights
        if _seq_is is None:
            _seq_is = torch.ones(B, dtype=torch.float32, device=self.device)
        _seq_is = _seq_is.to(device=self.device, dtype=torch.float32).reshape(B)
        def _priority_weighted_mean(term):
            if term.shape[0] == B:
                per_sequence = term.reshape(B, -1).mean(-1)
            elif term.shape[0] == B * T:
                per_sequence = term.reshape(B, T, -1).mean((1, 2))
            else:
                raise RuntimeError(
                    f'cannot align PER weights: leading dim {term.shape[0]}, B={B}, T={T}'
                )
            return (per_sequence * _seq_is).sum() / _seq_is.sum().clamp_min(1e-8)
        encoded = self.jepa_world_model.encode_obs(data)
        post_stoch, post_deter = self.jepa_world_model.observe(
            encoded,
            data["action"],
            initial,
            data["is_first"],
        )
        feat = self.jepa_world_model.get_feat(post_stoch, post_deter)

        losses["rew"] = _priority_weighted_mean(
            -self.reward(feat).log_prob(to_f32(data["reward"]))
        )
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = _priority_weighted_mean(-self.cont(feat).log_prob(cont))
        if self.action_masking:
            losses["avail"] = _priority_weighted_mean(
                -self.avail_head(feat).log_prob(to_f32(data["avail_actions"]))
            )
            losses["alive"] = _priority_weighted_mean(
                -self.alive_head(feat).log_prob(to_f32(data["agent_alive_mask"]))
            )
            from smacdreamer.masked_actions import mask_quality_metrics
            _avail_logits = self.avail_head.last(self.avail_head.mlp(feat)).detach()
            _mq = mask_quality_metrics(
                _avail_logits, to_f32(data["avail_actions"]), self._mask_threshold_logit)
            metrics["mask_precision"] = _mq["precision"]
            metrics["mask_recall"] = _mq["recall"]
            metrics["mask_fpr"] = _mq["fpr"]

        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, post_deter.shape[-1]).detach(),
        )
        imag_out = self._imagine(start, self.imag_horizon + 1)
        if self.tactical_enabled:
            imag_feat, imag_action, imag_tactic = imag_out
            imag_tactic = imag_tactic.detach()
        else:
            imag_feat, imag_action = imag_out
            imag_tactic = None
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()
        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        if self.action_masking:
            start_valid = ((1.0 - to_f32(data["is_first"])) * (1.0 - to_f32(data["is_last"]))).reshape(-1, 1, 1)
            weight = weight * start_valid
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale
        if self.action_masking:
            from smacdreamer.masked_actions import (
                MaskedMultiOneHotDist, invalid_mass_and_greedy_rate, empty_mask_rate,
                hard_mask_from_logits)
            base_policy_logits = self.actor.last(
                self.actor.mlp(imag_feat)
            )
            if self.tactical_enabled:
                tactic_logits = self.tactical_policy.selector_logits(
                    imag_feat
                )
                tactic_dist = torch.distributions.Categorical(
                    logits=tactic_logits
                )
                policy_logits = self.tactical_policy.combine_logits(
                    base_policy_logits, imag_feat, imag_tactic
                )
            else:
                tactic_logits = None
                tactic_dist = None
                policy_logits = base_policy_logits
            _amask, _aactive = self._predicted_action_mask(imag_feat)
            policy = MaskedMultiOneHotDist(
                policy_logits, _amask, _aactive, self._actor_shape, self._actor_unimix)
            _mass, _grate = invalid_mass_and_greedy_rate(policy_logits.detach(), _amask, _aactive)
            metrics["imag_pre_mask_invalid_mass"] = _mass
            metrics["imag_pre_mask_invalid_sample_rate"] = _grate
            metrics["imag_invalid_rate"] = _grate
            metrics["imag_post_mask_invalid_sample_rate"] = policy.post_mask_invalid_sample_rate()
            _pred_avail = hard_mask_from_logits(
                self._frozen_avail_head.last(self._frozen_avail_head.mlp(imag_feat)),
                self._mask_threshold_logit,
            ).reshape(*imag_feat.shape[:-1], self._mask_A, self._mask_C)
            metrics["imag_empty_mask_rate"] = empty_mask_rate(_pred_avail, _aactive)
        else:
            policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        primitive_policy_loss = _priority_weighted_mean(
            weight[:, :-1].detach()
            * -(logpi * adv.detach() + self.act_entropy * entropy)
        )
        if self.tactical_enabled:
            tactic_logpi = tactic_dist.log_prob(imag_tactic)[
                :, :-1
            ].unsqueeze(-1)
            tactic_entropy = tactic_dist.entropy()[
                :, :-1
            ].unsqueeze(-1)
            tactical = self.tactical_policy.settings
            tactic_policy_loss = _priority_weighted_mean(
                weight[:, :-1].detach()
                * -(
                    tactical.tactic_pg_scale
                    * tactic_logpi
                    * adv.detach()
                    + tactical.tactic_entropy_scale * tactic_entropy
                )
            )
            start_is = _seq_is[:, None].expand(B, T).reshape(
                B * T, 1
            )
            tactic_aux_weight = (
                weight[:, :-1, 0].detach() * start_is
            )
            tactic_stats = self.tactical_policy.usage_statistics(
                tactic_logits[:, :-1],
                sampled_tactic=imag_tactic[:, :-1],
                state_weights=tactic_aux_weight,
            )
            collapse_loss = tactic_stats["collapse_loss"]
            effect_stats = self.tactical_policy.effect_statistics(
                imag_feat[:, :-1].detach(),
                base_policy_logits[:, :-1].detach(),
                _amask[:, :-1].detach(),
                _aactive[:, :-1].detach(),
                self._actor_shape,
                tactic_aux_weight,
            )
            effect_js = effect_stats["js_mean"]
            effect_loss = torch.relu(
                torch.as_tensor(
                    tactical.effect_target,
                    device=effect_js.device,
                    dtype=effect_js.dtype,
                )
                - effect_js
            )
            residual_ratio = (
                effect_stats["residual_rms"]
                / effect_stats["base_rms"].clamp_min(1e-6)
            )
            residual_guard_loss = torch.relu(
                residual_ratio
                - torch.as_tensor(
                    tactical.max_residual_to_base,
                    device=residual_ratio.device,
                    dtype=residual_ratio.dtype,
                )
            ).square()
            base_kl_loss = effect_stats["base_kl_loss"]
            losses["policy"] = (
                primitive_policy_loss
                + tactic_policy_loss
                + tactical.collapse_loss_scale * collapse_loss
                + tactical.effect_loss_scale * effect_loss
                + tactical.residual_guard_scale * residual_guard_loss
                + tactical.base_kl_scale * base_kl_loss
            )
            metrics["tactic/policy_loss"] = tactic_policy_loss
            metrics["tactic/entropy"] = tactic_entropy.mean()
            metrics["tactic/entropy_normalized"] = (
                tactic_entropy.mean()
                / math.log(self.tactical_policy.num_tactics)
            )
            metrics["tactic/conditional_entropy"] = tactic_stats[
                "conditional_entropy"
            ]
            metrics["tactic/marginal_entropy"] = tactic_stats[
                "marginal_entropy"
            ]
            metrics["tactic/mutual_information"] = tactic_stats[
                "mutual_information"
            ]
            metrics["tactic/mutual_information_normalized"] = tactic_stats[
                "mutual_information_normalized"
            ]
            metrics["tactic/selector_max_probability"] = tactic_stats[
                "selector_max_probability"
            ]
            metrics["tactic/selector_logit_std"] = tactic_stats[
                "selector_logit_std"
            ]
            metrics["tactic/collapse_loss"] = collapse_loss
            # Compatibility panel name; semantics are now collapse-only.
            metrics["tactic/balance_loss"] = collapse_loss
            metrics["tactic/effect_loss"] = effect_loss
            metrics["tactic/effect_js"] = effect_js
            metrics["tactic/effect_js_min"] = effect_stats["js_min"]
            metrics["tactic/effect_js_max"] = effect_stats["js_max"]
            metrics["tactic/residual_rms"] = effect_stats["residual_rms"]
            metrics["tactic/residual_to_base_ratio"] = residual_ratio
            metrics["tactic/residual_guard_loss"] = residual_guard_loss
            metrics["tactic/base_kl_loss"] = base_kl_loss
            metrics["tactic/base_kl_mean"] = effect_stats["base_kl_mean"]
            metrics["tactic/base_kl_max"] = effect_stats["base_kl_max"]
            metrics["tactic/action_flip_rate"] = effect_stats[
                "action_flip_rate"
            ]
            metrics["tactic/mi_shortfall"] = tactic_stats["mi_shortfall"]
            metrics["tactic/usage_max"] = tactic_stats["usage_max"]
            metrics["tactic/effective_count"] = tactic_stats[
                "effective_count"
            ]
            for tactic_index in range(self.tactical_policy.num_tactics):
                metrics[f"tactic/usage_{tactic_index}"] = tactic_stats[
                    "marginal"
                ][tactic_index]
                metrics[
                    f"tactic/sampled_usage_{tactic_index}"
                ] = tactic_stats["sampled_usage"][tactic_index]
                metrics[
                    f"tactic/argmax_usage_{tactic_index}"
                ] = tactic_stats["argmax_usage"][tactic_index]
                metrics[
                    f"tactic/residual_rms_{tactic_index}"
                ] = effect_stats[f"residual_rms_{tactic_index}"]
        else:
            losses["policy"] = primitive_policy_loss
        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = _priority_weighted_mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )

        last, term, reward = to_f32(data["is_last"]), to_f32(data["is_terminal"]), to_f32(data["reward"])
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        weight_replay = 1.0 - last
        ret_replay = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret_replay, 0 * ret_replay[:, -1:]], 1)
        value_dist = self.value(feat)
        losses["repval"] = _priority_weighted_mean(
            weight_replay[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        _priority_value = value_dist.mode()
        _priority_error = (ret_replay.detach() - _priority_value[:, :-1].detach()).abs()
        _priority_valid = weight_replay[:, :-1].detach()
        _priority_num = (_priority_error * _priority_valid).reshape(B, -1).sum(-1)
        _priority_den = _priority_valid.reshape(B, -1).sum(-1).clamp_min(1.0)
        self._priority_sequence_priorities = _priority_num / _priority_den
        if "log_map_id" in data:
            _map_ids = data["log_map_id"][:, :_priority_error.shape[1]].detach()
            _map_feedback_valid = (
                _priority_valid * _seq_is.reshape(B, 1, 1)
            )
            self._priority_map_feedback = (
                _map_ids, _priority_error, _map_feedback_valid
            )
            metrics["priority/map_feedback_is_weight_mean"] = _seq_is.mean()
        metrics["priority/critic_error_mean"] = _priority_error.mean()
        metrics["priority/sequence_priority_mean"] = self._priority_sequence_priorities.mean()
        metrics["priority/sequence_priority_max"] = self._priority_sequence_priorities.max()
        if self.action_masking:
            from smacdreamer.masked_actions import (
                MaskedMultiOneHotDist, build_action_mask, invalid_mass_and_greedy_rate)
            real_logits = self.actor.last(self.actor.mlp(feat)).detach()
            _ract = to_f32(data["agent_slot_mask"]) * to_f32(data["agent_alive_mask"])
            _rmask, _ractive = build_action_mask(to_f32(data["avail_actions"]), _ract, self._mask_A, self._mask_C)
            _rmass, _ = invalid_mass_and_greedy_rate(real_logits, _rmask, _ractive)
            metrics["real_pre_mask_invalid_mass"] = _rmass
            _rpolicy = MaskedMultiOneHotDist(real_logits, _rmask, _ractive, self._actor_shape, self._actor_unimix)
            metrics["real_post_mask_invalid_sample_rate"] = _rpolicy.post_mask_invalid_sample_rate()
            metrics.update(self._horizon_mask_diagnostics(data, post_stoch, post_deter))

        memory, entity_mask, _, _ = _jepa_unpack_state(
            post_deter.reshape(-1, post_deter.shape[-1]), self.jepa_world_model.state_spec)
        metrics["jepa/latent_norm"] = post_stoch.detach().float().norm(dim=-1).mean()
        metrics["jepa/latent_std"] = post_stoch.detach().float().std()
        metrics["jepa/memory_norm"] = memory.detach().float().norm(dim=-1).mean()
        metrics["jepa/presence_rate"] = entity_mask.detach().float().mean()
        metrics["jepa/predicted_entity_count"] = entity_mask.detach().float().sum(dim=-1).mean()
        metrics["jepa/feature_norm"] = feat.detach().float().norm(dim=-1).mean()
        metrics["jepa/frozen_parameter_count"] = torch.tensor(
            float(sum(p.numel() for p in self.jepa_world_model.parameters_frozen())), device=feat.device)
        metrics["jepa/adapter_total_parameter_count"] = torch.tensor(
            float(sum(
                p.numel()
                for p in self.jepa_world_model.feature_adapter.parameters()
            )),
            device=feat.device,
        )
        metrics["jepa/trainable_adapter_parameter_count"] = torch.tensor(
            float(sum(
                p.numel()
                for p in self.jepa_world_model.feature_adapter.parameters()
                if p.requires_grad
            )),
            device=feat.device,
        )
        metrics["ret"] = torch.mean((ret - ret_offset) / ret_scale)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))
        metrics.update(tools.tensorstats(ret_replay, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        if self.hierarchical_enabled:
            for legacy_key in ("policy", "value", "repval"):
                losses.pop(legacy_key, None)
            metrics["option/legacy_behavior_losses_disabled"] = (
                torch.ones((), device=feat.device)
            )
        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    def _predicted_action_mask(self, feat):
        """(amask, aactive) from the FROZEN avail/alive heads at ``feat``.

        A DETACHED hard mask (the >= threshold is non-differentiable), so it never leaks gradient
        into the heads. NOOP is guaranteed valid by build_action_mask. Used to mask imagination
        actions and the actor-critic policy in imagination (P0.2)."""
        from smacdreamer.masked_actions import hard_mask_from_logits, build_action_mask
        avail_logits = self._frozen_avail_head.last(self._frozen_avail_head.mlp(feat))
        alive_logits = self._frozen_alive_head.last(self._frozen_alive_head.mlp(feat))
        pred_avail = hard_mask_from_logits(avail_logits, self._mask_threshold_logit)
        pred_active = hard_mask_from_logits(alive_logits, self._mask_threshold_logit)
        return build_action_mask(pred_avail, pred_active, self._mask_A, self._mask_C)

    @torch.no_grad()
    def _horizon_mask_diagnostics(self, data, post_stoch, post_deter, max_h=5):
        """Mask precision/recall/FPR by regime: posterior (h0), then OPEN-LOOP prior states
        rolled with the REAL actions, vs the true avail at each future step. Isolates how the
        predicted mask degrades with imagination horizon (world-model drift) — posterior should
        be high; long-horizon priors may fall off even while posterior stays high."""
        from smacdreamer.masked_actions import mask_quality_metrics
        thr = self._mask_threshold_logit
        out = {}
        if self.world_model_backend == "jepa":
            feat0 = self.jepa_world_model.get_feat(post_stoch, post_deter)
        else:
            feat0 = self.rssm.get_feat(post_stoch, post_deter)
        m0 = mask_quality_metrics(
            self.avail_head.last(self.avail_head.mlp(feat0)), to_f32(data["avail_actions"]), thr)
        out["maskh0_posterior_precision"] = m0["precision"]
        out["maskh0_posterior_recall"] = m0["recall"]
        out["maskh0_posterior_fpr"] = m0["fpr"]
        T = data["avail_actions"].shape[1]
        stoch, deter = post_stoch[:, 0], post_deter[:, 0]
        for h in range(1, min(max_h, T)):
            if self.world_model_backend == "jepa":
                stoch, deter = self.jepa_world_model.img_step(stoch, deter, data["action"][:, h - 1])
                hfeat = self.jepa_world_model.get_feat(stoch, deter)
            else:
                stoch, deter = self.rssm.img_step(stoch, deter, data["action"][:, h - 1])
                hfeat = self.rssm.get_feat(stoch, deter)
            ah = self.avail_head.last(self.avail_head.mlp(hfeat))
            mh = mask_quality_metrics(ah, to_f32(data["avail_actions"][:, h]), thr)
            tag = "prior1" if h == 1 else "openloop"
            out[f"maskh{h}_{tag}_precision"] = mh["precision"]
            out[f"maskh{h}_{tag}_recall"] = mh["recall"]
            out[f"maskh{h}_{tag}_fpr"] = mh["fpr"]
        return out

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        tactics = []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            if self.world_model_backend == "jepa":
                feat = self._frozen_jepa_world_model.get_feat(stoch, deter)
            else:
                feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            if self.action_masking:
                from smacdreamer.masked_actions import MaskedMultiOneHotDist
                raw_logits = self._frozen_actor.last(
                    self._frozen_actor.mlp(feat)
                )
                if self.tactical_enabled:
                    tactic = self._frozen_tactical_policy.select_tactic(
                        feat
                    )
                    raw_logits = self._frozen_tactical_policy.combine_logits(
                        raw_logits, feat, tactic
                    )
                amask, aactive = self._predicted_action_mask(feat)
                action = MaskedMultiOneHotDist(
                    raw_logits, amask, aactive, self._actor_shape, self._actor_unimix).rsample()
            else:
                action = self._frozen_actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            if self.tactical_enabled:
                tactics.append(tactic)
            if self.world_model_backend == "jepa":
                stoch, deter = self._frozen_jepa_world_model.img_step(stoch, deter, action)
            else:
                stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A)
        if self.tactical_enabled:
            return (
                torch.stack(feats, dim=1),
                torch.stack(actions, dim=1),
                torch.stack(tactics, dim=1),
            )
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
