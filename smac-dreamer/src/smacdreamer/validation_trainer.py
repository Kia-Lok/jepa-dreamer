"""ValidationTrainer (P0.4): replaces the old worker-based periodic evaluator.

On each validation tick it runs an explicit ``map × fixed_seed`` validation pass over the
held-out VALIDATION maps (ORIGINAL reward), logs per-map + macro/micro metrics, and saves
``best_val_macro_winrate.pt`` selected by MACRO validation win rate, tie-broken by MACRO
original return — never by shaped return.

NOTE: imports ``OnlineTrainer`` from R2-Dreamer, so it requires ``external/r2dreamer`` on
``sys.path`` (the training script sets this up before importing this module). The pure
selection rule lives in ``smacdreamer.evaluation.is_validation_improvement`` so it can be
unit-tested without R2-Dreamer.
"""

import math
import pathlib

import torch

from trainer import OnlineTrainer  # requires external/r2dreamer on sys.path

from smacdreamer.evaluation import evaluate_heldout, is_validation_improvement

# OPTION_CRITIC_HIERARCHY_V2

# TACTICAL_MIXTURE_HARDENING_V1_1

# Non-None sentinel: the base training loop only calls self.eval() when eval_envs is not None
# (and eval_episode_num > 0). We pass this up and override eval(); eval_envs is never used.
_VALIDATION_SENTINEL = object()


class ValidationEvery:
    """Local validation cadence with explicit step-zero behavior and resume safety."""

    def __init__(self, every: int, *, run_at_start: bool, initial_step: int = 0):
        self.every = int(every)
        self.run_at_start = bool(run_at_start)
        self._next = None
        self._initial_step = int(initial_step)

    def __call__(self, step):
        if self.every <= 0:
            return 0
        step = int(step)
        if self._next is None:
            if self.run_at_start and self._initial_step == 0:
                self._next = 0
            else:
                self._next = self.every
                while self._next <= self._initial_step:
                    self._next += self.every
        if step < self._next:
            return 0
        count = 0
        while step >= self._next:
            count += 1
            self._next += self.every
        return count

    def state_dict(self):
        return {
            "every": self.every,
            "run_at_start": self.run_at_start,
            "next": self._next,
            "initial_step": self._initial_step,
        }

    def load_state_dict(self, state):
        self.every = int(state["every"])
        self.run_at_start = bool(state["run_at_start"])
        self._next = state["next"]
        self._initial_step = int(state["initial_step"])


class ValidationTrainer(OnlineTrainer):
    """OnlineTrainer whose periodic eval is an explicit map×seed validation + best-ckpt save."""

    def __init__(self, config, replay_buffer, logger, logdir, train_envs, *,
                 validation_entries, pad_dims, seeds, device, gamma, max_episode_steps, obs_mode,
                 run_at_start=False, shutdown_timeout_seconds=5.0):
        super().__init__(config, replay_buffer, logger, logdir, train_envs, _VALIDATION_SENTINEL)
        self._val_entries = list(validation_entries)
        self._val_pad = pad_dims
        self._val_seeds = [int(s) for s in seeds]
        self._val_device = str(device)
        self._val_gamma = float(gamma)
        self._val_max_steps = int(max_episode_steps)
        self._val_obs_mode = str(obs_mode)
        self._val_shutdown_timeout = float(shutdown_timeout_seconds)
        self._logdir = pathlib.Path(logdir)
        self._best_macro_wr = -1.0
        self._best_macro_ret = -math.inf
        initial_step = int(replay_buffer.count()) * int(getattr(config, "action_repeat", 1))
        self._should_eval = ValidationEvery(
            int(getattr(config, "eval_every", 0)),
            run_at_start=bool(run_at_start),
            initial_step=initial_step,
        )

    def eval(self, agent, train_step):
        from smacdreamer.system_metrics import log_system_metrics, rss_bytes

        if getattr(agent, 'hierarchical_enabled', False):
            agent.set_hierarchy_training_step(train_step)
        agent.eval()
        before_rss = rss_bytes()
        log_system_metrics(
            self.logger,
            train_step,
            worker_infos=self.train_envs.worker_infos() if hasattr(self.train_envs, "worker_infos") else (),
            replay_count=self.replay_buffer.count(),
            replay_backend=getattr(self.replay_buffer, "storage_backend", None),
            completed_episodes=getattr(self.train_envs, "completed_episodes", None),
            worker_restarts=getattr(self.train_envs, "worker_restarts", None),
        )
        self.logger.write(train_step)
        report = evaluate_heldout(
            agent, self._val_entries, self._val_pad,
            seeds=self._val_seeds, device=self._val_device, gamma=self._val_gamma,
            max_episode_steps=self._val_max_steps, obs_mode=self._val_obs_mode,
            include_jepa_obs=(getattr(agent, "world_model_backend", "rssm") == "jepa"),
            jepa_visibility_config=getattr(getattr(agent, "jepa_world_model", None), "visibility_config", None),
            shutdown_timeout_seconds=self._val_shutdown_timeout,
            progress=False,
        )
        after_rss = rss_bytes()
        if before_rss is not None and after_rss is not None:
            self.logger.scalar("val/rss_before_gb", before_rss / (1024.0 ** 3))
            self.logger.scalar("val/rss_after_gb", after_rss / (1024.0 ** 3))
            self.logger.scalar("val/rss_delta_gb", (after_rss - before_rss) / (1024.0 ** 3))
        macro, micro = report["macro"], report["micro"]
        for k in ("win_rate", "original_return", "length", "timeout_rate",
                  "final_ally_ehp_frac", "final_enemy_ehp_frac"):
            self.logger.scalar(f"val/macro_{k}", float(macro[k]))
            self.logger.scalar(f"val/micro_{k}", float(micro[k]))
        self.logger.scalar("val/n_maps", float(report["n_maps"]))
        self.logger.scalar("val/n_episodes", float(report["n_episodes_total"]))

        wr, ret = float(macro["win_rate"]), float(macro["original_return"])
        if is_validation_improvement(wr, ret, self._best_macro_wr, self._best_macro_ret):
            self._best_macro_wr, self._best_macro_ret = wr, ret
            best_payload = {
                "agent_state_dict": agent.state_dict(),
                "val_macro_win_rate": wr,
                "val_macro_original_return": ret,
                "step": int(train_step),
                "obs_mode": self._val_obs_mode,
            }
            if hasattr(agent, "tactical_metadata"):
                best_payload["tactical_mixture_metadata"] = (
                    agent.tactical_metadata()
                )
            if hasattr(agent, "hierarchical_metadata"):
                best_payload["hierarchical_options_metadata"] = (
                    agent.hierarchical_metadata()
                )
            torch.save(
                best_payload,
                self._logdir / "best_val_macro_winrate.pt",
            )
            print(f"  [val step {train_step}] NEW BEST macro win_rate={wr:.3f} "
                  f"(orig_return={ret:.3f}) -> best_val_macro_winrate.pt")
        else:
            print(f"  [val step {train_step}] macro win_rate={wr:.3f} "
                  f"orig_return={ret:.3f} (best {self._best_macro_wr:.3f})")
        self.logger.write(train_step)
        agent.train()

    def state_dict(self):
        return {
            "best_macro_wr": self._best_macro_wr,
            "best_macro_ret": self._best_macro_ret,
            "validation_scheduler": self._should_eval.state_dict(),
        }

    def load_state_dict(self, state):
        self._best_macro_wr = float(state.get("best_macro_wr", self._best_macro_wr))
        self._best_macro_ret = float(state.get("best_macro_ret", self._best_macro_ret))
        if "validation_scheduler" in state:
            self._should_eval.load_state_dict(state["validation_scheduler"])


__all__ = ["ValidationTrainer"]
