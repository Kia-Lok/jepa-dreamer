#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--source-run-meta", type=Path)
parser.add_argument(
    "--expected-checkpoint-sha256",
    default=None,
)
args = parser.parse_args()
repo = args.repo.resolve()
config_path = (repo / args.config).resolve()
checkpoint_path = args.checkpoint.resolve()
source_meta_path = args.source_run_meta.resolve() if args.source_run_meta else None

for path, label in (
    (repo, "repo"),
    (config_path, "config"),
    (checkpoint_path, "checkpoint"),
):
    if not path.exists():
        fail(f"{label} missing: {path}")
if source_meta_path is not None:
    if not source_meta_path.exists():
        fail(f"source run metadata missing: {source_meta_path}")
    if checkpoint_path.parent != source_meta_path.parent:
        fail("checkpoint and source run metadata are not from the same run directory")
    try:
        json.loads(source_meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"source run metadata is not valid JSON: {exc}")

config = OmegaConf.load(config_path)
tactical = OmegaConf.load(repo / "configs/r2_2100_jepa_tactical_mixture_v1_2.yaml")
h = config.hierarchical_options
checks = {
    "enabled": bool(h.enabled),
    "eight_options": int(h.num_options) == 8,
    "two_frozen_source_groups": int(h.source_manager_group_count) == 2,
    "duration_1_4": int(h.min_duration) == 1 and int(h.max_duration) == 4,
    "fixed_horizon_15": (
        int(config.imag_horizon) == 15
        and int(h.imag_horizon_initial_max) == 15
        and int(h.imag_horizon_final_max) == 15
        and int(h.imag_horizon_window) == 1
    ),
    "learned_termination_disabled": (
        float(h.termination_loss_scale) == 0.0
        and float(h.termination_entropy_scale) == 0.0
        and float(h.termination_collapse_scale) == 0.0
        and int(h.termination_warmup_steps) >= 800_000
    ),
    "world_model_exactly_frozen": (
        float(h.world_model_grad_scale_initial) == 0.0
        and float(h.world_model_grad_scale_final) == 0.0
    ),
    "worker_before_manager": (
        int(h.worker_pg_warmup_steps) == 20_000
        and int(h.worker_pg_full_steps) == 150_000
        and int(h.manager_pg_warmup_steps) == 100_000
        and int(h.manager_pg_full_steps) == 300_000
    ),
    "fixed_anchor_floor": abs(float(h.slot_anchor_floor) - 0.40) < 1.0e-12,
    "critic_noise_guard": abs(float(h.option_critic_consistency_scale) - 1.0) < 1.0e-12,
    "all_slots_available": (
        int(h.slot_pair_unlock_initial_steps) == 0
        and int(h.slot_pair_unlock_interval_steps) == 1
        and int(h.slot_unlock_ramp_steps) == 1
    ),
    "bounded_child_delta": 0.0 < float(h.slot_delta_scale_max) <= 0.10,
    "no_task_independent_diversity_pressure": (
        float(h.manager_collapse_scale) == 0.0
        and float(h.manager_mi_scale) == 0.0
        and float(h.action_diversity_scale) == 0.0
        and float(h.residual_cosine_scale) == 0.0
    ),
    "validation_start_and_200k": (
        bool(config.validation.run_at_start)
        and int(config.validation.every) == 200_000
    ),
    "fresh_run_local_replay": str(config.buffer.scratch_dir) == "replay_v9_anchor_safe_h15",
    "tactical_module_disabled": not bool(config.tactical_mixture.enabled),
}

# The v9 config is derived from the successful Tactical-v1.2 training regime.
# Compare the complete resolved configuration and allow only the small, explicit
# set of paths that define this controlled hierarchy phase. This is stronger
# than checking a handful of guessed sampler keys because priority/map settings
# may be nested differently across local revisions.
def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            out.update(flatten(child, path))
    else:
        out[prefix] = value
    return out

config_flat = flatten(resolved(config))
tactical_flat = flatten(resolved(tactical))
all_paths = set(config_flat) | set(tactical_flat)
allowed_change_prefixes = (
    "hierarchical_options",
    "tactical_mixture.enabled",
    "buffer.scratch_dir",
    "validation.run_at_start",
    "validation.every",
    "imag_horizon",
    "compile",
    "model.compile",
    "wandb.run_name",
)
unexpected_changes = sorted(
    path for path in all_paths
    if config_flat.get(path, object()) != tactical_flat.get(path, object())
    and not any(path == prefix or path.startswith(prefix + ".") for prefix in allowed_change_prefixes)
)
checks["complete_tactical_training_regime_preserved"] = not unexpected_changes
if unexpected_changes:
    fail(f"unexpected Tactical-v1.2 config changes: {unexpected_changes}")

for name, passed in checks.items():
    if not passed:
        fail(f"config invariant failed: {name}")

actual_checkpoint_sha256 = sha256(checkpoint_path)
if args.expected_checkpoint_sha256 and actual_checkpoint_sha256 != args.expected_checkpoint_sha256:
    fail(
        "source checkpoint SHA-256 mismatch: "
        f"expected={args.expected_checkpoint_sha256}; actual={actual_checkpoint_sha256}"
    )
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metadata = checkpoint.get("tactical_mixture_metadata") or {}
state = checkpoint.get("agent_state_dict")
if not isinstance(state, dict) or not state:
    fail("source checkpoint lacks a non-empty agent_state_dict")
if metadata.get("architecture") != "tactical_mixture_v1_2":
    fail(f"source architecture is {metadata.get('architecture')!r}, not Tactical Mixture v1.2")
if int(metadata.get("num_tactics", -1)) != 2:
    fail("source checkpoint does not contain exactly two Tactical-v1.2 modes")
if any(key.startswith("hierarchical_options.") for key in state):
    fail("source checkpoint already contains Option-Critic parameters")
def method_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1 or matches[0].end_lineno is None:
        fail(f"expected exactly one method/function named {name}, found {len(matches)}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


installed = {
    "options": repo / "external/r2dreamer/hierarchical_options.py",
    "hierarchy": repo / "external/r2dreamer/hierarchical_dreamer.py",
    "critic": repo / "external/r2dreamer/option_critic.py",
    "dreamer": repo / "external/r2dreamer/dreamer.py",
    "trainer": repo / "external/r2dreamer/trainer.py",
    "tools": repo / "external/r2dreamer/tools.py",
    "runner": repo / "scripts/train_r2dreamer_smaclite_multimap.py",
    "validation": repo / "src/smacdreamer/validation_trainer.py",
    "rl_launcher": repo / "scripts/run_option_critic_v9_anchor_safe_800k.sh",
    "forecast_launcher": repo / "scripts/run_exp45_full_train_eval_resilient.sh",
    "master_launcher": repo / "scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh",
}
for label, path in installed.items():
    if not path.is_file():
        fail(f"installed {label} source missing: {path}")

source = {label: path.read_text(encoding="utf-8") for label, path in installed.items()}
for label in ("options", "hierarchy", "critic", "dreamer", "trainer", "tools", "runner", "validation"):
    try:
        ast.parse(source[label], filename=str(installed[label]))
    except SyntaxError as exc:
        fail(f"installed {label} source does not parse: {exc}")

required_option_tokens = (
    'ARCHITECTURE = "dreamer_option_critic_v9_anchor_safe_8slot"',
    "def interruption_mask(",
    "def switch_value_for_source_group(",
    "nn.init.orthogonal_(self.manager_slot[0].weight",
    "nn.init.orthogonal_(self.slot_delta[0].weight",
    "residual_mse.clamp_min(1.0e-12).sqrt()",
    "return torch.ones(",
    "slot_anchor_floor",
    "anchor_floor * anchor",
)
required_integration_tokens = (
    "def interruptible_option_bootstrap(",
    "manager_loss = slot_manager_loss",
    "worker_weight = weight * trainable_child",
    "target.manager_slot[2].bias.zero_()",
    "two_frozen_source_anchors_plus_six_anchor_floor_interruptible_children",
    "if s.action_diversity_scale:",
    "switch_value_for_source_group(",
    "within_group_option_consistency_loss(",
    "critic_consistency_blend",
    "if worker_pg_blend:",
    "Attach worker safety losses only when the worker PG",
    "target.data = source.data",
    "agent._source_hierarchical_options = copy.deepcopy",
)
for token in required_option_tokens:
    if token not in source["options"]:
        fail(f"missing option-controller contract: {token}")
for token in required_integration_tokens:
    if token not in source["hierarchy"]:
        fail(f"missing hierarchy integration contract: {token}")
for forbidden in (
    "target.manager_slot[0].weight.zero_()",
    "target.slot_delta[0].weight.zero_()",
    "manager_v = manager_value(manager_probs.detach()",
    "math.log(0.94)",
):
    if forbidden in source["hierarchy"]:
        fail(f"forbidden stale implementation remains: {forbidden}")

# Verify the already-integrated Dreamer hot path, not merely the new helper
# modules. This catches a repository that contains the v9 files but no longer
# calls them in acting, learning, freezing, checkpoint migration, or optimizer
# guarding.
for method_name, tokens in {
    "__init__": ("build_hierarchical_modules(self, config)",),
    "act": ("hierarchical_act_logits(",),
    "clone_and_freeze": ("clone_and_freeze_hierarchy(self)",),
    "_update_slow_target": ("update_slow_option_critic(self)",),
}.items():
    segment = method_source(source["dreamer"], method_name)
    for token in tokens:
        if token not in segment:
            fail(f"Dreamer.{method_name} missing integration token: {token}")

step_method = method_source(source["dreamer"], "set_hierarchy_training_step")
for token in (
    "self.hierarchical_options.set_training_step(step)",
    "self._frozen_hierarchical_options.set_training_step(step)",
):
    if token not in step_method:
        fail(f"Dreamer.set_hierarchy_training_step missing token: {token}")

update = method_source(source["dreamer"], "update")
positions = {
    "auxiliary": update.find("hierarchical_auxiliary_loss("),
    "backward": update.find("self._scaler.scale(hierarchy_loss).backward()"),
    "guard": update.find("apply_hierarchy_gradient_guards(self)"),
    "unscale": update.find("self._scaler.unscale_("),
    "step": update.find("self._scaler.step("),
}
if any(value < 0 for value in positions.values()):
    fail(f"Dreamer.update hierarchy/optimizer integration incomplete: {positions}")
if not (
    positions["auxiliary"] < positions["backward"] < positions["guard"]
    < positions["unscale"] < positions["step"]
):
    fail(f"Dreamer.update hierarchy/optimizer ordering is unsafe: {positions}")

jepa_grad = method_source(source["dreamer"], "_cal_grad_jepa")
legacy_disable = jepa_grad.find("losses.pop(legacy_key, None)")
total_loss = jepa_grad.find("total_loss = sum([v * self._loss_scales[k]")
if legacy_disable < 0 or total_loss < 0 or legacy_disable > total_loss:
    fail("legacy Dreamer behavior losses are not disabled before total-loss construction")

trainer_begin = method_source(source["trainer"], "begin")
for token in (
    'trans["action"] = act * ~done.unsqueeze(-1)',
    'getattr(agent, "hierarchical_enabled", False)',
    'trans[option_key] = agent_state[option_key]',
    "agent.set_hierarchy_training_step(step)",
):
    if token not in trainer_begin:
        fail(f"Trainer.begin missing hierarchy collection contract: {token}")

if "torch.bfloat16" not in source["tools"] or ".float()" not in method_source(source["tools"], "to_np"):
    fail("tools.to_np is missing the BF16-to-FP32 NumPy logging guard")

for token in (
    "load_hierarchical_compatible_state_dict(",
    "hierarchical_options_metadata",
    "config.model.hierarchical_options",
    "resume-start-step",
):
    if token not in source["runner"]:
        fail(f"multimap runner missing hierarchy checkpoint/config contract: {token}")
for token in (
    "agent.set_hierarchy_training_step(train_step)",
    "hierarchical_options_metadata",
):
    if token not in source["validation"]:
        fail(f"validation trainer missing hierarchy contract: {token}")

for token in (
    "FINAL_STEP=\"${FINAL_STEP:-800000}\"",
    'SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT',
    "imag_horizon=15",
    "learned_termination=disabled",
    "EXPECTED_SOURCE_CHECKPOINT_SHA256",
    "slot_anchor_floor=0.40",
    "--resume-start-step 0",
    '--steps "$FINAL_STEP"',
):
    if token not in source["rl_launcher"]:
        fail(f"v9 RL launcher missing contract: {token}")
for token in (
    "run_exp45_full_train_eval_resilient.sh",
    "run_option_critic_v9_anchor_safe_800k.sh",
    "CONTINUE_ON_FAILURE",
    "TACTICAL_V12_CHECKPOINT",
):
    if token not in source["master_launcher"]:
        fail(f"forecast-first master launcher missing contract: {token}")
for token in (
    "ordinary_eval",
    "hidden_eval",
    "STRICT_EXIT",
    "source_verify",
):
    if token not in source["forecast_launcher"]:
        fail(f"resilient forecast launcher missing contract: {token}")

print("[OK] Option-Critic v9 anchor-safe source/config audit passed")
print(
    json.dumps(
        {
            "repo": str(repo),
            "config": str(config_path),
            "source_run_meta": str(source_meta_path) if source_meta_path else None,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": actual_checkpoint_sha256,
                "step": checkpoint.get("step"),
                "val_macro_win_rate": checkpoint.get("val_macro_win_rate"),
                "val_macro_original_return": checkpoint.get("val_macro_original_return"),
            },
            "checks": checks,
        },
        indent=2,
        sort_keys=True,
    )
)
