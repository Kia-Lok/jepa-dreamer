"""R2-Dreamer + SMAClite standalone debug training script.

Runs the full online training loop (collect -> world-model update -> actor-critic
update) without requiring Hydra.  Config is built programmatically so all values
are pre-resolved — no interpolation machinery needed.

Usage (from project root, smac-r2 conda env active):
    python scripts\\train_r2dreamer_smaclite_debug.py
    python scripts\\train_r2dreamer_smaclite_debug.py --scenario 2s3z --steps 1000
    python scripts\\train_r2dreamer_smaclite_debug.py --logdir logs\\r2dreamer\\run1 --env-num 2

Phase 1B acceptance criteria
-----------------------------
  * Script starts without import errors.
  * World-model losses (rew, con, dyn, rep) appear in logs after ~10 env steps.
  * A checkpoint is written to <logdir>/latest.pt on completion.
  * No crash for the full --steps run.
"""

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Path setup — must happen before any r2dreamer / smaclite import.
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (
    str(ROOT / "src"),
    str(ROOT / "external" / "r2dreamer"),
    str(ROOT / "external" / "smaclite"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from omegaconf import OmegaConf

import tools
from buffer import Buffer
from dreamer import Dreamer
from trainer import OnlineTrainer
from smacdreamer.r2dreamer_factory import make_smaclite_envs
from smacdreamer.wandb_logger import WandbLogger
from smacdreamer.checkpointing import PeriodicCheckpointer, attach_checkpointing

# Use high-precision matmul on CPU (matches train.py behaviour).
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_config(args):
    """Build a fully-resolved OmegaConf DictConfig for the debug run.

    All Hydra interpolations (${device}, ${model.units}, …) are replaced with
    concrete values so the config can be used without Hydra's resolver.

    Model is intentionally small (deter=256, units=128) for fast CPU iteration.
    Raise --deter / --units to match production sizes once the loop is verified.
    """
    device       = "cpu"
    batch_size   = args.batch_size
    batch_length = args.batch_length
    units        = args.units
    deter        = args.deter

    return OmegaConf.create({
        "device":     device,
        "seed":       0,
        "batch_size": batch_size,
        "batch_length": batch_length,

        # ------------------------------------------------------------------ #
        # Replay buffer                                                        #
        # ------------------------------------------------------------------ #
        "buffer": {
            "batch_size":      batch_size,
            "batch_length":    batch_length,
            "max_size":        20_000,
            "device":          device,
            "storage_device":  device,
        },

        # ------------------------------------------------------------------ #
        # Online trainer                                                       #
        # ------------------------------------------------------------------ #
        "trainer": {
            "steps":            args.steps,
            "pretrain":         0,
            "eval_every":       args.steps + 1,   # effectively disabled
            "eval_episode_num": 0,
            "batch_size":       batch_size,
            "batch_length":     batch_length,
            # train_ratio controls updates-per-env-step:
            #   updates_needed = (batch_size * batch_length) / train_ratio
            # 32 / 32 = 1.0  → one WM update per env step (fine for debug).
            "train_ratio":      32,
            "video_pred_log":   False,
            "params_hist_log":  False,
            "update_log_every": 50,
            "action_repeat":    1,
        },

        # ------------------------------------------------------------------ #
        # Model (Dreamer) — values match size12M except deter/units are       #
        # shrunk for CPU debug.  Set compile=False; torch.compile needs Triton #
        # which is unavailable on Windows CPU.                                 #
        # ------------------------------------------------------------------ #
        "model": {
            # Dreamer top-level hyperparameters
            "act_entropy":          3e-4,
            "kl_free":              1.0,
            "imag_horizon":         args.imag_horizon,
            "horizon":              333,
            "lamb":                 0.95,
            "compile":              False,
            "log_grads":            False,
            "device":               device,
            "rep_loss":             "r2dreamer",
            # Optimiser (LaProp + AGC)
            "lr":    4e-5,
            "agc":   0.3,
            "pmin":  1e-3,
            "eps":   1e-20,
            "beta1": 0.9,
            "beta2": 0.999,
            "warmup": 100,
            "slow_target_update":   1,
            "slow_target_fraction": 0.02,
            # Loss weights
            "loss_scales": {
                "barlow": 0.05, "infonce": 1.0, "recon": 1.0,
                "rew": 1.0, "con": 1.0, "dyn": 1.0, "rep": 0.1,
                "policy": 1.0, "value": 1.0, "repval": 0.3,
                "swav": 1.0, "temp": 1.0, "norm": 1.0,
                # P0.2 auxiliary predicted-mask heads (low weight; representation targets only).
                "avail": 0.1, "alive": 0.1,
            },
            "r2dreamer": {"lambd": 5e-4},

            # Model size (mini for CPU debug; size12M uses deter=2048, units=256)
            "deter":    deter,
            "hidden":   units,
            "discrete": 8,       # stochastic discrete categories
            "depth":    16,      # CNN depth (unused; no image obs)
            "units":    units,
            "act":      "SiLU",
            "norm":     True,

            # ---- RSSM ---------------------------------------------------- #
            "rssm": {
                "stoch":       16,
                "deter":       deter,
                "hidden":      units,
                "discrete":    8,
                "img_layers":  2,
                "obs_layers":  1,
                "dyn_layers":  1,
                "blocks":      8,
                "act":         "SiLU",
                "norm":        True,
                "unimix_ratio": 0.01,
                "initial":     "learned",
                "device":      device,
            },

            # ---- Encoder (MLP only; state + avail_actions) --------------- #
            # mlp_keys ".*" matches everything; MultiEncoder explicitly
            # excludes is_first/is_last/is_terminal/reward and log_* keys.
            "encoder": {
                "mlp_keys": ".*",
                "cnn_keys":  "$^",          # no image observations
                "mlp": {
                    "shape":          None,
                    "layers":         3,
                    "units":          units,
                    "act":            "SiLU",
                    "norm":           True,
                    "device":         device,
                    "outscale":       None,
                    "symlog_inputs":  True,
                    "name":           "mlp_encoder",
                },
                "cnn": {
                    "act":         "SiLU",
                    "norm":        True,
                    "kernel_size": 5,
                    "minres":      4,
                    "depth":       16,
                    "mults":       [2, 3, 4, 4],
                },
            },

            # ---- Decoder (unused with rep_loss=r2dreamer) ---------------- #
            "decoder": {
                "mlp_keys": "$^",
                "cnn_keys":  "$^",
                "mlp_dist": {"name": "symlog_mse"},
                "cnn_dist": {"name": "mse"},
                "mlp": {
                    "shape":         None,
                    "layers":        3,
                    "units":         units,
                    "act":           "SiLU",
                    "norm":          True,
                    "dist":          {"name": "identity"},
                    "device":        device,
                    "outscale":      1.0,
                    "symlog_inputs": False,
                    "name":          "mlp_decoder",
                },
                "cnn": {
                    "depth":       16,
                    "units":       units,
                    "bspace":      8,
                    "mults":       [2, 3, 4, 4],
                    "act":         "SiLU",
                    "norm":        True,
                    "kernel_size": 5,
                    "minres":      4,
                    "outscale":    1.0,
                },
            },

            # ---- Reward head --------------------------------------------- #
            "reward": {
                "shape":         [255],
                "layers":        1,
                "units":         units,
                "act":           "SiLU",
                "norm":          True,
                "dist":          {"name": "symexp_twohot", "bin_num": 255},
                "outscale":      0.0,
                "device":        device,
                "symlog_inputs": False,
                "name":          "reward",
            },

            # ---- Continuation head --------------------------------------- #
            "cont": {
                "shape":         [1],
                "layers":        1,
                "units":         units,
                "act":           "SiLU",
                "norm":          True,
                "dist":          {"name": "binary"},
                "outscale":      1.0,
                "device":        device,
                "symlog_inputs": False,
                "name":          "cont",
            },

            # ---- Actor head (shape + dist overwritten by Dreamer.__init__)  #
            "actor": {
                "shape":         None,
                "layers":        3,
                "units":         units,
                "act":           "SiLU",
                "norm":          True,
                "device":        device,
                "dist": {
                    "cont":       {"name": "bounded_normal", "min_std": 0.1, "max_std": 1.0},
                    "disc":       {"name": "onehot",      "unimix_ratio": 0.01},
                    "multi_disc": {"name": "multi_onehot", "unimix_ratio": 0.01},
                },
                "outscale":      0.01,
                "symlog_inputs": False,
                "name":          "actor",
            },

            # ---- Critic head --------------------------------------------- #
            "critic": {
                "shape":         [255],
                "layers":        3,
                "units":         units,
                "act":           "SiLU",
                "norm":          True,
                "device":        device,
                "dist":          {"name": "symexp_twohot", "bin_num": 255},
                "outscale":      0.0,
                "symlog_inputs": False,
                "name":          "value",
            },
        },
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="R2-Dreamer + SMAClite debug training (Phase 1B)"
    )
    parser.add_argument("--scenario",          default="2s3z",
                        help="SMAClite scenario (default: 2s3z)")
    parser.add_argument("--steps",             type=int, default=500,
                        help="Total training env steps (default: 500)")
    parser.add_argument("--logdir",            default="logs/r2dreamer/debug",
                        help="TensorBoard / checkpoint log directory")
    parser.add_argument("--env-num",           type=int, default=1,
                        help="Parallel training envs (default: 1)")
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="Max steps per SMAClite episode (default: 200)")
    parser.add_argument("--batch-size",        type=int, default=4,
                        help="Replay buffer batch size (default: 4)")
    parser.add_argument("--batch-length",      type=int, default=8,
                        help="Replay buffer sequence length (default: 8)")
    parser.add_argument("--units",             type=int, default=128,
                        help="MLP unit width (default: 128, size12M=256)")
    parser.add_argument("--deter",             type=int, default=256,
                        help="RSSM deterministic state size (default: 256, size12M=2048)")
    parser.add_argument("--imag-horizon",      type=int, default=3,
                        help="Imagination rollout length (default: 3, full=15)")
    parser.add_argument("--wandb-project",     default=None,
                        help="Weights & Biases project name; omit to disable wandb")
    parser.add_argument("--wandb-run",         default=None,
                        help="W&B run display name (auto-generated if omitted)")
    parser.add_argument("--checkpoint-every-minutes", type=float, default=10.0,
                        help="Wall-clock minutes between checkpoints (default: 10; 0 disables)")
    parser.add_argument("--keep-snapshots",    action="store_true",
                        help="Also keep immutable step_<N>.pt snapshots, not just latest.pt")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    config = make_config(args)
    device = config.device

    tools.set_seed_everywhere(config.seed)

    print(f"\n{'='*60}")
    print(f"R2-Dreamer + SMAClite  -  Phase 1B debug run")
    print(f"{'='*60}")
    print(f"  scenario     : {args.scenario}")
    print(f"  env_num      : {args.env_num}")
    print(f"  steps        : {args.steps}")
    print(f"  batch        : {args.batch_size} x {args.batch_length}")
    print(f"  model        : deter={args.deter}  units={args.units}  imag={args.imag_horizon}")
    print(f"  logdir       : {logdir}")
    print(f"{'='*60}\n")

    if args.wandb_project:
        logger = WandbLogger(
            logdir,
            project=args.wandb_project,
            run_name=args.wandb_run or f"{args.scenario}-{args.steps}steps",
            config=config,
        )
    else:
        logger = tools.Logger(logdir)
    replay_buffer = Buffer(config.buffer)

    # ---- Create environments ---------------------------------------------- #
    print("Creating environments...")
    train_envs, eval_envs, obs_space, act_space = make_smaclite_envs(
        scenario          = args.scenario,
        env_num           = args.env_num,
        eval_episode_num  = 0,
        device            = device,
        max_episode_steps = args.max_episode_steps,
        seed              = config.seed,
    )
    print(f"  obs_space keys : {sorted(obs_space.spaces)}")
    print(f"  act_space      : {act_space}  multi_discrete={hasattr(act_space, 'multi_discrete')}")

    # ---- Build agent ------------------------------------------------------ #
    print("\nBuilding Dreamer agent...")
    agent = Dreamer(config.model, obs_space, act_space).to(device)
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Parameters : {total_params:,}")

    # ---- Periodic checkpointing ------------------------------------------- #
    # Save every N wall-clock minutes by hooking agent.act (called each env step).
    # env step proxy: transitions in the buffer * action_repeat (matches trainer).
    checkpointer = None
    if args.checkpoint_every_minutes > 0:
        checkpointer = PeriodicCheckpointer(
            agent,
            logdir,
            interval_seconds = args.checkpoint_every_minutes * 60.0,
            step_fn          = lambda: replay_buffer.count() * config.trainer.action_repeat,
            keep_snapshots   = args.keep_snapshots,
        )
        attach_checkpointing(agent, checkpointer)
        print(f"  Checkpoints : every {args.checkpoint_every_minutes:g} min -> {logdir / 'latest.pt'}"
              + ("  (+ step_<N>.pt snapshots)" if args.keep_snapshots else ""))

    # ---- Train ------------------------------------------------------------ #
    print(f"\nStarting training ({args.steps} env steps)...\n")
    policy_trainer = OnlineTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs
    )
    policy_trainer.begin(agent)

    # ---- Final checkpoint ------------------------------------------------- #
    if checkpointer is not None:
        checkpointer.save(final=True)
    else:
        # Checkpointing disabled; still save a final weights-only file.
        torch.save({"agent_state_dict": agent.state_dict()}, logdir / "latest.pt")
        print(f"\nCheckpoint saved -> {logdir / 'latest.pt'}")

    if args.wandb_project:
        logger.finish()

    print("\nPhase 1B debug run complete.")


if __name__ == "__main__":
    main()
