"""Weights & Biases logger that extends R2-Dreamer's TensorBoard Logger.

Subclasses tools.Logger so the trainer loop needs no changes.  Scalars are
captured from the parent's buffer before write() clears them, then forwarded
to wandb.  TensorBoard, console, and jsonl outputs are preserved unchanged.

Usage
-----
    from smacdreamer.wandb_logger import WandbLogger
    logger = WandbLogger(logdir, project="smac-r2", run_name="2s3z-debug", config=cfg)
    # use exactly like tools.Logger — trainer.begin(agent) etc. unchanged
    logger.finish()   # call at script end to flush wandb
"""

import pathlib
import os

import tools  # r2dreamer's tools module (must be on sys.path)


class WandbLogger(tools.Logger):
    """TensorBoard + Weights & Biases dual logger.

    Parameters
    ----------
    logdir      : pathlib.Path | str — passed to tools.Logger unchanged
    project     : wandb project name (required)
    run_name    : wandb run display name; None → wandb generates one
    config      : any dict or OmegaConf DictConfig to log as hyperparameters
    wandb_kwargs: any extra keyword args forwarded to wandb.init()
    """

    def __init__(self, logdir, project, run_name=None, config=None, **wandb_kwargs):
        super().__init__(logdir)

        import wandb
        self._wandb = wandb
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

        # Flatten OmegaConf config to a plain dict for wandb hparams panel.
        flat_config = None
        if config is not None:
            try:
                from omegaconf import OmegaConf
                flat_config = OmegaConf.to_container(config, resolve=True)
            except Exception:
                flat_config = dict(config)

        self._run = wandb.init(
            project=project,
            name=run_name,
            config=flat_config,
            dir=str(logdir),
            **wandb_kwargs,
        )

        # Do NOT pass an explicit step= to wandb.log(). The trainer's env step is
        # sparse and is reused across two interleaved call sites (episode writes
        # use write(step+i), train writes use write(step)); feeding those as
        # wandb's internal _step causes history points to be dropped while only
        # the summary survives — the "panel exists but no data on _step" symptom.
        #
        # Instead let wandb auto-increment its own monotonic _step on every log,
        # and record the real env step as a regular metric ("global_step") that
        # we declare as the default x-axis for all charts.
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

        # Surface where data is going so "no graphs" is diagnosable at a glance.
        mode = getattr(self._run.settings, "mode", "unknown")
        url = getattr(self._run, "url", None)
        print(f"  [wandb] mode={mode}")
        if mode == "offline":
            print("  [wandb] OFFLINE - metrics are NOT syncing to wandb.ai.")
            print("  [wandb] Run 'wandb login' (one time), then re-launch to see graphs online.")
            print(f"  [wandb] To upload this run later: wandb sync {self._run.dir}")
        elif url:
            print(f"  [wandb] View run at: {url}")

    def write(self, step, fps=False):
        # Snapshot scalars BEFORE parent clears self._scalars. (fps is computed
        # inside the parent's local scope and only reaches TensorBoard; not
        # forwarded here.)
        wandb_scalars = dict(self._scalars)

        # Forward to wandb FIRST, so logging can never be blocked by a failure in
        # the parent's console-print / TensorBoard path. No explicit step=
        # (wandb auto-increments _step); the env step rides along as the
        # "global_step" x-axis metric.
        if wandb_scalars:
            wandb_scalars["global_step"] = int(step)
            self._wandb.log(wandb_scalars)

        # Parent: TensorBoard + console print + metrics.jsonl (also clears buffers).
        super().write(step, fps=fps)

    def finish(self):
        """Flush and close the wandb run.  Call once at the end of training."""
        self._wandb.finish()
