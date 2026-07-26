import torch

import tools

# OPTION_CRITIC_HIERARCHY_V2


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.start_step = int(getattr(config, 'start_step', 0) or 0)
        self.current_step = self.start_step
        # UNIFIED_PRIORITY_V1
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        self.system_log_every = int(getattr(config, "system_log_every", 0) or 0)
        self._should_log_system = tools.Every(self.system_log_every) if self.system_log_every > 0 else None

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        if getattr(agent, 'hierarchical_enabled', False):
            agent.set_hierarchy_training_step(train_step)
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            envs._last_step = int(step)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.
        """
        envs = self.train_envs
        video_cache = []
        step = self.start_step + self.replay_buffer.count() * self._action_repeat
        self.current_step = int(step)
        if hasattr(self.replay_buffer, 'set_env_step'):
            self.replay_buffer.set_env_step(step)
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            if self._should_log_system is not None and self._should_log_system(step):
                try:
                    from smacdreamer.system_metrics import log_system_metrics

                    log_system_metrics(
                        self.logger,
                        step,
                        worker_infos=envs.worker_infos() if hasattr(envs, "worker_infos") else (),
                        replay_count=self.replay_buffer.count(),
                        replay_backend=getattr(self.replay_buffer, "storage_backend", None),
                        completed_episodes=getattr(envs, "completed_episodes", None),
                        worker_restarts=getattr(envs, "worker_restarts", None),
                    )
                    self.logger.write(step)
                except Exception as exc:
                    print(f"[telemetry] skipped system metrics at step {step}: {exc}", flush=True)
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            self.current_step = int(step)
            if hasattr(self.replay_buffer, 'set_env_step'):
                self.replay_buffer.set_env_step(step)
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            if getattr(agent, 'hierarchical_enabled', False):
                agent.set_hierarchy_training_step(step)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            if getattr(agent, "hierarchical_enabled", False):
                for option_key in (
                    "option_id", "option_age", "option_has",
                    "option_before_id", "option_before_age",
                    "option_before_has", "option_action_age",
                    "option_started", "option_terminated",
                    "option_termination_eligible",
                    "option_termination_prob",
                ):
                    trans[option_key] = agent_state[option_key]
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            if hasattr(self.replay_buffer, 'record_collection'):
                self.replay_buffer.record_collection(trans, env_step=step)
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            # Resume uses a new replay, so warm-up must depend on replay contents,
            # never on the restored absolute environment step.
            if self.replay_buffer.count() // envs.env_num > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)

                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics

                update_count += update_num
            # Log training metrics
            if self._should_log(step):
                if hasattr(self.replay_buffer, "metrics"):
                    train_metrics.update(self.replay_buffer.metrics())

                for name, value in train_metrics.items():
                    value = (
                        tools.to_np(value)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    self.logger.scalar(f"train/{name}", value)

                self.logger.scalar("train/opt/updates", update_count)

                if self.video_pred_log:
                    data, _, initial = self.replay_buffer.sample()
                    self.logger.video(
                        "open_loop",
                        tools.to_np(agent.video_pred(data, initial)),
                    )

                if self.params_hist_log:
                    for name, param in agent._named_params.items():
                        self.logger.histogram(name, tools.to_np(param))

                self.logger.write(step, fps=True)
