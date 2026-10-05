import abc

import gtimer as gt
from rlkit.core import eval_util, logger
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.samplers.data_collector import (VecMdpPathCollector,
                                           VecMdpStepCollector)
import numpy as np


class VecOnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: VecMdpStepCollector,
            evaluation_data_collector: VecMdpPathCollector,
            replay_buffer: TorchReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_paths_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_paths_per_epoch = num_eval_paths_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        """Modified train function for bus environment like sac_v2_bus.py"""
        self.training_mode(False)
        
        # Initialize environment and get environment info
        env = self.expl_env.envs[0] if hasattr(self.expl_env, 'envs') else self.expl_env
        
        # Initial exploration if needed
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training // self.expl_env.env_num,
                discard_incomplete_paths=False,
                random=True,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        # Main training loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print(f"Epoch: {epoch}")
            
            # Reset environment and initialize state
            env.reset()
            state_dict, reward_dict, done = env.initialize_state(render=False)
            
            # Initialize action dictionaries
            action_dict = {key: None for key in range(env.max_agent_num)}
            episode_reward = 0
            episode_steps = 0
            training_steps = 0
            
            # Environment interaction loop (similar to sac_v2_bus.py)
            while not done:
                # Handle agents with single state (first time at station)
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            # Get action from policy
                            state_input = np.array(state_dict[key][0])
                            # Use policy to get action (deterministic=False for exploration)
                            action, _ = self.trainer.policy.get_action(state_input, deterministic=False)
                            action_dict[key] = action

                    # Handle agents with two states (transition between stations)
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:  # Station changed
                            # Store transition in replay buffer
                            old_state = np.array(state_dict[key][0])
                            new_state = np.array(state_dict[key][1])
                            action = action_dict[key]
                            reward = reward_dict[key]
                            
                            # Add to replay buffer
                            self.replay_buffer.add_sample(
                                observation=old_state,
                                action=action,
                                reward=reward,
                                next_observation=new_state,
                                terminal=done,
                                env_info={}
                            )
                            
                            episode_reward += reward
                            episode_steps += 1
                        
                        # Update state and get new action
                        state_dict[key] = state_dict[key][1:]
                        state_input = np.array(state_dict[key][0])
                        action, _ = self.trainer.policy.get_action(state_input, deterministic=False)
                        action_dict[key] = action

                # Step environment
                state_dict, reward_dict, done = env.step(action_dict, debug=False, render=False)

                # Training
                if (self.replay_buffer.num_steps_can_sample() > self.min_num_steps_before_training and 
                    episode_steps % 5 == 0):  # Train every 5 steps
                    
                    self.training_mode(True)
                    for _ in range(1):  # Train once per update
                        train_data = self.replay_buffer.random_batch(self.batch_size)
                        self.trainer.train(train_data)
                        training_steps += 1
                    self.training_mode(False)

            print(f"Epoch: {epoch}, Episode Reward: {episode_reward:.2f}, "
                  f"Episode Steps: {episode_steps}, Training Steps: {training_steps}")

            # Evaluation
            if epoch % 10 == 0:  # Evaluate every 10 epochs
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_paths_per_epoch,
                    discard_incomplete_paths=True,
                )
                gt.stamp('evaluation sampling')

            self._end_epoch(epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        """
        Evaluation
        """
        if hasattr(self.eval_data_collector, 'get_diagnostics'):
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            logger.record_dict(
                eval_util.get_generic_path_information(eval_paths),
                prefix="evaluation/",
            )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)