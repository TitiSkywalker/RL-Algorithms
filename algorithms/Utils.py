"""
This file implements wrappers for environments.
- EnvSingle wraps a single Gymnasium or ALE environment.
- EnvBatch is designed for PPO, it manages many independent environments.
"""

import torch
import torchvision.transforms as T
import gymnasium as gym
import ale_py
import random
import matplotlib.pyplot as plt

from collections import deque

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs


#####################################################################
#                           Environment                             #
#####################################################################
class EnvSingle:
    def __init__(
        self, env_name, vision=False, num_frames=4, skip_frames=3, is_ale=False
    ) -> None:
        self.vision = vision
        self.num_frames = num_frames

        self.lives = 0
        self.steps = 0
        self.max_steps = 1000  # maximum rollout steps
        self.nops = 0
        self.max_nops = 128  # terminate after too many nops
        self.skip_frames = skip_frames  # skip some frames

        # image preprocessing
        self.transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),  # convert to grayscale (1 channel)
                T.Resize((84, 84)),  # rescale to 84x84
            ]
        )

        if is_ale:
            self.env = gym.make("ALE/" + env_name, render_mode="rgb_array_list")
        else:
            self.env = gym.make(env_name, render_mode="rgb_array_list")

        if self.vision:
            self.frame_stack = []
        else:
            self.frame_stack = None

    def reset(self):
        self.steps = 0
        self.nops = 0
        init_state, info = self.env.reset()
        init_state = torch.Tensor(init_state)

        if "lives" in info.keys():
            self.lives = info["lives"]

        if self.vision:
            self.frame_stack = []
            # reshape into (channel, width, height)
            init_state = init_state.permute(2, 0, 1)
            init_state = self.transform(init_state)

            # pad empty frames
            for _ in range(self.num_frames):
                self.frame_stack.append(torch.Tensor(init_state))

            # merge into channel dimension
            return torch.cat(self.frame_stack, dim=0)
        else:
            # simple observation
            return init_state

    def step(self, action) -> tuple[torch.Tensor, float, bool, bool, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = torch.Tensor(next_state)

        # in case the game won't stop
        self.steps += 1
        if self.steps > self.max_steps:
            truncated = True

        # for Atari, there are more considerations
        if self.vision:
            # reshape reward
            if reward > 0:
                reward = 10
            else:
                reward = 0

            # no-op reset
            if action == 0:
                self.nops += self.skip_frames
                if self.nops > self.max_nops:
                    truncated = True
            else:
                self.nops = 0

            # a life is lost
            if "lives" in info.keys() and info["lives"] < self.lives:
                if info["lives"] < 1:
                    # game over
                    reward -= 10
                    terminated = True
                else:
                    reward -= 10
                    self.lives = info["lives"]

            # skip frames and cumulate rewards
            for _ in range(self.skip_frames):
                if terminated or truncated:
                    break

                (
                    extra_state,
                    extra_reward,
                    extra_terminated,
                    extra_truncated,
                    extra_info,
                ) = self.env.step(action)

                if extra_reward > 0:
                    reward += 10

                terminated = terminated or extra_terminated
                truncated = truncated or extra_truncated

                if "lives" in extra_info.keys() and extra_info["lives"] < self.lives:
                    if extra_info["lives"] < 1:
                        reward -= 10
                        terminated = True
                    else:
                        reward -= 10
                        self.lives = extra_info["lives"]

                next_state = torch.Tensor(extra_state)

            next_state = next_state.permute(2, 0, 1)
            next_state = self.transform(next_state)

            # skipped frames are discarded
            self.frame_stack.pop(0)
            self.frame_stack.append(next_state)

            # concatenate in channel dimension
            return (
                torch.cat(self.frame_stack, dim=0),
                reward,
                terminated,
                truncated,
                info,
            )
        else:
            return next_state, reward, terminated, truncated, info

    def render(self):
        return self.env.render(), self.env.metadata["render_fps"]

    def action_size(self):
        return self.env.action_space.n

    def status_size(self):
        if self.vision:
            # grayscale image with size 84x84
            return (self.num_frames, 84, 84)
        else:
            return self.env.observation_space._shape[0]


# a batch of environments that can be used for parallel rollout
class EnvBatch:
    def __init__(
        self,
        env_name,
        num_envs,
        vision=False,
        num_frames=4,
        skip_frames=3,
        is_ale=False,
    ):
        self.num_envs = num_envs
        self.vision = vision
        self.num_frames = num_frames

        self.steps = [0] * self.num_envs
        self.lives = [0] * self.num_envs
        self.nops = [0] * self.num_envs
        self.max_steps = 1000  # maximum rollout steps
        self.max_nops = 128  # no-op reset mechanism
        self.skip_frames = skip_frames  # skip frames

        # image preprocessing
        self.transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),  # convert to grayscale (1 channel)
                T.Resize((84, 84)),  # rescale to 84x84
            ]
        )

        if is_ale:
            self.envs = [
                gym.make("ALE/" + env_name, render_mode="rgb_array_list")
                for _ in range(num_envs)
            ]
        else:
            self.envs = [
                gym.make(env_name, render_mode="rgb_array_list")
                for _ in range(num_envs)
            ]

        if self.vision:
            self.frame_stack = [None] * self.num_envs
        else:
            self.frame_stack = None

    def reset_single(self, index) -> torch.Tensor:
        self.steps[index] = 0
        self.nops[index] = 0

        init_state, info = self.envs[index].reset()
        init_state = torch.Tensor(init_state)

        if "lives" in info.keys():
            self.lives[index] = info["lives"]

        if self.vision:
            self.frame_stack[index] = []
            # reshape into (channel, width, height)
            init_state = init_state.permute(2, 0, 1)
            init_state = self.transform(init_state)

            # pad empty frames
            for _ in range(self.num_frames):
                self.frame_stack[index].append(torch.Tensor(init_state))

            # merge into channel dimension
            return torch.cat(self.frame_stack[index], dim=0)
        else:
            # simple observation
            return init_state

    def step_single(
        self, index, action
    ) -> tuple[torch.Tensor, float, bool, bool, dict]:
        next_state, reward, terminated, truncated, info = self.envs[index].step(action)
        next_state = torch.Tensor(next_state)

        # in case the game won't stop
        self.steps[index] += 1
        if self.steps[index] > self.max_steps:
            truncated = True

        # there are more considerations for Atari
        if self.vision:
            # reshape reward
            if reward > 0:
                reward = 10
            else:
                reward = 0

            # no-op reset
            if action == 0:
                self.nops[index] += self.skip_frames
                if self.nops[index] > self.max_nops:
                    truncated = True
            else:
                self.nops[index] = 0

            # a life is lost
            if "lives" in info.keys() and info["lives"] < self.lives[index]:
                if info["lives"] < 1:
                    # game over
                    reward -= 10
                    terminated = True
                else:
                    reward -= 10
                    self.lives[index] = info["lives"]

            # skip frames and cumulate rewards
            for _ in range(self.skip_frames):
                if terminated or truncated:
                    break

                (
                    extra_state,
                    extra_reward,
                    extra_terminated,
                    extra_truncated,
                    extra_info,
                ) = self.envs[index].step(action)

                if extra_reward > 0:
                    reward += 10

                terminated = terminated or extra_terminated
                truncated = truncated or extra_truncated

                if (
                    "lives" in extra_info.keys()
                    and extra_info["lives"] < self.lives[index]
                ):
                    if extra_info["lives"] < 1:
                        reward -= 10
                        terminated = True
                    else:
                        reward -= 10
                        self.lives[index] = extra_info["lives"]

                next_state = torch.Tensor(extra_state)

            next_state = next_state.permute(2, 0, 1)
            next_state = self.transform(next_state)

            # skipped frames are discarded
            self.frame_stack[index].pop(0)
            self.frame_stack[index].append(next_state)

            if terminated or truncated:
                # default behavior after gameover: reset
                self.reset_single(index)

            # concatenate into channel dimension
            return (
                torch.cat(self.frame_stack[index], dim=0),
                reward,
                terminated,
                truncated,
                info,
            )
        else:
            if terminated or truncated:
                # default behavior after gameover: reset
                self.reset_single(index)
            return next_state, reward, terminated, truncated, info

    def reset(self) -> torch.Tensor:
        states = [None] * self.num_envs
        for index in range(self.num_envs):
            states[index] = self.reset_single(index)

        return torch.stack(states)

    def step(
        self, actions
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool], list[bool], list[dict]]:
        next_state_all = [None] * self.num_envs
        reward_all = [None] * self.num_envs
        terminated_all = [None] * self.num_envs
        truncated_all = [None] * self.num_envs
        info_all = [None] * self.num_envs

        # step on each environment
        for index in range(self.num_envs):
            action = actions[index]

            next_state, reward, terminated, truncated, info = self.step_single(
                index, action
            )

            next_state_all[index] = next_state
            reward_all[index] = reward
            terminated_all[index] = terminated
            truncated_all[index] = truncated
            info_all[index] = info

        return (
            torch.stack(next_state_all),
            torch.tensor(reward_all),
            terminated_all,
            truncated_all,
            info_all,
        )

    def action_size(self):
        return self.envs[0].action_space.n

    def status_size(self):
        if self.vision:
            return (self.num_frames, 84, 84)
        else:
            return self.envs[0].observation_space._shape[0]


#####################################################################
#                          Replay Buffer                            #
#####################################################################


# replay buffer stores all experiences
# store data in CPU to reduce GPU memory usage
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def add(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def minibatch(self):
        # choose unique data
        data = random.sample(self.buffer, self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        terminates = []

        for d in data:
            states.append(d[0])
            actions.append(d[1])
            rewards.append(d[2])
            next_states.append(d[3])
            terminates.append(d[4])

        # torch.tensor is used to create a tensor of scalars
        # torch.stack is used to create a tensor of tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        terminates = torch.tensor(terminates).int().to(self.device)
        return states, actions, rewards, next_states, terminates

    def __len__(self):
        return len(self.buffer)
