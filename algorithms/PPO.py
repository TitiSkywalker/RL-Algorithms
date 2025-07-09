"""
This file implements the proximal policy optimization algorithm, proposed in the legend paper "Proximal Policy Optimization Algorithms" by John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. This is the most complicated algorithm in our project. The policy rollout is done in several independent environments. We can implemented easily using batched operations in PyTorch.

The surrogate objective is computed as:

    r = π(a | s) / π_old(a | s)
    L = min(r * A, clip(r, 1 - ε, 1 + ε) * A)

Where A is the generalized advantage estimation (GAE):

    A = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
    δ = reward + γ * V(s') - V(s)      (TD-error)

There are many ways to define target value. One simple way is using the discounted return. This can lead to problem because roll-outs are truncated in PPO, leading to incorrect value estimation. According to other implementations, we can define the target value R as A + V. If we are using discounted return, then we can define A as R - V.

We have done experiments both in OpenAI Gymnasium and Arcade Learning Environment.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torch.distributions.categorical import Categorical
from moviepy.editor import ImageSequenceClip
from PIL import Image
from Utils import EnvBatch, EnvSingle
from Networks import PPONet, PPONet2D_conv

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# agent with proximal policy optimization method
class PPOAgent:
    def __init__(self, hyperparams):
        self.env_name = hyperparams["env_name"]
        self.vision = hyperparams["vision"]
        self.is_ale = hyperparams["ALE"]
        self.frames = hyperparams["frames"]
        self.skip_frames = hyperparams["skip_frames"]
        self.reward_bound = hyperparams["reward_bound"]
        self.max_reward = hyperparams["max_reward"]
        self.gamma = hyperparams["gamma"]
        self.gae = hyperparams["gae"]
        self.gae_lambda = hyperparams["gae_lambda"]
        self.clip_coefficient = hyperparams["clip_coefficient"]
        self.entropy_coefficient = hyperparams["entropy_coefficient"]
        self.value_coefficient = hyperparams["value_coefficient"]
        self.lr = hyperparams["lr"]
        self.num_envs = hyperparams["num_envs"]
        self.rollout_steps = hyperparams["rollout_steps"]
        self.minibatch_size = hyperparams["minibatch_size"]
        self.num_optimize_epochs = hyperparams["num_optimize_epochs"]

        self.envs = EnvBatch(
            self.env_name,
            self.num_envs,
            self.vision,
            self.frames,
            self.skip_frames,
            self.is_ale,
        )
        self.status_size = self.envs.status_size()
        self.action_size = self.envs.action_size()

        if self.vision:
            self.ppoNet = PPONet2D_conv(self.status_size, self.action_size).to(device)
        else:
            self.ppoNet = PPONet(self.status_size, self.action_size).to(device)
        self.ppo_optimizer = torch.optim.Adam(self.ppoNet.parameters(), lr=self.lr)

        print(f"Agent information: ")
        print(f"- algorithm   : proximal policy optimization")
        print(f"- device      : {device}")
        print(f"- environment : {self.env_name}")
        print(f"- ALE         : {self.is_ale}")
        print(f"- observation : {self.status_size} dimension space")
        print(f"- action      : {self.action_size} dimension space")
        print(f"- envs        : {self.num_envs} independent environments")
        print(f"- rollout     : >= {self.rollout_steps} timesteps")
        print(f"- minibatch   : {self.minibatch_size} data units")
        print(f"- optimize    : {self.num_optimize_epochs} epochs")

    def action_value(
        self, state, action=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_logits, value = self.ppoNet(state)
        probs = Categorical(logits=policy_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def train(self, num_train_episodes=100):
        print("Start training")

        reward_history = []
        clip_loss_history = []
        value_loss_history = []
        entropy_loss_history = []

        for episode in range(num_train_episodes):
            # every epoch is a new day
            states = []
            actions = []
            prob_logits = []
            rewards = []
            terminates = []
            values = []
            current_state = torch.Tensor(self.envs.reset()).to(device)

            # policy rollout
            total_steps = 1
            max_rewards = np.zeros(self.num_envs)
            current_rewards = np.zeros(self.num_envs)
            while True:
                # store current state
                states.append(current_state)
                # batched action
                with torch.no_grad():
                    action, prob_logit, entropy, value = self.action_value(
                        current_state
                    )
                    values.append(value.flatten())
                    actions.append(action)
                    prob_logits.append(prob_logit)
                # batched environmnts
                next_state, reward, terminated, truncated, info = self.envs.step(
                    action.cpu().numpy()
                )

                rewards.append(reward.to(device).flatten())
                terminated = (
                    torch.logical_or(torch.tensor(terminated), torch.tensor(truncated))
                    .to(torch.bool)
                    .to(device)
                )
                terminates.append(terminated)
                current_state = next_state.to(device)

                # compute maximum reward
                current_rewards += reward.numpy()
                max_rewards = np.maximum(max_rewards, current_rewards)
                for index in range(self.num_envs):
                    if terminated[index] or truncated[index]:
                        current_rewards[index] = 0

                # whether stop current rollout
                finished = False
                for terminated, truncated in zip(terminated, truncated):
                    if terminated or truncated:
                        finished = True
                        break
                if finished and total_steps > self.rollout_steps:
                    break
                else:
                    total_steps += 1

            # combine a tensor
            # shape = (steps, num_envs, status_shape)
            states = torch.stack(states).to(device)
            # shape = (steps, num_envs)
            actions = torch.stack(actions).to(device)
            # shape = (steps, num_envs)
            prob_logits = torch.stack(prob_logits).to(device)
            # shape = (steps, num_envs)
            rewards = torch.stack(rewards).to(device)
            # shape = (steps, num_envs)
            terminates = torch.stack(terminates).to(device).int()
            # shape = (steps, num_envs)
            values = torch.stack(values).to(device)

            if self.gae:
                # calculate GAE
                with torch.no_grad():
                    advantages = torch.zeros_like(rewards).to(device)

                    current_advantage = 0
                    next_value = torch.zeros_like(values[0])

                    for step in reversed(range(len(rewards))):
                        reward = rewards[step]
                        terminated = terminates[step]
                        value = values[step]

                        mask = torch.ones_like(terminated) - terminated
                        delta = reward + mask * self.gamma * next_value - value
                        current_advantage = (
                            delta
                            + mask * self.gamma * self.gae_lambda * current_advantage
                        )
                        next_value = value

                        advantages[step] = current_advantage
                returns = advantages + values
            else:
                # calculate cumulated rewards
                with torch.no_grad():
                    returns = torch.zeros_like(rewards).to(device)
                    current_return = torch.zeros(self.num_envs).to(device)

                    for step in reversed(range(len(rewards))):
                        reward = rewards[step]
                        terminated = terminates[step]
                        # if terminated, then G(t) = 0
                        # else         , G(t) = r(t) + gamma*G(t+1)
                        mask = torch.ones_like(terminated) - terminated
                        current_return = reward + mask * self.gamma * current_return

                        returns[step] = current_return
                advantages = returns - values

            # flatten the batch
            batch_size = len(states) * self.num_envs
            b_states = states.view(
                len(states) * self.num_envs, *tuple(states.shape)[2:]
            )
            b_prob_logits = prob_logits.reshape(-1)
            b_actions = actions.reshape(-1).int()
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_indices = np.arange(batch_size)

            # start optimization
            for optimize_epoch in range(self.num_optimize_epochs):
                # shuffle the batch
                np.random.shuffle(b_indices)
                # go through the whole batch
                for start_index in range(0, batch_size, self.minibatch_size):
                    # get a minibatch
                    end_index = min(start_index + self.minibatch_size, batch_size)
                    mb_indices = b_indices[start_index:end_index]
                    mb_states = b_states[mb_indices]
                    mb_actions = b_actions[mb_indices]
                    mb_prob_logits = b_prob_logits[mb_indices]
                    mb_returns = b_returns[mb_indices]
                    mb_advantages = b_advantages[mb_indices]
                    mb_values = b_values[mb_indices]

                    # compute policy loss
                    _, new_prob_logits, new_entropy, new_value = self.action_value(
                        mb_states, mb_actions
                    )
                    # ratio = torch.exp(new_prob_logits.gather(1, mb_actions) - mb_prob_logits.gather(1, mb_actions))
                    ratio = torch.exp(new_prob_logits - mb_prob_logits)

                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clip(
                        ratio, 1 - self.clip_coefficient, 1 + self.clip_coefficient
                    )
                    pg_loss = torch.mean(torch.min(pg_loss1, pg_loss2))

                    # compute value loss
                    new_value = new_value.flatten()
                    value_loss = torch.mean((new_value - mb_returns) ** 2)

                    # value_loss_unclipped=(new_value-mb_returns)**2
                    # value_clipped=mb_values+torch.clamp(
                    #     new_value-mb_values,
                    #     -self.clip_coefficient,
                    #     self.clip_coefficient,
                    # )
                    # value_loss_clipped=(value_clipped-mb_returns)**2
                    # value_loss_max=torch.max(value_loss_unclipped, value_loss_clipped)
                    # value_loss=torch.mean(value_loss_max)

                    # compute entropy loss
                    entropy_loss = torch.mean(new_entropy)

                    loss = (
                        -pg_loss
                        + self.value_coefficient * value_loss
                        - self.entropy_coefficient * entropy_loss
                    )

                    # gradient descent
                    self.ppo_optimizer.zero_grad()
                    loss.backward()
                    self.ppo_optimizer.step()

            # evaluate_reward = self.evaluate()
            print(
                f"Training episode {episode}, reward = {np.max(max_rewards)}, steps = {total_steps}"
            )
            reward_history.append(np.max(max_rewards))

            clip_loss_history.append(-pg_loss.item())
            value_loss_history.append(value_loss.item() * self.value_coefficient)
            entropy_loss_history.append(-entropy_loss.item() * self.entropy_coefficient)

        plt.plot(reward_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.ylim(0, self.reward_bound)
        plt.axhline(y=self.max_reward, color="orange", linestyle="--")
        plt.title("Proximal policy optimization")
        plt.savefig(f"../results/PPO_reward_{self.env_name}.png")
        plt.close()

        plt.plot(clip_loss_history, label="clip loss")
        plt.plot(value_loss_history, label="value loss")
        plt.plot(entropy_loss_history, label="entropy loss")
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.title("Proximal policy optimization")
        plt.legend()
        plt.savefig(f"../results/PPO_loss_{self.env_name}.png")
        plt.close()

    def save(self):
        print("Save model")
        torch.save(self.ppoNet.state_dict(), f"../models/PPO_{self.env_name}.pth")

    def reload(self):
        print("Load model")
        try:
            state_dict = torch.load(
                f"../models/PPO_{self.env_name}.pth", weights_only=True
            )
            self.ppoNet.load_state_dict(state_dict)
        except:
            print(f"Cannot load model from ../models/PPO_{self.env_name}.pth")

    def evaluate(self, generate_video=False):
        # do a single rollout in one environment
        env = EnvSingle(
            self.env_name, self.vision, self.frames, self.skip_frames, self.is_ale
        )
        state = torch.Tensor(torch.stack([env.reset()])).to(device)
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            with torch.no_grad():
                action, log_prob, entropy, value = self.action_value(state)

            next_state, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()[0]
            )
            next_state = torch.Tensor(torch.stack([next_state])).to(device)
            reward = reward
            terminated = terminated
            truncated = truncated

            state = next_state
            total_reward += reward

        print(f"Total reward = {total_reward}")
        if generate_video:
            frames, fps = env.render()
            if self.vision:
                # video game is too slow
                clip = ImageSequenceClip(sequence=frames, fps=fps * 1.5)
            else:
                clip = ImageSequenceClip(sequence=frames, fps=fps)
            clip.write_videofile("../results/evaluate.mp4", codec="libx264")

        return total_reward

    def visualizeGradient(self, max_step=100):
        print(f"Visualize gradient map in {max_step} steps")

        if not self.vision:
            print("Only video games can be visualized")
            return

        env = EnvSingle(
            self.env_name, self.vision, self.frames, self.skip_frames, self.is_ale
        )
        state = (
            torch.stack([env.reset()]).clone().detach().requires_grad_(True).to(device)
        )
        state.retain_grad()
        terminated = False
        truncated = False

        policy_frames = []
        value_frames = []

        for step in range(max_step):
            action, log_prob, entropy, value = self.action_value(state)

            # 1. plot gradients produced by value network
            value.backward(retain_graph=True)
            gradient = state.grad[0][3].data.cpu()

            # normalize gradient for visualization
            gradient = gradient.abs().squeeze().numpy()
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

            # create jet colormap of gradients
            gradient_colormap = cm.jet(gradient)
            # convert from RGBA into RGB
            gradient_colormap = gradient_colormap[:, :, :3] * 255
            # overlay them together
            image = state[0][3].squeeze().detach().cpu().numpy()
            image = np.stack((image,) * 3, axis=-1)

            overlay_image = np.clip(
                0.5 * gradient_colormap + 0.6 * image, 0, 255
            ).astype(np.uint8)
            value_frames.append(overlay_image)

            state.grad.zero_()

            # 2. plot gradients produced by policy network
            log_prob.backward(retain_graph=False)
            gradient = state.grad[0][3].data.cpu()

            # normalize gradient for visualization
            gradient = gradient.abs().squeeze().numpy()
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

            # create jet colormap of gradients
            gradient_colormap = cm.jet(gradient)
            # convert from RGBA into RGB
            gradient_colormap = gradient_colormap[:, :, :3] * 255
            # overlay them together
            image = state[0][3].squeeze().detach().cpu().numpy()
            image = np.stack((image,) * 3, axis=-1)

            overlay_image = np.clip(
                0.5 * gradient_colormap + 0.6 * image, 0, 255
            ).astype(np.uint8)
            policy_frames.append(overlay_image)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = (
                torch.stack([next_state])
                .clone()
                .detach()
                .requires_grad_(True)
                .to(device)
            )

            state = next_state.clone().detach().requires_grad_(True).to(device)

            if terminated or truncated:
                break

        frames, fps = env.render()

        all_frames = []
        for i in range(len(policy_frames)):
            frame1 = np.array(Image.fromarray(policy_frames[i]).resize((210, 210)))
            frame2 = np.array(Image.fromarray(value_frames[i]).resize((210, 210)))

            for j in range(self.skip_frames + 1):
                frame3 = frames[i * (self.skip_frames + 1) + j]
                new_frame = np.hstack([frame3, frame1, frame2])
                all_frames.append(new_frame)

        clip = ImageSequenceClip(sequence=all_frames, fps=fps * 1.5)
        clip.write_videofile("../results/gradient.mp4", codec="libx264")


hyperparams_1 = {
    "env_name": "CartPole-v1",
    "vision": False,
    "ALE": False,
    "frames": 0,
    "skip_frames": 0,
    "reward_bound": 600,
    "max_reward": 500,
    # whether do GAE
    "gae": True,
    # discount rate
    "gamma": 0.99,
    # GAE parameter
    "gae_lambda": 1,
    # clipping parameter
    "clip_coefficient": 0.05,
    # entrypy loss weight
    "entropy_coefficient": 0.01,
    # value loss weight
    "value_coefficient": 0.5,
    # learning rate
    "lr": 0.0005,
    # number of independent environments
    "num_envs": 2,
    # minimum rollout steps
    "rollout_steps": 100,
    # size of each minibatch
    "minibatch_size": 32,
    # PPO optimize epoches
    "num_optimize_epochs": 3,
}

# play Atari Breakout
# this environment is better than Breakout-v5 because is simpler
hyperparams_2 = {
    "env_name": "BreakoutNoFrameskip-v4",
    "vision": True,
    "ALE": False,
    "frames": 4,
    "skip_frames": 8,
    "reward_bound": 1000,
    "max_reward": 1000,
    "gae": True,
    # discount rate
    "gamma": 0.99,
    # GAE parameter
    "gae_lambda": 0.95,
    # clipping parameter
    "clip_coefficient": 0.05,
    # entrypy loss weight
    "entropy_coefficient": 0.1,
    # value loss weight
    "value_coefficient": 0.5,
    # learning rate
    "lr": 0.001,
    # number of independent environments
    "num_envs": 8,
    # minimum rollout steps
    "rollout_steps": 128,
    # size of each minibatch
    "minibatch_size": 256,
    # PPO optimize epoches
    "num_optimize_epochs": 4,
}

if __name__ == "__main__":
    agent = PPOAgent(hyperparams_2)
    # agent.train(500)
    # agent.save()
    agent.reload()
    agent.evaluate(True)
    agent.visualizeGradient(200)
