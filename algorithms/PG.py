"""
This file implements the REINFORCE policy gradient algorithm. Compared to value based algorithms, this algorithms requires significantly less hyperparameters.

The gradient of the objective function is:

    ∇J(θ) = E[∑∇log(π(a | s)) ⋅ R]

Where R is the discounted return. The network is updated after each rollout.
"""

import matplotlib.pyplot as plt
import torch

from torch.distributions.categorical import Categorical
from moviepy.editor import ImageSequenceClip
from Utils import EnvSingle
from Networks import PolicyNet

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# agent with policy gradient method
class PGAgent:
    def __init__(self, hyperparams):
        self.env_name = hyperparams["env_name"]
        self.reward_bound = hyperparams["reward_bound"]
        self.max_reward = hyperparams["max_reward"]
        self.gamma = hyperparams["gamma"]
        self.lr = hyperparams["lr"]

        self.env = EnvSingle(self.env_name)
        self.status_size = self.env.status_size()
        self.action_size = self.env.action_size()

        self.memory = []

        self.policy_net = PolicyNet(self.status_size, self.action_size).to(device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        print(f"Agent information: ")
        print(f"- algorithm   : policy gradient")
        print(f"- device      : {device}")
        print(f"- environment : {self.env_name}")
        print(f"- observation : shape = {self.status_size}")
        print(f"- action      : {self.action_size} actions")

    def action(self, state):
        policy_logits = self.policy_net(state).view(-1)
        probs = Categorical(logits=policy_logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action)

    def add(self, reward, log_prob):
        self.memory.append((reward, log_prob))

    def update(self):
        self.optimizer.zero_grad()

        loss = torch.tensor([0.0], dtype=float).to(device)

        # total discounted return
        G = torch.tensor([0.0], dtype=float).to(device)
        for reward, log_prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss -= log_prob * G

        loss.backward()

        self.optimizer.step()

        # clear memory after each episode
        self.memory = []

    def train(self, num_train_episodes=100):
        print("Start training")
        reward_history = []

        for episode in range(num_train_episodes):
            state = torch.Tensor(self.env.reset()).to(device)
            terminated = False
            truncated = False
            total_reward = 0

            # rollout in the environment
            while not terminated and not truncated:
                action, log_prob = self.action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = torch.Tensor(next_state).to(device)

                self.add(reward, log_prob)

                state = next_state
                total_reward += reward

            # compute gradient and update network after each episode
            self.update()

            reward_history.append(total_reward)
            print(f"Training episode {episode}, reward = {total_reward}")

        plt.plot(reward_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.ylim(0, self.reward_bound)
        plt.axhline(y=self.max_reward, color="orange", linestyle="--")
        plt.title("Policy Gradient")
        plt.savefig(f"../results/PG_{self.env_name}.png")

    def save(self):
        print("Save model")
        torch.save(self.policy_net.state_dict(), f"../models/PG_{self.env_name}.pth")

    def reload(self):
        print("Load model")
        try:
            state_dict = torch.load(
                f"../models/PG_{self.env_name}.pth", weights_only=True
            )
            self.policy_net.load_state_dict(state_dict)
        except:
            print(f"Cannot load model from ../models/PG_{self.env_name}.pth")

    def evaluate(self):
        # do a single rollout in one environment
        print(f"Evaluate agent on {self.env_name}")

        env = EnvSingle(self.env_name)
        state = env.reset().to(device).detach()
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            with torch.no_grad():
                action, log_prob = self.action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.Tensor(next_state).to(device)
            reward = reward
            terminated = terminated
            truncated = truncated

            state = next_state
            total_reward += reward

        print(f"Total reward = {total_reward}")

        frames, fps = env.render()
        clip = ImageSequenceClip(sequence=frames, fps=fps)
        clip.write_videofile("../results/evaluate.mp4", codec="libx264")


hyperparams = {
    "env_name": "CartPole-v1",
    "reward_bound": 600,
    "max_reward": 500,
    "gamma": 0.98,
    "lr": 0.0005,
}

if __name__ == "__main__":
    agent = PGAgent(hyperparams)
    agent.train(500)
    agent.save()
    agent.reload()
    agent.evaluate()
