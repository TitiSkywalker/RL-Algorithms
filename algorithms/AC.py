"""
This file implements the actor-critic algorithm. REINFORCE updates the network with discounted return, actor-critic just replaces it with TD-error. To compute the TD-error, we need to estimate the value of states. Thus, actor-critic has 2 independent networks.

The policy network is updated with policy gradient:

    ∇J(θ) = E[∑∇log(π(a|s)) ⋅ D]
    D = reward + gamma * V(s') - V(s)

The benefit of doing so is that we can update the network at every step immediately, but the price is biased estimation.The value network is updated directly with TD-error:

    V <- V + alpha * D

Maybe it's better to calculate value and policy using a single network, as proposed in the paper "Dueling Network Architectures for Deep Reinforcement Learning" by Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas. We have implemented this idea in PPO, but not in actor-critic.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from moviepy.editor import ImageSequenceClip
from Utils import EnvSingle
from Networks import PolicyNet, ValueNet

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# agent with actor-critic method
class ACAgent:
    def __init__(self, hyperparams):
        self.env_name     = hyperparams["env_name"]
        self.reward_bound = hyperparams["reward_bound"]
        self.max_reward   = hyperparams["max_reward"]
        self.gamma        = hyperparams["gamma"]
        self.lr_p         = hyperparams["policy_lr"]
        self.lr_v         = hyperparams["value_lr"]

        self.env=EnvSingle(self.env_name)
        self.status_size=self.env.status_size()
        self.action_size=self.env.action_size()

        self.policy_net=PolicyNet(self.status_size, self.action_size).to(device)
        self.value_net=ValueNet(self.status_size).to(device)

        self.optimizer_p=torch.optim.Adam(self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v=torch.optim.Adam(self.value_net.parameters(), lr=self.lr_v)

        print(f"Agent information: ")
        print(f"- algorithm   : actor-critic")
        print(f"- device      : {device}")
        print(f"- environment : {self.env_name}")
        print(f"- observation : shape = {self.status_size}")
        print(f"- action      : {self.action_size} actions")
    
    def action(self, state):
        policy_logits = self.policy_net(state).view(-1)
        probs = Categorical(logits=policy_logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action)
    
    def update(self, state, log_prob, reward, next_state, terminated):
        state = state.to(device)
        next_state = next_state.to(device)

        next_value = self.value_net(next_state)
        value = self.value_net(state)
        target_value = (reward+self.gamma*next_value*(1-terminated))

        # compute value gradient
        MSE_func = nn.MSELoss()
        loss_v = MSE_func(value, target_value)

        # compute policy gradient with TD-error
        delta = (target_value-value).detach()
        loss_p = -log_prob * delta

        total_loss = loss_p + loss_v

        self.optimizer_p.zero_grad()
        self.optimizer_v.zero_grad()

        total_loss.backward()
        
        self.optimizer_p.step()
        self.optimizer_v.step()
    
    def train(self, num_train_episodes=100):
        print("Start training")
        reward_history=[]

        for episode in range(num_train_episodes):
            state=torch.Tensor(self.env.reset()).to(device)
            terminated=False
            truncated=False
            total_reward=0
            
            while not terminated and not truncated:
                action, log_prob=self.action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state=torch.Tensor(next_state).to(device)

                self.update(state, log_prob, reward, next_state, terminated)
                
                state=next_state
                total_reward+=reward

            reward_history.append(total_reward)
            print(f"Training episode {episode}, reward = {total_reward}")

        plt.plot(reward_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.ylim(0, self.reward_bound)
        plt.axhline(y=self.max_reward, color="orange", linestyle="--")
        plt.title("Actor Critic")
        plt.savefig(f"../results/AC_{self.env_name}.png")

    def save(self):
        print("Save model")
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
        }, f"../models/AC_{self.env_name}.pth")

    def reload(self):
        print("Load model")
        try:
            state_dict=torch.load(f"../models/AC_{self.env_name}.pth", weights_only=True)
            self.policy_net.load_state_dict(state_dict["policy_net"])
            self.value_net.load_state_dict(state_dict["value_net"])
        except:
            print(f"Cannot load model from ../models/AC_{self.env_name}.pth")
    
    def evaluate(self):
        # do a single rollout in one environment
        print(f"Evaluate agent on {self.env_name}")

        env = EnvSingle(self.env_name)
        state = torch.Tensor(env.reset()).to(device)
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

hyperparams={
    "env_name": "CartPole-v1",
    "reward_bound": 600,
    "max_reward": 500,
    "gamma": 0.98,
    "policy_lr": 0.0005,
    "value_lr": 0.0005,
}

if __name__ == "__main__":
    agent=ACAgent(hyperparams)
    agent.train(500)
    agent.save()
    agent.reload()
    agent.evaluate()