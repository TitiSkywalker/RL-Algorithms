"""
This file implements the double Q-learning algorithm, proposed in the paper "Deep Reinforcement Learning with Double Q-learning" by Hado van Hasselt, Arthur Guez, and David Silver.

The Q network is updated by:

    Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q_t(s', argmax_a Q(s', a)) - Q(s, a))

Unlike traditional double Q-learning where we have 2 independent Q function, we are now using a target network that copies the old parameters from time to time.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt

from moviepy.editor import ImageSequenceClip
from Utils import ReplayBuffer, EnvSingle
from Networks import QNet

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# agent with double DQN
class DDQNAgent:
    def __init__(self, hyperparams):
        self.env_name       = hyperparams["env_name"]
        self.reward_bound   = hyperparams["reward_bound"]
        self.max_reward     = hyperparams["max_reward"]
        self.gamma          = hyperparams["gamma"]
        self.lr             = hyperparams["lr"]
        
        self.start_epsilon  = hyperparams["start_epsilon"]
        self.end_epsilon    = hyperparams["end_epsilon"]

        self.buffer_size    = hyperparams["buffer_size"]
        self.batch_size     = hyperparams["batch_size"]

        self.sync_interval  = hyperparams["sync_interval"]
        self.train_interval = hyperparams["train_interval"]

        self.env = EnvSingle(self.env_name)
        self.status_size = self.env.status_size()
        self.action_size = self.env.action_size()

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        self.qnet = QNet(self.status_size, self.action_size).to(device)
        self.qnet_target = QNet(self.status_size, self.action_size).to(device)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        print(f"Agent information: ")
        print(f"- algorithm   : double deep Q network")
        print(f"- device      : {device}")
        print(f"- environment : {self.env_name}")
        print(f"- observation : shape = {self.status_size}")
        print(f"- action      : {self.action_size} actions")
        print(f"- synchronize : every {self.sync_interval} episodes")
        print(f"- buffer      : {self.buffer_size} units")
        print(f"- minibatch   : {self.batch_size} units")
    
    def synchronize(self):
        # deepcopy can copy everything, not just references
        self.qnet_target = copy.deepcopy(self.qnet)
    
    def action(self, state, epsilon):
        # ε-greedy policy
        if np.random.rand() < epsilon:
            # exploration
            return np.random.choice(self.action_size)
        else:
            # exploitation
            qs = self.qnet(state.to(device))
            return torch.argmax(qs).item()

    def add(self, state, action, reward, next_state, terminated):
        # store a time unit
        self.replay_buffer.add(state, action, reward, next_state, terminated)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # notice: they are batched
        state, action, reward, next_state, terminated=self.replay_buffer.minibatch()

        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action] 

        # double DQN uses argmax, rather than max to compute future gains
        next_action = torch.argmax(self.qnet(next_state), dim=1).view(-1, 1)
        next_qs = self.qnet_target(next_state)
        next_q = torch.gather(next_qs, 1, next_action).view(-1)

        # if Q network is good enough, "target" should roughly be equal to "q"
        target = reward+self.gamma*(1-terminated)*next_q

        # update parameters
        self.optimizer.zero_grad()
        criterion=nn.MSELoss()
        loss=criterion(q, target)
        loss.backward()
        self.optimizer.step()
    
    def train(self, num_train_episodes=100):
        print("Start training")
        reward_history = []
        global_steps = 0

        for episode in range(num_train_episodes):
            state = self.env.reset().detach()
            terminated=False
            truncated=False
            total_reward=0

            while not terminated and not truncated:
                global_steps += 1
                with torch.no_grad():
                    # linear interpolation of epsilon
                    progress = episode / num_train_episodes
                    epsilon = (1-progress)*self.start_epsilon + progress*self.end_epsilon
                    
                    action = self.action(state, epsilon)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = next_state.detach()

                self.add(state, action, reward, next_state, terminated)

                if global_steps % self.train_interval == 0:
                    # compute gradient and update network
                    self.update()
                
                state=next_state
                total_reward+=reward
            
            if episode % self.sync_interval == 0:
                # copy parameters into target network
                self.synchronize()

            reward_history.append(total_reward)
            print(f"Training episode {episode}, reward = {total_reward}")

        plt.plot(reward_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.ylim(0, self.reward_bound)
        plt.axhline(y=self.max_reward, color="orange", linestyle="--")
        plt.title("Double Deep Q Network")
        plt.savefig(f"../results/DDQN_{self.env_name}.png")

    def save(self):
        print("Save model")
        torch.save(self.qnet.state_dict(), f"../models/DDQN_{self.env_name}.pth")

    def reload(self):
        print("Load model")
        try:
            state_dict=torch.load(f"../models/DDQN_{self.env_name}.pth", weights_only=True)
            self.qnet.load_state_dict(state_dict)
            self.qnet_target=copy.deepcopy(self.qnet)
        except:
            print(f"Cannot load model from ../models/DDQN_{self.env_name}.pth")

    def action_evaluate(self, state):
        # greedy policy
        qs = self.qnet(state.to(device))
        return torch.argmax(qs).item() 
    
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
                action = self.action_evaluate(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state.to(device).detach()
            reward = reward
            terminated = terminated
            truncated = truncated

            total_reward+=reward

        print(f"Total reward = {total_reward}")

        frames, fps=env.render()
        clip = ImageSequenceClip(sequence=frames, fps=fps)
        clip.write_videofile("../results/evaluate.mp4", codec="libx264")

# the structure is the same with Q learning
hyperparams = {
    "env_name": "CartPole-v1",
    "reward_bound": 600,
    "max_reward": 500,
    "gamma": 0.98,                  
    "lr": 0.0005,                   
    "start_epsilon": 1,
    "end_epsilon": 0.1,
    "buffer_size": 10000,
    "batch_size": 32,
    "sync_interval": 20, 
    "train_interval": 1, 
}

if __name__ == "__main__":
    agent=DDQNAgent(hyperparams)
    agent.train(500)
    agent.save()
    agent.reload()
    agent.evaluate()