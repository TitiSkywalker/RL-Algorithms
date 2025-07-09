import matplotlib.pyplot as plt
import torch
import numpy as np

from torch.distributions.categorical import Categorical
from FootballEnv import FootballSingleEnv
from FootballParallelEnv import FootballParallelEnv
from FootballNetwork import FootballNet
from tqdm import tqdm

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# agent with proximal policy optimization method
class FootballAgent:
    def __init__(self, hyperparams):
        self.num_envs = hyperparams["num_envs"]
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
        self.rollout_steps = hyperparams["rollout_steps"]
        self.minibatch_size = hyperparams["minibatch_size"]
        self.num_optimize_epochs = hyperparams["num_optimize_epochs"]

        self.ppoNet = FootballNet(self.frames).to(device)
        self.ppo_optimizer = torch.optim.Adam(self.ppoNet.parameters(), lr=self.lr)

        print(f"Agent information: ")
        print(f"- algorithm   : PPO for football")
        print(f"- device      : {device}")
        print(f"- envs        : {self.num_envs} independent environments")
        print(f"- rollout     : >= {self.rollout_steps} timesteps")
        print(f"- minibatch   : {self.minibatch_size} data units")
        print(f"- optimize    : {self.num_optimize_epochs} epochs")

    def action_value(
        self, parameter, minimap, action=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_logits, value = self.ppoNet(parameter, minimap)
        probs = Categorical(logits=policy_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
        # return torch.Tensor([0]*self.num_envs).to(int), torch.Tensor([0]*self.num_envs), torch.Tensor([0]*self.num_envs), torch.Tensor([0]*self.num_envs)

    def train(self, num_train_episodes=100):
        print("Start training")
        envs = FootballParallelEnv(self.num_envs, self.frames, self.skip_frames)

        max_reward_history = []
        min_reward_history = []
        mean_reward_history = []
        score_history = []
        clip_loss_history = []
        value_loss_history = []
        entropy_loss_history = []

        # countdown = 5
        for episode in range(num_train_episodes):
            # every episode will reset the match
            parameters = []
            minimaps = []
            actions = []
            prob_logits = []
            rewards = []
            terminates = []
            values = []

            # policy rollout
            total_steps = 1
            current_rewards = torch.zeros(self.num_envs)

            current_parameter, current_minimap = envs.reset()

            # if countdown == 0:
            #     print("\nreset environment\n")
            #     countdown = 5
            #     current_parameter, current_minimap = self.envs.reset()
            # else:
            #     countdown -= 1

            while True:
                # store current state
                parameters.append(current_parameter)
                minimaps.append(current_minimap)

                # batched action
                with torch.no_grad():
                    current_parameter = current_parameter.to(device)
                    current_minimap = current_minimap.to(device)
                    action, prob_logit, entropy, value = self.action_value(
                        current_parameter, current_minimap
                    )
                    values.append(value.flatten())
                    actions.append(action)
                    prob_logits.append(prob_logit)
                # batched environmnts
                next_parameter, next_minimap, reward, terminated, info = envs.step(
                    action.cpu().tolist()
                )

                rewards.append(reward.to(device).flatten())
                terminated = torch.Tensor(terminated).flatten().to(torch.bool)
                terminates.append(terminated)

                current_parameter = next_parameter
                current_minimap = next_minimap

                # compute maximum reward
                current_rewards += reward

                total_steps += 1
                if total_steps > self.rollout_steps:
                    break

            # combine a tensor
            # shape = (steps, num_envs, parameter shape)
            parameters = torch.stack(parameters).to(device)
            # shape = (steps, num_envs, minimap shape)
            minimaps = torch.stack(minimaps).to(device)
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

            # checkpoint
            # print(f"total steps = {total_steps}")
            # print(parameters.shape)
            # print(minimaps.shape)
            # print(actions.shape)
            # print(prob_logits.shape)
            # print(rewards.shape)
            # print(terminates.shape)
            # print(values.shape)
            # exit(0)

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
            batch_size = len(parameters) * self.num_envs
            b_parameters = parameters.view(batch_size, *tuple(parameters.shape)[2:])
            b_minimaps = minimaps.view(batch_size, *tuple(minimaps.shape)[2:])
            b_prob_logits = prob_logits.reshape(-1)
            b_actions = actions.reshape(-1).int()
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # checkpoint
            # print(f"batch size = {batch_size}")
            # print(b_parameters.shape)
            # print(b_minimaps.shape)
            # print(b_prob_logits.shape)
            # print(b_actions.shape)
            # print(b_advantages.shape)
            # print(b_returns.shape)
            # print(b_values.shape)
            # exit(0)

            b_indices = np.arange(batch_size)

            # start optimization
            for optimize_epoch in range(self.num_optimize_epochs):
                # shuffle the batch
                np.random.shuffle(b_indices)
                # go through the whole batch
                for start_index in range(0, batch_size, self.minibatch_size):
                    # get a minibatch
                    end_index = min(start_index + self.minibatch_size, batch_size)
                    mb_indices = b_indices[start_index:end_index].tolist()

                    mb_parameters = b_parameters[mb_indices]
                    mb_minimaps = b_minimaps[mb_indices]
                    mb_actions = b_actions[mb_indices]
                    mb_prob_logits = b_prob_logits[mb_indices]
                    mb_returns = b_returns[mb_indices]
                    mb_advantages = b_advantages[mb_indices]
                    mb_values = b_values[mb_indices]

                    # compute policy loss
                    _, new_prob_logits, new_entropy, new_value = self.action_value(
                        mb_parameters, mb_minimaps, mb_actions
                    )
                    ratio = torch.exp(new_prob_logits - mb_prob_logits)

                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clip(
                        ratio, 1 - self.clip_coefficient, 1 + self.clip_coefficient
                    )
                    pg_loss = torch.mean(torch.min(pg_loss1, pg_loss2))

                    # compute value loss
                    new_value = new_value.flatten()
                    value_loss = torch.mean((new_value - mb_returns) ** 2)

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
            score_board = envs.score_board()
            print(
                f"Training episode {episode}, mean reward = {torch.mean(current_rewards).item()}, max reward = {torch.max(current_rewards).item()}, min reward = {torch.min(current_rewards).item()}, total score = {score_board}"
            )

            max_reward_history.append(torch.max(current_rewards).item())
            min_reward_history.append(torch.min(current_rewards).item())
            mean_reward_history.append(torch.mean(current_rewards).item())
            score_history.append(score_board)

            clip_loss_history.append(-pg_loss.item())
            value_loss_history.append(value_loss.item())
            entropy_loss_history.append(-entropy_loss.item())

        plt.plot(mean_reward_history, label="Mean reward")
        plt.plot(max_reward_history, label="Max reward")
        plt.plot(min_reward_history, label="Min reward")
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.ylim(-300, 300)
        plt.legend()
        plt.title("Proximal policy optimization on football")
        plt.savefig("../results/Football_reward.png")
        plt.close()

        plt.plot(clip_loss_history, label="clip loss")
        plt.plot(value_loss_history, label="value loss")
        plt.plot(entropy_loss_history, label="entropy loss")
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.title("Proximal policy optimization on football")
        plt.legend()
        plt.savefig("../results/Football_loss.png")
        plt.close()

        left_score, right_score = zip(*score_history)
        right_score = [-score for score in right_score]
        plt.plot(left_score)
        plt.plot(right_score)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.ylim(-5, 5)
        plt.title("Proximal policy optimization on football")
        plt.savefig("../results/Football_score.png")
        plt.close()

        envs.close()

    def save(self):
        print("Save model")
        torch.save(self.ppoNet.state_dict(), "../models/football.pth")

    def reload(self):
        print("Load model")
        try:
            state_dict = torch.load("../models/football.pth", weights_only=True)
            self.ppoNet.load_state_dict(state_dict)
        except:
            print(f"Cannot load model from ../models/football.pth")

    def evaluate(self, total_steps):
        env = FootballSingleEnv(4, 1, True)

        current_parameter, current_minimap = env.reset()

        for step in tqdm(range(total_steps)):
            # batched action
            with torch.no_grad():
                current_parameter = torch.stack([current_parameter])
                current_minimap = torch.stack([current_minimap])
                current_parameter = current_parameter.to(device)
                current_minimap = current_minimap.to(device)
                action, prob_logit, entropy, value = self.action_value(
                    current_parameter, current_minimap
                )
            # batched environmnts
            next_parameter, next_minimap, reward, terminated, info = env.step(
                action.cpu().tolist()[0]
            )

            current_parameter = next_parameter
            current_minimap = next_minimap

            if terminated:
                break

        env.save_video()


hyperparams = {
    "frames": 4,
    "skip_frames": 1,
    "reward_bound": 100,
    "max_reward": 100,
    "gae": True,
    # discount rate
    "gamma": 0.99,
    # GAE parameter
    "gae_lambda": 0.95,
    # clipping parameter
    "clip_coefficient": 0.05,
    # entrypy loss weight
    "entropy_coefficient": 0.05,
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
    agent = FootballAgent(hyperparams)
    # agent.train(500)
    # agent.save()
    agent.reload()
    agent.evaluate(1000)
