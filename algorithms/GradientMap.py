"""
This file plots the gradient map of the agent in Atari games. We can see what the agent is "looking at".
"""

import torch

from Networks import PPONet2D_conv

if __name__ == "__main__":
    ppoNet = PPONet2D_conv((4, 84, 84), 4)

    try:
        state_dict=torch.load(f"../models/PPO_BreakoutNoFrameskip-v4.pth", weights_only=True)
        ppoNet.load_state_dict(state_dict)
    except:
        print(f"Cannot load model from ../models/PPO_BreakoutNoFrameskip-v4.pth")