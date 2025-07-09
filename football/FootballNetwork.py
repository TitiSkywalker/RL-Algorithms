"""
This file implements the neural network for football PPO. It receives both parameterized input and super minimap as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FootballNet(nn.Module):
    def __init__(self, frame_num=4):
        super(FootballNet, self).__init__()
        self.frame_num = frame_num

        self.param_fc1 = nn.Linear(185, 256)
        self.param_fc2 = nn.Linear(256, 256)

        # convolutional layers
        self.conv1 = nn.Conv2d(16, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1
        )  # Additional layer
        # residual connection
        self.residual = nn.Conv2d(128, 128, kernel_size=1)
        # MLP after convolution
        self.minimap_fc = nn.Linear(128 * 5 * 8, 256)

        # shared MLPs
        self.shared_fc1 = nn.Linear(256 * frame_num + 256, 512)
        self.shared_fc2 = nn.Linear(512, 512)

        # policy MLPs
        self.policy_fc1 = nn.Linear(512, 128)
        self.policy_fc2 = nn.Linear(128, 19)

        # value MLPs
        self.value_fc1 = nn.Linear(512, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, parameter, minimap):
        # parameterized input
        param_out = F.relu(self.param_fc1(parameter))
        param_out = F.relu(self.param_fc2(param_out))
        param_out = param_out.view(param_out.size(0), -1)

        # minimap input
        minimap_out = minimap.view(
            minimap.size(0), -1, minimap.size(3), minimap.size(4)
        )
        minimap_out = F.relu(self.conv1(minimap_out))
        minimap_out = F.relu(self.conv2(minimap_out))
        minimap_out = F.relu(self.conv3(minimap_out))
        minimap_res = self.residual(minimap_out)  # residual connection
        minimap_out = F.relu(self.conv4(minimap_out + minimap_res))  # add residual
        # flatten
        minimap_out = minimap_out.view(minimap_out.size(0), -1)
        minimap_out = F.relu(self.minimap_fc(minimap_out))

        # concatenate together
        combined = torch.cat((param_out, minimap_out), dim=1)

        # pass through shared network
        shared_out = F.relu(self.shared_fc1(combined))
        shared_out = F.relu(self.shared_fc2(shared_out))

        # policy output
        logits = F.relu(self.policy_fc1(shared_out))
        logits = self.policy_fc2(logits)

        # value output
        value = F.relu(self.value_fc1(shared_out))
        value = self.value_fc2(value).squeeze(-1)

        return logits, value
