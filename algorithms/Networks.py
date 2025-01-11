"""
This file is the library for neural networks. We have implemented many different networks. All networks are listed below. Empty status means that the networks are bug-free, but they are still experimental. 

| Name              | Structure | Dm | Status              |
| ----------------- | --------- | -- | ------------------- |
| QNet              | MLP       | 1D | tested in Gymnasium |
| PolicyNet         | MLP       | 1D | tested in Gymnasium |
| ValueNet          | MLP       | 1D | tested in Gymnasium |
| PPONet            | MLP       | 1D | tested in Gymnasium |
| Qnet2D            | Resnet    | 2D |                     |
| PolicyNet2D       | Resnet    | 2D |                     |
| ValueNet2D        | Resnet    | 2D |                     |
| PPONet2D          | Resnet    | 2D |                     |
| Qnet2D_conv       | CNN       | 2D |                     |
| PolicyNet2D_conv  | CNN       | 2D |                     |
| ValueNet2D_conv   | CNN       | 2D |                     |
| PPONet2D_conv     | CNN       | 2D | tested in ALE       |

For the last 4 CNN networks, they are proposed in the paper "Playing Atari with Deep Reinforcement Learning" by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. 

The networks for PPO are special, because they need to produce both policy and value at eh same time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from Resnet import WideResNet, WideResNet_small, WideResNet_small_small

#####################################################################
#                           1D Networks                             #
#####################################################################
# input: 1D, parameterized state

# output: predicted Q function Q(a, s) for all a
class QNet(nn.Module):
    def __init__(self, status_size, action_size, hidden_size=128, device=None):
        super().__init__()
        self.l1=nn.Linear(status_size, hidden_size)
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.l3=nn.Linear(hidden_size, action_size)

        torch.nn.init.kaiming_uniform_(self.l1.weight)
        torch.nn.init.kaiming_uniform_(self.l2.weight)
        torch.nn.init.kaiming_uniform_(self.l3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=F.leaky_relu(self.l1(x))
        x=F.leaky_relu(self.l2(x))
        x=self.l3(x)
        return x
    
# output: action probability distribution P(a|s)
class PolicyNet(nn.Module):
    def __init__(self, status_size, action_size, hidden_size=128):
        super().__init__()
        self.l1=nn.Linear(status_size, hidden_size)
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.l3=nn.Linear(hidden_size, action_size)

        torch.nn.init.kaiming_uniform_(self.l1.weight)
        torch.nn.init.kaiming_uniform_(self.l2.weight)
        torch.nn.init.kaiming_uniform_(self.l3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            # add a batch dimension
            x = torch.unsqueeze(x, dim=0)
        x=F.leaky_relu(self.l1(x))
        x=F.leaky_relu(self.l2(x))
        x=self.l3(x)
        return x

# output: estimated value V(s)
class ValueNet(nn.Module):
    def __init__(self, status_size, hidden_size=128):
        super().__init__()
        self.l1=nn.Linear(status_size, hidden_size)
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.l3=nn.Linear(hidden_size, 1)

        torch.nn.init.kaiming_uniform_(self.l1.weight)
        torch.nn.init.kaiming_uniform_(self.l2.weight)
        torch.nn.init.kaiming_uniform_(self.l3.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=F.leaky_relu(self.l1(x))
        x=F.leaky_relu(self.l2(x))
        x=self.l3(x)
        return x
    
# output: action probability P(a|s) and value V(s)
# policy and value share part of the network
class PPONet(nn.Module):
    def __init__(self, status_size, action_size, hidden_size=128):
        super().__init__()
        self.l1=nn.Linear(status_size, hidden_size)
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.value_pre=nn.Linear(hidden_size, hidden_size)
        self.value_out=nn.Linear(hidden_size, 1)
        self.policy_pre=nn.Linear(hidden_size, hidden_size)
        self.policy_out=nn.Linear(hidden_size, action_size)

        torch.nn.init.kaiming_uniform_(self.l1.weight)
        torch.nn.init.kaiming_uniform_(self.l2.weight)
        torch.nn.init.kaiming_uniform_(self.value_pre.weight)
        torch.nn.init.kaiming_uniform_(self.value_out.weight)
        torch.nn.init.kaiming_uniform_(self.policy_pre.weight)
        torch.nn.init.kaiming_uniform_(self.policy_out.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x=F.leaky_relu(self.l1(x))
        x=F.leaky_relu(self.l2(x))

        policy=F.leaky_relu(self.policy_pre(x))
        policy_logits=self.policy_out(policy)
        value=F.leaky_relu(self.value_pre(x))
        value=self.value_out(value)

        return policy_logits, value

#####################################################################
#                           2D Networks                             #
#####################################################################
# input: stacked images, input shape = (channels, width, height)

# output: Q function Q(a, s) for all a
class QNet2D(nn.Module):
    def __init__(self, status_shape, action_size):
        super().__init__()
        self.WideResNet = WideResNet_small_small(
            input_channels=status_shape[0],
            output_size=action_size,
            is_ppo=False,
            need_softmax=False
        )
        self.normalize = transforms.Normalize(mean=[0.5]*status_shape[0], std=[0.5]*status_shape[0])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add batch_size dimension
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = self.normalize(x)
        x = self.WideResNet(x)
        return x

# output: action probability distribution P(a|s)
class PolicyNet2D(nn.Module):
    def __init__(self, status_shape, action_size):
        super().__init__()
        self.WideResNet = WideResNet_small_small(
            input_channels=status_shape[0],
            output_size=action_size,
            is_ppo=False,
            need_softmax=False
        )
        self.normalize = transforms.Normalize(mean=[0.5]*status_shape[0], std=[0.5]*status_shape[0])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add batch_size dimension
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = self.normalize(x)
        x = self.WideResNet(x)       
        return x

# output: estimated value V(s)
class ValueNet2D(nn.Module):
    def __init__(self, status_shape):
        super().__init__()
        self.WideResNet = WideResNet_small_small(
            input_channels=status_shape[0],
            output_size=1,
            is_ppo=False,
            need_softmax=False
        )
        self.normalize = transforms.Normalize(mean=[0.5]*status_shape[0], std=[0.5]*status_shape[0])
    def forward(self, x) -> torch.Tensor:
        # add batch_size dimension
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = self.normalize(x)
        x = self.WideResNet(x)
        return x

# output: P(a|s) and V(s)
# policy and value share part of the network
# input: stacked images, shape = (batch, channels, width, height)
class PPONet2D(nn.Module):
    def __init__(self, status_shape, action_size):
        super().__init__()
        self.WideResNet = WideResNet_small_small(
            input_channels=status_shape[0],
            output_size=action_size,
            is_ppo=True,
            need_softmax=False,
        )
        self.normalize = transforms.Normalize(mean=[0.5]*status_shape[0], std=[0.5]*status_shape[0])
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # x is guaranteed to be batched
        x = self.normalize(x)
        policy_logits, value = self.WideResNet(x)
        return policy_logits, value

# output: Q function Q(a, s) for all a
class QNet2D_conv(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# output: action probability distribution P(a|s)
class PolicyNet2D_conv(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x) -> torch.Tensor:
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        return logits

# output: estimated value V(s)
class ValueNet2D_conv(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x) -> torch.Tensor:
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# output: P(a|s) and V(s)
# policy and value share part of the network
# input: stacked images, shape = (batch, channels, width, height)
class PPONet2D_conv(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Shared fully connected layer
        self.shared_fc = nn.Linear(64 * 7 * 7, 512)
        
        # Policy head
        self.policy_fc = nn.Linear(512, action_size)
        
        # Value head
        self.value_fc = nn.Linear(512, 1)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layer
        x = F.relu(self.shared_fc(x))
        
        # Policy and value heads
        policy_logits = self.policy_fc(x)  # Policy logits
        value = self.value_fc(x)    # State value

        return policy_logits, value