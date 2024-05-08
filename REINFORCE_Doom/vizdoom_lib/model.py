"""

This Python script implements the Reinforce algorithm, following RL Hugh's tutorial series on YouTube (2022). 
Ensure that the required libraries and VizDoom scenarios are correctly installed and configured to avoid runtime errors.

References:
Perkins, H., 2022. youtube-rl-demos/vizdoom at vizdoom18 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom [Accessed 8 May 2024].
RL Hugh, n.d. ViZDoom: reinforcement learning using PyTorch - YouTube [Online]. www.youtube.com. Available from: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 [Accessed 8 May 2024].

"""

from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Model architecture: 

            Input shape : [120][160][3]
            ---- CONV BLOCK 1
            conv2d(kernel size 3, 16 feature planes, padding=1)
            [120][160][16]
            maxpooling(kernel size 3)
            [40][53][16]
            ReLU()
            
            ---- CONV BLOCK 2
            conv2d(kernel size 3, 16 feature planes, padding=1)
            [40][53][16]
            maxpooling(kernel size 2)
            [20][26][16]
            ReLU()

            ---- CONV BLOCK 2
            conv2d(kernel size 3, 16 feature planes, padding=1)
            [40][53][16]
            maxpooling(kernel size 2)
            [10][13][16]
            ReLU()

            linear(7 * 10 * 16, 3)

            Outputs 3 values representing preferences for actions 

    """

    def __init__(self, image_height: int, image_width: int, num_actions: int):
        super().__init__()
        h = image_height
        w = image_width
        self.c1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        h //= 3
        w //= 3
        self.c2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        h //= 2
        w //= 2
        self.c3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        h //= 2
        w //= 2

        self.output = nn.Linear(h * w * 16, num_actions)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.c1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.c2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = self.pool3(x)
        x = F.relu(x)

        x = x.view(batch_size, -1)
        x = self.output(x)
        return x