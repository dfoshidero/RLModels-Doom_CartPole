import torch
from torch import nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, n_output):
        super(ActorCriticNetwork, self).__init__()

        # Defining the convolutional and fully connected layers
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(), # Flatten as input to dense layers
            nn.Linear(2304,512),
            nn.ReLU(), 
        ]
        
        # Define the actor and critic output layers
        self.network = nn.Sequential(*network)
        self.actor_output = nn.Sequential(
            nn.Linear(512, n_output), # Fully connected layer for actor
            nn.Softmax(dim=1)  # Softmax activation function for probability distribution
        )
        self.critic_output = nn.Sequential(
            nn.Linear(512, 1), # Fully connected layer for critic
        )

    def forward(self, state):
        '''
        Perform forward pass through the network
        '''
        network_output = self.network(state)

        # Compute the value (critic output) and log probabilities (actor output)
        value = self.critic_output(network_output)  # Value estimation for the state
        log_probs = self.actor_output(network_output) # Log probabilities for each action
        return log_probs, value