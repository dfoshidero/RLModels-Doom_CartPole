import matplotlib.pyplot as plt
import torch

# GAE estimation considers not only the direct rewards obtained by the agent
#  but also the state values and a generalization over multiple steps in time 
#  to provide more stable and balanced estimations of action advantages, 
#  even in environments with uncertain or noisy rewards.

# Function to compute GAE
def compute_gae(rewards, values, dones, gamma, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) for a sequence of rewards, state values, and episode terminations.

    Input:
    - rewards (list): List of rewards obtained at each time step.
    - values (list): List of state values estimated by the critic network at each time step.
    - dones (list): List of episode termination flags indicating whether each episode has ended.
    - gamma (float): Discount factor for future rewards.
    - lam (float, optional): Lambda parameter for controlling the trade-off between bias and variance in GAE computation. Default is 0.95.

    Output:
    - advantage (list): List of computed advantages using GAE for each time step.
    """

    # Initialise empty list of GAE
    advantage = []
    # Initialise GAE values
    gae = torch.tensor(0.0)

    # Iterate through times steps in reverse order
    for t in reversed(range(len(rewards)-1)):

        # If end of episode, then re-initialise GAE
        if dones[t]:
            gae = torch.tensor(0.0)
        
        # Compute TD for advantage estimation
        delta_t = rewards[t] + (gamma * values[t+1] * int(not dones[t+1])) - values[t] # Temporal difference
        gae = delta_t + gamma * lam * gae  # Update GAE

        # Append GAE to list of advantages
        advantage.append(gae)
    
    # Reverse the list to align with chronological order of advantages
    advantage.reverse()

    return advantage


# Function to plot historical scores
def draw_plot(history_score, color = "red"):
    plt.title("Scores")
    plt.xlabel("Cycles")
    plt.ylabel("History")
    plt.plot(history_score, color)
    plt.show()