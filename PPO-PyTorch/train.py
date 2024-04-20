from agent import Agent
from frames_env import FramesEnv
from vizdoomenv import Vizdoomenv
from runner import Runner
import torch
import numpy as np
from auxiliars import compute_gae

# Function to train agent, takes in argument object which contains hyper parameters
def train(arguments):
    # Reading parameters form argument object
    learning_rate = arguments.lr
    gamma = arguments.gamma
    n_updates = arguments.epochs
    clip = arguments.clip
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size

    # Setting other parameters
    batch_size = 129
    cycles = 10
    lam = 0.95
    in_channels = 1
    #in_channels = 4
    n_outputs = 3
    actors = 8
    #actors = 1

    # Initialise agent with parameters declared above
    agent = Agent(in_channels, n_outputs, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2)

    # Creating multiple environment runners to run asynchronously
    env_runners = []
    for _ in range(actors):
        raw_env = Vizdoomenv()
        env = FramesEnv(raw_env)
        env_runners.append(Runner(env, agent, batch_size))

    # Iterates over the specified number of cycles
    for _ in range(cycles):
        batch_observations, batch_actions, batch_advantages, batch_old_action_prob = None, None, None, None
        
        
        for env_runner in env_runners:
            obs, actions, rewards, dones, values, old_action_prob = env_runner.run()

            advantages = compute_gae(rewards, values, dones, lam)
            # Collecting batch data
            batch_observations = torch.stack(obs[:-1]) if batch_observations == None else torch.cat([batch_observations,torch.stack(obs[:-1])])
            batch_actions = np.stack(actions[:-1]) if batch_actions is None else np.concatenate([batch_actions,np.stack(actions[:-1])])
            batch_advantages = advantages if batch_advantages is None else np.concatenate([batch_advantages,advantages])
            batch_old_action_prob = torch.stack(old_action_prob[:-1]) if batch_old_action_prob == None else torch.cat([batch_old_action_prob,torch.stack(old_action_prob[:-1])])

        # Update agent with batch data collected
        agent.update(batch_observations, batch_actions, batch_advantages, batch_old_action_prob)
        # Saving models
        agent.save_models()

# Main execution
if __name__ == '__main__':
    train()