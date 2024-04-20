import torch

class Runner():
    def __init__(self, env, agent, batch_size):
        self.env = env # Environment
        self.agent = agent # Agent
        self.observation = self.env.reset() # Reset environment for initial observation
        self.batch_size = batch_size # Batch size
        
    def run(self):
        # Method to run the environment and agent for a batch of steps
        observations = []  # List to store observations
        actions = []       # List to store actions taken by agent
        rewards = []       # List to store rewards obtained form the environment
        done_flag = []         # List to store episode termination flags
        values = []        # List to store state values predicted by the critic network
        log_probs = []     # List to store probabilities of action taken by the agent
        
        for _ in range(self.batch_size):

            # Iterate over batch size 
            self.observation = torch.FloatTensor(self.observation).unsqueeze(0)
            # Get action, log probability and value prediction from agent
            log_prob, value, action = self.agent.get_action(self.observation)
            # Take the action the environment and obtain the next observation, reward, and termination flag
            next_observation, reward, done = self.env.step(action.item())

            # if Episode is terminated, reset the environment
            if done:
                next_observation = self.env.reset()

            # Save current step's information to the corresponding lists
            observations.append(self.observation.squeeze(0)) # Remove batch dimension from observation
            actions.append(action)  # Append action taken
            values.append(value.detach())  # Append predicted value, and detach to prevent gradients flow
            log_probs.append(log_prob.detach()) # Append log probability of action (detach to prevent gradients flow)
            rewards.append(reward) # Append reward obtained
            done_flag.append(done)  # Append episode termination flag

            self.observation = next_observation
        
        return [observations, actions, rewards, done_flag, values, log_probs]