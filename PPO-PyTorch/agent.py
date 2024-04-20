import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader
from actor_critic_network import ActorCriticNetwork
from batch_dataset import Batch_DataSet
import os

# Class representing the agent
class Agent():
    def __init__(self, in_channels, n_output, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2):
        self.actor_critic = ActorCriticNetwork(in_channels, n_output)
        
        self.gamma = gamma
        self.n_updates_per_iteration = n_updates
        self.clip = clip
        self.minibatch_size = minibatch_size
        self.c1 = c1
        self.c2 = c2
        # Name model
        self.model_name = 'actor-critic-model.pt'

        # Initializing the ACN and Adam optimizer
        self.actor_critic_optimizer = Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Load previously saved model
        self.load_models()

    def save_models(self):
        '''
        Saves model.
        '''
        torch.save(self.actor_critic.state_dict(), self.model_name)
    
    def load_models(self):
        '''
        Loads previously saved model if exists.
        '''
        if(os.path.isfile(self.model_name)):
            print('A model has been loaded.')
            self.actor_critic.load_state_dict(torch.load(self.model_name))
        else:
            print('No model found.')
    
    def get_action(self, observation):
        '''
        Takes an observation as input, and returns the log probability, value, and action sampled from the ACN.
        '''
        distribution, value = self.actor_critic(observation)
        m = Categorical(distribution.squeeze(0))
        action = m.sample()
        log_prob = m.log_prob(action)

        return log_prob, value.squeeze(0).squeeze(0), action

    def get_action_max_prob(self, observation):
        '''
        Takes in an observation as input, and returns the action with the maximum probability from the ACN.
        '''
        distribution, _ = self.actor_critic(observation)
        action = torch.argmax(distribution)

        return action

    def get_log_probs_batch(self, observations, actions):
        '''
        Takes as input observations and corresponding actions and returns the log probabilities, values, and entropy of the ACN.
        '''
        distributions, values = self.actor_critic(observations)
        m = Categorical(distributions)
        log_probs = m.log_prob(actions)     
        entropy = m.entropy()

        return log_probs, values, entropy

    def update(self, observations, actions, advantage_values, old_logprobs):
        '''
        Updates the actor-critic network using the provided observations, actions, advantage values, and old log probabilities. 
        Computes loss, performs optimization, and updates the network parameters.
        '''
        print('Updating...')
        
        # Create dataset for batch processing 
        dataset = Batch_DataSet(observations, actions, advantage_values, old_logprobs)
        # Create dataloader for batch processing
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)
        
        # Loop through the number of updates per iteration
        for _ in range(self.n_updates_per_iteration):
            
            # Loop through all batches (total:8) in dataloader 
            for i, batch in enumerate(dataloader):
                
                if i > 8:
                    break

                # Load a batch into observation, action, advantage and old action probability
                observations_batch, actions_batch, advantages_batch, old_action_prob_batch = batch 

                # Normalize advantages
                advantages_batch = (advantages_batch - torch.mean(advantages_batch) ) / ( torch.std(advantages_batch) + 1e-8)

                # Compute log probabilities, values and entropy
                #   Log probabilities: log likelihoods of actions given observations
                #   Values: estimated values of states by the critic network
                #   Entropy: entropy of the action distribution, i.e., uncertainty/randomness of policy
                current_log_probs, current_values, entropy = self.get_log_probs_batch(observations_batch, actions_batch)

                current_values = current_values.squeeze(1) 

                # Si vas a utilizar el LOGARITMO de las probabilidades entonces el ratio se calcula con el exponente
                # Pero si vas a dividir las probabilidades entonces debes usar las probabilidades a secas sin calcular su logaritmo
                # current_probs / old_probs == torch.export(torch.log(current_probs) - torch.log(old_probs))

                # Compute ratios for policy gradient 
                ratios = torch.exp(current_log_probs - old_action_prob_batch)

                # Compute surrogate loss
                surr1 = ratios * advantages_batch 
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages_batch
                actor_loss = -torch.min(surr1,surr2) # clipped to prevent excessively large updates

                # Compute critic loss
                critic_loss = torch.pow(advantages_batch - current_values,2) # MSE between advantages and current values estimated by critic network

                # Compute total loss
                total_loss = actor_loss.mean() + self.c1 * critic_loss.mean() - self.c2 * entropy.mean()

                # Set optimiser gradients to zero
                self.actor_critic_optimizer.zero_grad()
                # Back propagation
                total_loss.backward()
                # Optimizer take a step of optimization to update network parameters
                self.actor_critic_optimizer.step()