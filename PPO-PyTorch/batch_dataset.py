import torch

class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, obs, actions, adv, old_action_prob):
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.adv = adv
        self.old_action_prob = old_action_prob
        
    def __len__(self):
        '''
        Returns length of dataset, which is determined by the number of observations.
        '''
        return self.obs.shape[0]
    
    def __getitem__(self, i):
        '''
        Returns a single item from the dataset at index i
        as a tuple containing the Observation, action, advantage and old action probability corresponding to index i.
        '''
        return self.obs[i], self.actions[i], self.adv[i], self.old_action_prob[i]