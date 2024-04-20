import gym
import numpy as np
import cv2

def show_img(observation):
    '''
    Displays the observation image in a window.
    '''
    window_title = "Game"

    # Create a window
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)   
    # Display observation in window     
    cv2.imshow(window_title, observation)
    # Wait for a key press and close window if 'q' is pressed
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

class FramesEnv(gym.Wrapper):
    def __init__(self, env):
        super(FramesEnv, self).__init__(env)
    
    def reset(self, **kwargs):
        # Reset the environment and process initial observation
        self.last_life_count = 0 # Set the number of lives as 0
        observation = self.env.reset(**kwargs) # Reset the environment
        observation = self.process_img(observation) # Preprocess (resize) the observation 
        observation = np.stack([observation]) # Stack the processed observation along a new axis
 
        return observation

    def step(self, action):
        # Take a step in the environment with the given action
        one_hot_action = np.zeros(3)  # Create one-hot encoded action space
        one_hot_action[action] = 1   # Set the index corresponding to the chosen action to 1
            
        # Take a step in the environment with the one-hot encoded action
        observation, reward, done = self.env.step(one_hot_action)

        # Preprocess the observation after the step
        observation = self.process_img(observation)
        # Display the processed observation
        show_img(observation)

        # Stack processed observation
        observation = np.stack([observation])

        # Modify reward to +4 if positive, -15 otherwise
        reward = 4 if reward > 0 else -15

        return observation, reward, done

    def process_img(self, image):
        '''
        Preprocess an image by resizing it to 80 x 80
        '''
        image = cv2.resize(image, (80,80))
        return  image