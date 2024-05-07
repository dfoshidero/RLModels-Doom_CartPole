# Reinforcement Learning Models for VizDoom and CartPole Environments
## Overview

This repository contains a collection of Reinforcement Learning (RL) models developed by Group 2 for the University of Bath's CM50270 module, focusing on two environments: VizDoom and CartPole. The primary goal is to train deep RL agents to effectively navigate and solve tasks in these complex simulations, abd evaluate how these models perform in simple versus more complex situations.

Conducted tests can be viewed here: LINK

### VizDoom: Defend the Center
In the "Defend the Center" scenario of VizDoom, the agent remains stationary at the center of a circular room and must shoot approaching enemies from various directions.

**State Space**: Includes the agent's first-person view via a screen buffer (RGB channels) and other game variables such as health, ammunition, alive status, and the number of enemies killed.  
**Action Space**: Consists of three discrete actions: (0) Turn left, (1) Turn right, (2) Shoot.  
**Rewards**: The agent receives +1 for each enemy killed and a -1 penalty upon death. The episode ends when the agent dies, with the objective to maximize the reward by effectively using the available ammunition and managing health against enemy attacks.

### CartPole: Benchmark
The CartPole environment serves as a simpler benchmark to test basic RL model functionality and performance before applying them to more complex scenarios like VizDoom.

**Objective**: The agent's task is to balance a pole mounted on a cart.  
**State Space**: The environment provides information about the cart's position and velocity, as well as the pole's angle and angular velocity.  
**Action Space**: Two discrete actions are available: move left or move right.  
**Rewards**: The agent receives a reward of +1 for each timestep the pole remains balanced, with the goal of maximizing the cumulative reward over the episode duration.

## Model Directories
The repository includes separate directories for each RL model applied to both the CartPole and VizDoom environments. Below is a list of all the available model implementations:

### CartPole Models
- **DDDQN_CartPole**: Implementation of Double Deuling Deep Q-Network for the CartPole environment.
- **DQRN_CartPole**: Implementation of the basic Deep Recurrent Q-Network for the CartPole environment.
- **PPO_CartPole**: Implementation of Proximal Policy Optimization for the CartPole environment.
- **REINFORCE_CartPole**: Implementation of the REINFORCE algorithm for the CartPole environment.

### VizDoom Models
- **DDDQN_Doom**: Implementation of Double Deuling Deep Q-Network for the VizDoom environment.
- **DQN_Doom**: Implementation of the basic Deep Recurrent Q-Network for the VizDoom environment.
- **PPO_Doom**: Implement of the Proximal Policy Optimization for the VizDoom environment.
- **REINFORCE_Doom**: Implement of the the REINFORCE algorithm for the VizDoom environment.

## How to Run Tests
To test any of the models, navigate to the respective model directory and run the Python script associated with that model. For example, to test the PPO model in the VizDoom environment:

1. Navigate to the `PPO_Doom` directory.
2. Run the script by executing `python PPO_Doom.py` in your terminal.

Parameters within each script can be modified to experiment with different configurations or to optimize performance based on the specific needs of the environment.

## Additional Information
For more details on the implementation and configuration of each model, refer to the documentation and comments within each directory's scripts. This will provide insights into the specifics of the algorithms and their application to the environments.

## Contributors
- Daniel Favour O.~Oshidero: dfoo20@bath.ac.uk
- Claire ~He: sh3278@bath.ac.uk
- Luis Eduardo D. A. ~Ballarati: ledab20@bath.ac.uk
- Yan Chun ~Yeung: ycy55@bath.ac.uk
- Ting-I ~Lei: til27@bath.ac.uk
- Yi-An ~Lin: yal29@bath.ac.uk

For any questions or further assistance, feel free to contact the contributors at their provided email addresses.

## Referenced Work
