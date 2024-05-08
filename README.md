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
- **DRQN_CartPole**: Implementation of the basic Deep Recurrent Q-Network for the CartPole environment.
- **PPO_CartPole**: Implementation of Proximal Policy Optimization for the CartPole environment.
- **REINFORCE_CartPole**: Implementation of the REINFORCE algorithm for the CartPole environment.

### VizDoom Models
- **DDDQN_Doom**: Implementation of Double Deuling Deep Q-Network for the VizDoom environment.
- **DRQN_Doom**: Implementation of the basic Deep Recurrent Q-Network for the VizDoom environment.
- **PPO_Doom**: Implement of the Proximal Policy Optimization for the VizDoom environment.
- **REINFORCE_Doom**: Implement of the the REINFORCE algorithm for the VizDoom environment.

## How to Run Tests
To test any of the models, navigate to the respective model directory and run the Python script associated with that model. For example, to test the PPO model in the VizDoom environment:

1. Navigate to the `PPO_Doom` directory.
2. Run the script by executing each cell consecutively in the Jupyter Notebook.

Parameters within each script can be modified to experiment with different configurations or to optimize performance based on the specific needs of the environment.

## Additional Information
For more details on the implementation and configuration of each model, refer to the documentation and comments within each directory's scripts. This will provide insights into the specifics of the algorithms and their application to the environments.

## Contributors
- Daniel Favour O. Oshidero: dfoo20@bath.ac.uk
- Claire He: sh3278@bath.ac.uk
- Luis Eduardo D. A. Ballarati: ledab20@bath.ac.uk
- Yan Chun Yeung: ycy55@bath.ac.uk
- Ting-I Lei: til27@bath.ac.uk
- Yi-An Lin: yal29@bath.ac.uk

For any questions or further assistance, feel free to contact the contributors at their provided email addresses.

## Referenced Work
bentrevett, 2022. pytorch-rl/5 - Proximal Policy Optimization (PPO) [CartPole].ipynb at master · bentrevett/pytorch-rl [Online]. GitHub. Available from: https://github.com/bentrevett/pytorch-rl/blob/master/5%20-%20Proximal%20Policy%20Optimization%20(PPO)%20%5BCartPole%5D.ipynb [Accessed 7 May 2024].

Chrysovergis, I., 2021. Keras documentation: Proximal Policy Optimization [Online]. keras.io. Available from: https://keras.io/examples/rl/ppo_cartpole/ [Accessed 7 May 2024].

Computer Science Stack Exchange, 2017. Why do we use the log in gradient-based reinforcement algorithms? [Online]. Available from: https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms [Accessed 7 May 2024].

Kang, C., 2021. REINFORCE on CartPole-v0 [Online]. Chan`s Jupyter. Available from: https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/12/REINFORCE-CartPole.html [Accessed 8 May 2024].

Machine Learning with Phil, 2020. Proximal Policy Optimization (PPO) is Easy With PyTorch | Full PPO Tutorial [Online]. www.youtube.com. Available from: https://www.youtube.com/watch?v=hlv79rcHws0 [Accessed 7 May 2024].

Mansar, Y., 2020. Learning to Play CartPole and LunarLander with Proximal Policy Optimization [Online]. Medium. Available from: https://towardsdatascience.com/learning-to-play-cartpole-and-lunarlander-with-proximal-policy-optimization-dacbd6045417 [Accessed 7 May 2024].

Minai, Y., 2023. Deep Q-learning (DQN) Tutorial with CartPole-v0 [Online]. Medium. Available from: https://medium.com/@ym1942/deep-q-learning-dqn-tutorial-with-cartpole-v0-5505dbd2409e [Accessed 8 May 2024].

Nicholas Renotte, 2022. Build a Doom AI Model with Python | Gaming Reinforcement Learning Full Course [Online]. www.youtube.com. Available from: https://www.youtube.com/watch?v=eBCU-tqLGfQ [Accessed 7 May 2024].

Paszke, A., n.d. Reinforcement Learning (DQN) Tutorial — PyTorch Tutorials 1.8.0 documentation [Online]. pytorch.org. Available from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html [Accessed 8 May 2024].

Perkins, H., 2022. youtube-rl-demos/vizdoom at vizdoom13 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/blob/vizdoom13/vizdoom/ [Accessed 8 May 2024].

Perkins, H., 2022a. youtube-rl-demos/vizdoom at vizdoom18 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom [Accessed 8 May 2024].

Perkins, H., 2022b. youtube-rl-demos/vizdoom/vizdoom_011.py at vizdoom13 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/blob/vizdoom13/vizdoom/vizdoom_011.py [Accessed 8 May 2024].

Ravichandiran, S., 2018. Hands-On-Reinforcement-Learning-With-Python/09. Playing Doom Game using DRQN/9.5 Doom Game Using DRQN.ipynb at 5440811df8da575eb41b131f897ddd8a7ce40d5f · sudharsan13296/Hands-On-Reinforcement-Learning-With-Python [Online]. GitHub. Available from: https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python/blob/5440811df8da575eb41b131f897ddd8a7ce40d5f/09.%20Playing%20Doom%20Game%20using%20DRQN/9.5%20Doom%20Game%20Using%20DRQN.ipynb [Accessed 7 May 2024].

RL Hugh, 2022. Old v1 Vizdoom Part 1: Introduction to using PyTorch to play Doom! [Online]. www.youtube.com. Available from: https://www.youtube.com/watch?v=I0tUl9TIcz8&list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1&index=1 [Accessed 7 May 2024].

RL Hugh, n.d. ViZDoom: reinforcement learning using PyTorch - YouTube [Online]. www.youtube.com. Available from: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 [Accessed 8 May 2024].

Schöpf, P., Auddy, S., Hollenstein, J. and Rodriguez-sanchez, A., 2022. Hypernetwork-PPO for Continual Reinforcement Learning [Online]. openreview.net. Available from: https://openreview.net/forum?id=s9wY71poI25 [Accessed 7 May 2024].

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal Policy Optimization Algorithms [Online]. arXiv.org. Available from: https://doi.org/10.48550/arXiv.1707.06347.

Seita, D., 2016. Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games [Online]. danieltakeshi.github.io. Available from: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/.
Simonini, T., 2018. An introduction to Deep Q-Learning: let’s play Doom [Online]. freeCodeCamp.org. Available from: https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8 [Accessed 7 May 2024].

Simonini, T., 2018. Deep_reinforcement_learning_Course/Dueling Double DQN with PER and fixed-q targets/Dueling Deep Q Learning with Doom (+ double DQNs and Prioritized Experience Replay).ipynb at master · simoninithomas/Deep_reinforcement_learning_Course [Online]. GitHub. Available from: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb [Accessed 7 May 2024].

Stable Baselines3, n.d. PPO — Stable Baselines3 1.4.1a3 documentation [Online]. stable-baselines3.readthedocs.io. Available from: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html.

Tabor, P., 2020. Youtube-Code-Repository/ReinforcementLearning/PolicyGradient/PPO/torch/main.py at master · philtabor/Youtube-Code-Repository [Online]. GitHub. Available from: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/main.py [Accessed 7 May 2024].

trtd56, 2022. trtd56/ppo-CartPole · Hugging Face [Online]. huggingface.co. Available from: https://huggingface.co/trtd56/ppo-CartPole [Accessed 7 May 2024].

Zakharenkov, A.I. and Makarov, I., 2021. Deep Reinforcement Learning with DQN vs. PPO in VizDoom [Online]. IEEE Xplore. IEEE. Available from: https://doi.org/10.1109/cinti53070.2021.9668479.
