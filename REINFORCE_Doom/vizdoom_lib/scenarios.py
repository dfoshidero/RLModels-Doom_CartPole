"""

This Python script sets up the scenario for the VizDoom environment using the Reinforce algorithm, following RL Hugh's tutorial series on YouTube (2022). 
Ensure that the required libraries and VizDoom scenarios are correctly installed and configured to avoid runtime errors.

References:
Perkins, H., 2022. youtube-rl-demos/vizdoom at vizdoom18 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom [Accessed 8 May 2024].
RL Hugh, n.d. ViZDoom: reinforcement learning using PyTorch - YouTube [Online]. www.youtube.com. Available from: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 [Accessed 8 May 2024].

"""

import vizdoom as vzd

scenarios = {
    'defend_the_center': {
        'buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK],
        'living_reward': 0,
        'scenario_filename': 'defend_the_center.wad',
        'episode_timeout': None,
        'reward_scaling': 1.0,
        'reward_baseline': 2.0
    }
}