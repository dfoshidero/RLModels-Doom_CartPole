# Implementation of Reinforce algorithm following the tutorial series on Youtube by RL Hugh (2022)
# Link: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 
# Original source code can be found at: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom 
# Accessed April 15th 2024.

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