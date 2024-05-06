# Implementation of Reinforce algorithm following the tutorial series on Youtube by RL Hugh (2022)
# Link: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 
# Original source code can be found at: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom 
# Accessed April 15th 2024.


import os
from time import sleep
from numpy import int8
import vizdoom as vzd
import torch
from typing import Optional
from torch import distributions, int64
import torch.nn.functional as F
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import vizdoom_settings


class ModelRunner:
    def __init__(self, scenario_name: str, relative_speed: float = 1.0):

        game = vzd.DoomGame()
        scenario = scenarios[scenario_name]
        game.set_doom_scenario_path(os.path.join(
            vzd.scenarios_path, scenario['scenario_filename']))

        game.set_doom_map("map01")
        game.set_sound_enabled(True)

        vizdoom_settings.setup_vizdoom(game)

        game.set_available_buttons(scenario['buttons'])
        print("Available buttons:", [b.name for b in game.get_available_buttons()])
        game.set_available_game_variables([vzd.GameVariable.AMMO2])
        print("Available game variables:", [v.name for v in game.get_available_game_variables()])

        if scenario['episode_timeout'] is not None:
            game.set_episode_timeout(scenario['episode_timeout'])

        game.set_episode_start_time(10)

        game.set_window_visible(True)

        game.set_living_reward(scenario['living_reward'])

        game.set_mode(vzd.Mode.PLAYER)

        game.init()

        self.actions = [
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ]

        self.sleep_time = 1.0 / vzd.DEFAULT_TICRATE / relative_speed 
        print('sleep time %.4f' % self.sleep_time)

        self.game = game

    def load_model(self, model_path: str):
        self.model = torch.load(model_path)
    
    def run_episode(self):
        game = self.game
        game.new_episode()
        step = 0
        while not game.is_episode_finished():

            state = game.get_state()
            screen_buf = state.screen_buffer

            screen_buf_t = torch.from_numpy(screen_buf) / 255

            # Transpose buffer image
            screen_buf_t = screen_buf_t.transpose(1, 2)
            screen_buf_t = screen_buf_t.transpose(0, 1)
            screen_buf_t = screen_buf_t.unsqueeze(0)
            action_logits = self.model(screen_buf_t)
            action_probs = F.softmax(action_logits, dim=-1)
            m = distributions.Categorical(action_probs)
            action = m.sample()
            action = action.item()

            game.make_action(self.actions[action])

            if self.sleep_time > 0:
                sleep(self.sleep_time)
            step += 1

        return {'reward': game.get_total_reward(), 'steps': step}

    def close(self):
        self.game.close()