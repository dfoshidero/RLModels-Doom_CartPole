#!/usr/bin/env python3
import os
from random import choice
from time import sleep
import vizdoom as vzd
import torch
import json
from torch import nn, optim, distributions
from vizdoom_lib.model import Net
import torch.nn.functional as F
import argparse
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import vizdoom_settings


def run(args):
    scenario = scenarios[args.scenario_name]

    game = vzd.DoomGame()
    game.set_doom_scenario_path(
        os.path.join('./VizDoom/scenarios/', scenario['scenario_filename']))

    game.set_doom_map("map01")

    vizdoom_settings.setup_vizdoom(game)

    game.set_available_buttons(scenario['buttons'])

    game.set_available_game_variables([
        vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2])

    if scenario['episode_timeout'] is not None:
        game.set_episode_timeout(scenario['episode_timeout'])

    game.set_episode_start_time(5)

    game.set_window_visible(args.visible)
    game.set_living_reward(scenario['living_reward'])

    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    actions = [
        [True, False, False],
        [False, True, False],
        [False, False, True]
    ]
    sleep_time = 0.0

    model = Net(image_height=120, image_width=160, num_actions=len(actions))
    opt = optim.RMSprop(lr=args.lr, params=model.parameters())
    out_f = open(args.log_path, 'w')

    episode = 0
    batch_loss = 0.0
    batch_reward = 0.0
    batch_argmax_action_prop = 0.0
    recorded_last_episode = False
    total_steps = 0

    while True:
        record_this_episode = (
            args.record_every is not None and
            args.replay_path_templ is not None and
            episode % args.record_every == 0
        )
        if recorded_last_episode != record_this_episode:
            game.close()
            game.init()
        
        record_filepath = ''
        if record_this_episode:
            record_filepath = args.replay_path_templ.format(episode=episode)
            print('    recording to ' + record_filepath)

        game.new_episode(record_filepath)

        action_log_probs = []
        last_var_values_str = ''
        if args.visible:
            print('=== new episode === ')
        episode_entropy = 0.0
        episode_steps = 0
        episode_argmax_action_taken = 0
        while not game.is_episode_finished():
            state = game.get_state()

            vars = state.game_variables
            screen_buf = state.screen_buffer  # Screen buffer is an array representing current state image

            var_values_str = ' '.join([
                f'{v:.3f}' for v in vars])

            if var_values_str != last_var_values_str:
                if args.visible:
                    print(var_values_str)
                last_var_values_str = var_values_str

            # Scale to range 0-1
            screen_buf_t = torch.from_numpy(screen_buf) / 255
            # Transpose buffer image array
            screen_buf_t = screen_buf_t.transpose(1, 2)
            screen_buf_t = screen_buf_t.transpose(0, 1)
         
            screen_buf_t = screen_buf_t.unsqueeze(0)

            action_logits = model(screen_buf_t)
            action_probs = F.softmax(action_logits, dim=-1)

            action_log_probs_product = action_probs * action_probs.log()
            step_entropy = (- action_log_probs_product).sum(1).sum()

            episode_entropy += step_entropy
            m = distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            action_log_probs.append(log_prob)
            action = action.item()
            _, argmax_action = action_probs.max(dim=-1)
            argmax_action = argmax_action.item()
            if argmax_action == action:
                episode_argmax_action_taken += 1

            # Makes an action (here random one) and returns a reward.
            r = game.make_action(actions[action])

            if sleep_time > 0:
                sleep(sleep_time)

            if record_this_episode:
                game.send_game_command('stop')
            episode_steps += 1
            total_steps += 1
        recorded_last_episode = record_this_episode

        reward_scaling = scenario['reward_scaling']
        reward_baseline = scenario['reward_baseline']
        episode_reward = (
            game.get_total_reward() * reward_scaling - reward_baseline)

        print(f'Episode {episode}    |    episode reward {game.get_total_reward()}')

        per_timestep_losses = [- log_prob * episode_reward for log_prob in action_log_probs]
        per_timestep_losses_t = torch.stack(per_timestep_losses)
        reward_loss = per_timestep_losses_t.sum()
        entropy_loss = - args.ent_reg * episode_entropy
        episode_argmax_action_prop = episode_argmax_action_taken / episode_steps
        total_loss = reward_loss + entropy_loss
        total_loss.backward()
        batch_loss += total_loss.item()
        batch_reward += game.get_total_reward()
        batch_argmax_action_prop += episode_argmax_action_prop

        # Accumulate reward per batch
        if (episode + 1) % args.accumulate_episodes == 0:
            b = episode // args.accumulate_episodes
            batch_avg_reward = batch_reward / args.accumulate_episodes
            batch_avg_loss = batch_loss / args.accumulate_episodes
            batch_avg_argmax_action_prop = batch_argmax_action_prop / args.accumulate_episodes

            opt.step()
            opt.zero_grad()

            # Write batch result to log for plotting
            out_f.write(json.dumps({
                'batch': b,
                'loss': batch_avg_loss,
                'argmax_action_prop': batch_avg_argmax_action_prop,
                'reward': batch_avg_reward,
                'total_steps': total_steps
            }) + '\n')
            out_f.flush()

            batch_loss = 0.0
            batch_reward = 0.0
            batch_argmax_action_prop = 0.0

        # Save model
        if episode % args.save_model_every == 0:
            save_path = args.model_path_templ.format(episode=episode)
            torch.save(model, save_path)
            print(f'saved model to {save_path}')
        episode += 1

        if episode > 100000:
            break

    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--accumulate-episodes', type=int, default=16,
        help='how many episodes to accumulate gradients over before opt step')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument(
        '--model-path-templ', type=str,
        default='./models/defend_the_center/model_ent_{episode}.pt',
        help='can use {episode} which will be replaced by episode')
    parser.add_argument(
        '--save-model-every', type=int, default=100,
        help='how often to save model, number of episodes')
    parser.add_argument('--log-path', type=str, default='./logs/log_center_ent.txt')
    parser.add_argument('--visible', action='store_true')
    parser.add_argument(
        '--record-every', type=int,
        help='record replay every this many episodes', default=100)
    parser.add_argument(
        '--replay-path-templ', type=str, default='./record/defend_the_center_ent_{episode}.lmp',
        help='eg vizdoom/replays_foo{episode}.lmp')
    parser.add_argument(
        '--ent-reg', type=float, default=0.001,
        help='entropy regularization, encourages exploration')
    parser.add_argument('--scenario-name', type=str, default='defend_the_center', help='name of scenario')
    
    args = parser.parse_args()

    assert args.scenario_name in scenarios
    run(args)