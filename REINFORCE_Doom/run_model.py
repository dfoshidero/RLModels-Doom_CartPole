# Implementation of Reinforce algorithm following the tutorial series on Youtube by RL Hugh (2022)
# Link: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 
# Original source code can be found at: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom 
# Accessed April 15th 2024.

import argparse
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import run_model_lib

def run(args):
    model_runner = run_model_lib.ModelRunner(
        scenario_name=args.scenario_name
    )
    model_runner.load_model(args.in_model_path)
    i = 0
    while True:
        res = model_runner.run_episode()
        print('reward', res['reward'])
        i += 1
        if i > args.episodes:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-model-path', type=str, required=True)
    parser.add_argument('--scenario-name', type=str, default='defend_the_center', help='name of scenario')
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    assert args.scenario_name in scenarios
    run(args)
