"""

This Python script runs the Reinforce algorithm, following RL Hugh's tutorial series on YouTube (2022). 
Ensure that the required libraries and VizDoom scenarios are correctly installed and configured to avoid runtime errors.

References:
Perkins, H., 2022. youtube-rl-demos/vizdoom at vizdoom18 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom [Accessed 8 May 2024].
RL Hugh, n.d. ViZDoom: reinforcement learning using PyTorch - YouTube [Online]. www.youtube.com. Available from: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 [Accessed 8 May 2024].

"""

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
