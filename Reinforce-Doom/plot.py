# Implementation of Reinforce algorithm following the tutorial series on Youtube by RL Hugh (2022)
# Link: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 
# Original source code can be found at: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom 
# Accessed April 15th 2024.

import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


def run(args):
	log_rows = []
	with open(args.in_logfile) as f:
		for line in f:
			row = json.loads(line)
			log_rows.append(row)
	log_rows = [
		row
		for row in log_rows if args.max_batch is None or
		row['batch'] <= args.max_batch]
	episodes = [row['batch'] for row in log_rows]
	values = [row[args.y_axis] for row in log_rows]

	sns.set_context("paper") 
	sns.lineplot(x=episodes, y=values, linewidth=2, color='blue') 
	plt.xlabel('Batch', fontdict={'fontsize': 9, 'fontweight': 'bold'})
	plt.ylabel('Average Reward per Episode',fontdict={'fontsize': 9, 'fontweight': 'bold'})
	plt.title("Reinforce Agent Reward by Batch (=16 Episodes)",fontdict={'fontsize': 11, 'fontweight': 'bold'}, pad=10)
	plt.tight_layout()
	plt.grid(True)
	plt.savefig(f'./graphs/{args.graph_name}.png')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-logfile', type=str, default='./logs/log_center_ent.txt')
	parser.add_argument('--graph-name', type=str, default="Reinforce reward graph")
	parser.add_argument('--max-batch', type=int)
	parser.add_argument('--y-axis', choices=['reward', 'loss'], default='reward')
	args = parser.parse_args()
	run(args)