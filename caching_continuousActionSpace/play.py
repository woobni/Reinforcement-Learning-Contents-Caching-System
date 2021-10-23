import gym
import time
import os
from tqdm import tqdm
import datetime
import pickle
import argparse

import tensorflow as tf
from ddpg import ddpgAgent
from gym_foo.envs.continuous_cache import ContinuousCache


NUM_EPISODES_ = 50

# Make directories for tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
proposed_caching_log_dir = 'logs/test/' + current_time + '/proposed_caching'
proposed_caching_summary_writer = tf.summary.create_file_writer(proposed_caching_log_dir)

popularity_caching_log_dir = 'logs/test/' + current_time + '/popularity_caching'
popularity_caching_summary_writer = tf.summary.create_file_writer(popularity_caching_log_dir)

random_caching_log_dir = 'logs/test/' + current_time + '/random_caching'
random_caching_summary_writer = tf.summary.create_file_writer(random_caching_log_dir)

optimal_caching_log_dir = 'logs/test/' + current_time + '/optimal_caching'
optimal_caching_summary_writer = tf.summary.create_file_writer(optimal_caching_log_dir)


def model_test(pretrained_):
	# Create Environments
	env = gym.make('foo-ContinuousCache-v0')

	# Create Agent model
	agent = ddpgAgent(env, batch_size=128, w_per=False)

	if not pretrained_ == None:
		agent.load_weights(pretrained_)

	# Initialize Environments
	steps = env.step_size  # steps per episode
	num_act_ = env.action_space.shape[0]
	num_obs_ = env.observation_space.shape[0]
	print("============ENVIRONMENT===============")
	print("num_of_action_spaces : %d"%num_act_)
	print("num_of_observation_spaces: %d"%num_obs_)
	print("max_steps_per_episode: %d"%steps)

	logger = dict(episode=[], contents=[], critics=[], users=[])

	for epi in range(NUM_EPISODES_):
		print("=========EPISODE # %d =========="%epi)

		obs = env.reset()

		epi_reward = 0
		epi_proposed_caching_score = 0
		epi_popularity_caching_score = 0
		epi_random_caching_score = 0
		for t in tqdm(range(steps)):
			# Make action from the current policy
			proto_action = agent.make_action(obs, t)

			# do step on gym at t-time
			new_obs, reward, done, info = env.step(proto_action)

			# grace finish and go to t+1 time
			obs = new_obs
			epi_reward = epi_reward + reward

			epi_proposed_caching_score += reward
			epi_random_caching_score += env.random_caching()
			epi_popularity_caching_score += env.popularity_caching()\

			# check if the episode is finished
			if done or (t == steps-1):
				print("Episode#%d, steps:%d, rewards:%f" % (epi, t, epi_reward))

				# save logs
				logger['episode'] = range(epi+1)
				logger['contents'].append(env.contents)
				logger['critics'].append(env.critics)
				logger['users'].append(env.users)

				# Draw score graph
				with proposed_caching_summary_writer.as_default():
					tf.summary.scalar('epi_score', epi_proposed_caching_score, step=epi)

				with popularity_caching_summary_writer.as_default():
					tf.summary.scalar('epi_score', epi_popularity_caching_score, step=epi)

				with random_caching_summary_writer.as_default():
					tf.summary.scalar('epi_score', epi_random_caching_score, step=epi)

				break

	# log saver
	dir_path = "%s/weights"%os.getcwd()
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)
	path = dir_path+'/'+'gym_ddpg_'
	pickle.dump(logger, open(path+'%s.pickle'%time.time(), 'wb'))

	env.close()


argparser = argparse.ArgumentParser(
	description='Train DDPG Agent on the openai gym')

argparser.add_argument(
	'-w',	'--weights',help='path to pretrained weights')


if __name__ == '__main__':
	#################################
	#   Parse Configurations
	#################################

	args = argparser.parse_args()
	weights_path = "/weights/"

	model_test(pretrained_=weights_path)