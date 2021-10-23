import gym
import numpy as np, time, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import datetime
import tensorflow as tf

from ddpg import ddpgAgent
from gym_foo.envs.continuous_cache import ContinuousCache

NUM_EPISODES_ = 200 #10000

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
proposed_caching_log_dir = 'logs/train/' + current_time + '/proposed_caching'
proposed_caching_summary_writer = tf.summary.create_file_writer(proposed_caching_log_dir)

popularity_caching_log_dir = 'logs/train/' + current_time + '/popularity_caching'
popularity_caching_summary_writer = tf.summary.create_file_writer(popularity_caching_log_dir)

random_caching_log_dir = 'logs/train/' + current_time + '/random_caching'
random_caching_summary_writer = tf.summary.create_file_writer(random_caching_log_dir)

optimal_caching_log_dir = 'logs/train/' + current_time + '/optimal_caching'
optimal_caching_summary_writer = tf.summary.create_file_writer(optimal_caching_log_dir)

test1_log_dir = 'logs/train/' + current_time + '/test1'
test1_summary_writer = tf.summary.create_file_writer(test1_log_dir)

test2_log_dir = 'logs/train/' + current_time + '/test2'
test2_summary_writer = tf.summary.create_file_writer(test2_log_dir)

test3_log_dir = 'logs/train/' + current_time + '/test3'
test3_summary_writer = tf.summary.create_file_writer(test3_log_dir)

def model_train():
    # Create Environments
    env = gym.make('foo-ContinuousCache-v0')

    # Create Agent model
    agent = ddpgAgent(env, batch_size=128, w_per=False)

    # Initialize Environments
    steps = env.step_size # steps per episode
    num_act_ = env.action_space.shape[0]
    num_obs_ = env.observation_space.shape[0]
    print("============ENVIRONMENT===============")
    print("num_of_action_spaces : %d"%num_act_)
    print("num_of_observation_spaces: %d"%num_obs_)
    print("max_steps_per_episode: %d"%steps)

    logger = dict(episode=[], reward=[], critic_loss=[], epi_proposed_caching_score=[],
                  epi_popularity_caching_score=[], epi_random_caching_score=[],
                  epi_optimal_caching_score=[])

    popularity_caching_score_list = []
    optimal_caching_score_list = []
    max_reward = 0
    for epi in range(NUM_EPISODES_):
        print("=========EPISODE # %d =========="%epi)

        obs = env.reset()
        print('contents => ', env.contents)

        epi_reward = 0
        epi_proposed_caching_score = 0
        epi_popularity_caching_score = 0
        epi_random_caching_score = 0
        epi_optimal_caching_score = 0

        popularity_caching_score_mean = 0
        optimal_caching_score_mean = 0

        test1_score = 0
        test2_score = 0
        test3_score = 0
        for t in tqdm(range(steps)):
            # Make action from the current policy
            proto_action = agent.make_action(obs, t)

            # do step on gym at t-time
            new_obs, reward, done, info = env.step(proto_action)

            # store the results to buffer
            agent.memorize(obs, proto_action, reward, done, new_obs)

            # grace finish and go to t+1 time
            obs = new_obs
            epi_reward = epi_reward + reward

            epi_proposed_caching_score = 0.9 * epi_proposed_caching_score + 0.1 * env.recommendation_rating_sum()
            epi_popularity_caching_score = env.popularity_caching()
            epi_random_caching_score = 0.9 * epi_random_caching_score + 0.1 * env.random_caching()
            epi_optimal_caching_score = env.optimal_caching()

            test1_score = 0.9 * test1_score + 0.1 * env.test1()
            test2_score = 0.9 * test2_score + 0.1 * env.test2()
            test3_score = 0.9 * test3_score + 0.1 * env.test3()

            agent.replay(1)

            # check if the episode is finished
            if done or (t == steps-1):
                print("Episode#%d, steps:%d, rewards:%f"%(epi, t, epi_reward))

                # save weights at the new records performance
                if epi_reward >= max_reward:
                    max_reward = epi_reward
                    dir_path = "%s/weights"%os.getcwd()
                    if not os.path.isdir(dir_path):
                        os.mkdir(dir_path)
                    path = dir_path+'/'+'gym_ddpg_'
                    agent.save_weights(path + 'ep%d_%f'%(epi,max_reward))


                # save reward logs
                logger['episode'] = range(epi+1)
                logger['reward'].append(epi_reward)
                logger['critic_loss'].append(agent.critic.critic_loss)
                logger['epi_proposed_caching_score'].append(epi_proposed_caching_score)
                logger['epi_popularity_caching_score'].append(epi_popularity_caching_score)
                logger['epi_random_caching_score'].append(epi_random_caching_score)
                logger['epi_optimal_caching_score'].append(epi_optimal_caching_score)

                popularity_caching_score_list.append(epi_popularity_caching_score)
                optimal_caching_score_list.append(epi_optimal_caching_score)

                popularity_caching_score_mean = sum(popularity_caching_score_list) / len(popularity_caching_score_list)
                optimal_caching_score_mean = sum(optimal_caching_score_list) / len(optimal_caching_score_list)

                break

        with proposed_caching_summary_writer.as_default():
            tf.summary.scalar('epi_score', epi_proposed_caching_score, step=epi)

        with popularity_caching_summary_writer.as_default():
            tf.summary.scalar('epi_score', popularity_caching_score_mean, step=epi)

        with random_caching_summary_writer.as_default():
            tf.summary.scalar('epi_score', epi_random_caching_score, step=epi)

        with optimal_caching_summary_writer.as_default():
            tf.summary.scalar('epi_score', optimal_caching_score_mean, step=epi)

        with test1_summary_writer.as_default():
            tf.summary.scalar('epi_score', test1_score, step=epi)

        with test2_summary_writer.as_default():
            tf.summary.scalar('epi_score', test2_score, step=epi)

        with test3_summary_writer.as_default():
            tf.summary.scalar('epi_score', test3_score, step=epi)
            
    # weight saver
    dir_path = "%s/weights"%os.getcwd()
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    path = dir_path+'/'+'gym_ddpg_'
    agent.save_weights(path +'lastest')
    env.close()

    # log saver
    import pickle
    pickle.dump(logger, open(path+'%s.pickle'%time.time(), 'wb'))


if __name__ == '__main__':
    model_train()