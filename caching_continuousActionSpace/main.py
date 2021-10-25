import gym
import time
import os
from tqdm import tqdm
import datetime
import pickle
import argparse

import tensorflow as tf
from ddpg import ddpgAgent
from gym_cachingSystem_env.envs.continuous_cache import ContinuousCache


NUM_EPISODES_ = 500

# Make directories for tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
proposed_caching_log_dir = 'logs/train/' + current_time + '/proposed_caching'
proposed_caching_summary_writer = tf.summary.create_file_writer(proposed_caching_log_dir)

popularity_caching_log_dir = 'logs/train/' + current_time + '/popularity_caching'
popularity_caching_summary_writer = tf.summary.create_file_writer(popularity_caching_log_dir)

random_caching_log_dir = 'logs/train/' + current_time + '/random_caching'
random_caching_summary_writer = tf.summary.create_file_writer(random_caching_log_dir)

optimal_caching_log_dir = 'logs/train/' + current_time + '/optimal_caching'
optimal_caching_summary_writer = tf.summary.create_file_writer(optimal_caching_log_dir)


def model_train():
    # Create Environments
    env = gym.make('ContinuousCache-v0')

    # Create Agent model
    agent = ddpgAgent(env, batch_size=128, w_per=False)

    # Initialize Environments
    steps = env.step_size  # steps per episode
    num_act_ = env.action_space.shape[0]
    num_obs_ = env.observation_space.shape[0]
    print("============ENVIRONMENT===============")
    print("num_of_action_spaces : %d"%num_act_)
    print("num_of_observation_spaces: %d"%num_obs_)
    print("max_steps_per_episode: %d"%steps)

    logger = dict(episode=[], contents=[], critics=[], users=[])

    popularity_caching_score_list = []
    optimal_caching_score_list = []
    max_reward = 0
    for epi in range(NUM_EPISODES_):
        print("=========EPISODE # %d =========="%epi)

        obs = env.reset()
        # print('contents => ', env.contents)

        epi_reward = 0
        epi_proposed_caching_score = 0
        epi_popularity_caching_score = 0
        epi_random_caching_score = 0
        epi_optimal_caching_score = 0

        popularity_caching_score_mean = 0
        optimal_caching_score_mean = 0
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

            epi_proposed_caching_score += reward
            epi_random_caching_score += env.random_caching()
            epi_popularity_caching_score += env.popularity_caching()

            # moving average score
            # epi_proposed_caching_score = 0.9 * epi_proposed_caching_score + 0.1 * env.proposed_caching_score
            # epi_random_caching_score = 0.9 * epi_random_caching_score + 0.1 * env.random_caching_score
            # epi_popularity_caching_score = 0.9 * epi_popularity_caching_score + 0.1 * env.popularity_caching_score
            # epi_optimal_caching_score = 0.9 * epi_optimal_caching_score + 0.1 * env.optimal_caching()

            # train models through memory replay
            agent.replay(1)

            # check if the episode is finished
            if done or (t == steps-1):
                print("Episode#%d, steps:%d, rewards:%f" % (epi, t, epi_reward))

                # save weights at the new records performance
                if epi_reward >= max_reward:
                    max_reward = epi_reward
                    dir_path = "%s/weights"%os.getcwd()
                    if not os.path.isdir(dir_path):
                        os.mkdir(dir_path)
                    path = dir_path+'/'+'gym_ddpg_'
                    agent.save_weights(path + 'ep%d_%f' % (epi, max_reward))

                # save logs
                logger['episode'] = range(epi+1)
                logger['contents'].append(env.contents)
                logger['critics'].append(env.critics)
                logger['users'].append(env.users)
                # logger['critic_loss'].append(agent.critic.critic_loss)

                # Find the average score when the user doesn't change during the episode.
                # popularity_caching_score_list.append(epi_popularity_caching_score)
                # optimal_caching_score_list.append(epi_optimal_caching_score)
                # popularity_caching_score_mean = sum(popularity_caching_score_list) / len(popularity_caching_score_list)
                # optimal_caching_score_mean = sum(optimal_caching_score_list) / len(optimal_caching_score_list)

                # Draw score graph
                with proposed_caching_summary_writer.as_default():
                    tf.summary.scalar('epi_score', epi_proposed_caching_score, step=epi)

                with popularity_caching_summary_writer.as_default():
                    # tf.summary.scalar('epi_score', popularity_caching_score_mean, step=epi)
                    tf.summary.scalar('epi_score', epi_popularity_caching_score, step=epi)

                with random_caching_summary_writer.as_default():
                    tf.summary.scalar('epi_score', epi_random_caching_score, step=epi)

                # with optimal_caching_summary_writer.as_default():
                #     # tf.summary.scalar('epi_score', optimal_caching_score_mean, step=epi)
                #     tf.summary.scalar('epi_score', epi_optimal_caching_score, step=epi)

                break

    # weight saver
    dir_path = "%s/weights"%os.getcwd()
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    path = dir_path+'/'+'gym_ddpg_'
    agent.save_weights(path +'lastest')

    # log saver
    pickle.dump(logger, open(path+'%s.pickle'%time.time(), 'wb'))

    env.close()


if __name__ == '__main__':
    model_train()