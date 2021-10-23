import gym
import numpy as np, time, os
from tqdm import tqdm
import datetime
import tensorflow as tf
from gym_foo.envs.continuous_cache import ContinuousCache

NUM_EPISODES_ = 500

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test1_log_dir = 'logs/train/' + current_time + '/test1'
test1_summary_writer = tf.summary.create_file_writer(test1_log_dir)

test2_log_dir = 'logs/train/' + current_time + '/test2'
test2_summary_writer = tf.summary.create_file_writer(test2_log_dir)

test3_log_dir = 'logs/train/' + current_time + '/test3'
test3_summary_writer = tf.summary.create_file_writer(test3_log_dir)

test4_log_dir = 'logs/train/' + current_time + '/test4'
test4_summary_writer = tf.summary.create_file_writer(test4_log_dir)

test5_log_dir = 'logs/train/' + current_time + '/test5'
test5_summary_writer = tf.summary.create_file_writer(test5_log_dir)


def model_train():
    # Create Environments
    env = gym.make('foo-ContinuousCache-v0')

    # Initialize Environments
    steps = env.step_size # steps per episode

    for epi in range(NUM_EPISODES_):
        print("=========EPISODE # %d =========="%epi)
        print('contents => ', env.contents)
        print('users => ', env.users)

        test1_score = 0
        test2_score = 0
        test3_score = 0
        test4_score = 0
        test5_score = 0
        for _ in tqdm(range(steps)):
            test1_score = 0.9 * test1_score + 0.1 * env.test1()
            # test2_score = 0.9 * test2_score + 0.1 * env.test2()
            # test3_score = 0.9 * test3_score + 0.1 * env.test3()
            # test4_score = 0.9 * test4_score + 0.1 * env.test4()
            # test5_score = 0.9 * test5_score + 0.1 * env.test5()

        with test1_summary_writer.as_default():
            tf.summary.scalar('epi_score', test1_score, step=epi)

        # with test2_summary_writer.as_default():
        #     tf.summary.scalar('epi_score', test2_score, step=epi)
        #
        # with test3_summary_writer.as_default():
        #     tf.summary.scalar('epi_score', test3_score, step=epi)
        #
        # with test4_summary_writer.as_default():
        #     tf.summary.scalar('epi_score', test4_score, step=epi)
        #
        # with test5_summary_writer.as_default():
        #     tf.summary.scalar('epi_score', test5_score, step=epi)


if __name__ == '__main__':
    model_train()