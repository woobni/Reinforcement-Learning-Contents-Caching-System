import numpy as np
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import dqn
from collections import deque
import gym
#from gym_foo.envs.helper import helperEnv
#from gym_foo.envs.select_content import SelectEnv
from gym_foo.envs.select_same import SelectSameEnv

dis = 0.99
REPLAY_MEMORY = 100000000

max_episodes = 10000
TEST = 10
STEP_SIZE = 250
replay_buffer= deque()

total_reward = 0
all_reward = []

env = gym.make('foo-SelectSame-v0')

input_size = env.observation_space.shape[0]
output_size = 2 #env.action_space.shape[0]

all_loss=[]
history_agent_h=[]


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def main():

    p_hold1 = tf.compat.v1.placeholder(dtype=tf.float32,name='p_hold1')
    p_hold2 = tf.compat.v1.placeholder(dtype=tf.float32, name='p_hold2')
    p_hold3 = tf.compat.v1.placeholder(dtype=tf.float32, name='p_hold3')

    r_s = p_hold1
    h_s = p_hold2
    d_s = p_hold3

    with tf.compat.v1.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main_h1")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target_h1")
        tf.compat.v1.global_variables_initializer().run()

        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state('model')
        # saver.restore(sess, ckpt.model_checkpoint_path)
        reward_scalar = tf.compat.v1.summary.scalar('total reward', r_s)
        hitratio_scalar = tf.compat.v1.summary.scalar('hit ratio', h_s)
        difference_scalar = tf.compat.v1.summary.scalar('average reward', d_s)
        writer = tf.compat.v1.summary.FileWriter("./log")
        writer.add_graph(sess.graph)
        merge = tf.compat.v1.summary.merge([reward_scalar, hitratio_scalar, difference_scalar])
        writer.flush()

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        sess.run(copy_ops)
        for episode in range(max_episodes):  # episode
            e = 1. / ((episode / 10) + 1)
            state = env.reset()
            total_reward = 0

            #while 1:
            for step in range(STEP_SIZE):
                print('EPISODE: {} STEP: {}'.format(episode, step))
                if np.random.rand(1) < e:
                    action = random.choice([0,1])
                else:
                    a = mainDQN.predict(state)
                    action = np.argmax(a)

                next_state, reward, done, agent_h = env.step(action)

                replay_buffer.append((state, action, reward, next_state, done))

                print('state        =>', list(state))
                print('next_state   =>', list(next_state))

                state = next_state

                total_reward += reward
                print('reward :', reward)
                print('\n')

            _, _, _, summary = sess.run([r_s, h_s, d_s, merge], feed_dict={p_hold1:total_reward, p_hold2 : agent_h, p_hold3 : total_reward/STEP_SIZE})
            writer.add_summary(summary,episode)

            print("Episode: {} ".format(episode))
            print("total_reward :", total_reward, "hit raito :", agent_h)

            print("\n")

            if episode % 10 == 9:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 32)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss h1: ", loss)
                all_loss.append(loss)

                sess.run(copy_ops)
            if episode % 1000 == 999:
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, "./model/my_test_model.ckpt")

        for episode in range(TEST):
            state = env.reset()

            total_reward = 0
            #while 1:
            for step in range(STEP_SIZE):
                print('EPISODE: {} STEP: {}'.format(episode, step))
                a = mainDQN.predict(state)
                action = np.argmax(a)

                next_state, reward, done, agent_h = env.step(action)

                print('state        =>', list(state))
                print('next_state   =>', list(next_state))

                state = next_state

                total_reward += reward
                print('reward :', reward)
                print('\n')

            history_agent_h.append(agent_h)
            print("TEST : {} Total_reward : {} hit ratio : {}".format(episode, total_reward, agent_h))
            print("\n")

        plt.plot(np.array(range(TEST)), history_agent_h, 'b', label='Agent')
        plt.title('Test last cache hit')
        plt.ylabel('hit ratio')
        plt.xlabel('episode')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()
        plt.plot(np.array(range(int(max_episodes/10))), all_loss, 'b', label='Agent')
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('episode')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()