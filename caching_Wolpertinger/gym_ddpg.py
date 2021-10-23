#import filter_env
#from gym_foo.envs.knapsackddpg import KnapSackDDPGEnv
#from gym_foo.envs.caching import CachingEnv
#from gym_foo.envs.hitratio import HitratioEnv
#from gym_foo.envs.state import StateEnv
#from gym_foo.envs.hitratio_no_onehot import HitratioNoEnv
#from gym_foo.envs.preference_change import PreferenceEnv
#from gym_foo.envs.recommend_con import RecommendEnv
#from gym_foo.envs.recommand_ddpg import RecommendDDPGEnv
#from gym_foo.envs.same_preference import SameDDPGEnv
#from gym_foo.envs.same_preference_wp import SameWPEnv

import gym
#from gym_foo.envs.action1 import Action1Env
from gym_foo.envs.test_marl_env import TestMarlEnv


from ddpg import *
import gc
import matplotlib.pyplot as plt
gc.enable()

#ENV_NAME = 'Pendulum-v0'
EPISODES = 1000 #10000
TEST = 20
STEP_SIZE = 250


load = 0

history_reward = []
history_last_reward = []
ave_reward = []

p_hold1 = tf.compat.v1.placeholder(dtype=tf.float32,name='p_hold1')
p_hold2 = tf.compat.v1.placeholder(dtype=tf.float32,name='p_hold2')
p_hold3 = tf.compat.v1.placeholder(dtype=tf.float32,name='p_hold3')


a_r_s = p_hold1
t_r_s = p_hold2
l_r_s = p_hold3

average_reward_scalar = tf.compat.v1.summary.scalar('average reward',a_r_s)
total_reward_scalar = tf.compat.v1.summary.scalar('total reward',t_r_s)
last_reward_scalar = tf.compat.v1.summary.scalar('last reward',l_r_s)

writer = tf.compat.v1.summary.FileWriter("./log")
sess = tf.compat.v1.Session()
writer.add_graph(sess.graph)
merge = tf.compat.v1.summary.merge([average_reward_scalar, total_reward_scalar, last_reward_scalar])
writer.flush()

#saver = tf.train.Saver()


def main():
    #env = gym.make('foo-Action1-v0')
    env = gym.make('foo-TestMarl-v0')
    agent = DDPG(env)
    last_cache_hit_count = 0
    for episode in range(EPISODES):
        state = env.reset()


        # Train
        t_reward = 0
        for step in range(STEP_SIZE):
        #step = 0
        #while 1:
            #out = env.BS()
            #if out:
            #    break
            print('EPISODE: {} STEP: {}'.format(episode, step))
            #print('state p:', state[0], state[1:6])
            #print('state p:', state[0], list(state[1:41]))
            #print('state 1:', state[41] ,list(state[42:82]))
            #print('state 2:', state[82], list(state[83:123]))
            #print('state 3:', state[123] ,list(state[124:164]))
            #print('state :', list(state))
            proto_action = agent.noise_action(state)
            action = agent.wolp_action(state, proto_action)
            print('proto action : ',proto_action)
            print('action  : ',action)
            next_state, reward, done, cache_hit = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state

            t_reward += reward
            print('reward :',reward)
            print('\n')
            #step+=1
            #if done:
            #if step >= STEP_SIZE-1:

             #   break
        last_cache_hit = cache_hit
        a_reward = t_reward/STEP_SIZE

        print('episode:', episode, ' Total Reward:', t_reward ,' Average Reward:', a_reward, 'Last cache hit:', last_cache_hit)
        print('\n')
        _, _, _, summary = sess.run([a_r_s, t_r_s, l_r_s, merge],  # h01234_s,
                                    feed_dict={p_hold1: a_reward, p_hold2: t_reward, p_hold3: last_cache_hit})  # p_hold2: h,
        writer.add_summary(summary, episode) # tensorboard 에 요약 내용 기록하기
        if (episode+1) % 1000 == 0 :
            pass
            #agent.actor_network.save_network(episode)
            #agent.critic_network.save_network(episode)
        # Testing:
        if last_cache_hit >= 7.3:
            last_cache_hit_count += 1


        #if load ==1:
        #   saver.save(sess, "./model/my_test_model.ckpt")
        if last_cache_hit_count > 100 or episode >= EPISODES-1:

            for i in range(TEST):
                state = env.reset()
                total_reward = 0
                for step in range(STEP_SIZE):
                    #env.render()
                    print("Test   EPISODE: {}   STEP: {}".format(i,step))
                    #print('state p:', state[0], state[1:6])
                    print('state p:', state[41], list(state[42:82]))

                    proto_action = agent.action(state)
                    action = agent.wolp_action(state, proto_action)
                    print('action  : ', action)
                     # direct action for test
                    next_state, reward, done, cache_hit = env.step(action)
                    #agent.perceive(state, action, reward, next_state, done)
                    total_reward += reward
                    print('reward :', reward)
                    print('\n')
                    state = next_state


                ave_reward.append(total_reward/step)
                history_reward.append(total_reward)
                history_last_reward.append(cache_hit)
                print('Test: {}    Last Hit: {}'.format(i, cache_hit))
            break

        #print('state        =>', env.reverse_show(state))

    print('Evaluation Average Reward: {}   Last Hit: {} '.format(ave_reward , history_last_reward))
    plt.plot(np.array(range(TEST)), history_last_reward, 'b', label='Agent')
    plt.title('Test last cache hit')
    plt.ylabel('hit ratio')
    plt.xlabel('episode')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    #env.monitor.close()
    plt.plot(np.array(range(TEST)), history_reward,'b', label='Agent')
    plt.title('Test total reward')
    plt.ylabel('hit ratio')
    plt.xlabel('episode')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()