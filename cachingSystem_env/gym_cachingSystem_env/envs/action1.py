import gym
import numpy as np
import math
import tensorflow as tf
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import random
import itertools
import pyflann
from collections import deque

content_total_num = 80
mobile_num = 1000
request_time = 1
ramda = 0.8
cache_size = 40

content_feature = []

'''
content_num = np.array(range(content_total_num))
content_preference = ((1 / (content_num + 1)) ** ramda) / sum((1 / (content_num + 1)) ** ramda)
content_preference_CDF = content_preference.cumsum()
'''

content_num = np.array(range(content_total_num))
content_preference = np.ones(content_total_num, dtype=float)
content_preference /= content_total_num
content_preference_CDF = content_preference.cumsum()
print(content_preference_CDF)


class Action1Env(gym.Env):
    metadata = {'render.models': ['human']}

    def __init__(self):
        self.low = np.array([0] * (164)) # 41*4
        self.high = np.array([10000] * (164))
        self.action_low = 0
        self.action_high = cache_size
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([self.action_low]),
                                       high=np.array([self.action_high]), dtype=np.float32)
        #self.min = 0.6091102500000002
        #self.max = 0.8346647500000004
        #self.dif = self.max - self.min
        self.request_1000 = deque()
        self.sess = tf.compat.v1.Session()
        self.reward_network_output = self.reward_network()
        self.train_method()
        self.sess.run(tf.compat.v1.initialize_all_variables())
        self.train_count = 0
        self.flann = pyflann.FLANN()
        self.point = 1
        self.space = self.init_uniform_space(low=np.array([-1]), high=np.array([1]),
                                             points=np.array([self.action_high + 1]))

    def step(self, action):
        # print('action')
        # mapping_action = list(self.space).index(action)
        mapping_action = np.where(self.space == action)
        #mapping_action = random.randint(0, 40)
        # print('space',self.space)
        print('mapping action', int(mapping_action[0]))

        # mapping_action = int(action)
        print('request =>', self.request)
        print('cache => ', list(self.cache))
        b_cdf, b_pdf = self.beta()
        #self.request_list(self.request)


        if int(mapping_action[0]) != 0:
            delete_content = self.cache[int(mapping_action[0])-1]
            #recent = self.recent_request(delete_content)
            self.cache[int(mapping_action[0])-1] = self.request

        print('cache => ', list(self.cache))
        self.count += 1
        a_cdf, a_pdf = self.beta()

        pdf_sum = 0
        reward_sum = 0
        for i in self.cache:
            pdf_sum += a_pdf[i]
            reward_sum += float(self.make_reward(i))


        if int(mapping_action[0]) != 0:
            req_reward = self.make_reward(self.request)
            del_reward = self.make_reward(delete_content)
            print('request_reward', float(req_reward))
            print('delete_reward', float(del_reward))
            if float(req_reward) - float(del_reward) > 0:
                reward = 1 + reward_sum#pdf_sum
            else:
                reward = -2
        else:
            reward = 0

#        if int(mapping_action[0]) != 0:
#            if recent == 1:
#                reward = recent + pdf_sum
#            else:
#                reward = -2
#        else:
#            reward = 0
        '''
        if int(mapping_action[0]) != 0:
            if a_pdf[self.request] - b_pdf[delete_content] > 0:
                reward = 1 + pdf_sum#(pdf_sum-self.min)/(self.dif)
            else:
                reward = -2# + (pdf_sum-self.min)/(self.dif)
            #reward = a_pdf[self.request] - b_pdf[delete_content]
        else:
            reward = 0
        '''

        #self.request = self.make_random_request()
        self.request = self.preference_request()
        result = self.make_state()

        if self.count == 250: #cache_size * 2:
            return result, reward, True, pdf_sum  # random_pdf_sum
        return result, reward, False, 0

    def reset(self):
        self.count = 0
        self.cache = self.random_init()
        while 1:
            self.request_list(self.preference_request())
            if len(self.request_1000) >= 1000:
                break
              # tmp_cdf, tmp_pdf = self.beta()
        self.request = self.preference_request()
        self.hit_rate = 0
        self.state = self.make_state()

        self.train()

        return self.state

    def make_random_request(self):
        while 1:
            tmp = random.randint(0, (cache_size * 2) - 1)
            if tmp not in self.cache:
                break
        return tmp

    def random_init(self):
        cache = []
        while 1:
            tmp = random.randint(0, 79)  # content_total_num - 1)
            if tmp not in cache:
                cache.append(tmp)
                if len(cache) >= 40:
                    break
        return cache

    def make_state(self):
        a = []
        b = []
        c = []
        p = []
        a.append(content_feature[self.request][0])
        b.append(content_feature[self.request][1])
        c.append(content_feature[self.request][2])
        p.append(content_preference[self.request])
        for i in self.cache:
            a.append(content_feature[i][0])
            b.append(content_feature[i][1])
            c.append(content_feature[i][2])
            p.append(content_preference[i])
        return np.hstack([p, a, b, c])

    def beta(self):
        # change_pre = ((1 / (content_num + 1)) ** ramda) / sum((1 / (content_num + 1)) ** ramda)
        change_pre = copy.deepcopy(content_preference)
        sum_pre = 0
        change_pre_sum = 0

        for i in self.cache:
            if i != -1:
                change_pre[i] = self.feature(content_feature[i], change_pre[i])
                change_pre_sum += change_pre[i]
                sum_pre += content_preference[i]
        if sum_pre != 0:
            increse = change_pre_sum / sum_pre - 1
        else:
            increse = 1
        for j in range(content_total_num):
            if j not in self.cache:
                change_pre[j] = change_pre[j] + ((-1) * increse * change_pre[j]) / (1 - sum_pre) + increse * change_pre[
                    j]
        change_pre_CDF = change_pre.cumsum()

        return change_pre_CDF, change_pre

    count = 0
    out = False
    for _ in range(content_total_num):
        for i in range(5):
            for j in range(4):
                for k in range(4):
                    content_feature.append([i, j, k])
                    count += 1
                    if count == content_total_num:
                        out = True
                        break
                if out == True:
                    break
            if out == True:
                break
        if out == True:
            break

    def feature(self, f, preference):
        a = -1
        b = -1
        c = -1
        '''
        if f[:][0] == 0:
            a = 0.41  # 4.1
            # romance
        elif f[:][0] == 1:
            a = 0.72  # 5.2
            # action
        elif f[:][0] == 2:
            a = 0.23  # 2.3
            # comedy
        elif f[:][0] == 3:
            a = 0.34  # 3.4
            # horror
        elif f[:][0] == 4:
            a = 0.1  # 1.0
            # thriller
        if f[:][1] == 0:
            b = 0.301  # 3.01
            # FHD
        elif f[:][1] == 1:
            b = 0.702  # 4.02
            # QHD
        elif f[:][1] == 2:
            b = 0.203  # 2.03
        elif f[:][1] == 3:
            b = 0.104  # 1.04
        if f[:][2] == 0:
            c = 1.0001
            # 12
        elif f[:][2] == 1:
            c = 2.0002
        elif f[:][2] == 2:
            c = 9.0003
        elif f[:][2] == 3:
            c = 3.0004
            # 15
        '''
        if f[:][0] == 0:
            a = 0.0001  # 4.1
            # romance
        elif f[:][0] == 1:
            a = 0.0003  # 5.2
            # action
        elif f[:][0] == 2:
            a = 0.0005  # 2.3
            # comedy
        elif f[:][0] == 3:
            a = 0.0004  # 3.4
            # horror
        elif f[:][0] == 4:
            a = 0.0002  # 1.0
            # thriller
        if f[:][1] == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            b = 0.801  # 3.01
            # FHD
        elif f[:][1] == 1:
            b = 0.902  # 4.02
            # QHD
        elif f[:][1] == 2:
            b = 0.203  # 2.03
        elif f[:][1] == 3:
            b = 0.104  # 1.04
        if f[:][2] == 0:
            c = 0.11
            # 12
        elif f[:][2] == 1:
            c = 0.22
        elif f[:][2] == 2:
            c = 9.3
        elif f[:][2] == 3:
            c = 8.4
            # 15
        plus = (1 + (0.1 * a + 0.1 * b + 0.1 * c))
        preference *= plus

        return preference

    '''
    def request_generation(self,content_preference_CDF):
        while 1:
            prob = np.random.rand(1)
            for j in range(content_total_num):
                if j == 0 and prob <= content_preference_CDF[j]:
                    mobile_req = content_num[j]
                    break
                elif prob > content_preference_CDF[j - 1] and prob <= content_preference_CDF[j]:
                    mobile_req = content_num[j]
                    break
            if mobile_req not in self.cache:
                break
        return mobile_req
    '''

    def KNN(self, proto_action):  # ,cache):

        self._index = self.flann.build_index(self.space, algorithm='kmeans', branching=16)
        p_in = np.array([proto_action])
        search_res, _ = self.flann.nn_index(p_in, self.point)

        knns = self.space[search_res]
        p_out = []
        for p in knns:
            p_out.append(p)
        return np.array(p_out)

    def init_uniform_space(self, low, high, points):  # , cache):
        dims = len(low)
        points_in_each_axis = np.array([points[0]])

        axis = []

        for i in range(dims):
            axis.append(list(np.linspace(low[i], high[i], points_in_each_axis[i])))

        space = []
        for _ in itertools.product(*axis):
            space.append(list(_))

        return np.array(space)

    def preference_request(self):
        cdf, pdf = self.beta() # cdf, pdf 둘 다 배열임
        while 1:
            prob = np.random.rand(1)
            for j in range(content_total_num):
                if j == 0 and prob <= cdf[j]:
                    re = content_num[j]
                    break
                elif prob > cdf[j - 1] and prob <= cdf[j]:
                    re = content_num[j]
                    break
            self.request_list(re)
            if re not in self.cache:
                break
        return re

    def recent_request(self, content):
        tmp=[]

        cdf, pdf = self.beta()

        while 1:
            prob = np.random.rand(1)
            for j in range(content_total_num):
                if j == 0 and prob <= cdf[j]:
                    tmp.append(content_num[j])
                    break
                elif prob > cdf[j - 1] and prob <= cdf[j]:
                    tmp.append(content_num[j])
                    break
            if len(tmp) == 1000:
                break
        if tmp.count(content) >= 19:
        #if content in tmp:
            reward = -1
        else:
            reward = 1
        return reward

    def request_list(self, content):
        if len(self.request_1000) >= 1000:
            self.request_1000.popleft()
        self.request_1000.append(content)

    def reward_network(self):
        input_dim = 4
        output_dim = 1
        layer1_size = 128
        layer2_size = 128

        self.X = tf.compat.v1.placeholder("float", [None, input_dim])
        self.Y = tf.compat.v1.placeholder("float", [None, output_dim])


        W1 = tf.Variable(tf.random.uniform([input_dim, output_dim], -1.0, 1.0))
        b1 = tf.Variable(tf.random.uniform([output_dim], -1.0, 1.0))
        # W1 = tf.Variable(tf.random.uniform([input_dim, layer1_size], -1.0, 1.0))
        # b1 = tf.Variable(tf.random.uniform([layer1_size], -1.0, 1.0))
        # W2 = tf.Variable(tf.random.uniform([layer1_size, layer2_size], -1.0, 1.0))
        # b2 = tf.Variable(tf.random.uniform([layer2_size], -1.0, 1.0))
        # W3 = tf.Variable(tf.random.uniform([layer2_size, output_dim], -1.0, 1.0))
        # b3 = tf.Variable(tf.random.uniform([output_dim], -1.0, 1.0))

        # layer1 = tf.nn.leaky_relu(tf.matmul(self.X, W1) + b1)
        # layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)
        output = (tf.matmul(self.X, W1) + b1)
        #output = (tf.matmul(layer2, W3) + b3)

        return output
    def train_method(self):
        rate = tf.Variable(0.1)
        cost = tf.reduce_mean(input_tensor=tf.square(self.reward_network_output - self.Y))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(rate).minimize(cost)

    def train(self):
        self.y_data = []
        self.x_data = []

        for i in range(content_total_num):
            self.x_data.append(np.hstack([0.0125, content_feature[i][0], content_feature[i][1], content_feature[i][2]]))
            self.y_data.append(self.request_1000.count(i) * 0.001)

        for step in range(10001):
            self.sess.run(self.optimizer, feed_dict={self.X: self.x_data, self.Y: np.reshape(self.y_data,[80,1])})


    def make_reward(self, content):
        a = []
        b = []
        c = []
        p = []
        a.append(content_feature[content][0])
        b.append(content_feature[content][1])
        c.append(content_feature[content][2])
        p.append(content_preference[content])
        tmp = np.hstack([p, a, b, c])
        return self.sess.run(self.reward_network_output,feed_dict={self.X: np.reshape(tmp,[1,4])})