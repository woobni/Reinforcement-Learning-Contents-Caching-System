import gym
import numpy as np
import math
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import random
import itertools
import pyflann
import tensorflow as tf
from collections import deque

content_total_num = 80
cache_size = 40
ramda = 0.8

content_feature = []

content_num = np.array(range(content_total_num))
content_preference = np.ones(content_total_num, dtype=float)
content_preference /= content_total_num
content_preference_CDF = content_preference.cumsum()

print(content_preference_CDF)


class SelectSameEnv(gym.Env):
    metadata = {'render.models': ['human']}

    def __init__(self):
        self.low = np.array([0] * ((4)*2))
        self.high = np.array([100] * ((4)*2))
        self.action_low = 0
        self.action_high = 1
        #self.cache = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]#[5, 6, 7, 8, 9]
        #self.high_content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]#[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78]
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([self.action_low]), high=np.array([self.action_high]),
                                       dtype=np.float32)
        self.weight = 0.01
        self.total_reward = 0
        self.count = 0
        self.hit_rate = 0
        self.request = 0
        self.sess = tf.compat.v1.Session()
        self.reward_network_output = self.reward_network()
        self.train_method()
        self.sess.run(tf.compat.v1.initialize_all_variables())
        self.request_1000 = deque()


    def step(self, action):
        print('action',action)
        print('delete content =>', self.delete_content)
        print('request =>', self.request)
        print('cache => ', list(self.cache))

        b_cdf, b_pdf = self.beta()
        if action != 0:
            self.cache[self.delete_content_index] = self.request

        print('cache => ', list(self.cache))
        self.count += 1

        a_pdf_sum = 0
        a_cdf, a_pdf = self.beta()
        reward_sum = 0
        for i in self.cache:
            a_pdf_sum += a_pdf[i]
            reward_sum += float(self.make_reward(i))

        if action != 0:
            req_reward = self.make_reward(self.request)
            del_reward = self.make_reward(self.delete_content)
            print('request_reward', float(req_reward))
            print('delete_reward', float(del_reward))
            reward = float(req_reward) - float(del_reward)
        else:
            reward = 0

        #if action != 0:
        #    reward = a_pdf[self.request] - b_pdf[self.delete_content]
        #else:
        #    reward = 0

        self.delete_content, self.delete_content_index = self.select_delete_count()
        #self.request = self.make_random_request()
        self.request = self.preference_request()
        print('delete content =>', self.delete_content)
        print('next request =>', self.request)
        result = self.make_state()

        if self.count == (250):#content_total_num * 2):
            return result, reward, True, a_pdf_sum #reward*0.1 #pdf_sum
        return result, reward, False, 0

    def reset(self):
        self.count = 0
        self.cache = self.random_init()#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]#[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]#self.random_init()#[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]#[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]#[5, 6, 7, 8, 9]
        while 1:
            self.request_list(self.preference_request())
            if len(self.request_1000) >= 1000:
                break
        self.request = self.preference_request()
        #self.request = self.make_random_request()
        change_pdf, change_cdf = self.beta()
        self.delete_content, self.delete_content_index = self.select_delete_count()
        self.hit_rate = 0
        self.state = self.make_state()
        self.train()
        return self.state

    def preference_request(self):
        cdf, pdf = self.beta()
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

    def request_list(self, content):
        if len(self.request_1000) >= 1000:
            self.request_1000.popleft()
        self.request_1000.append(content)

    def select_delete(self,pre):
        cache_pre = []
        for i in self.cache:
            cache_pre.append(pre[i])

        min_con_index = cache_pre.index(min(cache_pre)) #np.where(cache_pre == min(cache_pre))
        min_con = self.cache[min_con_index]
        return min_con, min_con_index

    def select_delete_count(self):
        cache_pre = []
        for i in self.cache:
            cache_pre.append(self.request_1000.count(i))

        min_con_index = cache_pre.index(min(cache_pre))  # np.where(cache_pre == min(cache_pre))
        min_con = self.cache[min_con_index]
        return min_con, min_con_index


    def make_random_request(self):
        #if self.count <= (cache_size-1):
        #    return self.count
        #else:
        while 1:
            tmp = random.randint(0, 79)#content_total_num - 1)
            if tmp not in self.cache:
                break
        return tmp
    def random_init(self):
        cache=[]
        while 1:
            tmp = random.randint(0, 79)#content_total_num - 1)
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

        a.append(content_feature[self.delete_content][0])
        b.append(content_feature[self.delete_content][1])
        c.append(content_feature[self.delete_content][2])
        p.append(content_preference[self.delete_content])
        return np.hstack([p, a, b, c])


    def beta(self):
        change_pre = copy.deepcopy(content_preference)#((1 / (content_num + 1)) ** ramda) / sum((1 / (content_num + 1)) ** ramda)
        sum_pre = 0
        change_pre_sum = 0

        for i in self.cache:
            if i != -1: # cache에 아무것도 없을 때
                change_pre[i] = self.feature(content_feature[i], change_pre[i])
                change_pre_sum += change_pre[i] # 바뀌기 후 선호도 (캐시에 있는 것들)
                sum_pre += content_preference[i] # 바뀌기 전 선호도 (캐시에 있는 것들)
        if sum_pre != 0:
            increase = change_pre_sum/sum_pre - 1
        else:
            increase = 1
        for j in range(content_total_num):
            if j not in self.cache:
                change_pre[j] = change_pre[j] + ((-1)*increase*change_pre[j])/(1-sum_pre) + increase * change_pre[j]
        change_pre_CDF = change_pre.cumsum()
        # 사용자가 컨텐츠를 요청할 때 바뀐 선호도를 고려하기 위해 누적 분포함수 사용
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
                        out=True
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
        plus = (1 + (0.1 * a + 0.1 * b + 0.1 * c))
        preference *= plus


        return preference

    def reward_network(self):
        input_dim = 4
        output_dim = 1
        layer1_size = 64
        layer2_size = 64

        self.X = tf.compat.v1.placeholder("float", [None, input_dim])
        self.Y = tf.compat.v1.placeholder("float", [None, output_dim])

        W1 = tf.Variable(tf.random.uniform([input_dim, layer1_size], -1.0, 1.0))
        b1 = tf.Variable(tf.random.uniform([layer1_size], -1.0, 1.0))
        W2 = tf.Variable(tf.random.uniform([layer1_size, layer2_size], -1.0, 1.0))
        b2 = tf.Variable(tf.random.uniform([layer2_size], -1.0, 1.0))
        W3 = tf.Variable(tf.random.uniform([layer2_size, output_dim], -1.0, 1.0))
        b3 = tf.Variable(tf.random.uniform([output_dim], -1.0, 1.0))

        layer1 = tf.nn.leaky_relu(tf.matmul(self.X, W1) + b1)
        layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)
        output = tf.matmul(layer2, W3) + b3

        #output = (tf.matmul(self.X, W1) + b1)

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