import gym
import numpy as np
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import random
import itertools
import pyflann
import tensorflow as tf
from collections import deque

content_total_num = 27
content_num = np.array(range(content_total_num))
cache_size = 9
request_list_size = 100

#make contents
# feature_total_num = 4
# genre_num = 3 #genre(romance, action, comedy)
# director_num = 3 #director(Steven_Spielberg, James_Cameron, Christopher_Nolan)
# actor_num = 3 #actor(Scarlett_Johansson, Brad_Pitt, Leonardo_DiCaprio)
#
# genre_list = np.random.randint(0, 2, (content_total_num, genre_num))
# director_list = np.array([])
# temp = np.eye(director_num)  # onehot
# for i in range(content_total_num):
#     director_list = np.append(director_list, temp[np.random.randint(director_num)])
# director_list = director_list.reshape(content_total_num, director_num)
# actor_list = np.random.randint(0, 2, (content_total_num, actor_num))
#
# content_feature = np.array([])
# for i in range(content_total_num):
#     temp = np.array([])
#     temp = np.append(temp, genre_list[i])
#     temp = np.append(temp, director_list[i])
#     temp = np.append(temp, actor_list[i])
#     content_feature = np.append(content_feature, temp, axis=0)
# content_feature = content_feature.reshape(content_total_num, len(temp))

content_feature = []
count = 0
out = False
for _ in range(content_total_num):
    for i in range(3):
        for j in range(3):
            for k in range(3):
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


class TestMarlEnv(gym.Env):
    metadata = {'render.models': ['human']}

    def __init__(self):
        self.low = np.array([0] * (4*2)) #(4*2) = (content_feature_num + preference)*(requested content + deleted content)
        self.high = np.array([content_total_num] * (4*2))
        self.action_low = 0
        self.action_high = 1
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([self.action_low]), high=np.array([self.action_high]), dtype=np.float32)
        self.request_list = deque()

        self.count = 0
        self.content_preference = np.ones(content_total_num)
        self.content_preference /= content_total_num
        self.cache = self.random_init()
        self.request= 0
        self.hit_ratio = 0
        self.short_hit_count = 0
        self.long_hit_count = 0
        self.delete_content, self.delete_content_index = self.select_delete_content()
        self.state = self.make_state()

    def step(self, action): #콘텐츠를 교체할지 말지가 한 스텝
        print('before cache => ', list(self.cache))
        print('request =>', self.request)
        print('min-preference content =>', self.delete_content)
        print('action', action)

        self.count += 1

        if action == 1:
            self.cache[self.delete_content_index] = self.request

            req_reward = self.content_preference[self.request]
            del_reward = self.content_preference[self.delete_content]
            print('request_reward ==> ', float(req_reward))
            print('delete_reward ==> ', float(del_reward))

            if (req_reward-del_reward) > 0:
                reward = 0.1 + self.hit_ratio
            else : reward = -1 + self.hit_ratio

        else: reward = self.hit_ratio

        print('after cache => ', list(self.cache))

        cdf = self.beta()
        #update reqeust and cache hit ratio
        while True:
            self.preference_request(cdf)
            self.add_request_list(self.request)

            if self.request in self.request_list:
                self.long_hit_count = self.request_list.count(self.request) #최근 100번의 요청 중 몇 개 히트했는지
            else : self.long_hit_count = 0

            if self.request in self.cache:
                self.short_hit_count = 1
                self.hit_ratio = self.short_hit_count + 0.2 * self.long_hit_count #short_hit_count에 더 가중치
                continue
            else:
                self.short_hit_count = 0
                self.hit_ratio = self.short_hit_count + 0.2 * self.long_hit_count
                break
        print('request list =>', self.request_list)
        print('next request =>', self.request)

        self.delete_content, self.delete_content_index = self.select_delete_content()
        next_state = self.make_state()

        pref_sum = 0
        for i in self.cache:
            pref_sum += self.content_preference[i]

        if self.count == (250):
            return next_state, reward, True, pref_sum

        return next_state, reward, False, 0


    def reset(self):
        self.count = 0
        self.content_preference = np.ones(content_total_num, dtype=float)
        self.content_preference /= content_total_num
        self.cache = self.random_init()
        self.request = random.randint(0, content_total_num - 1)
        self.hit_ratio = 0
        self.short_hit_count = 0
        self.long_hit_count = 0
        self.delete_content, self.delete_content_index = self.select_delete_content()
        self.state = self.make_state()

        return self.state


    def preference_request(self, cdf):
        #콘텐츠 선호도를 기반으로 사용자의 콘텐츠 요청 생성
        prob = np.random.rand(1) #[0, 1) 범위에서 균일한 분포에서  주어진 형태의 난수 어레이를 생성
        for j in range(content_total_num):
            if j == 0 and prob <= cdf[j]:
                self.request = content_num[j]
                break
            elif prob > cdf[j - 1] and prob <= cdf[j]:
                self.request = content_num[j]
                break


    def beta(self):
        # 캐시 안에 있는 게 추천이 될 확률에 따라 캐시된 콘텐츠 모두 선호도가 증가함
        #changed_pref : 추천이 됐을시 바뀌는 콘텐츠 선호도 증가량
        changed_pref_sum = 0
        pref_sum = 0
        changed_pref = self.content_preference.copy()
        for i in self.cache:
            changed_pref[i] = self.increased_preference(content_feature[i], self.content_preference[i])
            changed_pref_sum += changed_pref[i]
            pref_sum += self.content_preference[i]
        increase = changed_pref_sum / pref_sum - 1

        for j in range(content_total_num):
            if j not in self.cache:
                changed_pref[j] = changed_pref[j] + ((-1) * increase * changed_pref[j]) / (1-pref_sum) + increase * changed_pref[j]
        change_pref_CDF = changed_pref.cumsum() #사용자가 컨텐츠를 요청할 때 바뀐 선호도를 고려하기 위해 누적 분포함수 사용

        self.content_preference = changed_pref.copy()

        return change_pref_CDF


    def add_request_list(self, content):
        if len(self.request_list) >= request_list_size:
            self.request_list.popleft()
        self.request_list.append(content)


    def select_delete_content(self):
        cache_pre = []
        for i in self.cache:
            cache_pre.append(self.request_list.count(i))

        min_content_index = cache_pre.index(min(cache_pre))
        min_content = self.cache[min_content_index]

        return min_content, min_content_index


    def random_init(self):
        cache=[]
        while 1:
            tmp = random.randint(0, content_total_num - 1)
            if tmp not in cache:
                cache.append(tmp)
                if len(cache) >= cache_size:
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
        p.append(self.content_preference[self.request])

        a.append(content_feature[self.delete_content][0])
        b.append(content_feature[self.delete_content][1])
        c.append(content_feature[self.delete_content][2])
        p.append(self.content_preference[self.delete_content])

        return np.hstack([p, a, b, c])


    def increased_preference(self, content_faeture, preference):
        a = 0
        b = 0
        c = 0
        max = 18
        min = 3

        #콘텐츠 특성에 따른 추천이 될 확률, 추천이 됐을 때 증가하는 콘텐츠 선호도 설정
        if content_faeture[:][0] == 0:
            a = 6
            # romance
        elif content_faeture[:][0] == 1:
            a = 3
            # action
        elif content_faeture[:][0] == 2:
            a = 1
            # comedy

        if content_faeture[:][1] == 0:
            b = 6
            # QHD
        elif content_faeture[:][1] == 1:
            b = 3
            # FHD
        elif content_faeture[:][1] == 2:
            b = 1
            # HD

        if content_faeture[:][2] == 0:
            c = 6
            # 19
        elif content_faeture[:][2] == 1:
            c = 3
            # 15
        elif content_faeture[:][2] == 2:
            c = 1
            # 7

        plus = (1 + ((a + b + c)-min) / (max-min))
        preference *= plus

        return preference
