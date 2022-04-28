######################
# python==3.7.9
# tensorflow==2.1
# keras==2.3.1
# cuda==10.1.2
# cudnn==10.1
######################
import numpy as np
import copy
import gym
from gym import spaces


class ContinuousCache(gym.Env):
    def __init__(self):
        self.step_size = 100
        self.contents_total_num = 500
        self.cache_size = 10
        self.cache = self.cache_init_random()
        self.max_rating = 5  # max ratings that factors give to contents.
        self.min_rating = 0  # min ratings that factors give to contents.
        self.factors_num = 5
        self.factors = self.generate_factors()
        self.contents = self.generate_contents()
        self.users_num = 4
        self.users, self.users_preference = self.update_users()
        self.recommendation_list_size = 5  # number of contents recommended to users within cache.
        self.recommendation_list = self.update_recommendation_list()
        self.state = self.update_state()
        self.min_action = 0.0
        self.max_action = 1.0
        self.proposed_caching_score = 0
        self.popularity_caching_score = 0
        self.random_caching_score = 0

        # The state is all the contents on the recommendation list of each users.
        self.low_state = np.array(
            [self.min_rating] * (self.factors_num+1) * self.recommendation_list_size * self.users_num,
            dtype=np.float32)  # (self.factors_num+1) => contents feature and contents expected rating.
        self.high_state = np.array(
            [self.max_rating] * (self.factors_num+1) * self.recommendation_list_size * self.users_num,
            dtype=np.float32)

        # The state is the top content on the recommendation list of each users.
        # self.low_state = np.array(
        #     [self.min_rating] * (self.factors_num+1) * self.users_num,
        #     dtype=np.float32)  # (self.factors_num+1) => contents feature and contents expected rating.
        # self.high_state = np.array(
        #     [self.max_rating] * (self.factors_num+1) * self.users_num,
        #     dtype=np.float32)

        # The state when we know contents factor preferences of users from recommendation system.
        # self.low_state = np.array(
        #     [self.min_rating] * self.factors_num * self.users_num,
        #     dtype=np.float32)  # (self.factors_num+1) => contents feature and contents expected rating.
        # self.high_state = np.array(
        #     [self.max_rating] * self.factors_num * self.users_num,
        #     dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.cache_size*self.factors_num,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

    def generate_factors(self):
        # Each factor has its own preference about contents feature(e.g. genre, director, actor, etc.).
        # Each factor gives a rating for each content.
        factors = []
        for _ in range(self.factors_num-2):
            ratings = []
            for _ in range(self.contents_total_num):
                # rating = np.random.uniform(self.min_rating, self.max_rating)
                rating = np.random.normal(2.5, 1)  # mu = 2.5, sd = 1
                if rating > 5.0:
                    rating = 5.0
                elif rating < 0.0:
                    rating = 0.0
                rating = np.round(rating, 2)
                ratings.append(rating)
            factors.append(ratings)

        # Generate factors who have an opposite preference to the factors created earlier.
        for i in range(2):
            opposite_ratings = []
            for j in range(self.contents_total_num):
                opposite_rating = self.max_rating - factors[i+1][j]
                opposite_rating = np.round(opposite_rating, 2)
                opposite_ratings.append(opposite_rating)
            factors.append(opposite_ratings)

        return factors

    def update_users(self):
        # feature of users with expected ratings.
        users = []
        preferences = []
        # preferences = [1, 2]
        for i in range(self.users_num):
            # Suppose the users continues to change. (Random factor preference)
            preference = np.random.randint(1, self.factors_num)
            preferences.append(preference)

            # Suppose the users continues to remain unchanged. (Fix factor preference)
            preference = preferences[i]

            # first factor has universal content preference. so, fix to
            expected_ratings = [0.5*x + 0.5*y for x, y in zip(self.factors[0], self.factors[preference])]
            # expected_ratings = self.factors[preference]
            expected_ratings = list(np.round(expected_ratings, 2))
            users.append(expected_ratings)

        return users, preferences

    def generate_contents(self):
        # Each content is placed in a continuous space consisting of factors' ratings dimension.
        contents = []
        for i in range(self.contents_total_num):
            content = []
            for j in range(self.factors_num):
                content.append(self.factors[j][i])
            contents.append(content)

        return contents

    def update_recommendation_list(self):
        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in self.cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else: continue
            recommendation_list.append(top_rating_contents)

        return recommendation_list

    def cache_init_random(self):
        cache = []
        while 1:
            content_index = np.random.randint(0, self.contents_total_num)
            if content_index not in cache:
                if len(cache) == self.cache_size: break
                cache.append(content_index)

        return cache

    # def recommendation_count_sum(self):
    #     # Count how much cached contents were recommended to users.
    #     recommendation_list = []
    #     count_sum = 0
    #     for i in range(self.users_num):
    #         recommendation_list = recommendation_list + self.recommendation_list[i]
    #     for i in range(self.cache_size):
    #         count_sum = count_sum + recommendation_list.count(self.cache[i])
    #
    #     return count_sum

    def recommendation_rating_sum(self):
        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if self.cache[j] in self.recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][self.cache[j]]
                else: continue

        return rating_sum

    def reset(self):
        self.factors = self.generate_factors()
        self.contents = self.generate_contents()
        self.cache = self.cache_init_random()
        self.users, self.users_preference = self.update_users()
        self.recommendation_list = self.update_recommendation_list()
        self.state = self.update_state()

        return self.state

    def rescaling_action(self, proto_action):
        action_bound = (self.action_space.high - self.action_space.low) / 2
        old_min = -action_bound
        old_max = action_bound
        new_min = self.min_action
        new_max = self.max_action

        proto_action = (proto_action-old_min) / (old_max-old_min) * (new_max-new_min) + new_min

        return proto_action

    def contents_mapping(self, proto_action):
        proto_action = self.rescaling_action(proto_action)  # Makes the range positive.
        proto_action = np.reshape(proto_action, (self.cache_size, self.factors_num))

        # Convert the output of the actor to the probability that the user prefers each factor.
        factor_prob = copy.deepcopy(proto_action)
        for i in range(self.cache_size):
            for j in range(self.factors_num):
                factor_prob[i][j] = proto_action[i][j] / sum(proto_action[i])

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        return factor_prob, cache

    def update_state(self):
        # The state is all the contents on the recommendation list of each user.
        state = np.array([])
        for i in range(self.users_num):
            for j in range(self.recommendation_list_size):
                content = self.recommendation_list[i][j]
                state = np.append(state, self.contents[content])
                state = np.append(state, self.users[i][content])
        state = state.flatten()

        # The state is the top contents on the recommendation list of each users.
        # state = np.array([])
        # for i in range(self.users_num):
        #     content = self.recommendation_list[i][0]
        #     state = np.append(state, self.contents[content])
        #     state = np.append(state, self.users[i][content])
        # state = state.flatten()

        # The state when we know contents factor preferences of users from recommendation system.
        # users_preference = np.array([])
        # for i in range(self.users_num):
        #     user_preference = [0 for _ in range(self.factors_num)]
        #     user_preference[0] = 0.5
        #     user_preference[self.users_preference[i]] = 0.5
        #     users_preference = np.append(users_preference, user_preference)
        # state = users_preference.flatten()

        return state

    def step(self, proto_action):
        print('proto action => ', proto_action)
        factor_prob, self.cache = self.contents_mapping(proto_action)
        self.recommendation_list = self.update_recommendation_list()

        # done = bool(self.count == self.step_size)
        # if not done:

        print('users preference => ', self.users_preference)
        print('factor preference => ', np.round(factor_prob, 2))
        print('cache => ', self.cache)
        print('recommendation list => ', self.recommendation_list)

        self.state = self.update_state()
        print('next state=> ', self.state)

        self.proposed_caching_score = self.recommendation_rating_sum()
        print('proposed caching score => ', np.round(self.proposed_caching_score, 2))

        self.popularity_caching_score = self.popularity_caching()
        self.random_caching_score = self.random_caching()

        reward = self.proposed_caching_score
        print('reward => ', np.round(reward, 2))

        return self.state, reward, {}, {}

    def popularity_caching(self):
        # Caching contents based on popular contents(universal preference).
        # first factor has universal content preference.
        factor_prob = [1] + [0]*(self.factors_num-1)
        factor_prob = factor_prob * self.cache_size
        factor_prob = np.reshape(factor_prob, (self.cache_size, self.factors_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('proto action =>', factor_prob)
        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('popularity caching score => ', np.round(rating_sum, 2))

        return rating_sum

    def random_factor_caching(self):
        # Caching contents based on random factor preference.
        proto_action = np.random.uniform(self.min_action, self.max_action, self.cache_size*self.factors_num)
        proto_action = np.reshape(proto_action, (self.cache_size, self.factors_num))

        # Convert the output of the actor to the probability that the user prefers each factor.
        factor_prob = copy.deepcopy(proto_action)
        for i in range(self.cache_size):
            for j in range(self.factors_num):
                factor_prob[i][j] = proto_action[i][j] / sum(proto_action[i])

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('proto action =>', factor_prob)
        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('random factor caching score => ', np.round(rating_sum, 2))

        return rating_sum

    def random_caching(self):
        # Caching contents randomly.
        cache = []
        while(1):
            content = np.random.randint(0, self.contents_total_num)
            if content not in cache:
                cache.append(content)
                if len(cache) == self.cache_size:
                    break

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('random caching score => ', np.round(rating_sum, 2))

        return rating_sum

    def optimal_caching(self):
        # Calculate the optimal cache when knowing the user's factor preference.
        users = []
        for i in range(self.users_num):
            user = [0.5] + [0]*(self.factors_num-1)
            user[self.users_preference[i]] = 0.5
            users.append(user)

        factor_prob = []
        for j in range(self.users_num):
            for k in range(int(self.recommendation_list_size)):
                factor_prob.append(users[j])

        factor_prob = np.reshape(factor_prob, (self.cache_size, self.factors_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('proto action =>', factor_prob)
        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('optimal caching score => ', np.round(rating_sum, 2))

        return rating_sum

#     def preference_request(self):
#         preference_content_user = self.preference(self.content_feature, self.user_feature)
#         preference_CDF = preference_content_user.cumsum()

#         while 1:
#             prob = np.random.rand(1)
#             for j in range(content_total_num):
#                 if j == 0 and prob <= preference_CDF[j]:
#                     request = content_index[j]
#                     break
#                 elif prob > preference_CDF[j - 1] and prob <= preference_CDF[j]:
#                     request = content_index[j]
#                     break

#         return request

    def test1(self):
        factor_prob = [0.2, 0.2, 0.2, 0.2, 0.2] * self.cache_size
        factor_prob = np.reshape(factor_prob, (self.cache_size, self.factors_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('test1 score => ', np.round(rating_sum, 2))

        return rating_sum

    def test2(self):
        factor_prob = [0, 0.25, 0.25, 0.25, 0.25] * self.cache_size
        factor_prob = np.reshape(factor_prob, (self.cache_size, self.factors_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('test2 score => ', np.round(rating_sum, 2))

        return rating_sum

    def test3(self):
        factor_prob = [0, 0, 0, 0.5, 0.5] * self.cache_size
        factor_prob = np.reshape(factor_prob, (self.cache_size, self.factors_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], factor_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        # Recommend contents within the cache to each user.
        recommendation_list = []
        for i in range(self.users_num):
            user = {j: self.users[i][j] for j in range(len(self.users[i]))}  # list to dict
            sorted_expected_ratings = sorted(user.items(), reverse=True,
                                             key=lambda item: item[1])  # Sort by value

            top_rating_contents = []
            for k in range(len(sorted_expected_ratings)):
                content = sorted_expected_ratings[k][0]  # the key is the content between key-value.
                rating = sorted_expected_ratings[k][1]  # the value is the rating of content between key-value.
                if content in cache:
                    top_rating_contents.append(content)
                    if len(top_rating_contents) == self.recommendation_list_size:
                        break
                else:
                    continue
            recommendation_list.append(top_rating_contents)

        # Calculate the expected ratings sum of users for recommended contents in cache.
        rating_sum = 0
        for i in range(self.users_num):
            for j in range(self.cache_size):
                if cache[j] in recommendation_list[i]:
                    rating_sum = rating_sum + self.users[i][cache[j]]
                else:
                    continue

        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('test3 score => ', np.round(rating_sum, 2))

        return rating_sum
