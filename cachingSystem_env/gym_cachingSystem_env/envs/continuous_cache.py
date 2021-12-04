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
        self.contents_total_num = 3000
        self.cache_size = 10
        self.cache = self.cache_init_random()
        self.max_rating = 5  # max ratings that critics give to contents.
        self.min_rating = 0  # min ratings that critics give to contents.
        self.critics_num = 5
        self.critics = self.generate_critics()
        self.contents = self.generate_contents()
        self.users_num = 4
        self.users, self.users_preference = self.update_users()
        self.recommendation_list_size = 5  # number of contents recommended to users within cache.
        self.recommendation_list = self.update_recommendation_list()
        self.state = self.update_state()
        self.min_action = 0.0
        self.max_action = 1.0

        # The state is all the contents on the recommendation list of each users.
        self.low_state = np.array(
            [self.min_rating] * (self.critics_num+1) * self.recommendation_list_size * self.users_num,
            dtype=np.float32)  # (self.critics_num+1) => contents feature and contents expected rating.
        self.high_state = np.array(
            [self.max_rating] * (self.critics_num+1) * self.recommendation_list_size * self.users_num,
            dtype=np.float32)

        # The state is the top content on the recommendation list of each users.
        # self.low_state = np.array(
        #     [self.min_rating] * (self.critics_num+1) * self.users_num,
        #     dtype=np.float32)  # (self.critics_num+1) => contents feature and contents expected rating.
        # self.high_state = np.array(
        #     [self.max_rating] * (self.critics_num+1) * self.users_num,
        #     dtype=np.float32)

        # The state when we know contents critic preferences of users from recommendation system.
        # self.low_state = np.array(
        #     [self.min_rating] * self.critics_num * self.users_num,
        #     dtype=np.float32)  # (self.critics_num+1) => contents feature and contents expected rating.
        # self.high_state = np.array(
        #     [self.max_rating] * self.critics_num * self.users_num,
        #     dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.cache_size*self.critics_num,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

    def generate_critics(self):
        # Each critic has its own preference about contents feature(e.g. genre, director, actor, etc.).
        # Each critic gives a rating for each content.
        critics = []
        for _ in range(self.critics_num-2):
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
            critics.append(ratings)

        # Generate critics who have an opposite preference to the critics created earlier.
        for i in range(2):
            opposite_ratings = []
            for j in range(self.contents_total_num):
                opposite_rating = self.max_rating - critics[i+1][j]
                opposite_rating = np.round(opposite_rating, 2)
                opposite_ratings.append(opposite_rating)
            critics.append(opposite_ratings)

        return critics

    def update_users(self):
        # feature of users with expected ratings.
        users = []
        preferences = []
        # preferences = [1, 2]
        for i in range(self.users_num):
            # Suppose the users continues to change. (Random critic preference)
            preference = np.random.randint(1, self.critics_num)
            preferences.append(preference)

            # Suppose the users continues to remain unchanged. (Fix critic preference)
            preference = preferences[i]

            # first critic has universal content preference. so, fix to
            expected_ratings = [0.7*x + 0.3*y for x, y in zip(self.critics[0], self.critics[preference])]
            # expected_ratings = self.critics[preference]
            expected_ratings = list(np.round(expected_ratings, 2))
            users.append(expected_ratings)

        return users, preferences

    def generate_contents(self):
        # Each content is placed in a continuous space consisting of critics' ratings dimension.
        contents = []
        for i in range(self.contents_total_num):
            content = []
            for j in range(self.critics_num):
                content.append(self.critics[j][i])
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
        self.critics = self.generate_critics()
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
        proto_action = np.reshape(proto_action, (self.cache_size, self.critics_num))

        # Convert the output of the actor to the probability that the user prefers each critic.
        critic_prob = copy.deepcopy(proto_action)
        for i in range(self.cache_size):
            for j in range(self.critics_num):
                critic_prob[i][j] = proto_action[i][j] / sum(proto_action[i])

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

            dot_contents_dict = {t: dot_contents[t] for t in range(len(dot_contents))}  # list to dict
            sorted_dot_contents_dict = sorted(dot_contents_dict.items(),
                                              key=(lambda item: item[1]), reverse=True)  # Sort by value

            for k in range(len(sorted_dot_contents_dict)):
                content = sorted_dot_contents_dict[k][0]  # the key is the content between key-value.
                if content not in cache:
                    cache.append(content)
                    break
                else: continue

        return critic_prob, cache

    def update_state(self):
        # The state is all the contents on the recommendation list of each users.
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

        # The state when we know contents critic preferences of users from recommendation system.
        # users_preference = np.array([])
        # for i in range(self.users_num):
        #     user_preference = [0 for _ in range(self.critics_num)]
        #     user_preference[0] = 0.5
        #     user_preference[self.users_preference[i]] = 0.5
        #     users_preference = np.append(users_preference, user_preference)
        # state = users_preference.flatten()

        return state

    def step(self, proto_action):
        critic_prob, self.cache = self.contents_mapping(proto_action)
        self.recommendation_list = self.update_recommendation_list()

        # done = bool(self.count == self.step_size)
        # if not done:

        # reward = self.proposed_caching_score - self.popularity_caching_score
        rating_sum = self.recommendation_rating_sum()
        reward = rating_sum

        self.state = self.update_state()

        print("\n")
        print("==============================================================")
        # print('critics => ', self.critics)
        # print('users => ', self.users)
        print('users preference => ', self.users_preference)
        print('proto action => ', np.round(critic_prob, 2))
        print('cache => ', self.cache)
        print('recommendation list => ', self.recommendation_list)
        print('reward => ', np.round(reward, 2))
        print('next state=> ', self.state)
        print('proposed caching score => ', np.round(rating_sum, 2))

        return self.state, reward, {}, {}

    def popularity_caching(self):
        # Caching contents based on popular contents(universal preference).
        # first critic has universal content preference.
        critic_prob = [1] + [0]*(self.critics_num-1)
        critic_prob = critic_prob * self.cache_size
        critic_prob = np.reshape(critic_prob, (self.cache_size, self.critics_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

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

        print('proto action =>', critic_prob)
        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('popularity caching score => ', np.round(rating_sum, 2))

        return rating_sum

    def random_critic_caching(self):
        # Caching contents based on random critic preference.
        proto_action = np.random.uniform(self.min_action, self.max_action, self.cache_size*self.critics_num)
        proto_action = np.reshape(proto_action, (self.cache_size, self.critics_num))

        # Convert the output of the actor to the probability that the user prefers each critic.
        critic_prob = copy.deepcopy(proto_action)
        for i in range(self.cache_size):
            for j in range(self.critics_num):
                critic_prob[i][j] = proto_action[i][j] / sum(proto_action[i])

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

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

        print('proto action =>', critic_prob)
        print('cache => ', cache)
        print('recommendation list => ', recommendation_list)
        print('random critic caching score => ', np.round(rating_sum, 2))

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
        # Calculate the optimal cache when knowing the user's critic preference.
        users = []
        for i in range(self.users_num):
            user = [0.5] + [0]*(self.critics_num-1)
            user[self.users_preference[i]] = 0.5
            users.append(user)

        critic_prob = []
        for j in range(self.users_num):
            for k in range(int(self.recommendation_list_size)):
                critic_prob.append(users[j])

        critic_prob = np.reshape(critic_prob, (self.cache_size, self.critics_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

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

        print('proto action =>', critic_prob)
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
        critic_prob = [0.2, 0.2, 0.2, 0.2, 0.2] * self.cache_size
        critic_prob = np.reshape(critic_prob, (self.cache_size, self.critics_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

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
        critic_prob = [0, 0.25, 0.25, 0.25, 0.25] * self.cache_size
        critic_prob = np.reshape(critic_prob, (self.cache_size, self.critics_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

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
        critic_prob = [0, 0, 0, 0.5, 0.5] * self.cache_size
        critic_prob = np.reshape(critic_prob, (self.cache_size, self.critics_num))

        # Find the content that users on the current network are most likely to prefer.
        cache = []
        for i in range(self.cache_size):
            dot_contents = []
            for j in range(self.contents_total_num):
                dot_contents.append(np.dot(self.contents[j], critic_prob[i]))

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
