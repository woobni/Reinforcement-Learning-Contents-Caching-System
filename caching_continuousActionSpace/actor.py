import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Lambda


class ActorNet():
    """ Actor Network for DDPG
    """
    def __init__(self, in_dim, out_dim, act_range, lr_, tau_):
        self.obs_dim = in_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.lr = lr_; self.tau = tau_

        # initialize actor network and target
        self.network = self.create_network()
        self.target_network = self.create_network()

        # initialize optimizer
        self.optimizer = Adam(self.lr)

        # copy the weights for initialization
        weights_ = self.network.get_weights()
        self.target_network.set_weights(weights_)


    def create_network(self):
        """ Create a Actor Network Model using Keras
        """
        # input layer(observations)
        input_ = Input(shape=self.obs_dim)

        # hidden layer 1
        h1_ = Dense(300,kernel_initializer=glorot_normal())(input_)
        h1_b = BatchNormalization()(h1_)
        h1 = Activation('relu')(h1_b)

        # hidden_layer 2
        h2_ = Dense(400,kernel_initializer=glorot_normal())(h1)
        h2_b = BatchNormalization()(h2_)
        h2 = Activation('relu')(h2_b)

        # output layer(actions)
        output_ = Dense(self.act_dim,kernel_initializer=glorot_normal())(h2)
        output_b = BatchNormalization()(output_)
        output = Activation('tanh')(output_b)
        scalar = self.act_range * np.ones(self.act_dim)
        out = Lambda(lambda i: i * scalar)(output)

        return Model(input_,out)

    def train(self, obs, critic, q_grads):
        """ training Actor's Weights
        """
        with tf.GradientTape() as tape:
            actions = self.network(obs)
            actor_loss = -tf.reduce_mean(critic([obs,actions]))
            # actor_grad = tape.gradient(self.network(obs), self.network.trainable_variables,-q_grads)
        actor_grad = tape.gradient(actor_loss,self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grad,self.network.trainable_variables))

    def target_update(self):
        """ soft target update for training target actor network
        """
        weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.target_network.set_weights(weights_t)

    def predict(self, obs):
        """ predict function for Actor Network
        """
        return self.network.predict(np.expand_dims(obs, axis=0))

    def target_predict(self, new_obs):
        """  predict function for Target Actor Network
        """
        return self.target_network.predict(new_obs)

    def save_network(self, path):
        self.network.save_weights(path + '_actor.h5')
        self.target_network.save_weights(path +'_actor_t.h5')

    def load_network(self, path):
        self.network.load_weights(path + '_actor.h5')
        self.target_network.load_weights(path + '_actor_t.h5')
        print(self.network.summary())


# for test
if __name__ == '__main__':
    actor = ActorNet()