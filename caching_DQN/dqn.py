import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h1_size=300, h2_size=300, l_rate=1e-3):
        with tf.compat.v1.variable_scope(self.net_name):
            self._X = tf.compat.v1.placeholder(
                tf.float32, [None, self.input_size], name="input_x")

            self.W1 = tf.compat.v1.get_variable("W1", shape=[self.input_size, h1_size],
                                 initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            self.b1 = tf.compat.v1.get_variable("b1", shape=[h1_size],
                                 initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            layer1 = tf.nn.relu(tf.matmul(self._X, self.W1)+self.b1)

            self.W2 = tf.compat.v1.get_variable("W2", shape=[h1_size, h2_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            self.b2 = tf.compat.v1.get_variable("b2", shape=[h2_size],
                                      initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            layer2 = tf.nn.relu(tf.matmul(layer1, self.W2)+self.b2)

            self.W3 = tf.compat.v1.get_variable("W3", shape=[h2_size, self.output_size],
                                 initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            self._Qpred = tf.matmul(layer2, self.W3)

        self._Y = tf.compat.v1.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(input_tensor=tf.square(self._Y - self._Qpred))

        self._train = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})



