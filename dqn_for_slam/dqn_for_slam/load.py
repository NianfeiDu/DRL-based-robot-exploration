import sys
from statistics import mode

import tensorflow as tf

sys.path.append("/home/nianfei/ros_ws/src/dqn_for_slam")
import gym
import tensorflow as tf
import tensorflow.keras.layers as L
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

import dqn_for_slam.environment
from dqn_for_slam.custom_policy import CustomEpsGreedy

model_path = (
    "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/models/nf_weights_51811.h5f"
)
ENV_NAME = "RobotEnv-v0"
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

model = tf.keras.Sequential()
model.add(
    L.Reshape(target_shape=(90, 90, 1), input_shape=(1,) + env.observation_space.shape)
)
model.add(L.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu"))
model.add(L.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
model.add(L.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))
model.add(L.Flatten())
model.add(L.Dense(625, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(L.Reshape((25, 25)))
model.add(
    L.LSTM(
        units=256,
    )
)
model.add(L.Dense(4, activation="linear"))
memory = SequentialMemory(limit=100000, window_length=1)
policy = CustomEpsGreedy(max_eps=0.6, min_eps=0.1, eps_decay=0.9997)

agent = DQNAgent(
    nb_actions=4,
    model=model,
    memory=memory,
    policy=policy,
    gamma=0.99,
    batch_size=64,
)
agent.compile(optimizer=Adam(learning_rate=1e-3), metrics=["mae"])
agent.load_weights(model_path)
agent.model.summary()
agent.test(env, nb_episodes=15, visualize=False)

# observation = env.reset()
# while env.done == False:
#     action = agent.forward(observation)
#     observation, r, d, info = env.step(action)
