import datetime
import json
import logging
import os
import pprint
import sys
import traceback
from statistics import mode

sys.path.append("/home/nianfei/ros_ws/src/dqn_for_slam")
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

import dqn_for_slam.environment
from dqn_for_slam.custom_policy import CustomEpsGreedy

ENV_NAME = "RobotEnv-v0"
file_path = __file__
dir_path = file_path[: (len(file_path) - len("rl_worker.py"))]
MODELS_PATH = dir_path + "models/"  # model save directory
FIGURES_PATH = dir_path + "figures/"


def kill_all_nodes() -> None:
    """
    kill all ros node except for roscore
    """
    nodes = os.popen("rosnode list").readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace("\n", "")
    for node in nodes:
        os.system("rosnode kill " + node)


if __name__ == "__main__":
    os.system("rosclean purge -y")
    model_dir = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/models/"
    model_path = model_dir + "nf_weights_51811.h5f"
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n

    model = tf.keras.Sequential()
    model.add(
        L.Reshape(
            target_shape=(90, 90, 1), input_shape=(1,) + env.observation_space.shape
        )
    )
    model.add(L.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu"))
    model.add(L.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
    model.add(L.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))
    model.add(L.Flatten())
    model.add(L.Dense(625, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(L.Reshape((25, 25)))
    model.add(L.LSTM(units=256))
    model.add(L.Dense(4, activation="linear"))

    memory = SequentialMemory(limit=100000, window_length=1)
    policy = CustomEpsGreedy(max_eps=0.6, min_eps=0.1, eps_decay=0.9997)

    agent = DQNAgent(
        nb_actions=nb_actions,
        model=model,
        memory=memory,
        policy=policy,
        gamma=0.99,
        batch_size=10,  # 64
        nb_steps_warmup=20,  # 500 1000
        train_interval=1,
        memory_interval=1,
        target_model_update=25,  # 1000 10000
        delta_range=None,
    )

    agent.compile(optimizer=Adam(learning_rate=1e-3), metrics=["mae"])
    agent.load_weights(model_path)

    history = agent.fit(
        env,
        nb_steps=150,  # 100000
        visualize=False,
        nb_max_episode_steps=15,  # 20 100 250
        log_interval=15,  # 20 250
        verbose=1,
    )

    kill_all_nodes()

    dt_now = datetime.datetime.now()
    agent.save_weights(
        MODELS_PATH
        + "nf_weights_{}{}{}.h5f".format(dt_now.month, dt_now.day, dt_now.hour),
        overwrite=True,
    )
    conf_path = MODELS_PATH + "nf_conf_{}{}{}.json".format(
        dt_now.month, dt_now.day, dt_now.hour
    )
    history_path = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/reward/maze10_{}{}{}.txt".format(
        dt_now.month, dt_now.day, dt_now.hour
    )
    with open(history_path, "a+") as f:
        f.write(pprint.pformat(history.history))
    config = agent.get_config()
    del config["model"]
    del config["target_model"]
    with open(conf_path, "w+") as f:
        json.dump(config, f)
