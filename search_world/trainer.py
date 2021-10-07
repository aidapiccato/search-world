"""Trainer for model on Hallway task.

The scalar, image, and figures logged during training can be visualized in
tensorboard by running the following command:
$ python3 tensorboard --log_dir=logs/$run_number/tensorboard
"""

import time
import logging
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, model, env, num_training_steps, render):
        self._model = model
        self._env = env
        self._num_training_steps = num_training_steps
        self._render = render

    def train_step():
        pass

    def __call__(self, log_dir):
        obs = self._env.reset()
        plot_every = 1
        plt.ion()
        fig, ax = plt.subplots()
        for step in range(self._num_training_steps):
            logging.info('Step: {} of {}'.format(
                step, self._num_training_steps))
            action = self._env.action_space.sample() # TODO: Implement action space property for env

            obs, reward, done, info = self._env.step(action)
            logging.info('action={}, obs={}, reward={}, done={}, info={}'.format(action, obs, reward, done, info))

            if self._render and step % plot_every == 0:
                ax.clear()
                self._env.render() 
                plt.show(block=False)
                plt.pause(0.1)

            if done: 
                self._env.reset()

        self._env.close() # TODO: Implement close method for env

