"""Trainer for model on Hallway task.

The scalar, image, and figures logged during training can be visualized in
tensorboard by running the following command:
$ python3 tensorboard --log_dir=logs/$run_number/tensorboard
"""

import time

class Trainer(object):
    def __init__(self, model, env, num_steps):
        self._model = model
        self._env = env
        self._num_steps = num_steps

    def train_step():
        pass

    def __call__(self, log_dir):
        obs = self._env.reset()
        for step in range(self._num_steps):
            action = self._env.action_space.sample() # TODO: Implement action space property for env

            obs, reward, done, info = self.env.step(action)

            self._env.render() # TODO: Implement render step for environment

            time.sleep(0.001)

            if done: 
                self._env.reset()

        self._env.close() # TODO: Implement close method for env

