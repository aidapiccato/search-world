"""Trainer for model on Hallway task.

The scalar, image, and figures logged during training can be visualized in
tensorboard by running the following command:
$ python3 tensorboard --log_dir=logs/$run_number/tensorboard
"""

import logging
import pickle as pkl
import os
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, model, model_kwargs, env, num_training_steps, render):
        self._model = model
        self._model_kwargs = model_kwargs
        self._env = env
        self._num_training_steps = num_training_steps
        self._render = render

    def train_step():
        pass

    def __call__(self, log_dir):
        obs, reward, done, info = self._env.reset()

        job_id = log_dir.rsplit('/', 1)[-1]
        model = self._model(self._env, **self._model_kwargs) 
        model.reset()
        action = [0, 0]
        plot_every = 1
        plt.ion()
        fig, axs = plt.subplots(nrows=2, ncols=1)
        vector_data = []
        step = -1

        if done: 
            # if done is true on the first timestep, don't run!
            return
        vector_data.append({'obs': obs, 'job_id': job_id, 'reward': reward, 'action': action, 'done': done, 'step': step, 'info': info})
        for step in range(self._num_training_steps):

            logging.info('Step: {} of {}'.format(
                step, self._num_training_steps))

            action = model(obs)
            # action will be one done in response to current obs, reward            

            obs, reward, done, info = self._env.step(action)

            logging.info('action={}, obs={}, reward={}, done={}, info={}'.format(action, obs, reward, done, info))            

            # savig vectorized data
            vector_data.append({'obs': obs, 'job_id': job_id, 'reward': reward, 'action': action, 'done': done, 'step': step, 'info': info})

            if self._render and step % plot_every == 0:
                for ax in axs:
                    ax.clear()
                self._env.render(ax=axs[0]) 
                model.render(ax=axs[1])
                plt.show(block=False)
                plt.pause(0.1)

            if done: 
                obs, reward, done, info = self._env.reset() 
                model.reset()
                action = [0, 0]
                vector_data.append({'obs': obs, 'job_id': job_id, 'reward': reward, 'action': action, 'done': done, 'step': step, 'info': info})

        self._env.close()
        scalar_data = {'env': self._env, 'model': model}
        logging.info('Writing data.')
        write_dir = os.path.join(log_dir, 'data')
        os.makedirs(write_dir)

        with open(os.path.join(write_dir, 'vector'), 'wb') as f:
            pkl.dump(vector_data, f)

        with open(os.path.join(write_dir, 'scalar'), 'wb') as f:
            pkl.dump(scalar_data, f)