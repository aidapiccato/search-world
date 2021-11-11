"""Trainer for model on Hallway task.

The scalar, image, and figures logged during training can be visualized in
tensorboard by running the following command:
$ python3 tensorboard --log_dir=logs/$run_number/tensorboard
"""

import shutil
import logging
import pickle as pkl
import os
import matplotlib.pyplot as plt
import imageio

class Trainer(object):
    def __init__(self, model, model_kwargs, env, num_training_steps, render, save_gif=False):
        self._model = model
        self._model_kwargs = model_kwargs
        self._env = env
        self._num_training_steps = num_training_steps
        self._render = render
        self._save_gif = save_gif

    def train_step():
        pass

    def __call__(self, log_dir):
        obs, reward, done, info = self._env.reset()
         
        model = self._model(self._env, **self._model_kwargs) 
        model.reset()
        action = [0, 0]
        plot_every = 1
        plt.ion()
        
        vector_data = []
        step = -1
        filenames = []

        if self._render and self._save_gif:
            images_dir = os.path.join(log_dir, 'images')
            os.makedirs(images_dir)

        if done: 
            # if done is true on the first timestep, don't run!
            return
        logging.info('Step: {} of {}'.format(step, self._num_training_steps))
        logging.info('action={}, obs={}, reward={}, done={}, info={}'.format(action, obs, reward, done, info))            
        vector_data.append({'obs': obs, 'reward': reward, 'action': action, 'done': done, 'step': step, 'info': info})
        for step in range(self._num_training_steps):
            # import pdb; pdb.set_trace()

            logging.info('Step: {} of {}'.format(
                step, self._num_training_steps))

            action = model(obs)
            # action will be one done in response to current obs, reward            

            obs, reward, done, info = self._env.step(action)

            logging.info('action={}, obs={}, reward={}, done={}, info={}'.format(action, obs, reward, done, info))            

            # savig vectorized data
            vector_data.append({'obs': obs, 'reward': reward, 'action': action, 'done': done, 'step': step, 'info': info})

            if self._render and step % plot_every == 0:
                fig, axs = plt.subplots(nrows=2, ncols=1)
                self._env.render(ax=axs[0]) 
                model.render(ax=axs[1])
                if self._save_gif:
                    for i in range(10):
                        filename = os.path.join(log_dir, f'images/frame_{step}_{i}.png') 
                        filenames.append(filename)
                        plt.savefig(filename, dpi=96)
                else:
                    plt.pause(0.5)
                plt.close()

            if done: 
                obs, reward, done, info = self._env.reset() 
                model.reset()
                action = [0, 0]
                vector_data.append({'obs': obs, 'reward': reward, 'action': action, 'done': done, 'step': step, 'info': info})

        self._env.close()

        if self._render and self._save_gif:
            with imageio.get_writer(os.path.join(log_dir, 'demo.gif'), mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            logging.info('GIF saved')
            shutil.rmtree(os.path.join(log_dir, 'images'))

        scalar_data = {'env': self._env.info(), 'model': model.info()}
        logging.info('Writing data.')
        write_dir = os.path.join(log_dir, 'data')
        os.makedirs(write_dir)

        with open(os.path.join(write_dir, 'vector'), 'wb') as f:
            pkl.dump(vector_data, f)

        with open(os.path.join(write_dir, 'scalar'), 'wb') as f:
            pkl.dump(scalar_data, f)