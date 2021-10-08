"""Configurations to train a model on Hallway task. 

"""

from search_world.trainer import Trainer
from search_world.envs.maze import Hallway

def model_config():
    config = {
        'module': 'search_world.models.random',
        'method': 'RandomAgent'
    }
    return config

def env_config():
    config = {
        'constructor': Hallway,
    }
    return config

def get_config():
    """Config used for main.py"""
    config = {
        'constructor': Trainer,
        'kwargs': {
            'model': model_config(),
            'env': env_config(),
            'num_training_steps': 1000, 
            'render': True
        }    
    }
    return config