"""Configurations to train a model on Hallway task. 

"""

from search_world.trainer import Trainer
from search_world.envs.maze import Maze
from search_world.utils.maze_utils import hallway

def model_config():
    config = {
        'module': 'search_world.models.random',
        'method': 'MLSAgent'
    }
    return config

def model_kwargs_config():
    config = {
        'horizon': 10,
        'discount_factor': 1
    }
    return config

def env_config():
    config = {
        'constructor': Maze,
        'kwargs':{
            'max_steps': 100,
            'maze_gen_func': hallway,
            'maze_gen_func_kwargs': {
                'length': 8,
                'agent_initial_position': 0, 
                'target_position': 6
            }
        }
    }
    return config

def get_config():
    """Config used for main.py"""
    config = {
        'constructor': Trainer,
        'kwargs': {
            'model': model_config(),
            'model_kwargs': model_kwargs_config(),
            'env': env_config(),
            'num_training_steps': 10, 
            'render': True,
            'save_gif': False
        }    
    }
    return config