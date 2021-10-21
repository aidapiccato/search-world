"""Configurations to train a model on Symmetric Corridors task. 

"""

from search_world.trainer import Trainer
from search_world.envs.maze import Maze
from search_world.utils.maze_utils import symmetric_corridors

def model_config():
    config = {
        'module': 'search_world.models.random',
        'method': 'MLSAgent',
    }
    return config

def model_kwargs_config():
    config = {
        'horizon': 5,
        'discount_factor': 0.9
    }
    return config
    
def env_config():
    config = {
        'constructor': Maze,
        'kwargs':{
            'max_steps': 100,
            'maze_gen_func': symmetric_corridors,
            'maze_gen_func_kwargs': {
                'length': 5, 
                'n_corridors': 4,
                'target_position': 9,
                'agent_initial_position': 9,                
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
            'num_training_steps': 200, 
            'render': False
        }    
    }
    return config