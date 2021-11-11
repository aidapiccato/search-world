"""Configurations to train a model on Symmetric Corridors task. 

"""

from search_world.trainer import Trainer
from search_world.envs.maze import Maze
from search_world.utils.maze_utils import symm_corr

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
            'maze_gen_func': symm_corr,
            'maze_gen_func_kwargs':  {'length': 3,  'n_corr': 2},
            'init_state': {'target_state': 1, 'agent_init_state': 2}
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