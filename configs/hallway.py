"""Configurations to train a model on Hallway task. 

"""

from search_world.trainer import Trainer

def model_config():
    config = {}
    return config

def task_config():
    config = {}
    return config

def get_config():
    """Config used for main.py"""
    config = {
        'constructor': Trainer,
        'kwargs': {
            'model': model_config(),
            'task': task_config(),
            'log_dir': 'logs'
        }    
    }
    return config