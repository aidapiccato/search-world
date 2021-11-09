import numpy as np
from search_world.envs.maze import Maze
from search_world.utils.maze_utils import symm_corr
import json
import pandas as pd 
from moog import maze_lib
from search_world.utils import common_utils

import numpy as np
import os



TIME_INDEX = 0
ACTION_INDEX = 3
STATE_INDEX = 5
METADATA_INDEX = 4
_Y_VERTEX_MIN = 0.25
_X_VERTEX_MIN = 0 
_DISPLAY_SIZE = 0.5
_EPSILON = 1e-45


def moog_generator(ambient_size):
    maze = Maze(max_steps=100, maze_gen_func=symm_corr, maze_gen_func_kwargs={'length': 5,  'n_corr': 3, 'target_pos': 12, 'agent_init_pos': 1})
    maze_obj = maze    
    maze.reset()

    maze = maze._maze
    size = maze.shape[0] 

    agent_init_pos = maze_obj._state_space[maze_obj._agent_initial_state]
    prey_pos = maze_obj._state_space[maze_obj._target_state]
    
    # Add wall border if necessary
    if ambient_size is not None and ambient_size > size:
        maze_with_border = np.ones((ambient_size, ambient_size))
        start_index = (ambient_size - size) // 2
        maze_with_border[start_index: start_index + size,
                         start_index: start_index + size] = maze
        orig_maze = maze
        maze = maze_with_border   
        agent_init_pos = agent_init_pos + start_index
        prey_pos = prey_pos + start_index 
    return orig_maze, maze, maze_obj, agent_init_pos, prey_pos


def get_maze(trial):
    # Selecting first timestep since maze and prey location are static throughout trial

    state = trial[0][STATE_INDEX]
    
    walls = list(filter(lambda sprite: sprite[0] == 'walls', state))
    walls = [common_utils.attributes_to_sprite(factors) for factors in walls[0][1]]
    maze = maze_lib.Maze.from_state({'walls': walls}, maze_layer='walls') 
    y_vertex_min = np.min(np.array([sprite.position[1] for sprite in walls]))
    prey = list(filter(lambda sprite: sprite[0] == 'prey', state))
    if len(prey[0][1]) > 0:
        prey = _get_inds(prey[0][1][0][:2], maze.maze_size, y_vertex_min - maze.half_grid_side)
    else: 
        prey = []
   
    return {'prey': prey, 'grid_side': maze.grid_side, 
    'half_grid_side': maze.half_grid_side, 'maze_size': maze.maze_size, 'y_vertex_min': y_vertex_min - maze.half_grid_side, 'maze_array': maze.maze}

def get_action_sequence(trial):
    action_tups = [(step[ACTION_INDEX][1], step[TIME_INDEX][1]) for step in trial]
    action_tups = list(filter(lambda a: a[0] != 4, action_tups))
    action_times = [action[1] for action in action_tups]
    actions = [action[0] for action in action_tups]
    return {'actions': actions, 'action_times': action_times}


def get_maze_kwargs(trial):
    state = trial[0][STATE_INDEX]
    agent = list(filter(lambda sprite: sprite[0] == 'agent', state))
    kwargs = agent[0][1][0][14]['env']
    if 'maze' in kwargs:
        del kwargs['maze']
    if 'max_steps' in kwargs:
        del kwargs['max_steps']
    return kwargs

def get_agent(trial, maze_size, y_vertex_min):
    state = [step[STATE_INDEX] for step in trial]    
    agent = [list(filter(lambda sprite: sprite[0] == 'agent', step)) for step in state]
    agent = list(map(lambda sprite: _get_inds(sprite[0][1][0][:2], maze_size, y_vertex_min), agent))
    return {'agent': agent}


def get_time(trial):
    times = [step[TIME_INDEX][1] for step in trial]
    times = [time - times[0] for time in times]
    return {'time': times}


def get_metadata(trial):
    # Selecting first timestep since metadata is static
    metadata = trial[0][METADATA_INDEX][1]
    return metadata

def _get_inds(position, maze_size, y_vertex_min, round=False): 
    # Find indices corresponding to given x and y coordinates in padded maze array
    position = np.asarray(position)
    offset = np.asarray([_X_VERTEX_MIN, y_vertex_min])
    position = position - offset

    grid_side = _DISPLAY_SIZE / maze_size
    half_grid_side = 0.5 * _DISPLAY_SIZE / maze_size


    if round:
        nearest_inds = (np.round(position  / grid_side - 0.5)).astype(int)
    else:
        nearest_inds = position/grid_side - 0.5
    rounded_position = half_grid_side   + nearest_inds * grid_side 

    on_grid = np.abs(rounded_position - position) < _EPSILON

    new_position = np.copy(position)

    new_position[on_grid] = rounded_position[on_grid] 

    # inds = ((position  - half_grid_side) // grid_side).astype(int)
    # inds[on_grid] = nearest_inds[on_grid]

    # Returning nearest_inds instead of inds to avoid rounding error
    return nearest_inds

def _flatten_dict(d):
    """Flattens any nested dictionaries in dictionary d into a single-level dictionary. Only flattens a single level"""
    d_copy = {}
    t = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for nested_k, nested_v in v.item():
                d_copy.update({nested_k: nested_v})
        else:
            d_copy.update({k: v})
    return d_copy

def get_maze_env(trial):
    kwargs = get_maze_kwargs(trial)
    env = Maze(maze_gen_func=symm_corr, maze_gen_func_kwargs=kwargs, max_steps=100)
    return {'env': env}

def get_trial_dataframe(trial_paths, **kwargs):
    """Create trial dataframe."""

    trial_dicts = []

    for i, p in enumerate(trial_paths):
        trial = json.load(open(p, 'r')) 
        d = {}
        d['trial_dict'] = trial
        # d.update(get_metadata(trial)) 
 
        d.update(get_time(trial))
        d.update(get_maze(trial))
        d.update(get_maze_kwargs(trial))
        d.update(get_maze_env(trial))        
        d.update(get_action_sequence(trial))
        env = d['env']
        vector_data = []
        step = 0
        obs, reward, done, info = env.reset()
        vector_data.append({'obs': obs, 'reward': reward, 'done': done, 'step': step, 'info': info})
        for step, action in enumerate(d['actions']):
            obs, reward, done, info = env.step(action)
            vector_data.append({'obs': obs, 'reward': reward, 'done': done, 'step': step, 'info': info})
        vector_dict = {k: [dic[k] for dic in vector_data] for k in vector_data[0]}
        d.update(vector_dict)
        d.update({'name': 'Human'}) 
        d['trial_index'] = i
        d['dataset_index'] = i
        trial_dicts.append(d)



    keys = list(trial_dicts[0].keys())
    trial_dict = {k: [d[k] for d in trial_dicts] for k in keys}
    for k, v in kwargs.items():
        trial_dict[k] = len(trial_dict['trial_index']) * [v]
    trial_df = pd.DataFrame(trial_dict)
    
    return trial_df

def get_trial_paths(data_path):

    # Chronological trial paths
    trial_paths = [
        os.path.join(data_path, x)
        for x in sorted(os.listdir(data_path)) if x.isnumeric()
    ]

    num_trials = len(trial_paths)
    print(f'Number of trials:  {num_trials}')

    return trial_paths
