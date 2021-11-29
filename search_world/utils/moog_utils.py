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
_EPSILON = 1e-4

def moog_generator():
    # length = np.random.choice([3, 5, 7, 9])
    # n_corr = np.random.choice([2, 3, 4, 5])
    n_corr = np.random.choice([2, 3, 4])
    length = np.random.choice([3, 5, 7, 9])
    agent_init_pos = np.random.choice(list(range(0, n_corr * length + n_corr - 1, 3)))
    target_pos = np.random.choice(list(range(0, n_corr * length + n_corr - 1, 3)))    
    while target_pos == agent_init_pos:
        target_pos = np.random.choice(list(range(0, n_corr * length + n_corr - 1, 3)))    
    maze = Maze(max_steps=100, maze_gen_func=symm_corr, maze_gen_func_kwargs={'length': length,  'n_corr': n_corr}, init_state={'target_state': target_pos, 'agent_init_state': agent_init_pos})

    maze_obj = maze    
    maze.reset()

    maze = maze._maze
    size_0 = maze.shape[0] 
    size_1 = maze.shape[1] 
    ambient_size = np.int64(13)

    agent_init_pos = maze_obj._state_space[maze_obj._agent_init_state]
    prey_pos = maze_obj._state_space[maze_obj._target_state]
    
    orig_maze = maze

    # Add wall border if necessary
    if ambient_size is not None and ambient_size > np.amax(maze.shape):
        maze_with_border = np.ones((ambient_size, ambient_size))
        start_index = (ambient_size - maze.shape) // 2 - 1
        maze_with_border[start_index[0]: start_index[0] + size_0,
                         start_index[1]: start_index[1] + size_1] = maze
        maze = maze_with_border   
        agent_init_pos = agent_init_pos + start_index
        prey_pos = prey_pos + start_index 

    return orig_maze, maze, maze_obj, agent_init_pos, prey_pos


def get_states(maze_array, agent):
    states = []
    states_coors = np.vstack(np.where(np.flip(maze_array) == 0)).T
    for pos in agent:
        states.append(np.where(np.all(states_coors == np.asarray(pos), axis=1))[0][0])
    states = np.asarray(states)
    return {'agent_state': states}



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
    'half_grid_side': maze.half_grid_side, 'maze_size': maze.maze_size, 'y_vertex_min': y_vertex_min - maze.half_grid_side, 'maze_array': np.flip(maze.maze)}

def get_action_sequence(trial):
    action_tups = [(step[ACTION_INDEX][1], step[TIME_INDEX][1]) for step in trial]
    # action_tups = list(filter(lambda a: a[0] != 4, action_tups))
    action_times = np.asarray([action[1] for action in action_tups])
    actions = np.asarray([action[0] for action in action_tups])
    return {'raw_actions': actions, 'raw_action_times': action_times}

def _flatten_dict(d):
    """Flattens any nested dictionaries in dictionary d into a single-level dictionary. Only flattens a single level"""
    d_copy = {}
    t = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for nested_k, nested_v in v.items():        
                d_copy.update({nested_k: nested_v})
        else:
            d_copy.update({k: v})      
    return d_copy

def get_maze_kwargs(trial):
    state = trial[0][STATE_INDEX]
    agent = list(filter(lambda sprite: sprite[0] == 'agent', state))
    kwargs = agent[0][1][0][14]['env'] 
    return kwargs

def get_agent(trial, maze_size, y_vertex_min):
    state = [step[STATE_INDEX] for step in trial]    
    agent = [list(filter(lambda sprite: sprite[0] == 'agent', step)) for step in state]
    on_grid = list(map(lambda sprite: _get_on_grid(sprite[0][1][0][:2], maze_size, y_vertex_min), agent))
    agent = np.asarray(list(map(lambda sprite: _get_inds(sprite[0][1][0][:2], maze_size, y_vertex_min), agent)))

    move_steps = np.asarray(np.where(np.diff(on_grid))[0])
    move_steps = move_steps[::2]+1
    move_steps = move_steps.astype(int)
    # use these to index into actions (all actinos, not just the nonzero ones)
    # use these to index into states 
    return {'agent': agent, 'on_grid': on_grid, 'move_steps': move_steps}


def get_time(trial):
    times = [step[TIME_INDEX][1] for step in trial]
    times = [time - times[0] for time in times]
    return {'time': times}


def get_metadata(trial):
    # Selecting first timestep since metadata is static
    metadata = trial[0][METADATA_INDEX][1]
    return metadata

def _get_on_grid(position, maze_size, y_vertex_min):
    position = np.asarray(position)
    offset = np.asarray([_X_VERTEX_MIN, y_vertex_min])
    position = position - offset

    grid_side = _DISPLAY_SIZE / maze_size
    half_grid_side = 0.5 * _DISPLAY_SIZE / maze_size
    half_grid_side = 0.5 * grid_side

    if round:
        nearest_inds = (np.round(position  / grid_side - 0.5)).astype(int)
    else:
        nearest_inds = position/grid_side - 0.5
    rounded_position = half_grid_side + nearest_inds * grid_side 

    on_grid = np.abs(rounded_position - position) < _EPSILON

    return np.all(on_grid)

def _get_inds(position, maze_size, y_vertex_min, round=True): 
    # Find indices corresponding to given x and y coordinates in padded maze array
    position = np.asarray(position)
    offset = np.asarray([_X_VERTEX_MIN, y_vertex_min])
    position = position - offset

    grid_side = _DISPLAY_SIZE / maze_size
    half_grid_side = 0.5 * _DISPLAY_SIZE / maze_size
    half_grid_side = 0.5 * grid_side

    if round:
        nearest_inds = (np.round(position  / grid_side - 0.5)).astype(int)
    else:
        nearest_inds = position/grid_side - 0.5
    rounded_position = half_grid_side + nearest_inds * grid_side 

    on_grid = np.abs(rounded_position - position) < _EPSILON

    new_position = np.copy(position)

    new_position[on_grid] = rounded_position[on_grid] 

    inds = ((position  - half_grid_side) // grid_side).astype(int)
    inds[on_grid] = nearest_inds[on_grid]

    # Returning nearest_inds instead of inds to avoid rounding error
    return np.flip(inds)



def get_maze_env(trial):
    kwargs = get_maze_kwargs(trial)
    env = Maze(maze_gen_func=symm_corr, maze_gen_func_kwargs=kwargs['maze_gen_func_kwargs'], init_state=kwargs['init_state'], max_steps=100)
    return {'env': env}

def get_agent_node(coors, grid_side, y_vertex_min):
    all_inds = []
    half_grid_side = 0.5 * grid_side
    for position in coors: 

        offset = [0, y_vertex_min]
        position = position - offset
        
        nearest_inds = (np.round(position  / grid_side - 0.5)).astype(int)
        
        rounded_position = half_grid_side + nearest_inds * grid_side 
        
        on_grid = np.abs(rounded_position - position) < _EPSILON
        
        new_position = np.copy(position)
        
        new_position[on_grid] = rounded_position[on_grid] 
        
        inds = ((position  - half_grid_side) // grid_side).astype(int)

        inds[on_grid] = nearest_inds[on_grid]
        
        all_inds.append([inds[1], inds[0]])
    return {'agent_node': all_inds}

def get_trial_dataframe(trial_paths, **kwargs):
    """Create trial dataframe."""

    trial_dicts = []

    for i, p in enumerate(trial_paths):
        trial = json.load(open(p, 'r')) 
        d = {}
        d.update(get_time(trial))
        d.update(get_maze(trial)) 
        d.update(get_maze_kwargs(trial))
        d.update(get_maze_env(trial))        
        d.update(get_action_sequence(trial))    
        d.update({'name': 'Human', 'horizon': None, 'lambda': None, 'model': None}) 
        d.update(get_agent(trial, maze_size=d['maze_size'], y_vertex_min=d['y_vertex_min']))
        d.update(get_states(d['maze_array'], d['agent']))
        move_indexes = d['move_steps']-1
        move_indexes = np.concatenate((move_indexes, [-1]))
        d.update({'discrete_states': d['agent_state'][move_indexes]})
        d.update({'action': d['raw_actions'][d['move_steps']], 'action_times': d['raw_action_times'][d['move_steps']]})
        d.update(_flatten_dict(d['maze_gen_func_kwargs']))
        d.update(_flatten_dict(d['init_state']))

        env = d['env']
        vector_data = []
        step = 0
        obs, reward, done, info = env.reset()
        vector_data.append({'obs': obs, 'reward': reward, 'done': done, 'step': step, 'info': info})
        for step, action in enumerate(d['action']):
            obs, reward, done, info = env.step(action)
            vector_data.append({'obs': obs, 'reward': reward, 'done': done, 'step': step, 'info': info})
        vector_dict = {k: [dic[k] for dic in vector_data] for k in vector_data[0]}
        d.update(vector_dict)
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
