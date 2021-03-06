"""PacMan-like task.

In this task all objects move in a maze with the same speed. The maze is
randomized every trial. The subject controls a green agent. Red ghost agents
wander the maze randomly without backtracking. The subject's goal is to collect
all yellow pellets in the maze. Ghosts only begin moving once agent moves.
"""

import collections
from re import S
from moog.maze_lib import maze
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import maze_lib
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks
from moog import shapes

from search_world.utils.moog_utils import moog_generator
_WALL_THICKNESS = 0.025
_N_TIMESTEPS = 6
_Y_VERTEX_MIN = 0.25
_X_VERTEX_MIN = 0 
_DISPLAY_SIZE = 0.5


def _get_config(maze_size_fun, occluder_rad_fun, occluder_opacity_fun, y_vertex_min_fun):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    # Agent
    agent_factors = dict(shape='circle', scale=0.02, c0=0.33, c1=1., c2=0.66)

    # Prey
    prey_factors = dict(shape='circle', scale=0.02, c0=0.2, c1=1., c2=1.)

    # Occluder
    occluder_factors = dict(c0=0, c1=0, c2=0.2)
    def state_initializer():    
        nonlocal maze_size_fun
        nonlocal occluder_opacity_fun
        nonlocal occluder_rad_fun
        nonlocal y_vertex_min_fun
        
        # maze_size = maze_size_fun()
        occluder_rad = occluder_rad_fun()
        y_vertex_min = y_vertex_min_fun()
        occluder_opacity = occluder_opacity_fun() 
        # Creating base search maze         

        orig_maze_array, maze_array, maze_obj, agent_init_pos, prey_pos = moog_generator()
        maze_size = orig_maze_array.shape
        ambient_size = maze_array.shape


        translation_x = np.random.randint(low=0, high=ambient_size[0]-(maze_size[0]-1)-2)
        translation_y = np.random.randint(low=0, high=ambient_size[1]-(maze_size[1]-1)-2) 
        print(maze_size, ambient_size, translation_x, translation_y)

        # Translating maze to produce search maze        
        search_maze_array = np.roll(maze_array, shift=translation_x, axis=0)
        search_maze_array = np.roll(search_maze_array, shift=translation_y, axis=1)
        
        search_maze = maze_lib.Maze(search_maze_array, y_vertex_min=y_vertex_min, x_vertex_min=_X_VERTEX_MIN, grid_size=_DISPLAY_SIZE)
        search_walls = search_maze.to_sprites(c0=0., c1=0., c2=0.4)
        
         # Creating reference maze
        ref_maze = maze_lib.Maze(maze_array, y_vertex_min=_Y_VERTEX_MIN, x_vertex_min=_X_VERTEX_MIN + 0.5, grid_size=_DISPLAY_SIZE)
        ref_walls = ref_maze.to_sprites(c0=0., c1=0., c2=0.4) 

        # Creating display walls to split the reference and search maze
        display_walls = [sprite.Sprite(shape=np.asarray(
            [[0.5 - _WALL_THICKNESS / 2, 1], [0.5 + _WALL_THICKNESS / 2, 1], 
            [0.5 + _WALL_THICKNESS / 2, 0], [0.5 - _WALL_THICKNESS / 2, 0]]), x=0, y=0, c0=0., c1=0., c2=0.5)]


        search_maze_offset = [y_vertex_min, _X_VERTEX_MIN] 

        # Agents
        agent_position = search_maze.grid_side * (np.array(agent_init_pos +  [translation_x, translation_y]) + 0.5) + search_maze_offset  
        agent = [sprite.Sprite(
            x=agent_position[1], y=agent_position[0], metadata={'env': maze_obj.info()}, **agent_factors)]

        # Creating occluder
        occluder_shape = shapes.annulus_vertices(occluder_rad, 1.)
        occluder = sprite.Sprite(x=agent_position[1], y=agent_position[0], shape=occluder_shape, opacity=occluder_opacity, **occluder_factors)

        # Creating black background for reference maze
        background_shape = np.asarray([[0.5, 1.], [1., 1.], [1., 0], [0.5, 0]])
        background = sprite.Sprite(x=0, y=0, shape=background_shape, scale=1, c0=0, c1=0, c2=0, opacity=255)
        
        # Placing prey in reference maze
        ref_prey = []
        ref_maze_offset = [_Y_VERTEX_MIN, _X_VERTEX_MIN]

        ref_prey_pos = ref_maze_offset + ref_maze.grid_side * (np.array(prey_pos) + 0.5)
        ref_prey.append(sprite.Sprite(x=ref_prey_pos[1]+0.5, y=ref_prey_pos[0], **prey_factors))

        # Place prey at a single location in maze
        prey = []

        target_prey_pos = search_maze_offset + search_maze.grid_side * (np.array(prey_pos + [translation_x, translation_y])  + 0.5)
        prey.append(sprite.Sprite(x=target_prey_pos[1], y=target_prey_pos[0], opacity=255, **prey_factors))
            

        agent[0].mass = _N_TIMESTEPS/ref_maze.grid_side
        occluder.mass = _N_TIMESTEPS/ref_maze.grid_side
        state = collections.OrderedDict([
            ('walls', search_walls),
            ('prey', prey),
            ('ghosts', []),
            ('agent', agent),
            ('occluder', [occluder]),
            ('background', [background]),
            ('ref_walls', ref_walls),
            ('ref_prey', ref_prey),
            ('display_walls', display_walls),
        ])
        return state

    ############################################################################
    # Physics
    ############################################################################

    maze_physics = physics_lib.MazePhysics(
        maze_layer='walls',
        avatar_layers=('agent', 'prey', 'occluder'),
    )

    physics = physics_lib.Physics(
        (physics_lib.RandomMazeWalk(speed=0.015), ['ghosts']),
        updates_per_env_step=1, corrective_physics=[maze_physics],
    )

    ############################################################################
    # Task
    ############################################################################

 
    prey_task = tasks.ContactReward(1, layers_0='agent', layers_1='prey')
    reset_task = tasks.Reset(
        condition=lambda state: len(state['prey']) == 0,
        steps_after_condition=5,
    )
    task = tasks.CompositeTask(prey_task, reset_task, timeout_steps=1000)

    ############################################################################
    # Action space
    ############################################################################
 
    action_space = Jump(
        n_timesteps=_N_TIMESTEPS,
        scaling_factor=1,
        action_layers=('agent', 'occluder'),
        control_velocity=True,
        momentum=0.,  # Value irrelevant, since maze_physics has constant speed
    )
    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(256, 256),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )

    ############################################################################
    # Game rules
    ############################################################################


    vanish_on_contact = game_rules.VanishOnContact(
        vanishing_layer='prey', contacting_layer='agent')

    rules = (vanish_on_contact, )

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'game_rules': rules,
    }
    return config


def get_config(level):
    """Get config dictionary of kwargs for environment constructor.
    
    Args:
        level: Int. Different values yield different maze sizes and numbers of
            ghosts.
    """
    if level == 2:
        return _get_config(
            maze_size_fun=lambda: np.random.randint(low=5, high=10),
            occluder_rad_fun = lambda: 0.0273,
            occluder_opacity_fun = lambda: 255,
            y_vertex_min_fun=lambda: np.random.uniform(low=0, high=0.5)
        )
    elif level == 1:
        return _get_config(
            maze_size_fun=lambda: np.random.randint(low=3, high=5),
            occluder_rad_fun = lambda: 0.03,
            occluder_opacity_fun = lambda: 255,
            y_vertex_min_fun=lambda: np.random.uniform(low=0, high=0.5)
        )
    elif level == 0:
        return _get_config(
            maze_size_fun=lambda: np.random.randint(low=4, high=10),
             occluder_rad_fun = lambda: 0.02,
            occluder_opacity_fun = lambda: 0,
            y_vertex_min_fun=lambda: np.random.uniform(low=0, high=0.5)
        )
    else:
        raise ValueError('Invalid level {}'.format(level))

class Jump(action_spaces.Grid):
    """Discrete, jump-like action space.
    """

    def __init__(self, n_timesteps, **kwargs):
        super(Jump, self).__init__(**kwargs)
        if not isinstance(kwargs['action_layers'], (list, tuple)):
            action_layers = (kwargs['action_layers'], )
        self._action_layers = kwargs['action_layers']
        self._n_timesteps = n_timesteps
        self._glue_until = -1
        self._jump_action = 4 

    def step(self, state, action):
        self._glue_until -= 1
        if self._glue_until > 0:
            super().step(state, self._jump_action)
            return
        else:
            if action != 4:
                self._glue_until = self._n_timesteps
                self._jump_action = action
                for action_layer in self._action_layers:
                    for sprite in state[action_layer]:
                        sprite.velocity = 0
                super().step(state, self._jump_action)
            else:
                self._jump_action = 4                
                for action_layer in self._action_layers:
                    for sprite in state[action_layer]:
                        sprite.velocity = 0
                super().step(state, self._jump_action)

    def set_n_timesteps(self, n_timesteps):
        self._n_timesteps = n_timesteps

    def reset(self, state):
        self._glue_until = -1
        self._jump_action = 4
        super().reset(state)



