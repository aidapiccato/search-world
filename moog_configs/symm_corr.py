"""Grid search task with one-dimensional maze.
TODO: fill in task docstring
"""

import collections
import numpy as np

from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog import tasks
from moog.action_spaces import abstract_action_space
from moog.physics import abstract_physics
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators
from search_world.utils.moog_utils import moog_generator
from dm_env import specs

_N_TIMESTEPS = 2
_MAX_LEN = 10
_MAX_RADIUS = 6
_EDGE_HEIGHT = 0.007
_WALL_THICKNESS = 0.05
_NODE_SCALE = 0.03
_AGENT_SPRITE_SCALE = 0.02
_GRID_SIZE = 1 / (2 * _MAX_LEN)
_REFERENCE_MAZE_Y = 0.75
_SEARCH_MAZE_Y = 0.25
_GRID_CENTER_X = 0.5


def get_config(_):
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    target_node_search_factors = dict(shape='circle', scale=_NODE_SCALE, c0=0.33, c1=1., c2=0.66, opacity=0)
    target_node_reference_factors = dict(shape='circle', scale=_NODE_SCALE, c0=0.33, c1=1., c2=0.66, opacity=255)
    search_node_factors = dict(shape='circle', scale=_NODE_SCALE, c0=0.6, c1=1., c2=1.0, opacity=0)
    reference_node_factors = dict(shape='circle', scale=_NODE_SCALE, c0=0.6, c1=1., c2=1.0)
    reference_edge_factors = dict(c0=0, c1=0, c2=.84, x=0, y=0, opacity=255)
    search_edge_factors = dict(c0=0, c1=0, c2=.84, x=0, y=0, opacity=0)

    def _scale_grid(nodes, grid_size):
        nodes = nodes * grid_size
        nodes[..., 0] += _GRID_CENTER_X
        return nodes

    def node_generator(coordinates, factors):
        return [sprite.Sprite(x=coordinates[idx_coor, ..., 0], y=coordinates[idx_coor, ..., 1], **factors)
                for idx_coor in range(coordinates.shape[0])]

    def edge_generator(coordinates, factors):
        def _make_edge(coordinates):
            coordinates[0, ..., 0] += _AGENT_SPRITE_SCALE
            coordinates[1, ..., 0] -= _AGENT_SPRITE_SCALE
            height = np.zeros_like(coordinates)
            height[..., 1] = _EDGE_HEIGHT

            coordinates = np.vstack((coordinates - height / 2, np.flip(coordinates + height / 2, axis=0)))
            return coordinates

        coordinates = [_make_edge(coordinates[idx_coor]) for idx_coor in range(coordinates.shape[0])]
        return [sprite.Sprite(shape=coor, **factors) for coor in coordinates]

    # Create callable initializer returning entire state
    def state_initializer():
        length = np.random.randint(low=3, high=_MAX_LEN)
        grid_size = 1 / (1.5 * _MAX_LEN)
        moog_specs = moog_generator()
        reference_nodes_coordinates = moog_specs['reference_nodes']
        search_nodes_coordinates = moog_specs['search_nodes']
        reference_edges_coordinates = moog_specs['reference_edges']
        search_edges_coordinates = moog_specs['search_edges']
        target_nodes_search_coordinates = moog_specs['target_nodes_search']
        target_nodes_reference_coordinates = moog_specs['target_nodes_reference']

        # # Agent
        # agent_sprite = [sprite.Sprite(x=0.5, y=_SEARCH_MAZE_Y, shape='circle', scale=_AGENT_SPRITE_SCALE, c0=1., c1=1.,
        #                               c2=1.)]

        # # Search nodes
        # search_nodes_coordinates = _scale_grid(search_nodes_coordinates, grid_size=grid_size)
        # search_nodes_coordinates[:, 1] += _SEARCH_MAZE_Y
        # search_nodes_sprites = node_generator(search_nodes_coordinates, search_node_factors)

        # # Target search nodes
        # target_nodes_search_coordinates = _scale_grid(target_nodes_search_coordinates, grid_size=grid_size)
        # target_nodes_search_coordinates[:, 1] += _SEARCH_MAZE_Y
        # target_nodes_search_sprites = node_generator(target_nodes_search_coordinates, target_node_search_factors)

        # # Target reference nodes
        # target_nodes_reference_coordinates = _scale_grid(target_nodes_reference_coordinates, grid_size=grid_size)
        # target_nodes_reference_coordinates[:, 1] += _REFERENCE_MAZE_Y
        # target_nodes_reference_sprites = node_generator(target_nodes_reference_coordinates,
        #                                                 target_node_reference_factors)

        # Reference nodes
        reference_nodes_coordinates = _scale_grid(reference_nodes_coordinates, grid_size=grid_size)
        reference_nodes_coordinates[:, 1] += _REFERENCE_MAZE_Y
        reference_nodes_sprites = node_generator(reference_nodes_coordinates, reference_node_factors)

        # # Search edges

        # search_edges_coordinates = _scale_grid(search_edges_coordinates, grid_size=grid_size)
        # search_edges_coordinates[..., 1] += _SEARCH_MAZE_Y
        # search_edges_sprites = edge_generator(search_edges_coordinates, search_edge_factors)

        # Reference edges
        reference_edges_coordinates = _scale_grid(reference_edges_coordinates, grid_size=grid_size)
        reference_edges_coordinates[..., 1] += _REFERENCE_MAZE_Y
        reference_edges_sprites = edge_generator(reference_edges_coordinates, reference_edge_factors)

        # Wall
        wall_color = dict(c0=0., c1=0., c2=0.5)
        boundary_walls = shapes.border_walls(visible_thickness=_WALL_THICKNESS, **wall_color)
        center_wall_coordinates = np.asarray(
            [[0, 0.5 - _WALL_THICKNESS / 2], [1, 0.5 - _WALL_THICKNESS / 2], [1, 0.5 + _WALL_THICKNESS / 2],
             [0, 0.5 + _WALL_THICKNESS / 2]])
        center_wall = [sprite.Sprite(shape=center_wall_coordinates, x=0, y=0, c0=0., c1=0., c2=0.5)]
        walls = np.concatenate((boundary_walls, center_wall))

        reference_maze_sprites = np.concatenate((reference_edges_sprites, reference_nodes_sprites))

        # search_maze_sprites = np.concatenate((search_edges_sprites, search_nodes_sprites))

        # target_nodes_sprites = np.concatenate((target_nodes_reference_sprites, target_nodes_search_sprites))
        # agent_sprite[0].mass = _N_TIMESTEPS / grid_size
        state = collections.OrderedDict([
            ('walls', walls),
            ('reference_maze', reference_maze_sprites),
            ('search_maze', []),
            ('target_nodes', []),
            ('agent', [])
            # ('search_maze', search_maze_sprites),
            # ('target_nodes', target_nodes_sprites),
            # ('agent', agent_sprite),
        ])

        return state

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.15)

    forces = (
        (agent_friction_force, 'walls'),
    )

    # constant_speed = physics_lib.ConstantSpeed(layer_names='agent', speed=(1/(1.5 * _MAX_LEN)))

    physics = physics_lib.Physics(*forces, updates_per_env_step=1)

    ############################################################################
    # Task
    ############################################################################

    agent_task = tasks.ContactReward(
        5, layers_0='agent', layers_1='target_nodes', reset_steps_after_contact=50)

    def _should_reset(state, meta_state):
        should_reset = (
                state['target_nodes'][1].opacity == 255
        )
        return should_reset

    reset_task = tasks.Reset(
        condition=_should_reset,
        steps_after_condition=5,
    )

    task = tasks.CompositeTask(agent_task, reset_task)

    ############################################################################
    # Action space
    ############################################################################

    action_space = Jump(
        n_timesteps=_N_TIMESTEPS,
        momentum=0,
        control_velocity=True,
        scaling_factor=1,
        action_layers='agent',
        # extract_borders_fn=_find_borders,
        # check_borders_fn=_within_borders
    )

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )

    ############################################################################
    # Game rules
    ############################################################################

    def _appear(sprite):
        sprite.opacity = 255

    appear_rule_search = game_rules.ModifyOnContact(
        layers_0='search_maze', layers_1='agent', modifier_0=_appear, modifier_1=None)

    appear_rule_target = game_rules.ModifyOnContact(
        layers_0='target_nodes', layers_1='agent', modifier_0=_appear, modifier_1=None)

    rules = [appear_rule_search, appear_rule_target]

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


class ClippedGrid(action_spaces.Grid):
    """Discrete grid action space.

    This action space has 5 actions {left, right, up, down, do-nothing}. These
    actions control either the force or the velocity of the agent(s).
    """

    def __init__(self, extract_borders_fn, check_borders_fn, **kwargs):
        """Constructor.

        Args:
            scaling_factor: Scalar. Scaling factor multiplied to the action.
            agent_layer: String or iterable of strings. Elements (or itself if
                string) must be keys in the environment state. All sprites in
                these layers will be acted upon by this action space.
            control_velocity: Bool. Whether to control velocity (True) or force
                (False).
            momentum: Float in [0, 1]. Discount factor for previous action. This
                should be zero if control_velocity is False, because imparting
                forces automatically gives momentum to the agent(s) being
                controlled. If control_velocity is True, setting this greater
                than zero gives the controlled agent(s) momentum. However, the
                velocity is clipped at scaling_factor, so the agent only retains
                momentum when stopping or changing direction and does not
                accelerate.
        """
        super(ClippedGrid, self).__init__(**kwargs)
        self._extract_border_fn = extract_borders_fn
        self._check_borders_fn = check_borders_fn
        self._borders = None

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict of layers of sprites. Environment state.
            action: Numpy float array of size (2). Force to apply.
        """
        self._action *= self._momentum

        self._action += ClippedGrid._ACTIONS[action]
        self._action = np.clip(
            self._action, -self._scaling_factor, self._scaling_factor)

        for action_layer in self._action_layers:
            for sprite in state[action_layer]:
                if self._check_borders_fn(self._borders, sprite, self._action):
                    if self._control_velocity:
                        sprite.velocity = self._action / sprite.mass
                    else:
                        sprite.velocity += self._action / sprite.mass
                else:
                    sprite.velocity = 0

    def reset(self, state):
        """Reset action space at start of new episode."""
        self._borders = _find_borders(state)
        super().reset(state)


def _within_borders(borders, sprite, action):
    new_position = sprite.position + _N_TIMESTEPS * action / sprite.mass
    within_x = borders[0] <= new_position[0] <= borders[1]
    within_y = borders[2] <= new_position[1] <= borders[3]
    return within_x and within_y


def _find_borders(state):
    x_positions = [s.position[0] for s in state['search_maze']]
    y_positions = [s.position[1] for s in state['search_maze']]
    padding = 2 * _GRID_SIZE
    return [np.amin(x_positions) - padding, np.amax(x_positions) + padding, np.amin(y_positions),
            np.amax(y_positions)]


class Jump(action_spaces.Grid):
    """Discrete, jump-like action space.
    """

    def __init__(self, n_timesteps, **kwargs):
        super(Jump, self).__init__(**kwargs)
        self._n_timesteps = n_timesteps
        self._glue_until = -1
        self._jump_action = 4
        self._min_x = 0
        self._max_x = 1

    def step(self, state, action):
        self._glue_until -= 1
        if self._glue_until > 0:
            super().step(state, self._jump_action)
            return
        else:
            if action != 4:
                self._glue_until = self._n_timesteps
                self._jump_action = action
                super().step(state, self._jump_action)
            else:
                self._jump_action = 4
                super().step(state, self._jump_action)

    def set_n_timesteps(self, n_timesteps):
        self._n_timesteps = n_timesteps

    def reset(self, state):
        self._glue_until = -1
        self._jump_action = 4
        super().reset(state)



