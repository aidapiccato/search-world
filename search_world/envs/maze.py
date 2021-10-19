import search_world
import numpy as np
import matplotlib.pyplot as plt
import sys
class MazeObservationModel(object):
    """Grid world observation model. 
    """
    def __init__(self, observation_space, state_space): 
        self._observation_space = observation_space
        self._state_space = state_space
        self._histogram = {str(state): {str(observation): np.all(self._observation_space[state_idx] == observation) for observation in self._observation_space}
        for (state_idx, state) in enumerate(self._state_space)}

    def __call__(self, observation, state):
        return self._histogram[str(state)][str(observation)]

class MazeRewardModel(object):
    """Grid world reward model"""
    def __init__(self, state_space, reward_func):
        self._reward_map = {str(state): reward_func(state) for state in state_space}

    def __call__(self, state):
        return self._reward_map[str(state)]


class MazeTransitionModel(object):
    """Grid world transition model
    """
    def __init__(self, state_space, action_space, transition_func) -> None:
        self._state_space = state_space
        self._action_space = action_space
        self._transition_func = transition_func
        self._histogram = {str(state): {str(action): {str(next_state): np.all(self._transition_func(state, action) == next_state) for next_state in self._state_space} for action in action_space} for state in self._state_space}
         

    def __call__(self, next_state, state, action):
        return self._histogram[str(state)][str(action)][str(next_state)]
    
class MazeActionSpace(search_world.Space):
    """A grid world action space with 4 movements
    """
    def __init__(self, seed=None):
        self._action_space = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]        
        super().__init__(len(self._action_space), np.ndarray, seed)

    def sample(self):
        return self._action_space[np.random.choice(len(self._action_space))]
    
    def __getitem__(self, index):
        return self._action_space[index]

    def __iter__(self):
        return iter(self._action_space)

    def contains(self, a):
        """Returns true if action performed by agent is contained in grid space.

        Args:
            a (object): Action performed by agent

        Returns:
            bool: True if agent action is valid, false otherwise
        """
        a = np.asarray(a)
        return (-1 <= a).all() and (a <= 1).all() and len(np.flatnonzero(np.abs(a) > 0)) == 1

class Maze(search_world.Env):
    def __init__(self, maze_gen_func, maze_gen_func_kwargs, max_steps) -> None:
        """Constructor for maze class

        Args:
            _maze_gen_func (func): Function to generate mazes
        """
        super().__init__() 
        self.action_space = MazeActionSpace()
        self._max_steps = max_steps
        self._maze_gen_func = maze_gen_func
        self._maze_gen_func_kwargs = maze_gen_func_kwargs
        self._maze = None


    def _observation(self, state) -> object:
        """Generates observation for given coordinate position. Useful for constructing entire observation space

        Args:
            state (tuple of ints): coordinates from which to generate observation. 
        """
        adjacent_nodes = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]]) + state
        adjacent_nodes = [self._maze[node_x, node_y] for (node_x, node_y) in adjacent_nodes]
        is_inf = np.any(np.all(np.isin(self._inf_positions, state, True), axis=1))
        is_target = np.all(state == self._target_position)
        return (adjacent_nodes, is_inf, is_target)

    def _agent_observation(self) -> object:
        """Generates observation based on current location of agent. The observation is a binary vector of length 4 corresponding adjacent nodes have a wall, 0 otherwise. 

        Returns:
            object: Observation corresponding to current position of agent.
        """ 
        return self._observation(self._agent_position)

    def agent_reward(self):
        return self._reward_func(state=self._agent_position)

    def info(self):
        info =  self._maze_gen_func_kwargs
        info.update({'min_dist': self._min_dist, 'max_dist': self._max_dist})
        return info
    
    def step(self, action):
        """ Executes one time step within the environment. 

        Args:
            action (object): Object corresponding to agent's current action            
        
        Returns:
            observation (object): observation corresponding to novel position in environment
            reward (float): positive reward if agent has found target, 0 otherwise
            done (bool): True if agent has found target or environment times out, False otherwise.
            info (dict): auxiliary information
        """ 

        self._take_action(action)

        done = False
        reward = self._reward_func(self._agent_position)


        if np.all(self._agent_position == self._target_position):
            done = True

        obs = self._agent_observation()

        self._num_steps += 1

        if self._num_steps >= self._max_steps:
            done = True
        
        info = {'agent_position': self._agent_position}
        return obs, reward, done, info


    def _find_shortest_path_is_valid(self,maze, visited, x, y):
        return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and not (maze[x][y] == 1 or visited[x][y])

    def _find_shortest_path_helper(self, maze, visited, x, y, dest, min_dist=sys.maxsize, dist=0):

        # if you've reached destination, return minimum
        if np.all((x, y) == dest):
            return min(dist, min_dist)

        visited[x][y] = 1

        if self._find_shortest_path_is_valid(maze, visited, x+1, y):
            min_dist = self._find_shortest_path_helper(maze, visited, x + 1, y, dest, min_dist, dist+1)

        if self._find_shortest_path_is_valid(maze, visited, x, y+1):
            min_dist = self._find_shortest_path_helper(maze, visited, x, y+1, dest, min_dist, dist+1)

        if self._find_shortest_path_is_valid(maze, visited, x-1, y):
            min_dist = self._find_shortest_path_helper(maze, visited, x-1, y, dest, min_dist, dist+1)

        if self._find_shortest_path_is_valid(maze, visited, x, y-1):
            min_dist = self._find_shortest_path_helper(maze, visited, x, y-1, dest, min_dist, dist+1)

        visited[x][y] = 0

        return min_dist


    def _find_shortest_path(self):
        # TODO: Document _find_shortest_path
        # TODO: Move to maze_utils 
        s_x, s_y = self._agent_initial_position        
        d_x, d_y = self._target_position

        visited = np.zeros_like(self._maze)

        min_dist = self._find_shortest_path_helper(self._maze, visited, s_x, s_y, (d_x, d_y))
        return min_dist


    def _transition_func(self, state, action):
        new_state = state + action
        if self._is_valid(new_state):
            return new_state
        return state

    def _reward_func(self, state):
        return np.all(state == self._target_position) * 10 + (1 - np.all(state == self._target_position)) * -3

    def _take_action(self, action):
        """Updates agent position. 

        Args:
            action (object): action performed by agent.
        """
        if action not in self.action_space:
            return ValueError("Agent action not in Maze environment action space")
        self._agent_position = self._transition_func(self._agent_position, action)


    def _is_valid(self, position):
        """Returns true if given position is valid for agent occupancy

        Args:
            position (object): coordinates of object whose position is being checked
        """
        if (0 <= position).all() and (position < self._maze.shape).all():
            return self._maze[position[0], position[1]] != 1
        return False
    
    def render(self, ax=None, mode="human"):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self._maze, cmap='gray')
        target = plt.Circle((self._target_position[1], self._target_position[0]), radius=0.5, color='yellow')
        ax.add_artist(target)
        agent = plt.Circle((self._agent_position[1], self._agent_position[0]), radius=0.5)        
        for inf_pos in self._inf_positions:
            inf = plt.Circle((inf_pos[1], inf_pos[0]), radius=0.5, color='red')
            ax.add_artist(inf)
        ax.add_artist(agent)
        ax.set_title('obs = {}, reward={}'.format(self._agent_observation(), self.agent_reward()))

    def _find_longest_path(self):
        # TODO: Make this not specific to the H mazes. Currently it is!
        v = self._state_space.shape[0]
        e = (self._maze_gen_func_kwargs['length'] - 1) * self._maze_gen_func_kwargs['n_corridors'] + self._maze_gen_func_kwargs['n_corridors']
        return 2 * v * e

    def reset(self):
        """Generates new maze, target position, and agent position
        Returns:
            observation (object): observation corresponding to initial state
        """
        # creating maze and setting initial conditions
        maze = self._maze_gen_func(**self._maze_gen_func_kwargs)
        self._maze = maze['maze']
        self._target_position = maze['target_position']
        self._inf_positions = maze['inf_positions']
        self._agent_initial_position = maze['agent_initial_position']
        self._agent_position = self._agent_initial_position

        # creating state and observation spaces
        self._state_space = np.vstack(np.where(self._maze == 0)).T
        self._observation_space = [self._observation(state) for state in self._state_space]        
        self._observation_model = MazeObservationModel(self._observation_space, self._state_space)
        self._transition_model = MazeTransitionModel(self._state_space, self.action_space, self._transition_func)
        self._reward_model = MazeRewardModel(self._state_space, self._reward_func)
        self._min_dist = self._find_shortest_path()
        self._max_dist = self._find_longest_path()
        self._num_steps = 0
        # obs, reward, done, info

        return self.step([0, 0])
