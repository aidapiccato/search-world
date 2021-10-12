import search_world
import hashlib
import numpy as np
import matplotlib.pyplot as plt

class MazeObservationModel(object):
    """Grid world observation model. 
    """
    def __init__(self, observation_space, state_space): 
        self._observation_space = observation_space
        self._state_space = state_space
        self._histogram = {str(state): {str(observation): np.all(self._observation_space[state_idx] == observation) for observation in self._observation_space}
        for (state_idx, state) in enumerate(self._state_space)}

    def probability(self, observation, state):
        return self._histogram[str(state)][str(observation)]

class MazeTransitionModel(object):
    """Grid world transition model
    """
    def __init__(self, state_space, action_space, transition_func) -> None:
        self._state_space = state_space
        self._action_space = action_space
        self._transition_func = transition_func
        self._histogram = {str(state): {str(action): {str(next_state): np.all(self._transition_func(state, action) == next_state) for next_state in self._state_space} for action in action_space} for state in self._state_space}
         

    def probability(self, next_state, state, action):
        return self._histogram[str(state)][str(action)][str(next_state)]
    
class MazeActionSpace(search_world.Space):
    """A grid world action space with 4 movements
    """
    def __init__(self, seed=None):
        self._action_space = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]        
        super().__init__(len(self._action_space), np.ndarray, seed)

    def sample(self):
        return self._action_space[np.random.choice(len(self._action_space))]
    
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
    def __init__(self, maze_gen_func) -> None:
        """Constructor for maze class

        Args:
            _maze_gen_func (func): Function to generate mazes
        """
        super().__init__()

        self.action_space = MazeActionSpace()
        self._maze_gen_func = maze_gen_func
        self._maze = None

    def _observation(self, state) -> object:
        """Generates observation for given coordinate position. Useful for constructing entire observation space

        Args:
            state (tuple of ints): coordinates from which to generate observation. 
        """
        adjacent_nodes = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]]) + state
        adjacent_nodes = [self._maze[node_x, node_y] for (node_x, node_y) in adjacent_nodes]
        is_inf = np.any(np.all(np.isin(self._inf_positions, state, True), axis=1))
        return (adjacent_nodes, is_inf)

    def _agent_observation(self) -> object:
        """Generates observation based on current location of agent. The observation is a binary vector of length 4 corresponding adjacent nodes have a wall, 0 otherwise. 

        Returns:
            object: Observation corresponding to current position of agent.
        """ 
        return self._observation(self._agent_position)

    def agent_reward(self):
        return np.all(self._agent_position == self._target_position) * 10

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
        import pdb; pdb.set_trace()
        self._take_action(action)

        done = False
        reward = self.agent_reward()


        if np.all(self._agent_position == self._target_position):
            done = True

        obs = self._agent_observation()

        return obs, reward, done, {}

    def _transition_func(self, state, action):
        new_state = state + action
        if self._is_valid(new_state):
            return new_state
        return state

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
    
    def render(self, mode="human"):
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

    def reset(self):
        """Generates new maze, target position, and agent position
        Returns:
            observation (object): observation corresponding to initial state
        """
        # creating maze and setting initial conditions
        maze = self._maze_gen_func()
        self._maze = maze['maze']
        self._target_position = maze['target_position']
        self._inf_positions = maze['inf_positions']
        self._agent_position = maze['agent_position']

        # creating state and observation spaces
        self._state_space = np.vstack(np.where(self._maze == 0)).T
        self._observation_space = [self._observation(state) for state in self._state_space]        
        self._observation_model = MazeObservationModel(self._observation_space, self._state_space)
        self._transition_model = MazeTransitionModel(self._state_space, self.action_space, self._transition_func)
        return self._agent_observation()