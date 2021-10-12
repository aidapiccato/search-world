from os import readv
import search_world
import itertools
import numpy as np
import matplotlib.pyplot as plt

# class MazeStateSpace(search_world.Space):
#     def __init__(self, shape=None, dtype=None, seed=None) -> None:
#         super().__init__(shape=shape, dtype=dtype, seed=seed)
        
#     def contains(self, state):
        
    

class MazeObservationSpace(search_world.Space):
    """Grid world observation space. Assuming only 4 adjacent blocks are visible. 
    """
    def __init__(self, seed=None): 
        self._observation_space = [list(i) for i in itertools.product([0, 1], repeat=4)]
        super().__init__(len(self._observation_space), np.ndarray, seed)
    
    def contains(self, obs):
        obs = np.asarray(obs)
        return len(list(filter(lambda o: (o == obs).all(), self._observation_space))) > 0
    
    def sample(self):
        return self._observation_space[np.random.choice(len(self._observation_space))]
        
class MazeActionSpace(search_world.Space):
    """A grid world action space with 4 movements
    """
    def __init__(self, seed=None):
        self._action_space = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]        
        super().__init__(len(self._action_space), np.ndarray, seed)

    def sample(self):
        return self._action_space[np.random.choice(len(self._action_space))]
    
    def contains(self, a):
        """Returns true if action performed by agent is contained in grid space.

        Args:
            a (object): Action performed by agent

        Returns:
            bool: True if agent action is valid, false otherwise
        """
        a = np.asarray(a)
        return (-1 <= a).all() and (a <= 1).all() and len(np.flatnonzero(np.abs(a) > 0)) == 1


class Hallway(search_world.Env):
    def __init__(self) -> None:
        """Constructor for hallway class. 
        """        
        super().__init__()
        self.action_space = MazeActionSpace()
        self.observation_space = MazeObservationSpace()
        self._maze = None

    def _observation(self, state) -> object:
        """Generates observation for given coordinate position. Useful for constructing entire observation space

        Args:
            state (tuple of ints): coordinates from which to generate observation. 
        """
        adjacent_nodes = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]]) + state
        adjacent_nodes = [self._maze[node_x, node_y] for (node_x, node_y) in adjacent_nodes]
        return adjacent_nodes

    def _agent_observation(self) -> object:
        """Generates observation based on current location of agent. The observation is a binary vector of length 4 corresponding adjacent nodes have a wall, 0 otherwise. 

        Returns:
            object: Observation corresponding to current position of agent.
        """ 
        return self._observation(self._agent_position)


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
        reward = 0

        if np.all(self._agent_position == self._target_position):
            done = True
            reward = 10

        obs = self._agent_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        """Updates agent position. 

        Args:
            action (object): action performed by agent.
        """

        if action not in self.action_space:
            return ValueError("Agent action not in Maze environment action space")
        new_agent_position = self._agent_position + action
        
        # only update position if tentative new position is a valid one 
        if self._is_valid(new_agent_position):
            self._agent_position = new_agent_position

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
        ax.add_artist(agent)

    def reset(self):
        """Generates new maze, target position, and agent position
        Returns:
            observation (object): observation corresponding to initial state
        """
        # TODO: Move all of this into a maze-generating function
        self._length = np.random.randint(low=3, high=12)
        leaf_nodes = np.arange(1, self._length, 2)
        # creating main empty corridor and leaves 
        self._maze = np.ones((2, self._length))
        self._maze[0] = np.zeros((1, self._length))
        self._maze[1, leaf_nodes] = 0
        # randomly selecting leaf node to be target node
        self._target_position = np.asarray([1, np.random.choice(leaf_nodes)])
        # adding horizontal and vertical padding
        self._maze = np.vstack([np.ones((1, self._length)), self._maze, np.ones((1, self._length))])
        self._maze = np.hstack([np.ones((4, 1)), self._maze, np.ones((4, 1))])
        self._target_position += [1, 1]  
        # start agent somewhere along main corridor
        self._agent_position = np.asarray([1, np.random.choice(self._length) + 1])
        # creating state and observation space
        self._state_space = np.vstack(np.where(self._maze == 0)).T
        self._observation_space = [self._observation(state) for state in self._state_space]
        return self._agent_observation()
        








