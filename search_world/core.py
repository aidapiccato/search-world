from abc import abstractmethod



class Space(object):
    """Defines observation and action space.     
    """
    # TODO: Fill out with more features as included here - https://github.com/openai/gym/blob/45902be4d082e4f2c289a449a30efbf76a0616a2/gym/spaces/space.py#L4

    def __init__(self, shape=None, dtype=None, seed=None) -> None:
        super().__init__()
        import numpy as np
        self._shape = None if shape is None else type(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        if seed is not None:
            self.seed(seed)

    @property
    def shape(self):
        """Return the shape of the space as an immutable property"""
        return self._shape

    @abstractmethod
    def sample(self):
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    @abstractmethod
    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)


class Env(object):
    """Abstract class for implementing an arbitrary environment dynamics. Can be partially or fully observed. Adapted from OpenAI Gym's abstract Env class. (https://github.com/openai/gym/blob/master/gym/core.py) """

    action_space = None
    observation_space = None
    state_space = None

    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of the episode is reached, you are responsible for calling reset() to reset environment's state

        Args:
            action (object):  action provided by agent

        Returns:
            observation (object):  agent's observation of current environmnet
            reward (float): amount of reward returned after previous action
            done (bool): whether episode has ended, in which case further step() calls will return undefined results
            info (dict): auxiliary information
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state and returns an initial observation. Should not reset environment's random number variables

        Returns:
            observation (object): the initial observation
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator 

        Args:
            seed (list<int>, optional): returns list of seeds used in env's random number generators. Defaults to None.
        """
        return

    @abstractmethod
    def close(self):
        """Override close in your subclass to perform any cleanup
        """
        pass

    @abstractmethod
    def render(self, mode="human"):
        """Render the environment

        Args:
            mode (str, optional): mode for rendering. If human, render to current display or terminal. Defaults to "human".
        """
        raise NotImplementedError


class Wrapper(Env):
    """Wraps the environment to allow modular transformations
    # TODO: Continue adding other properties that we might want to add to a wrapper - e.g. spec, etc. -  https://github.com/openai/gym/blob/45902be4d082e4f2c289a449a30efbf76a0616a2/gym/core.py#L344
    """
    def __init__(self, env):
        self.env = env
        self._action_space = None
        self._state_space = None
        self._observation_space = None
        self._reward_range = None
        self._metadata = None

    @property 
    def state_space(self):
        if self._state_space is None:
            return self.env.state_space
        return self._state_space
    
    @state_space.setter
    def state_space(self, space):
        self._state_space = space
        
    @property
    def action_space(self):
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self._action_space = space

    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        self._observation_space = space

    @property
    def metadata(self):
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)




    

