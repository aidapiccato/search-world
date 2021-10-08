class RandomAgent(object):
    def __init__(self, env):
        self._env = env
        self._action_space = self._env.action_space

    def __call__(self, obs):
        """[summary]

        Args:
            obs ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self._action_space.sample()