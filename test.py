from search_world.envs import hallway

if __name__ == '__main__':
    hallway_env = hallway.Hallway()
    hallway_env.reset()
    action = [1, 0]
    print(hallway_env.step(action))