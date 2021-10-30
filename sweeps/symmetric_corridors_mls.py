"""Generate config overrides for a job array over configs.complex.py.
This defines all of the parameter overrides for a sweep over config parameters.
See python_utils/configs/sweep.py for functions you can use to generate sweeps.
Warning: Do not add or alter any print statements in this file. The launch
script openmind_launch.sh relies on the printing behavior of this file (config
first, then serialized sweep elements) to parse these prints --- the print
statements are the only way this file communicates the sweep to the launch
script.
"""

from python_utils.configs import sweep

_CONFIG_NAME = 'configs.symm_corr_mls'


def _get_param_sweep():
    """Return the sweep we want to launch."""
    n_corridors = [2, 3, 4, 5]
    lengths = [3, 5, 7, 9] 

    param_sweep = []
    for c in n_corridors:
        for l in lengths:
            states = list(range(0, c * l + c, 3)) 
            maze_initial_condition_sweep = sweep.product(
                sweep.zipper(
                    sweep.discrete(('kwargs', 'env', 'kwargs', 'maze_gen_func_kwargs', 'length'), [l]),
                    sweep.discrete(('kwargs', 'env', 'kwargs', 'maze_gen_func_kwargs', 'n_corridors'), [c])),
                sweep.product(
                    sweep.discrete(('kwargs', 'env', 'kwargs', 'maze_gen_func_kwargs', 'agent_initial_position'), states), 
                    sweep.discrete(('kwargs', 'env', 'kwargs', 'maze_gen_func_kwargs', 'target_position'), states)) 
            )            
            param_sweep = sweep.chain(param_sweep, maze_initial_condition_sweep)

    return param_sweep


def main():
    """Generate and write sweep of config overrides."""

    print(_CONFIG_NAME)    

    param_sweep = _get_param_sweep()

    param_sweep = sweep.product(
        param_sweep,
        sweep.discrete(('kwargs', 'model', 'method'), ['MLSAgent']),
    )
    
    param_sweep = sweep.add_log_dir_sweep(param_sweep)
    
   
    for json_spec in sweep.serialize_sweep_elements(param_sweep):
        print(json_spec) 


if __name__ == '__main__':
    main()