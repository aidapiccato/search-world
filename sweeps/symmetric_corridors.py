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

_CONFIG_NAME = 'configs.symmetric_corridors'


def _get_param_sweep():
    """Return the sweep we want to launch."""
    param_sweep = sweep.product(
        sweep.discrete(('kwargs', 'env', 'kwargs', 'maze_gen_func_kwargs', 'length'), ['12', '13']),
        sweep.discrete(('kwargs', 'env', 'kwargs', 'maze_gen_func_kwargs', 'n_corridors'), ['4', '5']))
    #     sweep.zipper(
    #         sweep.discrete(
    #             ('kwargs', 'sub_module', 'kwargs', 'd', 'kwargs', 'e'),
    #             [16, 17, 18]),
    #         sweep.discrete(
    #             ('kwargs', 'sub_module', 'kwargs', 'd', 'kwargs', 'f'),
    #             [17, 18, 19]),
    #     ),
    # )

    # param_sweep_0 = sweep.product(
    #     sweep.discrete(('kwargs', 'a'), [0.11, 0.12]),
    #     sub_module_param_sweep,
    # )
    # param_sweep_1 = sweep.discrete(('kwargs', 'a'), [0.13, 0.14])

    # param_sweep = sweep.chain(param_sweep_0, param_sweep_1)

    return param_sweep


def main():
    """Generate and write sweep of config overrides."""


    print(_CONFIG_NAME)

    # Define the sweep we want to launch:
    param_sweep = _get_param_sweep()

    param_sweep = sweep.add_log_dir_sweep(param_sweep)

    # Note, it is absolutely fine to add more overrides to the sweep here that
    # you don't want to include in the log_dir (to keep the log_dir short). But
    # since the log_dir override has already been added be sure to not add any
    # additional elements to the sweep, i.e. make sure that anything added here
    # is a singleton sweep. For example, I often add parameters like how often
    # scalars/images should be logged, batch size, etc. that I don't want to
    # sweep over but want to override for array launches from the values in the
    # config. Here's an example:
    # param_sweep = sweep.product(
    #     param_sweep,
    #     sweep.discrete(('kwargs', 'b'), [-0.4]),
    # )

    # Print one spec per line. It is important to print these out line by line,
    # because openmind_launch.sh relies on these prints, piping them into an
    # array that it uses to launch to job array.
    for json_spec in sweep.serialize_sweep_elements(param_sweep):
        print(json_spec)


if __name__ == '__main__':
    main()