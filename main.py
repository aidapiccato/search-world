
  
"""Main entry point to run this project.
Before running, first install the dependencies listed in ../README.md.
To train a model locally, run the following command:
$ python3 main.py --config=$your_config_name
Consider also adding the flag --pdb_post_mortem to enter a breakpoint upon
error.
Once the model is training (or has trained) locally, you can visualize logged
scalars and images in Tensorboard by running the following command (in a
different terminal window):
$ tensorboard --logdir=logs/$run_number/tensorboard
where $run_number is the directory in which logs are written (you can find this
printed at the beginning of model training).
If running a sweep on openmind, see the openmind_launch.sh file.
Also, see template project here for a similar example with openmind sweeps:
https://github.mit.edu/jazlab/python_utils/tree/master/template_project
"""

from absl import flags
from absl import app

import importlib
import logging
import os

from python_utils.configs import build_from_config
from python_utils.configs import override_config

FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    'configs.lif_rnn',
                    'Module name of task config to use.')
flags.DEFINE_string(
    'config_overrides', '',
    'JSON-serialized config overrides. This is typically not used locally, '
    'only when running sweeps on Openmind.')
flags.DEFINE_string('log_directory', 'logs', 'Prefix for the log directory.')
flags.DEFINE_string(
    'metadata', '',
    'Metadata to write to metadata.log file. Often used for slurm task ID.')


def main(_):

    ############################################################################
    # Load config
    ############################################################################

    config_module = importlib.import_module(FLAGS.config)
    config = config_module.get_config()
    logging.info(FLAGS.config_overrides)

    # Apply config overrides
    config = override_config.override_config_from_json(config,
                                                       FLAGS.config_overrides)

    ############################################################################
    # Create logging directory
    ############################################################################

    log_dir = FLAGS.log_directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # If log_dir is not empty, create a new enumerated sub-directory in it for
    # logging.
    list_log_dir = os.listdir(log_dir)
    if len(list_log_dir) != 0:  # For safety, explicitly use len instead of bool
        existing_log_subdirs = [
            int(filename) for filename in list_log_dir if filename.isdigit()]
        if not existing_log_subdirs:
            existing_log_subdirs = [-1]
        new_log_subdir = str(max(existing_log_subdirs) + 1)
        log_dir = os.path.join(log_dir, new_log_subdir)
        os.mkdir(log_dir)

    logging.info('Log directory: {}'.format(log_dir))

    ############################################################################
    # Log config name, config overrides, config, and metadata
    ############################################################################

    def _log(log_filename, thing_to_log):
        f_name = os.path.join(log_dir, log_filename)
        logging.info('In file {} will be written:'.format(log_filename))
        logging.info(thing_to_log)
        f_name_open = open(f_name, 'w+')
        f_name_open.write(thing_to_log)

    _log('config_name.log', FLAGS.config)
    _log('config_overrides.log', FLAGS.config_overrides)
    _log('config.log', str(config))
    _log('metadata.log', FLAGS.metadata)

    ############################################################################
    # Build and run experiment
    ############################################################################

    experiment = build_from_config.build_from_config(config)
    experiment(log_dir=log_dir)


if __name__ == '__main__':
    app.run(main)