# search-world

Library for running RL agents on search tasks in maze-like environments. 


## Code organization

The entry point for model training is main.py, which runs a config. Configs are in the configs directory and fully specify the model-training experiment. 

In the search-world directory are also trainer.py, which generates a general model trainer. All of the models can be found in models/. All of the environments can be found in envs/

## Running locally

`python3 main.py --config='configs.hallway'`
