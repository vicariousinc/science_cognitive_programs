[![](data/vicarious_logo.png)](https://www.vicarious.com)

# Implementation of concept induction for Science Robotics paper

## Setup

To run the example commands below:

- Place the oracle_cache folder into the root directory (same as this file); download the compressed file under "*Code and data --> Cached emulator data for our datasets [required]*" towards the bottom of this post https://www.vicarious.com/2019/01/18/a-thought-is-a-program/
- Install the following python packages: numpy, scipy, keras, tensorflow, shapely, numba, and parmap
- Add the cognitive_programs directory to your PYTHONPATH

## Example Commands

- `python cognitive_programs/program_induction/search.py --arg_prediction_type mix_fixation_provided_and_cnn_arg_predictor --n_progs 500000`

- `python cognitive_programs/program_induction/search.py --arg_prediction_type mix_fixation_provided_and_cnn_arg_predictor --n_progs 3000000`

- `python  cognitive_programs/program_induction/search.py --arg_prediction_type cnn_arg_predictor --n_progs 1000000`

- `python  cognitive_programs/program_induction/search.py --arg_prediction_type uniform_arg --n_progs 1000000`
