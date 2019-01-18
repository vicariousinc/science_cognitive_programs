import os
import cognitive_programs

COGPROGS_PATH = cognitive_programs.__path__[0]

SYMBOLIC_DATA_PATH = os.path.join(COGPROGS_PATH, '../', 'data', 'symbolic',
                                  'training_examples.pkl')
ARG_PREDICTION_MODEL_PATH = os.path.join(COGPROGS_PATH, 'arg_prediction',
                                         'model')
TARGET_PROGRAMS_PATH = os.path.join(COGPROGS_PATH, '../', 'data', 'symbolic',
                                    'target_programs.pkl')
