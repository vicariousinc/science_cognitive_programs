import copy
import os

import numpy as np
from keras import backend
from keras.models import model_from_json
from tensorflow import keras

from cognitive_programs.arg_prediction.feature_and_label_generator import \
    MIN_NUM_EXS, NUM_FEATURES, IMG_RESOLUTION, \
    convert_one_example_set_to_feature_vector
from cognitive_programs.program_induction import utils as pu
from cognitive_programs.tools.system import ARG_PREDICTION_MODEL_PATH

PROB_THRESHOLD = 0.55


def predict_arg_order_all_tasks_with_prob(
        concepts_dict,
        depth=5,
        is_complete_dep_graph=True):
    """
    Similar to the function predict_arg_order_all_tasks, but output arg
    choices with probabilities
    """
    features, task_names = convert_all_examples_to_features(concepts_dict)
    num_tasks = features.shape[0]

    target_funcs = [5, 6, 13]

    arg_order_with_task_names = {}
    for task_name in task_names:
        arg_order_with_task_names[task_name] = {}

    for target_func in target_funcs:
        cmds = pu.create_commands_dict()
        args = cmds[target_func][1]

        y_pred_all_args = np.zeros((num_tasks, len(args)))

        for arg_idx in xrange(len(args)):
            model_path = ARG_PREDICTION_MODEL_PATH + \
                '/model_func={' \
                '}_numFeatures={' \
                '}_imgResolution={' \
                '}_depth={' \
                '}_isCompleteDepGraph={' \
                '}_arg={}'.format(
                    target_func,
                    NUM_FEATURES,
                    IMG_RESOLUTION,
                    depth,
                    is_complete_dep_graph,
                    args[arg_idx])
            loaded_model = load_model(model_path)
            y_pred_one_arg = loaded_model.predict(features)
            y_pred_all_args[:, arg_idx:(arg_idx + 1)] = y_pred_one_arg
            backend.clear_session()

        for task_idx in xrange(num_tasks):
            task_name = task_names[task_idx]

            args_with_prob = []
            for arg_idx in xrange(len(args)):
                prob = y_pred_all_args[task_idx][arg_idx]

                args_with_prob.append((args[arg_idx], prob))

            arg_order_with_task_names[task_name][target_func] = args_with_prob
    return arg_order_with_task_names


def augment_examples(examples_one_concept, min_num_exs=MIN_NUM_EXS):
    num_exs = len(examples_one_concept)
    if num_exs < min_num_exs:
        for i in xrange(num_exs, MIN_NUM_EXS):
            idx_random = np.random.choice(num_exs)
            ex_sampled = copy.deepcopy(examples_one_concept[idx_random])
            examples_one_concept.append(ex_sampled)

    return examples_one_concept


def convert_all_examples_to_features(concepts_dict):
    features = []
    task_names = []
    for task_name, examples_one_concept in concepts_dict.iteritems():
        # if task_name in TARGET_PROGRAMS:
        examples_augmented = augment_examples(examples_one_concept[
                                              :MIN_NUM_EXS])
        feature = convert_one_example_set_to_feature_vector(
            examples_augmented)
        features.append(feature)
        task_names.append(task_name)

    print(features[0].shape)
    features_np = np.zeros((len(features),) + (features[0].shape))
    for feat_idx, x in enumerate(features):
        features_np[feat_idx] = features[feat_idx]

    num_exs = MIN_NUM_EXS
    num_channels = NUM_FEATURES
    img_size = (IMG_RESOLUTION + 1)
    features = features_np.reshape(
        (features_np.shape[0],
         num_exs * img_size * img_size, num_channels))
    return features, task_names


def load_model(model_name):
    # load json and create model
    json_file = open(
        os.path.join(ARG_PREDICTION_MODEL_PATH,
                     '{}.json'.format(model_name)), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={
        'keras': keras
    })
    # load weights into new model
    loaded_model.load_weights(os.path.join(ARG_PREDICTION_MODEL_PATH,
                                           "{}.h5".format(model_name)))
    return loaded_model
