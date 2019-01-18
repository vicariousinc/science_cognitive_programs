import numpy as np

STATE_FEATURE_NAMES = ['square_shape', 'circle_shape', 'diamond_shape',
                       'triangle_shape',
                       'star_shape', 'r', 'g', 'b', 'y', 'pointer']
MIN_NUM_EXS = 10
NUM_FEATURES = 21
DIST_DELTA = 0.1
IMG_RESOLUTION = 14  # the distance between objects is 0.1: 1/(0.1/sqrt(2))


def emulator_state_to_array(objs, pointer, names=STATE_FEATURE_NAMES, resolution=IMG_RESOLUTION):
    arr = np.zeros((len(names), resolution + 1, resolution + 1))

    for obj in objs:
        loc = np.floor((obj.get_center() * resolution)).astype('int')
        if loc[0] > IMG_RESOLUTION or loc[1] > IMG_RESOLUTION:
            return None
        shape_name = obj.shape_name.__name__
        arr[names.index(shape_name), loc[0], loc[1]] = 1
        color = obj.color
        arr[names.index(color), loc[0], loc[1]] += 1

        if pointer is not None:
            ploc = np.floor((pointer * resolution)).astype('int')
            arr[names.index('pointer'), ploc[0], ploc[1]] = 1
    return arr


def convert_one_example_set_to_feature_vector(examples,
                                              names=STATE_FEATURE_NAMES,
                                              resolution=IMG_RESOLUTION):
    """ Convert a set of examples for the same program to a feature vector. The
    feature vector is derived from the summary of the difference
    between input and output state. Specifically, the max and the sum of (
    output_state_array - input_state_array) """

    num_exs = len(examples)
    img_size = resolution + 1

    num_channels = 2 * len(names) + 1  # 3*len(names) + 1

    features = np.zeros((num_exs, num_channels, img_size, img_size))

    if len(examples[0]) == 3:
        examples = [(input_state, output_state) for (input_state,
                                                     output_state, _) in examples]

    for ex_idx, (input_state, output_state) in enumerate(examples):

        input_state_array = emulator_state_to_array(input_state.obj_list,
                                                    input_state.pointer,
                                                    names,
                                                    resolution=resolution)
        output_state_array = emulator_state_to_array(output_state.obj_list,
                                                     output_state.pointer,
                                                     names,
                                                     resolution=resolution)

        in_out_state_diff = output_state_array - input_state_array

        features_one_example = np.zeros((num_channels, img_size, img_size))
        features_one_example[:len(names)] = in_out_state_diff
        features_one_example[len(names): 2 * len(names)] = input_state_array

        features_one_example[-1] = np.array(np.abs(in_out_state_diff).sum(0) >= 1,
                                            dtype=float)
        features[ex_idx] = features_one_example

    features = features.transpose([0, 2, 3, 1])

    return features
