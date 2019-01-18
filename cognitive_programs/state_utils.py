import numpy as np


DELTA = 0.02


class State:
    def __init__(self, obj_list, pointer, buffer):
        self.obj_list = obj_list
        self.pointer = pointer
        self.buffer = buffer


def is_two_encoded_objs_close(encoded_obj1, encoded_obj2, delta=DELTA):
    return encoded_obj1[2] == encoded_obj2[2] and \
        encoded_obj1[3] == encoded_obj2[3] and \
        distance(encoded_obj1[:2], encoded_obj2[:2]) < delta


def is_two_encoded_obj_lists_close(obj_list1, obj_list2, delta=DELTA):
    assert len(obj_list1) == len(obj_list2)
    num_objs = len(obj_list1)
    is_not_matched_flags = np.ones(num_objs)

    for encoded_obj1 in obj_list1:
        is_obj1_matched = False

        for obj_idx in xrange(num_objs):
            if is_not_matched_flags[obj_idx] and \
                    is_two_encoded_objs_close(encoded_obj1, obj_list2[
                        obj_idx], delta=delta):
                is_not_matched_flags[obj_idx] = 0
                is_obj1_matched = True
                break
        if not is_obj1_matched:
            return False
    return not np.any(is_not_matched_flags)


def encode_object(obj):
    return (obj.shape.centroid.bounds[0], obj.shape.centroid.bounds[1],
            ['circle_shape', 'diamond_shape', 'wedge_shape', 'square_shape',
                'triangle_shape', 'star_shape'].index(obj.shape_name.__name__),
            'yrgb'.index(obj.color))


def cache_list(obj_list):
    return [encode_object(obj) for obj in obj_list]


def is_nearly_same_obj_list_cached(obj_cache, obj_list):
    if len(obj_cache) != len(obj_list):
        return False
    encoded_obj_list2 = [encode_object(obj) for obj in obj_list]
    return is_two_encoded_obj_lists_close(obj_cache, encoded_obj_list2)


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) +
                   (point1[1] - point2[1]) * (point1[1] - point2[1]))
