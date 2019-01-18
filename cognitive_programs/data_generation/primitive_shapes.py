"""Some basic shape and sampling functions"""
import collections

import numpy as np
import shapely.geometry as sg


WIDTH_MAX = 0.15
WIDTH_MIN = 0.05

ObjTargetAttributes = collections.namedtuple('ObjTargetAttributes', ['shape',
                                                                     'color',
                                                                     'more_than_one'])


class Obj:
    def __init__(self, shape, color=None, shape_name=None, size=None, id=None):
        self.shape = shape
        self.color = color
        self.size = size
        self.shape_name = shape_name
        self.id = id

    def get_center(self):
        return np.array([self.shape.centroid.bounds[0],
                         self.shape.centroid.bounds[1]])


def circle_shape(center=None, width=None, width_max=WIDTH_MAX,
                 target_color=None,
                 exclude_colors=None, direction=None):
    # Center is sampled randomly.
    center, width, target_color = sample_loc_scale_color(center=center,
                                                         width=width,
                                                         width_max=width_max,
                                                         target_color=target_color,
                                                         exclude_colors=exclude_colors,
                                                         direction=direction)
    radius = width / 2.0
    circ = sg.Point(center).buffer(radius)
    return Obj(circ, target_color, circle_shape)


def diamond_shape(center=None, width=None, width_max=WIDTH_MAX,
                  target_color=None,
                  exclude_colors=None, direction=None):
    # Center is sampled randomly.

    center, width, color = sample_loc_scale_color(center=center, width=width,
                                                  target_color=target_color,
                                                  width_max=width_max,
                                                  exclude_colors=exclude_colors,
                                                  direction=direction)

    ref_diamond = np.array([[-0.5, 0], [0, 0.5], [0.5, 0], [0, -0.5]])
    dia_coords = ref_diamond * width + center

    diamond = sg.Polygon(dia_coords)
    return Obj(diamond, color, diamond_shape)


def square_shape(center=None, width=None, width_max=WIDTH_MAX,
                 target_color=None,
                 exclude_colors=None, direction=None):
    # Center is sampled randomly.

    center, width, color = sample_loc_scale_color(center=center, width=width,
                                                  target_color=target_color,
                                                  width_max=width_max,
                                                  exclude_colors=exclude_colors,
                                                  direction=direction)

    ref_diamond = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    sqr_coords = ref_diamond * width + center

    sqr = sg.Polygon(sqr_coords)
    return Obj(sqr, color, square_shape)


def triangle_shape(center=None, width=None, width_max=WIDTH_MAX,
                   target_color=None,
                   exclude_colors=None, direction=None):
    # Center is sampled randomly.
    center, width, color = sample_loc_scale_color(center=center,
                                                  width=width,
                                                  width_max=width_max,
                                                  target_color=target_color,
                                                  exclude_colors=exclude_colors,
                                                  direction=direction)

    ref_diamond = np.array([[0, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    sqr_coords = ref_diamond * width + center

    sqr = sg.Polygon(sqr_coords)
    return Obj(sqr, color, triangle_shape)


def star_shape(center=None, width=None, width_max=WIDTH_MAX, target_color=None,
               exclude_colors=None, direction=None):
    # Center is sampled randomly.

    center, width, color = sample_loc_scale_color(center=center,
                                                  width=width,
                                                  width_max=width_max,
                                                  target_color=target_color,
                                                  exclude_colors=exclude_colors,
                                                  direction=direction)

    ref_star = np.array([[0.0, 0.6],
                         [0.2, 0.2],
                         [0.7, 0.2],
                         [0.3, 0.0],
                         [0.5, -0.45],
                         [0.0, -0.2],
                         [-0.5, -0.45],
                         [-0.3, 0.0],
                         [-0.7, 0.2],
                         [-0.2, 0.2]])
    star_coords = ref_star * width + center

    sqr = sg.Polygon(star_coords)
    return Obj(sqr, color, star_shape)


def sample_loc_scale_color(center=None, width=None, target_color=None,
                           exclude_colors=None, width_max=WIDTH_MAX,
                           width_min=WIDTH_MIN, direction=None):
    if width is None:
        width = np.random.random() * (width_max - width_min) + width_min
    if center is None:
        if direction is None:
            center = np.random.random(2) * (1 - 2 * width) + width
        elif direction == 'horizontal':
            center = np.random.random(1) * (1 - 2 * width) + width
            center = np.insert(center, 1, 0.5)
        elif direction == 'vertical':
            center = np.random.random(1) * (1 - 2 * width) + width
            center = np.insert(center, 0, 0.5)
        else:
            raise ValueError("Invalid direction")

    if target_color is None:
        if exclude_colors is None:
            color_candidates = ALL_COLORS
        else:
            color_candidates = remove_targets(ALL_COLORS, exclude_colors)
        target_color = np.random.choice(color_candidates)

    return center, width, target_color


def remove_targets(all, targets):
    return list(set(all) - set(targets))


ALL_COLORS = ['r', 'g', 'b', 'y']
ALL_SHAPES = [square_shape, circle_shape,
              diamond_shape,
              triangle_shape, star_shape]
ALL_SHAPE_NAMES = ['square_shape', 'circle_shape',
                   'diamond_shape',
                   'triangle_shape', 'star_shape']
