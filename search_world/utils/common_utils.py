"""Functions to show trial videos and images."""

import sys
sys.path.append('../../task')

import collections
import colorsys
import json
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
from moog import observers
from moog import sprite as sprite_lib
import numpy as np
import os

META_STATE_INDEX = 4  # Index of meta-state in log file step strings
STATE_INDEX = 5  # Index of state in log file step strings
PREY_LAYER_INDEX = 0  # Index of prey layer in state log

ATTRIBUTES_FULL = list(sprite_lib.Sprite.FACTOR_NAMES)
ATTRIBUTES_PARTIAL = ['x', 'y', 'x_vel', 'y_vel', 'opacity', 'metadata']
ATTRIBUTES_PARTIAL_INDICES = {k: i for i, k in enumerate(ATTRIBUTES_PARTIAL)}


def create_new_sprite(sprite_kwargs, vertices=None):
    """Create new sprite from factors.

    Args:
        sprite_kwargs: Dict. Keyword arguments for sprite_lib.Sprite.__init__().
            All of the strings in sprite_lib.Sprite.FACTOR_NAMES must be keys of
            sprite_kwargs.
        vertices: Optional numpy array of vertices. If provided, are used to
            define the shape of the sprite. Otherwise, sprite_kwargs['shape'] is
            used.

    Returns:
        Instance of sprite_lib.Sprite.
    """
    if vertices is not None:
        # Have vertices, so must invert the translation, rotation, and
        # scaling transformations to get the original sprite shape.
        center_translate = mpl_transforms.Affine2D().translate(
            -sprite_kwargs['x'], -sprite_kwargs['y'])
        x_y_scale = 1. / np.array([
            sprite_kwargs['scale'],
            sprite_kwargs['scale'] * sprite_kwargs['aspect_ratio']
        ])
        transform = (
            center_translate +
            mpl_transforms.Affine2D().rotate(-sprite_kwargs['angle']) +
            mpl_transforms.Affine2D().scale(*x_y_scale)
        )
        vertices = mpl_path.Path(vertices)
        vertices = transform.transform_path(vertices).vertices

        sprite_kwargs['shape'] = vertices

    return sprite_lib.Sprite(**sprite_kwargs)


def attributes_to_sprite(a):
    """Create sprite with given attributes."""
    attributes = {x: a[i] for i, x in enumerate(ATTRIBUTES_FULL)}
    if len(a) > len(ATTRIBUTES_FULL):
        vertices = np.array(a[-1])
    else:
        vertices = None
    return create_new_sprite(attributes, vertices=vertices)


def get_initial_state(trial):
    """Get initial state OrderedDict."""
    def _attributes_to_sprite_list(sprite_list):
        return [attributes_to_sprite(s) for s in sprite_list]
    
    state_layers = [(k, _attributes_to_sprite_list(v)) for k, v in trial[0]]
    state = collections.OrderedDict(state_layers)

    return state


def update_state(state, step_string, translucent=False):
    """Update the state in place given a step string."""
    transparent = 96 if translucent else 0

    # Deal with state changes
    meta_state = step_string[4][1]
    if meta_state['phase'] != 'fixation':
        state['screen'][0].opacity = 0
        state['fixation'][0].opacity = 0

    if meta_state['phase'] == 'response':
        state['background'][0].opacity = 255

        # Find prey_ind
        prey_ind = -1
        for i, p in enumerate(state['prey']):
            if p.metadata['response'] is None:
                prey_ind = i
                break
        if prey_ind == -1:
            prey_ind = i
            
        prey = state['prey'][prey_ind]
        state['background'][0].c0 = prey.c0
        state['background'][0].c1 = prey.c1
        state['background'][0].c2 = 0.4 * prey.c2
    
    # Loop through sprite layers
    for x in step_string[-1]:

        # Update sprite attributes
        if x[0] in ['prey', 'eye']:
            sprites = state[x[0]]
            for s, s_attr in zip(sprites, x[1]):
                attributes = {k: v for v, k in zip(s_attr, ATTRIBUTES_PARTIAL)}
                s.position = [attributes['x'], attributes['y']]
                s.velocity = [attributes['x_vel'], attributes['y_vel']]
                s.metadata = attributes['metadata']
                if x[0] == 'prey':
                    s.opacity = (
                        transparent if attributes['opacity'] == 0 else 255)
                else:
                    s.opacity = attributes['opacity']


def observer():
    """Get a renderer."""
    observer = observers.PILRenderer(
        image_size=(256, 256),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )

    return observer
    