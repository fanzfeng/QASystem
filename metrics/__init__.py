# -*- coding: utf-8 -*-
import six
from keras.utils.generic_utils import deserialize_keras_object
import os, sys
root_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(root_path)
sys.path.append(root_path)
from metrics.evaluations import *


def serialize(generator):
    return generator.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
