#!/usr/bin/env python
# coding: utf-8

"""
tfrecord handling and optimizer configuration
"""
import tensorflow as tf
import numpy as np
import os, sys
from copy import deepcopy
import yaml
from fileio import iovar

from functools import partial

_type_map = {
    "tf.float32": tf.float32,
    "tf.float16" : tf.float16,
    "np.float32": np.float32,
    "np.float16": np.float16,
}
_inverse_type_map = {
    tf.float32 : "tf.float32",
    tf.float16 : "tf.float16",
    np.float32 : "np.float32",
    np.float16 : "np.float16",
}


def _fill_format(vv):
    """
    replace environment variables
    """
    if isinstance(vv, str):
        if vv in _type_map :
            vv = _type_map[vv]
        else:
            if  "$" in vv:
                vv= os.path.expandvars(vv)
            if  "~" in vv:
                vv= os.path.expanduser(vv)

    elif isinstance(vv, dict):
        for kk, _vv in vv.items():
            vv[kk] = _fill_format(_vv)
    return vv


def get_list_parameter(val, n_elements, n_repeater=None):
    """
    create a list of parameters with n_elements form a list or a scalar

    In case val is a scalar or has len 1 the value is duplicated n_elements times
    In case val is of len n_elements//n_repeater each of its elements will be repeated n_repeater times
    in case the resulting list is longer than n_elements the exceeding elements are discarded

    """

    try:
        val_list = val[:]
    except TypeError:
        val_list = [val]

    if len(val_list) == 1:
        val_list = val_list * n_elements
    elif (n_repeater is not None) and (len(val_list) * n_repeater < n_elements + n_repeater):
        val_list = [vv for vv in val_list for _ in range(n_repeater)]
        # repeat last element if necessary
        if len(val_list) < n_elements:
             val_list = val_list + [val_list[-1] for _ in range(n_elements - len(val_list))]
        # cut list if  last element if necessary
        val_list = val_list[:n_elements]
    elif len(val_list) != n_elements:
        raise RuntimeError(f"traing_utils::error:: cannot contstruct list of {n_elements} "
                           f"from {val} with sub repeater  {n_repeater}")
    return val_list

def read_config(config_file):
    """
    read and fill config with self references
    """
    with open(config_file, "r") as fi:
        config = yaml.safe_load(fi)
    for kk, vv in config.items():
        config[kk] = _fill_format(vv)
    return config




