#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-23
"""

import numpy as np


class IndexMapping(object):

    def __init__(self, coll):
        self._g2l = {}
        self._l2g = {}
        for doc in coll.find():
            global_index = doc['label_index']
            tuples = doc['task_label_indexes']
            for task_index, local_index in tuples:
                if global_index not in self._g2l:
                    self._g2l[global_index] = {}
                if task_index not in self._l2g:
                    self._l2g[task_index] = {}
                self._g2l[global_index][task_index] = local_index
                self._l2g[task_index][local_index] = global_index
        self._g2t = {}
        for global_index in self._g2l:
            target = np.zeros(shape=(len(self._l2g, )), dtype=np.float32)
            self._g2t[global_index] = target
            for task_index in self._g2l[global_index]:
                target[task_index] = 1.0

    def to_local(self, task_index, global_index):
        try:
            return self._g2l[global_index][task_index]
        except KeyError:
            return None

    def to_global(self, task_index, local_index):
        try:
            return self._l2g[task_index][local_index]
        except KeyError:
            return None

    def to_target(self, global_index):
        try:
            return self._g2t[global_index]
        except KeyError:
            return None
