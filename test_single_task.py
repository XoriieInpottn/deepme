#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-22
"""

import argparse
import pickle

import numpy as np
import pymongo
from tqdm import tqdm


class Mapping(object):

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


def main(args):
    with pymongo.MongoClient('sis2.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        db = conn[args.db]
        coll = db[f'result_{args.task_index:02d}_test']
        mapping = Mapping(db['clusters'])

        hit = 0
        tot = 0
        progress = tqdm(total=coll.count(), ncols=96)
        for doc in coll.find():
            progress.update()
            global_index = doc['index']
            local_index = mapping.to_local(args.task_index, global_index)
            if local_index is None:
                continue
            y = pickle.loads(doc['y'])
            label_index = int(np.argmax(y))
            # print(label_index, local_index)
            tot += 1
            if label_index == local_index:
                hit += 1
        progress.close()
        print(hit / tot)
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--db', required=True)
    _parser.add_argument('--task-index', type=int, required=True)
    #
    _args = _parser.parse_args()
    exit(main(_args))
