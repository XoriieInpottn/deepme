#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-23
"""

import argparse
import multiprocessing
import pickle

import numpy as np
import pymongo
import tqdm

import utils

mapping = None
queue_doc = multiprocessing.Queue(maxsize=100)
queue_result = multiprocessing.Queue(maxsize=100)


def mmm():
    while True:
        docs = queue_doc.get()
        if docs is None:
            break
        feature = np.zeros(shape=(10184,), dtype=np.float)
        for task_index, doc in enumerate(docs):
            y = pickle.loads(doc['y'])
            phi = y[-1]
            for local_index, value in enumerate(y[:-1] * (1 - phi)):
                global_index = mapping.to_global(task_index, local_index)
                feature[global_index] += value
        label_index = docs[0]['index']
        _id = docs[0]['_id']
        queue_result.put((_id, feature, label_index))


def rrr(coll_name):
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        coll = conn['imagenet_dmde'][f'fusion_{coll_name}']
        buffer = []

        while True:
            r = queue_result.get()
            if r is None:
                break
            _id, y, label_index = r
            buffer.append({
                '_id': _id,
                'y': pickle.dumps(y),
                'label_index': label_index
            })
            if len(buffer) >= 1000:
                coll.insert_many(buffer)
                buffer.clear()
        if len(buffer) != 0:
            coll.insert_many(buffer)
            buffer.clear()


def main(args):
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        db = conn['imagenet_dmde']
        global mapping
        mapping = utils.IndexMapping(db['clusters'])
        colls = [db[f'newresult_{i:02d}_{args.coll_name}'] for i in range(20)]

        p_list = [multiprocessing.Process(target=mmm) for _ in range(10)]
        for p in p_list:
            p.start()
        rp = multiprocessing.Process(target=rrr, args=(args.coll_name,))
        rp.start()

        progress = tqdm.tqdm(total=colls[0].count(), ncols=96)
        for docs in zip(*(coll.find() for coll in colls)):
            queue_doc.put(docs)
            progress.update()
        progress.close()

        for _ in p_list:
            queue_doc.put(None)
        for p in p_list:
            p.join()
        queue_result.put(None)
        rp.join()
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--coll-name', required=True)
    #
    _args = _parser.parse_args()
    exit(main(_args))
