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

queue_doc = multiprocessing.Queue(maxsize=100)
queue_result = multiprocessing.Queue(maxsize=100)


def mmm(mapping):
    while True:
        docs = queue_doc.get()
        if docs is None:
            break

        fusion = np.zeros(shape=(10184,), dtype=np.float)
        for task_index, doc in enumerate(docs):
            y = pickle.loads(doc['y'])
            for local_index, value in enumerate(y):
                global_index = mapping.to_global(task_index, local_index)
                fusion[global_index] += value
        fusion = fusion / np.sum(np.exp(fusion))

        _id = docs[0]['_id']
        true_index = docs[0]['index']
        pred_index = int(np.argmax(fusion))
        result = (_id, true_index, pred_index, fusion)
        queue_result.put(result)


def rrr():
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        coll = conn['imagenet_deepme']['final_deepme_no_gate']
        buffer = []

        hit = tot = 0
        i = 1
        while True:
            r = queue_result.get()
            if r is None:
                break
            _id, true_index, pred_index, fusion = r

            buffer.append({
                '_id': _id,
                'label_index': true_index,
                'y': pickle.dumps(fusion)
            })
            if len(buffer) >= 1000:
                coll.insert_many(buffer)
                buffer.clear()

            hit += 1 if pred_index == true_index else 0
            tot += 1
            if i % 10000 == 0:
                print(hit / tot)
            i += 1
        if len(buffer) != 0:
            coll.insert_many(buffer)
            buffer.clear()
        print(hit / tot)


def main(args):
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        db = conn['imagenet_deepme']
        mapping = utils.IndexMapping(db['clusters'])
        colls = [db[f'result_{i:02d}_test'] for i in range(15)]

        p_list = [multiprocessing.Process(target=mmm, args=(mapping,)) for _ in range(args.num_process)]
        rp = multiprocessing.Process(target=rrr)
        for p in p_list:
            p.start()
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

    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--num-process', type=int, default=10)
    #
    _args = _parser.parse_args()
    exit(main(_args))
