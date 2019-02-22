#!/usr/bin/env python3


"""
@author: xi
@since: 2019-01-17
"""

import argparse

import pymongo
from tqdm import tqdm


def foo(conn, wnid_assign, num_tasks, name):
    colls = [conn['imagenet_deepme'][f'task_{i:02d}_{name}'] for i in range(num_tasks)]
    for coll in colls:
        coll.create_index('label_index')
        coll.remove()
    buffers = [[] for _ in range(num_tasks)]

    print(f'Creating {name}...')
    coll_input = conn['imagenet_vgg'][f'{name}']
    bar = tqdm(total=coll_input.count(), ncols=96, desc=f'Creating {name}')
    for doc in coll_input.find():
        doc['global_label_index'] = doc['label_index']
        wnid = doc['wnid']
        for task_index, label_index in wnid_assign[wnid]:
            doc = doc.copy()
            doc['label_index'] = label_index
            coll = colls[task_index]
            buffer = buffers[task_index]
            buffer.append(doc)
            if len(buffer) >= 1000:
                coll.insert_many(buffer)
                buffer.clear()
        bar.update()
    for coll, buffer in zip(colls, buffers):
        if len(buffer) != 0:
            coll.insert_many(buffer)
            buffer.clear()
    bar.close()


def main(args):
    with pymongo.MongoClient('sis2.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')

        # db = conn['imagenet_deepme']
        # for name in db.collection_names():
        #     if name.startswith('task_'):
        #         print(f'Dropping {name}...')
        #         db[name].drop()
        # exit()

        print('Loading clusters...')
        wnid_assign = {}
        task_indexes = set()
        coll_cluster = conn['imagenet_deepme']['clusters']
        for doc in coll_cluster.find():
            task_label_indexes = doc['task_label_indexes']
            wnid_assign[doc['wnid']] = task_label_indexes
            for task_index, _ in task_label_indexes:
                task_indexes.add(task_index)

        foo(conn, wnid_assign, len(task_indexes), 'train')
        foo(conn, wnid_assign, len(task_indexes), 'valid')

    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _args = _parser.parse_args()
    exit(main(_args))
