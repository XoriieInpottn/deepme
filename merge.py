#!/usr/bin/env python3

"""
@author: xi
@since: 2018-03-30
"""

import argparse

import pymongo


class IndexMapping(list):

    def __init__(self, coll):
        super(IndexMapping, self).__init__(
            (doc['index'], [
                (i, doc['index%d' % i])
                for i in range(20)
                if doc['index%d' % i] != 0
            ])
            for doc in coll.find()
        )
        self.sort(key=lambda a: a[0])


class TaskMapping(list):

    def __init__(self, coll):
        super(TaskMapping, self).__init__(
            (doc['label_index'], doc['task_label_indexes'])
            for doc in coll.find()
        )
        self.sort(key=lambda a: a[0])



def _main(args):
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        db = conn['imagenet_dmde']
        mapping = TaskMapping(db['clusters'])

        colls = [db[f'result_{i:02d}_{args.split}'] for i in range(20)]
        for docs in zip(*(coll.find() for coll in colls)):
            pass


def main(args):
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        index_mapping = IndexMapping(conn['imagenet']['labels'])
        coll = conn['imagenet']['trainset_ooxx']
        coll_ = conn['base']['train']
        if 'y' != input('This will clear target collections, continue?'):
            return 1
        coll_.remove()
        buffer = []
        count = coll.count()
        for k, doc in enumerate(coll.find({}, {'img': 0}, batch_size=3)):
            print('[%d/%d]' % (k + 1, count))
            _id = doc['_id']
            groups = set(i for i, _ in index_mapping[doc['index']][1])
            pred_list = [
                doc['pred_index'] if doc is not None else None
                for doc in (
                    conn['base']['train_pred_cluster%d' % i].find_one({'_id': _id}) if i in groups else None
                    for i in range(20)
                )
            ]
            fusioned = [
                sum(
                    pred[index_] * (1.0 - pred[0]) / (1 + pred[0]) if pred is not None else 0.0
                    for pred, index_ in ((pred_list[j], index_) for j, index_ in indexes)
                )
                for _, indexes in index_mapping
            ]
            buffer.append({
                '_id': _id,
                'x': fusioned,
                'y': doc['index']
            })
            if len(buffer) >= 100:
                coll_.insert_many(buffer)
                buffer.clear()
        if len(buffer) != 0:
            coll_.insert_many(buffer)
            buffer.clear()
    print('Complete.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _args = _parser.parse_args()
    exit(main(_args))
