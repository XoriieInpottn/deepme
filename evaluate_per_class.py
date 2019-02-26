#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-25
"""

import argparse
import pickle

import pymongo
import tqdm


class ClassStat(object):

    def __init__(self):
        self.hit = 0
        self.total = 0

    @property
    def acc(self):
        return self.hit / self.total


stats = [ClassStat() for _ in range(10184)]


def main(args):
    with pymongo.MongoClient('sis3.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        db = conn['imagenet_deepme']
        coll = db[f'final_{args.model}']

        progress = tqdm.tqdm(total=coll.count(), ncols=96)
        for i, doc in enumerate(coll.find(), 1):
            label_index = doc['label_index']
            y = pickle.loads(doc['y'])
            r = [(i, v) for i, v in enumerate(y)]
            r.sort(key=lambda a: -a[1])
            top5 = set(i for i, v in r[:5])
            stat = stats[label_index]
            stat.total += 1
            if label_index in top5:
                stat.hit += 1
            progress.update()
        progress.close()

        for label_index, stat in enumerate(stats):
            print(f'{label_index}={stat.acc}')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--model', required=True)
    #
    _args = _parser.parse_args()
    exit(main(_args))
