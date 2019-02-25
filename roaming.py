#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-25
"""

import argparse
import gzip
import pickle

import pymongo
import tqdm


def main(args):
    with pymongo.MongoClient('sis3.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        coll = conn['imagenet_vgg']['final_test']
        coll_output = conn['imagenet_deepme']['final_vgg']

        progress = tqdm.tqdm(total=coll.count(), ncols=96)
        buffer = []
        for doc in coll.find():
            y = doc['y']
            y = gzip.decompress(y)
            doc['y'] = y
            buffer.append(doc)
            if len(buffer) >= 1000:
                coll_output.insert_many(buffer)
                buffer.clear()
            progress.update()
        if len(buffer) != 0:
            coll_output.insert_many(buffer)
            buffer.clear()
        progress.close()
    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    #
    _args = _parser.parse_args()
    exit(main(_args))
