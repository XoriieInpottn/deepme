#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-23
"""

import argparse

import pymongo
import tqdm


def main(args):
    with pymongo.MongoClient('sis2.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')

        docs = []
        mapping1 = {}
        for doc in conn['imagenet_deepme']['bak.clusters'].find():
            docs.append(doc)
            wnid = doc['wnid']
            label_index = doc['label_index']
            mapping1[wnid] = label_index

        progress = tqdm.tqdm(total=len(mapping1), ncols=96)
        mapping2 = {}
        for doc in conn['imagenet_vgg']['train'].find({}, {'data': 0}):
            wnid = doc['wnid']
            label_index = doc['label_index']
            if wnid not in mapping2:
                mapping2[wnid] = label_index
                progress.update()
            if len(mapping2) == len(mapping1):
                break
        progress.close()

        for doc in docs:
            wnid = doc['wnid']
            new_index = mapping2[wnid]
            doc['label_index'] = new_index

        conn['imagenet_deepme']['clusters'].insert_many(docs)

    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    #
    _args = _parser.parse_args()
    exit(main(_args))
