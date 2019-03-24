#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-26
"""

import argparse

import bson
import pymongo


def main(args):
    with pymongo.MongoClient('sis3.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        coll = conn['imagenet']['imagenet_10k']
        doc = coll.find_one({'_id': bson.ObjectId(args.image_id)})
        if doc is not None:
            data = doc['data']
            with open(f'{args.image_id}.jpg', 'wb') as f:
                f.write(data)
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('image_id')
    #
    _args = _parser.parse_args()
    exit(main(_args))
