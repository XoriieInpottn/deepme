#!/usr/bin/env python3

"""
@author: xi
@since: 2019-02-03
"""

import argparse
import pymongo
import cv2 as cv
import numpy as np

import threading
import queue
from tqdm import tqdm

n = 10

queue = queue.Queue()
lock = threading.Semaphore(1)
results = [None for _ in range(n)]
finished = threading.Semaphore(0)


def convert(data):
    data = np.asarray(bytearray(data), np.byte)
    image = cv.imdecode(data, cv.IMREAD_UNCHANGED)
    image = cv.resize(image, (224, 224))
    data = cv.imencode('.jpg', image, (cv.IMWRITE_JPEG_QUALITY, 80))[1].tobytes()
    return data


def foo():
    while True:
        task = queue.get()
        if task is None:
            break
        index, doc = task
        doc['data'] = convert(doc['data'])
        doc['size'] = len(doc['data'])
        with lock:
            results[index] = doc
        finished.release()


def main(args):
    ts = [threading.Thread(target=foo) for _ in range(n)]
    for t in ts:
        t.setDaemon(True)
        t.start()
    with pymongo.MongoClient('sis4.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        coll_input = conn['imagenet'][f'imagenet_10k_{args.db}']
        coll_output = conn['imagenet_vgg'][args.db]
        i = 0
        buffer = []
        progress = tqdm(total=coll_input.count(), ncols=96)
        for doc in coll_input.find():
            queue.put((i, doc))
            i += 1
            if i == n:
                for _ in range(i):
                    finished.acquire()
                for j in range(i):
                    doc_ = results[j]
                    buffer.append(doc_)
                    if len(buffer) >= 1000:
                        coll_output.insert_many(buffer)
                        buffer.clear()
                i = 0
            progress.update()
        if i != 0:
            for _ in range(i):
                finished.acquire()
            for j in range(i):
                doc_ = results[j]
                buffer.append(doc_)
                if len(buffer) >= 1000:
                    coll_output.insert_many(buffer)
                    buffer.clear()
        if len(buffer) != 0:
            coll_output.insert_many(buffer)
            buffer.clear()
        progress.close()
    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--db', required=True)
    #
    _args = _parser.parse_args()
    exit(main(_args))
