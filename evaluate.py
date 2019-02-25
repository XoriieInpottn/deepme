#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-25
"""

import argparse
import gzip
import multiprocessing
import pickle

import pymongo
import tqdm

queue1 = multiprocessing.Queue(maxsize=100)
queue2 = multiprocessing.Queue(maxsize=100)

lock = multiprocessing.Semaphore(1)

hit_top1 = multiprocessing.Value('L')
hit_top3 = multiprocessing.Value('L')
hit_top5 = multiprocessing.Value('L')
total = multiprocessing.Value('L')


def foo():
    while True:
        doc = queue1.get()
        if doc is None:
            break
        label_index = doc['label_index']
        # y = pickle.loads(doc['y'])
        y = pickle.loads(gzip.decompress(doc['y']))
        r = [(i, v) for i, v in enumerate(y)]
        r.sort(key=lambda a: -a[1])

        top1 = set(i for i, v in r[:1])
        top3 = set(i for i, v in r[:3])
        top5 = set(i for i, v in r[:5])

        with lock:
            if label_index in top1:
                hit_top1.value += 1
            if label_index in top3:
                hit_top3.value += 1
            if label_index in top5:
                hit_top5.value += 1
            total.value += 1


def main(args):
    with pymongo.MongoClient('sis3.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')
        db = conn[f'imagenet_{args.model}']
        coll = db['final_test']

        ps = [multiprocessing.Process(target=foo, daemon=True) for _ in range(args.num_process)]
        for p in ps:
            p.start()

        progress = tqdm.tqdm(total=coll.count(), ncols=64)
        for i, doc in enumerate(coll.find(), 1):
            queue1.put(doc)

            if i % 100 == 0:
                with lock:
                    if total.value != 0:
                        acc_top1 = hit_top1.value / total.value
                        acc_top3 = hit_top3.value / total.value
                        acc_top5 = hit_top5.value / total.value
                        progress.set_description(
                            f'top1={acc_top1:.06f}, top3={acc_top3:.06f}, top5={acc_top5:.06f}',
                            False
                        )
            progress.update()
        progress.close()

        for _ in ps:
            queue1.put(None)
        for p in ps:
            p.join()

        acc_top1 = hit_top1.value / total.value
        acc_top3 = hit_top3.value / total.value
        acc_top5 = hit_top5.value / total.value
        print(f'Final top1={acc_top1:.06f}, top3={acc_top3:.06f}, top5={acc_top5:.06f}')
    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--model', required=True)
    _parser.add_argument('--num-process', type=int, default=10)
    #
    _args = _parser.parse_args()
    exit(main(_args))
