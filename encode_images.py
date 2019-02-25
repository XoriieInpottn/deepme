#!/usr/bin/env python3

"""
@author: xi
@since: 2019-02-18
"""

import argparse
import gzip
import os
import pickle

import pymongo
from tqdm import tqdm

import photinia as ph
from photinia.apps.imagenet import vgg


class Model(ph.Model):

    def __init__(self, name):
        super(Model, self).__init__(name)

    def _build(self):
        encoder = vgg.VGG16('encoder')
        image = ph.placeholder('image', (None, vgg.HEIGHT, vgg.WIDTH, 3), ph.float)
        encoder.setup(image)
        h7 = encoder['h7']

        self.step = ph.Step(inputs=image, outputs=h7)
        self.predict = self.step


class DataSource(ph.io.ThreadBufferedSource):

    def __init__(self, coll, random_order, batch_size):
        ds = ph.io.MongoSource(
            ['_id', 'data', 'label_index'],
            coll,
            random_order=random_order,
            max_buffer_size=5_000_000
        )
        ds = ph.io.BatchSource(ds, batch_size)
        super(DataSource, self).__init__(ds, num_thread=5)

    def _next(self, row):
        _id, data, label_index = row
        image = [ph.utils.image.load_as_array(data_i) for data_i in data]
        return _id, image, label_index


def main(args):
    ph.set_tf_log_level(ph.TF_LOG_NO_WARN)
    with pymongo.MongoClient('sis3.ustcdm.org') as conn:
        conn['admin'].authenticate('root', 'SELECT * FROM users;')

        db = conn['imagenet_vgg']
        coll = db[args.coll]
        coll_output = db[f'h7_{args.coll}']
        coll_output.create_index('label_index')

        ds_test = DataSource(coll, False, args.batch_size)

        model = Model('model')
        ph.initialize_global_variables()
        ph.io.load_model_from_file(model['encoder'], args.vgg16, 'vgg16')

        progress = tqdm(total=coll.count(), ncols=96)
        buffer = []
        for _id, image, label in ds_test:
            h7, = model.predict(image)
            for _id_i, h7_i, label_i in zip(_id, h7, label):
                buffer.append({
                    '_id': _id_i,
                    'h7': gzip.compress(pickle.dumps(h7_i), 5),
                    'label_index': label_i
                })
                if len(buffer) >= 1000:
                    coll_output.insert_many(buffer)
                    buffer.clear()
            progress.update(len(image))
        if len(buffer) != 0:
            coll_output.insert_many(buffer)
            buffer.clear()
        progress.close()

    print('All clear.')
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-size', type=int, default=128)
    _parser.add_argument('--vgg16', required=True)
    _parser.add_argument('--coll', required=True)
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
