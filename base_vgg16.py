#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-23
"""

import argparse
import os
import pickle

import pymongo
import tensorflow as tf
from tqdm import tqdm

import photinia as ph
from photinia.apps.imagenet import vgg


class Model(ph.Model):

    def __init__(self, name, num_classes):
        self._num_classes = num_classes
        super(Model, self).__init__(name)

    def _build(self):
        input_image = ph.placeholder('input_image', (None, vgg.HEIGHT, vgg.WIDTH, 3), ph.float)
        encoder = vgg.VGG16('encoder')
        encoder.setup(input_image)
        h = encoder['h7']
        dense = ph.Linear('dense', encoder.fc7.output_size, self._num_classes)
        y = dense.setup(h)
        y = tf.nn.softmax(y)
        label = tf.argmax(y, axis=1)

        self.predict = ph.Step(
            inputs=input_image,
            outputs=(label, y)
        )

        input_label = ph.placeholder('input_label', (None,), ph.int)
        y_target = tf.one_hot(input_label, self._num_classes)
        loss = ph.ops.cross_entropy(y_target, y)
        loss = tf.reduce_mean(loss)

        var_list = dense.get_trainable_variables()
        reg = ph.reg.L2Regularizer(1e-6)
        reg.setup(var_list)
        self.train = ph.Step(
            inputs=(input_image, input_label),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-5, 0.9, 0.9).minimize(loss + reg.get_loss(), var_list=var_list)
        )

        var_list = self.get_trainable_variables()
        reg = ph.reg.L2Regularizer(1e-6)
        reg.setup(var_list)
        self.fine_tune = ph.Step(
            inputs=(input_image, input_label),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-6, 0.9, 0.9).minimize(loss + reg.get_loss(), var_list=var_list)
        )


class DataSource(ph.io.ThreadBufferedSource):

    def __init__(self, coll, random_order, batch_size):
        ds = ph.io.MongoSource(['_id', 'data', 'label_index'], coll, random_order=random_order)
        ds = ph.io.BatchSource(ds, batch_size)
        super(DataSource, self).__init__(ds, buffer_size=100)

    def _next(self, row):
        _id, data, label_index = row
        image = [
            ph.utils.image.load_as_array(data_i)
            for data_i in data
        ]
        return _id, image, label_index


class Main(ph.Application):

    def _main(self, args):
        ph.set_tf_log_level(ph.TF_LOG_NO_WARN)
        with pymongo.MongoClient('sis3.ustcdm.org') as conn:
            conn['admin'].authenticate('root', 'SELECT * FROM users;')

            db = conn[args.db_name]
            coll_train = db[f'task_{args.task_index:02d}_train']
            coll_valid = db[f'task_{args.task_index:02d}_valid']
            num_classes = len(coll_train.distinct('label_index'))
            print(f'Found {num_classes} classes.')

            ds_train = DataSource(coll_train, True, args.batch_size)
            ds_valid = DataSource(coll_valid, False, args.batch_size)

            model = Model('model', num_classes)
            ph.initialize_global_variables()
            ph.io.load_model_from_file(model['encoder'], args.vgg16, 'vgg16')

            #
            # train the last layer
            bar = tqdm(total=args.num_train, ncols=96, desc='Training')
            for i in range(args.num_train):
                self.checkpoint()
                try:
                    _, image, label = ds_train.next()
                except StopIteration:
                    _, image, label = ds_train.next()
                loss, = model.train(image, label)
                bar.update()
                bar.set_description(f'Training loss={loss:.06f}')
            bar.close()

            #
            # validation
            bar = tqdm(total=coll_valid.count(), ncols=96, desc='Validating')
            cal = ph.train.AccCalculator()
            for _, image, label in ds_valid:
                label_pred, _ = model.predict(image)
                cal.update(label_pred, label)
                bar.update(len(image))
            bar.close()
            print(f'Validation acc={cal.accuracy}')

            #
            # fine tuning all the parameters
            bar = tqdm(total=args.num_loops, ncols=96, desc='Fine tuning')
            es = ph.train.EarlyStopping(3)
            for i in range(args.num_loops):
                self.checkpoint()
                try:
                    _, image, label = ds_train.next()
                except StopIteration:
                    _, image, label = ds_train.next()
                loss, = model.fine_tune(image, label)
                bar.set_description(f'Fine tuning loss={loss:.06f}')

                if (i + 1) % 1000 == 0:
                    bar1 = tqdm(total=coll_valid.count(), ncols=96, desc='Validating')
                    cal = ph.train.AccCalculator()
                    for _, image, label in ds_valid:
                        label_pred, _ = model.predict(image)
                        cal.update(label_pred, label)
                        bar1.update(len(image))
                    bar1.close()
                    bar.clear()
                    print(f'Validation acc={cal.accuracy}')
                    if es.convergent(1 - cal.accuracy):
                        break
                bar.update()
            bar.close()
            ds_train = None
            ds_valid = None

            coll = conn['imagenet_vgg']['train']
            coll_output = db[f'result_{args.task_index:02d}_train']
            self._write_result(model, coll, coll_output)

            coll = conn['imagenet_vgg']['valid']
            coll_output = db[f'result_{args.task_index:02d}_valid']
            self._write_result(model, coll, coll_output)

            coll = conn['imagenet_vgg']['test']
            coll_output = db[f'result_{args.task_index:02d}_test']
            self._write_result(model, coll, coll_output)

        print('All clear.')
        return 0

    @staticmethod
    def _write_result(model, coll, coll_output):
        ds_test = DataSource(coll, False, 96)
        bar = tqdm(total=coll.count(), ncols=96, desc=f'Predicting for {coll.name}')
        buffer = []
        for _id, image, label_index in ds_test:
            _, y = model.predict(image)
            for _id_i, label_i, y_i in zip(_id, label_index, y):
                doc = {
                    '_id': _id_i,
                    'index': label_i,
                    'y': pickle.dumps(y_i)
                }
                buffer.append(doc)
                if len(buffer) >= 1000:
                    coll_output.insert_many(buffer)
                    buffer.clear()
            bar.update(len(image))
        bar.close()
        if len(buffer) != 0:
            coll_output.insert_many(buffer)
            buffer.clear()


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-size', type=int, default=64)
    _parser.add_argument('--num-train', type=int, default=10000)
    _parser.add_argument('--num-loops', type=int, default=30000)
    _parser.add_argument('--task-index', type=int, required=True)
    _parser.add_argument('--vgg16', required=True)
    _parser.add_argument('--db-name', default='imagenet_deepme')
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(Main().run(_args))
