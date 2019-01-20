#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-23
"""

import argparse
import gzip
import os
import pickle

import pymongo
import tensorflow as tf
from tqdm import tqdm

import photinia as ph
from photinia.apps.imagenet import vgg


class Model(ph.Model):

    def __init__(self, name, num_classes, num_gpus, batch_size):
        self._num_classes = num_classes
        self._num_gpus = num_gpus
        self._batch_size = batch_size
        super(Model, self).__init__(name)

    def _build(self):
        encoder = vgg.VGG16('encoder')
        dense1 = ph.Linear('dense1', encoder.fc7.output_size, 4096, w_init=ph.init.TruncatedNormal(0, 1e-3))
        dense2 = ph.Linear('dense2', 4096, self._num_classes, w_init=ph.init.TruncatedNormal(0, 1e-3))
        input_image = ph.placeholder('input_image', (None, vgg.HEIGHT, vgg.WIDTH, 3), ph.float)
        input_label = ph.placeholder('input_label', (None,), ph.int)

        self._num_gpus -= 1
        batch_size = tf.shape(input_image)[0]
        num_per_device = tf.cast(tf.ceil(batch_size / self._num_gpus), tf.int32)

        var_list1 = [
            *dense1.get_trainable_variables(),
            *dense2.get_trainable_variables()
        ]
        var_list2 = self.get_trainable_variables()

        y_list = []
        loss_list = []
        grad_list_list1 = []
        grad_list_list2 = []
        for i in range(self._num_gpus):
            with tf.device(f'/gpu:{i + 1}'):
                input_image_i = input_image[i * num_per_device:(i + 1) * num_per_device]
                encoder.setup(input_image_i)
                h = encoder['h7'] if i == 0 else encoder[f'h7_{i}']
                y = ph.ops.lrelu(dense1.setup(h) + h)
                y = tf.nn.softmax(dense2.setup(y))
                y_list.append(y)

                input_label_i = input_label[i * num_per_device:(i + 1) * num_per_device]
                y_target = tf.one_hot(input_label_i, self._num_classes)
                loss = ph.ops.cross_entropy(y_target, y)
                loss = tf.reduce_mean(loss)
                loss_list.append(loss)

                reg1 = ph.reg.L2Regularizer(1e-6)
                reg1.setup(var_list1)
                grad_list1 = tf.gradients(loss + reg1.get_loss(), var_list1)
                grad_list_list1.append(grad_list1)

                reg2 = ph.reg.L2Regularizer(1e-6)
                reg2.setup(var_list2)
                grad_list2 = tf.gradients(loss + reg2.get_loss(), var_list2)
                grad_list_list2.append(grad_list2)

        y = tf.concat(y_list, axis=0)
        loss = tf.reduce_mean(loss_list)

        grad_list1 = [
            tf.reduce_mean(grads, axis=0)
            for grads in zip(*grad_list_list1)
        ]
        self.train = ph.Step(
            inputs=(input_image, input_label),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-5, 0.9, 0.9).apply_gradients(zip(grad_list1, var_list1))
        )

        grad_list2 = [
            tf.reduce_mean(grads, axis=0)
            for grads in zip(*grad_list_list2)
        ]
        self.fine_tune = ph.Step(
            inputs=(input_image, input_label),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-6, 0.9, 0.9).apply_gradients(zip(grad_list2, var_list2))
        )

        label = tf.argmax(y, axis=1)
        self.predict = ph.Step(
            inputs=input_image,
            outputs=(label, y)
        )


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
        image = [
            ph.utils.image.load_as_array(data_i)
            for data_i in data
        ]
        return _id, image, label_index


class Main(ph.Application):

    def _main(self, args):
        # ph.get_session_config().log_device_placement = True
        ph.set_tf_log_level(ph.TF_LOG_NO_WARN)
        with pymongo.MongoClient('sis4.ustcdm.org') as conn:
            conn['admin'].authenticate('root', 'SELECT * FROM users;')

            db = conn['imagenet_vgg']
            coll_train = db[f'train']
            coll_valid = db[f'valid']
            num_classes = len(coll_train.distinct('label_index'))
            print(f'Found {num_classes} classes.')

            ds_train = DataSource(coll_train, True, args.batch_size)
            ds_valid = DataSource(coll_valid, False, args.batch_size)

            model = Model('model', num_classes, len(args.gpu.split(',')), args.batch_size)
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

                if (i + 1) % 10000 == 0:
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

            coll_test = db[f'test']
            bar = tqdm(total=coll_test.count(), ncols=96, desc='Testing')
            ds_test = DataSource(coll_test, False, args.batch_size)
            coll_output = db['result_vgg']
            buffer = []
            cal = ph.train.AccCalculator()
            for _id, image, label_index in ds_test:
                label_pred, y = model.predict(image)
                cal.update(label_pred, label_index)
                for _id_i, label_index_i, y_i in zip(_id, label_index, y):
                    doc = {
                        '_id': _id_i,
                        'label_index': label_index_i,
                        'y': gzip.compress(pickle.dumps(y_i), 7)
                    }
                    buffer.append(doc)
                    if len(buffer) >= 1000:
                        coll_output.insert_many(buffer)
                        buffer.clear()
                bar.update(len(image))
            if len(buffer) != 0:
                coll_output.insert_many(buffer)
                buffer.clear()
            bar.close()
            print(f'Final acc={cal.accuracy}')

        print('All clear.')
        return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-size', type=int, default=64)
    _parser.add_argument('--num-train', type=int, default=100000)
    _parser.add_argument('--num-loops', type=int, default=500000)
    _parser.add_argument('--vgg16', required=True)
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(Main().run(_args))
