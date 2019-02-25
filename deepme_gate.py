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
import utils
from photinia.apps.imagenet import vgg

NUM_CLASSES = 15


class Model(ph.Model):

    def __init__(self, name, keep_prob):
        self._keep_prob = keep_prob
        super(Model, self).__init__(name)

    def _build(self):
        image = ph.placeholder('input_image', (None, vgg.HEIGHT, vgg.WIDTH, 3), ph.float)
        encoder = vgg.VGG16('encoder')
        encoder.setup(image)
        h = encoder['h7']

        dropout = ph.Dropout('dropout')
        h = dropout.setup(h)

        dense = ph.Linear('dense', encoder.fc7.output_size, NUM_CLASSES)
        y = dense.setup(h)
        y = tf.nn.sigmoid(y)

        self.predict = ph.Step(
            inputs=image,
            outputs=y,
            givens={dropout.keep_prob: 1.0}
        )

        target = ph.placeholder('target', (None, NUM_CLASSES), ph.float)
        loss = ph.ops.cross_entropy(target, y)
        loss = tf.reduce_mean(loss)

        var_list = dense.get_trainable_variables()
        reg = ph.reg.L2Regularizer(1e-6)
        reg.setup(var_list)
        grad_list = [
            tf.clip_by_value(grad, -10, 10)
            for grad in tf.gradients(loss + reg.get_loss(), var_list)
        ]
        lr = ph.train.ExponentialDecayedValue('lr_train', 1e-4, num_loops=3e3, min_value=1e-5)
        update = tf.train.AdamOptimizer(lr.value).apply_gradients(zip(grad_list, var_list))
        self.train = ph.Step(
            inputs=(image, target),
            outputs=loss,
            updates=(update, lr.update_op),
            givens={dropout.keep_prob: self._keep_prob}
        )

        var_list = self.get_trainable_variables()
        reg = ph.reg.L2Regularizer(1e-7)
        reg.setup(var_list)
        grad_list = [
            tf.clip_by_value(grad, -10, 10)
            for grad in tf.gradients(loss + reg.get_loss(), var_list)
        ]
        lr = ph.train.ExponentialDecayedValue('lr_fine_tune', 2e-5, num_loops=2e4, min_value=1e-6)
        update = tf.train.AdamOptimizer(lr.value).apply_gradients(zip(grad_list, var_list))
        self.fine_tune = ph.Step(
            inputs=(image, target),
            outputs=loss,
            updates=(update, lr.update_op),
            givens={dropout.keep_prob: self._keep_prob}
        )


class DataSource(ph.io.ThreadBufferedSource):

    def __init__(self,
                 coll,
                 mapping: utils.IndexMapping,
                 random_order,
                 batch_size):
        ds = ph.io.MongoSource(['_id', 'data', 'label_index'], coll, random_order=random_order)
        ds = ph.io.BatchSource(ds, batch_size)
        self._mapping = mapping
        super(DataSource, self).__init__(ds, buffer_size=100, num_thread=4)
        self._aug_filter = ph.utils.image.default_augmentation_filter()

    def _next(self, row):
        _id, data, label_index = row
        image = [
            self._aug_filter(ph.utils.image.load_as_array(data_i))
            for data_i in data
        ]
        target = [
            self._mapping.to_target(label_index_i)
            for label_index_i in label_index
        ]
        return _id, image, target


class Main(ph.Application):

    def _main(self, args):
        ph.set_tf_log_level(ph.TF_LOG_NO_WARN)
        with pymongo.MongoClient('sis2.ustcdm.org') as conn:
            conn['admin'].authenticate('root', 'SELECT * FROM users;')

            coll_train = conn['imagenet_vgg']['train']
            # coll_valid = conn['imagenet_vgg']['valid']

            mapping = utils.IndexMapping(conn['imagenet_deepme']['clusters'])

            ds_train = DataSource(coll_train, mapping, True, args.batch_size)
            # ds_valid = DataSource(coll_valid, mapping, False, args.batch_size)

            model = Model('model', args.keep_prob)
            ph.initialize_global_variables()
            ph.io.load_model_from_file(model['encoder'], args.vgg16, 'vgg16')

            #
            # train the last layer
            progress = tqdm(total=args.num_train, ncols=96, desc='Training')
            for i in range(args.num_train):
                self.checkpoint()
                try:
                    _, image, target = ds_train.next()
                except StopIteration:
                    _, image, target = ds_train.next()
                loss, = model.train(image, target)
                progress.set_description(f'Training loss={loss:.06f}', refresh=False)
                progress.update()
            progress.close()

            #
            # fine tuning all the parameters
            progress = tqdm(total=args.num_loops, ncols=96, desc='Fine tuning')
            for i in range(args.num_loops):
                self.checkpoint()
                try:
                    _, image, target = ds_train.next()
                except StopIteration:
                    _, image, target = ds_train.next()
                loss, = model.fine_tune(image, target)
                progress.set_description(f'Fine tuning loss={loss:.06f}', refresh=False)
                progress.update()
            progress.close()
            ds_train = None
            ds_valid = None

            if args.write_results:
                coll = conn['imagenet_vgg']['test']
                coll_output = conn['imagenet_deepme']['gate_test']
                self._write_result(model, coll, mapping, coll_output)

        print('All clear.')
        return 0

    @staticmethod
    def _write_result(model, coll, mapping, coll_output):
        ds_test = DataSource(coll, mapping, False, 96)
        bar = tqdm(total=coll.count(), ncols=96, desc=f'Predicting for {coll.name}')
        buffer = []
        for _id, image, target in ds_test:
            y, = model.predict(image)
            for _id_i, label_i, y_i in zip(_id, target, y):
                doc = {
                    '_id': _id_i,
                    'index': pickle.dumps(label_i),
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
    _parser.add_argument('--batch-size', type=int, default=96)
    _parser.add_argument('--num-train', type=int, default=5000)
    _parser.add_argument('--num-loops', type=int, default=50000)
    _parser.add_argument('--vgg16', required=True)
    _parser.add_argument('--keep-prob', type=float, default=1.0)
    _parser.add_argument('--write-results', action='store_true', default=False)
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(Main().run(_args))
