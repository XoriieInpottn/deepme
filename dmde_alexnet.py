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
from photinia.apps.imagenet import alexnet


class Model(ph.Model):

    def __init__(self,
                 name,
                 hidden_size,
                 num_classes,
                 keep_prob,
                 reg,
                 grad_clip,
                 learning_rate_1,
                 learning_rate_2,
                 num_loops_1,
                 num_loops_2):
        self._hidden_size = hidden_size
        self._num_classes = num_classes
        self._keep_prob = keep_prob
        self._reg = reg
        self._grad_clip = grad_clip
        self._learning_rate_1 = learning_rate_1
        self._learning_rate_2 = learning_rate_2
        self._num_loops_1 = num_loops_1
        self._num_loops_2 = num_loops_2
        super(Model, self).__init__(name)

    def _build(self):
        input_image = ph.placeholder('input_image', (None, alexnet.HEIGHT, alexnet.WIDTH, 3), ph.float)
        encoder = alexnet.AlexNet('encoder', ph.ops.swish)
        dropout = ph.Dropout('dropout')
        dense = ph.Linear('dense', encoder['dense_7'].output_size, self._hidden_size)
        output_layer = ph.Linear('output_layer', dense.output_size, self._num_classes + 1)

        encoder.setup(input_image)
        y = ph.setup(
            encoder['feature_7'], [
                dense, ph.ops.swish,
                dropout,
                output_layer, tf.nn.softmax
            ]
        )
        label = tf.argmax(y, axis=1)

        self.predict = ph.Step(
            inputs=input_image,
            outputs=(label, y),
            givens={dropout.keep_prob: 1.0}
        )

        input_label = ph.placeholder('input_label', (None,), ph.int)
        y_target = tf.one_hot(input_label, self._num_classes + 1)
        loss = -ph.ops.log_likelihood(y_target, y)
        loss = tf.reduce_mean(loss)

        ################################################################################
        # pre-train
        ################################################################################
        vars_new = [
            *dense.get_trainable_variables(),
            *output_layer.get_trainable_variables()
        ]
        reg = ph.reg.L2Regularizer(self._reg)
        reg.setup(vars_new)
        lr = ph.train.ExponentialDecayedValue(
            'lr_1',
            init_value=self._learning_rate_1,
            num_loops=self._num_loops_1,
            min_value=self._learning_rate_1 / 10
        )
        update_1 = tf.train.AdamOptimizer(lr.value).apply_gradients([
            (tf.clip_by_value(g, -self._grad_clip, self._grad_clip), v)
            for g, v in zip(tf.gradients(loss + reg.get_loss(), vars_new), vars_new) if g is not None
        ])
        # with tf.control_dependencies([update_1]):
        #     update_2 = ph.train.L2Regularizer(self._reg).apply(vars_new)
        self.train = ph.Step(
            inputs=(input_image, input_label),
            outputs=(loss, lr.variable),
            updates=update_1,
            givens={dropout.keep_prob: self._keep_prob}
        )

        ################################################################################
        # fine tune
        ################################################################################
        vars_all = self.get_trainable_variables()
        reg = ph.reg.L2Regularizer(self._reg)
        reg.setup(vars_all)
        lr = ph.train.ExponentialDecayedValue(
            'lr_2',
            init_value=self._learning_rate_2,
            num_loops=self._num_loops_2,
            min_value=self._learning_rate_2 / 10
        )
        update_1 = tf.train.AdamOptimizer(lr.value).apply_gradients([
            (tf.clip_by_value(g, -self._grad_clip, self._grad_clip), v)
            for g, v in zip(tf.gradients(loss + reg.get_loss(), vars_all), vars_all) if g is not None
        ])
        # with tf.control_dependencies([update_1]):
        #     update_2 = ph.train.L2Regularizer(self._reg).apply(vars_all)
        self.fine_tune = ph.Step(
            inputs=(input_image, input_label),
            outputs=(loss, lr.variable),
            updates=update_1,
            givens={dropout.keep_prob: self._keep_prob}
        )


class DataSource(ph.io.DataSource):

    def __init__(self, coll, random_order, batch_size):
        ds = ph.io.MongoSource(
            ('_id', 'data', 'label_index'),
            coll=coll,
            random_order=random_order
        )
        ds = ph.io.BatchSource(ds, batch_size)
        ds = ph.io.ThreadBufferedSource(ds, fn=self._fn, num_thread=5, buffer_size=10)
        self._ds = ds
        self._random_order = random_order
        if random_order:
            self._aug_filter = ph.utils.image.default_augmentation_filter()
        super(DataSource, self).__init__(('_id', 'image', 'label'))

    def next(self):
        return self._ds.next()

    def _fn(self, row):
        _id, data, label_index = row
        image = [ph.utils.image.load_as_array(data_i, size=(227, 227)) for data_i in data]
        if self._random_order:
            image = [self._aug_filter(image_i) for image_i in image]
        return self._data_model(_id, image, label_index)


class Main(ph.Application):

    def _main(self, args):
        ph.set_tf_log_level(ph.TF_LOG_NO_WARN)
        with pymongo.MongoClient('sis3.ustcdm.org') as conn:
            conn['admin'].authenticate('root', 'SELECT * FROM users;')

            ################################################################################
            # define data source and init model
            ################################################################################
            db = conn['imagenet_dmde']
            coll_train = db[f'task_{args.task_index:02d}_train']
            coll_valid = db[f'task_{args.task_index:02d}_valid']
            num_classes = len(coll_train.distinct('label_index'))
            print(f'Found {num_classes} classes.')

            ds_train = DataSource(coll_train, True, args.batch_size)
            ds_valid = DataSource(coll_valid, False, args.batch_size)

            model = Model(
                'model',
                hidden_size=args.hidden_size,
                num_classes=num_classes,
                keep_prob=args.keep_prob,
                reg=args.reg,
                grad_clip=args.grad_clip,
                learning_rate_1=args.learning_rate_1,
                learning_rate_2=args.learning_rate_2,
                num_loops_1=args.num_loops_1,
                num_loops_2=args.num_loops_2
            )
            ph.initialize_global_variables()
            ph.io.load_model_from_file(model['encoder'], args.alexnet, 'alexnet')

            ################################################################################
            # pre-train
            ################################################################################
            progress = tqdm(total=args.num_loops_1, ncols=96)
            for i in range(args.num_loops_1):
                self.checkpoint()
                try:
                    _, image, label = ds_train.next()
                except StopIteration:
                    _, image, label = ds_train.next()
                loss, lr = model.train(image, label)
                progress.set_description(f'Pre-train loss={loss:.03e}, lr={lr:.03e}', refresh=False)
                progress.update()
            progress.close()

            progress = tqdm(total=coll_valid.count(), ncols=96, desc='Validating')
            cal = ph.train.AccCalculator()
            for _, image, label in ds_valid:
                label_pred, _ = model.predict(image)
                cal.update(label_pred, label)
                progress.update(len(image))
            progress.close()
            print(f'Validation acc={cal.accuracy}')

            ################################################################################
            # fine tune
            ################################################################################
            progress = tqdm(total=args.num_loops_2, ncols=96)
            monitor = ph.train.EarlyStopping(5, model)
            for i in range(args.num_loops_2):
                self.checkpoint()
                try:
                    _, image, label = ds_train.next()
                except StopIteration:
                    _, image, label = ds_train.next()
                loss, lr = model.fine_tune(image, label)
                progress.set_description(f'Fine tune loss={loss:.03e}, lr={lr:.03e}', refresh=False)

                if (i + 1) % 1000 == 0:
                    progress_valid = tqdm(total=coll_valid.count(), ncols=96, desc='Validating')
                    cal = ph.train.AccCalculator()
                    for _, image, label in ds_valid:
                        label_pred, _ = model.predict(image)
                        cal.update(label_pred, label)
                        progress_valid.update(len(image))
                    progress_valid.close()
                    progress.clear()
                    print(f'[{i + 1}] Validation acc={cal.accuracy}')
                    if monitor.convergent(1 - cal.accuracy):
                        model.set_parameters(monitor.best_parameters)
                        break
                progress.update()
            progress.close()
            ds_train = None
            ds_valid = None

            ################################################################################
            # write results
            ################################################################################
            if args.write_results:
                coll = conn['imagenet']['imagenet_10k_224_train']
                coll_output = db[f'alexnet_result_{args.task_index:02d}_train']
                self._write_result(model, coll, coll_output)

                coll = conn['imagenet']['imagenet_10k_224_valid']
                coll_output = db[f'alexnet_result_{args.task_index:02d}_valid']
                self._write_result(model, coll, coll_output)

                coll = conn['imagenet']['imagenet_10k_224_test']
                coll_output = db[f'alexnet_result_{args.task_index:02d}_test']
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

    _parser.add_argument('--num-loops-1', type=int, default=5000)
    _parser.add_argument('--num-loops-2', type=int, default=100000)

    _parser.add_argument('--keep-prob', type=float, required=True)
    _parser.add_argument('--reg', type=float, required=True)
    _parser.add_argument('--grad-clip', type=float, required=True)
    _parser.add_argument('--learning-rate-1', type=float, required=True)
    _parser.add_argument('--learning-rate-2', type=float, required=True)

    _parser.add_argument('--hidden-size', type=int, required=True)
    _parser.add_argument('--task-index', type=int, required=True)
    _parser.add_argument('--alexnet', required=True)
    _parser.add_argument('--write-results', action='store_true', default=False)
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(Main().run(_args))
