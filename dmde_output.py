#!/usr/bin/env python3


"""
@author: xi
@since: 2019-02-25
"""

import argparse
import os
import pickle

import pymongo
import tensorflow as tf
from tqdm import tqdm

import photinia as ph


class Model(ph.Model):

    def __init__(self, name, input_size, num_classes, keep_prob):
        self._input_size = input_size
        self._num_classes = num_classes
        self._keep_prob = keep_prob
        self._hidden_size = 4096
        super(Model, self).__init__(name)

    def _build(self):
        x = ph.placeholder('x', shape=(None, self._input_size), dtype=ph.float)

        hidden_layer = ph.Linear(
            'hidden_layer',
            input_size=self._input_size,
            output_size=self._hidden_size
        )
        out_layer = ph.Linear(
            'out_layer',
            input_size=self._hidden_size,
            output_size=self._num_classes
        )
        dropout = ph.Dropout('dropout')

        y = ph.setup(
            x, [
                hidden_layer, ph.ops.lrelu,
                dropout,
                out_layer, tf.nn.softmax
            ]
        )
        label = tf.argmax(y, axis=1)

        self.predict = ph.Step(
            inputs=x,
            outputs=(label, y),
            givens={dropout.keep_prob: 1.0}
        )

        true_label = ph.placeholder('true_label', shape=(None,), dtype=ph.int)
        target = tf.one_hot(true_label, self._num_classes)
        loss = ph.ops.cross_entropy(target, y)
        loss = tf.reduce_mean(loss)

        var_list = self.get_trainable_variables()
        reg = ph.reg.L2Regularizer(1e-6)
        reg.setup(var_list)
        grad_list = [
            tf.clip_by_value(grad, -10, 10)
            for grad in tf.gradients(loss + reg.get_loss(), var_list)
        ]
        lr = ph.train.ExponentialDecayedValue('lr_train', 1e-4, num_loops=2e4, min_value=1e-6)
        update = tf.train.AdamOptimizer(lr.value).apply_gradients(zip(grad_list, var_list))
        self.train = ph.Step(
            inputs=(x, true_label),
            outputs=loss,
            updates=(update, lr.update_op),
            givens={dropout.keep_prob: self._keep_prob}
        )


class DataSource(ph.io.ThreadBufferedSource):

    def __init__(self, coll, random_order, batch_size):
        ds = ph.io.MongoSource(['_id', 'y', 'label_index'], coll, random_order=random_order)
        ds = ph.io.BatchSource(ds, batch_size)
        super(DataSource, self).__init__(ds, buffer_size=10000)

    def _next(self, row):
        _id, y, label_index = row
        y = [pickle.loads(a) for a in y]
        return _id, y, label_index


class Main(ph.Application):

    def _main(self, args):
        with pymongo.MongoClient('sis4.ustcdm.org') as conn:
            conn['admin'].authenticate('root', 'SELECT * FROM users;')
            db = conn['imagenet_dmde']
            coll_train = db['fusion_train']
            coll_valid = db['fusion_valid']
            coll_test = db['fusion_test']

            ds_train = DataSource(coll_train, True, args.batch_size)
            ds_valid = DataSource(coll_valid, False, args.batch_size)

            model = Model('model', 10184, 10184, args.keep_prob)
            ph.initialize_global_variables()

            progress = tqdm(total=args.num_loops, ncols=96, desc='Training')
            monitor = ph.train.EarlyStopping(5, model)
            for i in range(args.num_loops):
                self.checkpoint()
                try:
                    _, x, label = ds_train.next()
                except StopIteration:
                    _, x, label = ds_train.next()
                loss, = model.train(x, label)
                progress.set_description(f'Training loss={loss:.06f}', refresh=False)

                if (i + 1) % 1000 == 0:
                    progress_valid = tqdm(total=coll_valid.count(), ncols=96, desc='Validating')
                    cal = ph.train.AccCalculator()
                    for _, x, label in ds_valid:
                        label_pred, _ = model.predict(x)
                        cal.update(label_pred, label)
                        progress_valid.update(len(x))
                    progress_valid.close()
                    progress.clear()
                    print(f'[{i + 1}] Validation acc={cal.accuracy}')
                    if monitor.convergent(1 - cal.accuracy):
                        model.set_parameters(monitor.best_parameters)
                        break

                progress.update()
            progress.close()

            if args.write_results:
                coll_output = conn['imagenet_deepme']['final_dmde']
                self._write_result(model, coll_test, coll_output)

        return 0

    @staticmethod
    def _write_result(model, coll, coll_output):
        ds_test = DataSource(coll, False, 96)
        bar = tqdm(total=coll.count(), ncols=96, desc=f'Predicting for {coll.name}')
        buffer = []
        cal = ph.train.AccCalculator()
        for _id, x, label_index in ds_test:
            pred_index, y = model.predict(x)
            cal.update(pred_index, label_index)
            for _id_i, label_i, y_i in zip(_id, label_index, y):
                doc = {
                    '_id': _id_i,
                    'label_index': label_i,
                    'y': pickle.dumps(y_i)
                }
                buffer.append(doc)
                if len(buffer) >= 1000:
                    bar.set_description(f'Predicting acc={cal.accuracy}', False)
                    coll_output.insert_many(buffer)
                    buffer.clear()
            bar.update(len(x))
        bar.close()
        print(f'Final acc={cal.accuracy}')
        if len(buffer) != 0:
            coll_output.insert_many(buffer)
            buffer.clear()


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-size', type=int, default=256)
    _parser.add_argument('--num-loops', type=int, default=5200)
    _parser.add_argument('--keep-prob', type=float, default=1.0)
    _parser.add_argument('--write-results', action='store_true', default=False)
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(Main().run(_args))
