import argparse
import mxnet as mx
import os, sys
import numpy as np
import logging

from crossentropy import *
import data_io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


def get_fine_tune_model(batch_size, sentence_size, vocab_size, num_embed, num_classes):
    input_x = mx.sym.Variable('data')
    input_y = mx.sym.Variable('softmax_label')

    # embedding layer
    embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
    conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, sentence_size, num_embed))

    conv1 = mx.symbol.Convolution(data=conv_input, kernel=(2,2), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    flatten = mx.symbol.Flatten(data=relu3)
    net = mx.symbol.FullyConnected(data = flatten, num_hidden = 128)

    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.sigmoid(data=net, name='sig')
    net = mx.symbol.Custom(data=net, name='softmax', op_type='CrossEntropyLoss')

    return net


def multi_factor_scheduler(begin_epoch, epoch_size, step=[5,10], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def train_model(gpus, epoch, num_epoch=20, kv='device', num_class=6, batch_size=50):
    num_embed = 300
    train_iter, valid_iter, sentence_size, embed_size, vocab_size = data_io.data_iter(64, num_embed)

    new_sym = get_fine_tune_model(batch_size, sentence_size, vocab_size, num_embed, num_class)
    
    num_examples = 36212
    epoch_size = max(int(num_examples / batch_size / kv.num_workers), 1)
    lr_scheduler=multi_factor_scheduler(0, epoch_size)

    optimizer_params = {
            'learning_rate': 0.0005,
            'momentum' : 0.9,
            'wd' : 0.0001,
            'lr_scheduler': lr_scheduler}
    
    initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]
        
    model = mx.mod.Module(
        context       = devs,
        symbol        = new_sym
    )
    
    
    save_result = os.path.join(os.path.dirname("__file__"), "result_model")
    checkpoint = mx.callback.do_checkpoint(save_result+"/check_pp")

    def acc(label, pred, label_width = num_class):
        return float((label == np.round(pred)).sum()) / label_width / pred.shape[0]

    def loss(label, pred):
        loss_all = 0
        for i in range(len(pred)):
            loss = 0
            loss -= label[i] * np.log(pred[i] + 1e-6) + (1.- label[i]) * np.log(1. + 1e-6 - pred[i])
            loss_all += np.sum(loss)
        loss_all = float(loss_all)/float(len(pred) + 0.000001)
        return  loss_all


    eval_metric = list()
    eval_metric.append(mx.metric.np(acc))
    eval_metric.append(mx.metric.np(loss))

    model.fit(train_iter,
              begin_epoch=epoch,
              num_epoch=num_epoch,
              eval_data=valid_iter,
              eval_metric=eval_metric,
              validation_metric=eval_metric,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              initializer=initializer,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(batch_size, 20),
              epoch_end_callback=checkpoint)
if __name__ == '__main__':
    """

    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus',          type=str, default='0')
    parser.add_argument('--batch-size',    type=int, default=200)
    parser.add_argument('--epoch',         type=int, default=0)
    parser.add_argument('--image-shape',   type=str, default='3,224,224')
    parser.add_argument('--num-classes',   type=int, default=6)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--num-epoch',     type=int, default=2)
    parser.add_argument('--kv-store',      type=str, default='device', help='the kvstore type')
    parser.add_argument('--num-examples',  type=int, default=20000)
    parser.add_argument('--mom',           type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=0.0001, help='weight decay for sgd')
    args = parser.parse_args()
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    kv = mx.kvstore.create("device")
    save_result = os.path.join(os.path.dirname("__file__"), "result_model")
    if not os.path.exists(save_result):
        os.mkdir(args.save_result)

    train_model(gpus="0", epoch=1, num_epoch=12, kv=kv, num_class=15, batch_size=64)